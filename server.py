"""Folder → YOLO dataset via SAM3 HTTP API (boxes only)."""

from __future__ import annotations

import base64
import io
import json
import os
import random
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from yolo_logic import (
    CLASS_ID_NEGATIVE,
    CLASS_ID_POSITIVE,
    CLASS_NAMES,
    DEFAULT_PROBE_PROMPT,
    DEFAULT_SUBJECT_PROMPT,
    class_label,
    classify_subject_boxes,
    detection_yolo_line,
    objects_for_phrase,
)

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
OUTPUT_DIR = ROOT / "output"
DEFAULT_SAM3_URL = os.environ.get("SAM3_API_URL", "http://127.0.0.1:8080").rstrip("/")
UI_PORT = int(os.environ.get("WEB_IMG_YOLO_PORT", "8091"))

app = FastAPI(title="WEB IMG → YOLO", docs_url=None, redoc_url=None)
_sessions: dict[str, dict[str, Any]] = {}


class DetectionPatch(BaseModel):
    kept: bool | None = None
    class_id: int | None = Field(None, ge=0, le=1)


def _target_base(request: Request) -> str:
    header = request.headers.get("x-sam3-base-url", "").strip()
    return (header or DEFAULT_SAM3_URL).rstrip("/")


def _session_dir(session_id: str) -> Path:
    path = OUTPUT_DIR / session_id
    if not path.is_dir():
        raise HTTPException(404, "session not found")
    return path


def _load_session(session_id: str) -> dict[str, Any]:
    if session_id in _sessions:
        return _sessions[session_id]
    path = OUTPUT_DIR / session_id / "session.json"
    if not path.is_file():
        raise HTTPException(404, "session not found")
    meta = json.loads(path.read_text(encoding="utf-8"))
    _sessions[session_id] = meta
    return meta


def _save_meta(session_id: str) -> None:
    meta = _sessions.get(session_id)
    if not meta:
        return
    (OUTPUT_DIR / session_id / "session.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _default_sam_progress() -> dict[str, Any]:
    return {"phase": "idle", "total": 0, "completed": 0, "current_file": ""}


def _sam_progress_public(meta: dict[str, Any]) -> dict[str, Any]:
    prog = meta.get("sam_progress") or _default_sam_progress()
    total = int(prog.get("total") or 0)
    completed = int(prog.get("completed") or 0)
    pct = round(100 * completed / total) if total > 0 else 0
    if prog.get("phase") == "running" and prog.get("current_file") and total > 0:
        # Show partial progress while the current file is on SAM.
        pct = min(99, round(100 * (completed + 0.35) / total))
    return {
        "phase": str(prog.get("phase") or "idle"),
        "total": total,
        "completed": completed,
        "current_file": str(prog.get("current_file") or ""),
        "percent": pct,
    }


def _sam_begin(meta: dict[str, Any], total: int) -> None:
    meta["sam_progress"] = {
        "phase": "running",
        "total": max(0, int(total)),
        "completed": 0,
        "current_file": "",
    }


def _sam_mark_current(meta: dict[str, Any], filename: str) -> None:
    prog = meta.setdefault("sam_progress", _default_sam_progress())
    if prog.get("phase") == "running":
        prog["current_file"] = filename


def _sam_finish_file(meta: dict[str, Any]) -> None:
    prog = meta.setdefault("sam_progress", _default_sam_progress())
    if prog.get("phase") == "running":
        prog["completed"] = int(prog.get("completed", 0)) + 1
        prog["current_file"] = ""


def _sam_complete(meta: dict[str, Any]) -> None:
    prog = meta.setdefault("sam_progress", _default_sam_progress())
    total = int(prog.get("total") or 0)
    prog["phase"] = "done"
    prog["completed"] = max(int(prog.get("completed", 0)), total)
    prog["current_file"] = ""


def _serialize_detection(d: dict[str, Any]) -> dict[str, Any]:
    cid = int(d.get("class_id", CLASS_ID_NEGATIVE))
    if cid not in (CLASS_ID_POSITIVE, CLASS_ID_NEGATIVE):
        cid = CLASS_ID_NEGATIVE
    return {
        "id": str(d["id"]),
        "bbox_xyxy": d.get("bbox_xyxy"),
        "class_id": cid,
        "label": class_label(cid),
        "kept": bool(d.get("kept", True)),
    }


def _validation_stats(meta: dict[str, Any]) -> dict[str, int]:
    kept_pos = kept_neg = dropped = 0
    frames_with_kept = 0
    frames = meta.get("frames") or []
    for frame in frames:
        k = 0
        for d in frame.get("detections") or []:
            if d.get("kept", True):
                k += 1
                if int(d.get("class_id", CLASS_ID_NEGATIVE)) == CLASS_ID_POSITIVE:
                    kept_pos += 1
                else:
                    kept_neg += 1
            else:
                dropped += 1
        if k > 0:
            frames_with_kept += 1
    return {
        "frames": len(frames),
        "frames_with_kept": frames_with_kept,
        "kept_helmet": kept_pos,
        "kept_person_head": kept_neg,
        "kept_total": kept_pos + kept_neg,
        "dropped": dropped,
    }


def _find_detection(meta: dict[str, Any], det_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for frame in meta.get("frames") or []:
        for det in frame.get("detections") or []:
            if str(det.get("id")) == det_id:
                return frame, det
    return None, None


def _write_yaml(path: Path, dataset_root: Path, names: list[str]) -> None:
    lines = [
        f"path: {dataset_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx, name in enumerate(names):
        safe = str(name).replace('"', "'")
        lines.append(f'  {idx}: "{safe}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def _sam_predict_json(
    *,
    sam_base: str,
    image_bytes: bytes,
    subject_prompt: str,
    probe_prompt: str,
    threshold: float,
    image_id: str,
) -> dict[str, Any]:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    prompt = f"{subject_prompt.strip()}, {probe_prompt.strip()}"
    body = {
        "segmentation": False,
        "threshold": threshold,
        "prompt": "",
        "images": [
            {
                "id": image_id,
                "image": b64,
                "prompt": prompt,
            }
        ],
    }
    url = f"{sam_base.rstrip('/')}/v1/predict/json"
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=15.0)) as client:
        resp = await client.post(url, json=body)
    if resp.status_code >= 400:
        detail = resp.text[:500]
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            pass
        raise HTTPException(502, f"SAM3 error {resp.status_code}: {detail}")
    data = resp.json()
    results = data.get("results") or []
    if not results:
        raise HTTPException(502, "SAM3 returned no results")
    return results[0]


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.api_route("/proxy/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(path: str, request: Request) -> Response:
    base = _target_base(request)
    url = f"{base}/{path}"
    body = await request.body()
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length", "x-sam3-base-url", "connection"}
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            upstream = await client.request(
                request.method,
                url,
                content=body,
                headers=headers,
                params=dict(request.query_params),
            )
    except httpx.ConnectError:
        return JSONResponse(
            {"detail": f"Cannot connect to SAM3 API at {base}"},
            status_code=502,
        )
    except httpx.TimeoutException:
        return JSONResponse({"detail": "SAM3 API request timed out"}, status_code=504)

    skip = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in skip}
    return Response(content=upstream.content, status_code=upstream.status_code, headers=resp_headers)


@app.post("/api/sessions")
async def create_session(
    val_ratio: float = Form(0.2),
    seed: int = Form(42),
    subject_prompt: str = Form(DEFAULT_SUBJECT_PROMPT),
    probe_prompt: str = Form(DEFAULT_PROBE_PROMPT),
    threshold: float = Form(0.5),
    sam_base_url: str = Form(DEFAULT_SAM3_URL),
) -> dict[str, Any]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_id = ts
    staging = OUTPUT_DIR / session_id / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    meta = {
        "session_id": session_id,
        "val_ratio": min(0.5, max(0.05, float(val_ratio))),
        "seed": int(seed),
        "subject_prompt": subject_prompt.strip() or DEFAULT_SUBJECT_PROMPT,
        "probe_prompt": probe_prompt.strip() or DEFAULT_PROBE_PROMPT,
        "threshold": min(0.99, max(0.0, float(threshold))),
        "sam_base_url": (sam_base_url or DEFAULT_SAM3_URL).strip().rstrip("/"),
        "classes": CLASS_NAMES,
        "frames": [],
        "stats": {
            "processed": 0,
            "positive_boxes": 0,
            "negative_boxes": 0,
            "skipped_no_heads": 0,
            "errors": 0,
        },
        "sam_progress": _default_sam_progress(),
        "finalized": False,
    }
    _sessions[session_id] = meta
    _save_meta(session_id)
    return {"session_id": session_id}


@app.post("/api/sessions/{session_id}/sam/begin")
async def sam_begin(session_id: str, total: int = Form(...)) -> dict[str, Any]:
    meta = _load_session(session_id)
    if meta.get("finalized"):
        raise HTTPException(400, "session already finalized")
    _sam_begin(meta, total)
    _save_meta(session_id)
    return {"ok": True, "sam_progress": _sam_progress_public(meta)}


@app.post("/api/sessions/{session_id}/sam/complete")
async def sam_complete(session_id: str) -> dict[str, Any]:
    meta = _load_session(session_id)
    _sam_complete(meta)
    _save_meta(session_id)
    return {
        "ok": True,
        "sam_progress": _sam_progress_public(meta),
        "validation": _validation_stats(meta),
        "frame_count": len(meta.get("frames") or []),
    }


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    meta = _load_session(session_id)
    return {
        "session_id": session_id,
        "finalized": bool(meta.get("finalized")),
        "classes": meta.get("classes") or CLASS_NAMES,
        "stats": meta.get("stats") or {},
        "sam_progress": _sam_progress_public(meta),
        "validation": _validation_stats(meta),
        "counts": meta.get("counts"),
        "download_url": f"/api/sessions/{session_id}/download" if meta.get("finalized") else None,
    }


@app.get("/api/sessions/{session_id}/frames")
async def list_frames(session_id: str) -> dict[str, Any]:
    meta = _load_session(session_id)
    items: list[dict[str, Any]] = []
    for frame in meta.get("frames") or []:
        dets = frame.get("detections") or []
        kept = sum(1 for d in dets if d.get("kept", True))
        items.append(
            {
                "index": int(frame["index"]),
                "stem": frame.get("stem"),
                "source_name": frame.get("source_name"),
                "box_count": len(dets),
                "kept_count": kept,
                "has_dropped": any(not d.get("kept", True) for d in dets),
            }
        )
    return {
        "session_id": session_id,
        "frames": items,
        "validation": _validation_stats(meta),
        "classes": meta.get("classes") or CLASS_NAMES,
    }


@app.get("/api/sessions/{session_id}/frames/{frame_index}")
async def get_frame(session_id: str, frame_index: int) -> dict[str, Any]:
    meta = _load_session(session_id)
    frames = meta.get("frames") or []
    frame = next((f for f in frames if int(f.get("index", -1)) == frame_index), None)
    if not frame:
        raise HTTPException(404, "frame not found")
    return {
        "index": int(frame["index"]),
        "stem": frame.get("stem"),
        "source_name": frame.get("source_name"),
        "width": int(frame.get("width") or 0),
        "height": int(frame.get("height") or 0),
        "image_url": f"/api/sessions/{session_id}/image/{frame_index}",
        "detections": [_serialize_detection(d) for d in frame.get("detections") or []],
        "validation": _validation_stats(meta),
        "classes": meta.get("classes") or CLASS_NAMES,
    }


@app.get("/api/sessions/{session_id}/image/{frame_index}")
async def get_frame_image(session_id: str, frame_index: int) -> FileResponse:
    meta = _load_session(session_id)
    frames = meta.get("frames") or []
    frame = next((f for f in frames if int(f.get("index", -1)) == frame_index), None)
    if not frame:
        raise HTTPException(404, "frame not found")
    stem = str(frame.get("stem") or "")
    suffix = str(frame.get("suffix") or ".jpg")
    path = OUTPUT_DIR / session_id / "staging" / f"{stem}{suffix}"
    if not path.is_file():
        raise HTTPException(404, "image not found")
    return FileResponse(path)


@app.patch("/api/sessions/{session_id}/detections/{det_id}")
async def patch_detection(
    session_id: str,
    det_id: str,
    body: DetectionPatch,
) -> dict[str, Any]:
    meta = _load_session(session_id)
    if meta.get("finalized"):
        raise HTTPException(400, "сессия уже собрана — правки недоступны")
    _frame, det = _find_detection(meta, det_id)
    if not det:
        raise HTTPException(404, "detection not found")
    if body.kept is not None:
        det["kept"] = bool(body.kept)
    if body.class_id is not None:
        cid = int(body.class_id)
        if cid not in (CLASS_ID_POSITIVE, CLASS_ID_NEGATIVE):
            raise HTTPException(400, "class_id must be 0 or 1")
        det["class_id"] = cid
    _save_meta(session_id)
    return {
        "ok": True,
        "detection": _serialize_detection(det),
        "validation": _validation_stats(meta),
    }


@app.post("/api/sessions/{session_id}/process")
async def process_image(
    session_id: str,
    file: UploadFile = File(...),
    relative_path: str = Form(""),
) -> dict[str, Any]:
    meta = _load_session(session_id)
    if meta.get("finalized"):
        raise HTTPException(400, "session already finalized")

    stats = meta["stats"]
    name = (relative_path or file.filename or "image").replace("\\", "/").split("/")[-1]
    _sam_mark_current(meta, name)
    _save_meta(session_id)

    finished = False

    def _finish_once() -> None:
        nonlocal finished
        if finished:
            return
        finished = True
        stats["processed"] = int(stats.get("processed", 0)) + 1
        _sam_finish_file(meta)
        _save_meta(session_id)

    stem = Path(name).stem
    suffix = Path(name).suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"

    try:
        raw = await file.read()
        if not raw:
            raise ValueError("empty file")
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            width, height = img.size
            if img.mode != "RGB":
                img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=95)
                raw = buf.getvalue()
                suffix = ".jpg"

        sam_result = await _sam_predict_json(
            sam_base=meta["sam_base_url"],
            image_bytes=raw,
            subject_prompt=meta["subject_prompt"],
            probe_prompt=meta["probe_prompt"],
            threshold=meta["threshold"],
            image_id=name,
        )
        w = int(sam_result.get("width") or width)
        h = int(sam_result.get("height") or height)
        objects = sam_result.get("objects") or []
        heads = objects_for_phrase(objects, meta["subject_prompt"])
        helmets = objects_for_phrase(objects, meta["probe_prompt"])
        if not heads:
            stats["skipped_no_heads"] = int(stats.get("skipped_no_heads", 0)) + 1
            _finish_once()
            return {
                "ok": True,
                "skipped": True,
                "reason": "no_subject_boxes",
                "stats": stats,
                "sam_progress": _sam_progress_public(meta),
            }
        classified = classify_subject_boxes(heads, helmets)
        detections: list[dict[str, Any]] = []
        for i, (subject, _lab, cid) in enumerate(classified):
            box = subject.get("bbox_xyxy") or []
            if len(box) < 4:
                continue
            line = detection_yolo_line(
                {"kept": True, "class_id": cid, "bbox_xyxy": box},
                w,
                h,
            )
            if line is None:
                continue
            if cid == CLASS_ID_POSITIVE:
                stats["positive_boxes"] = int(stats.get("positive_boxes", 0)) + 1
            else:
                stats["negative_boxes"] = int(stats.get("negative_boxes", 0)) + 1
            detections.append(
                {
                    "bbox_xyxy": [float(x) for x in box[:4]],
                    "class_id": cid,
                    "kept": True,
                    "_i": i,
                }
            )

        if not detections:
            stats["skipped_no_heads"] = int(stats.get("skipped_no_heads", 0)) + 1
            _finish_once()
            return {
                "ok": True,
                "skipped": True,
                "reason": "no_valid_subject_boxes",
                "stats": stats,
                "sam_progress": _sam_progress_public(meta),
            }

        frame_index = len(meta["frames"])
        unique_stem = f"{stem}_{frame_index:05d}"
        for det in detections:
            det["id"] = f"{unique_stem}_d{det.pop('_i')}"

        staging = OUTPUT_DIR / session_id / "staging"
        img_path = staging / f"{unique_stem}{suffix}"
        img_path.write_bytes(raw)
        meta["frames"].append(
            {
                "index": frame_index,
                "stem": unique_stem,
                "suffix": suffix,
                "width": w,
                "height": h,
                "source_name": name,
                "detections": detections,
            }
        )
        _finish_once()
        return {
            "ok": True,
            "skipped": False,
            "frame_index": frame_index,
            "boxes": len(detections),
            "stats": stats,
            "validation": _validation_stats(meta),
            "sam_progress": _sam_progress_public(meta),
        }
    except HTTPException:
        stats["errors"] = int(stats.get("errors", 0)) + 1
        _finish_once()
        raise
    except Exception as exc:
        stats["errors"] = int(stats.get("errors", 0)) + 1
        _finish_once()
        return {
            "ok": False,
            "error": str(exc),
            "stats": stats,
            "sam_progress": _sam_progress_public(meta),
        }


@app.post("/api/sessions/{session_id}/finalize")
async def finalize_session(session_id: str) -> dict[str, Any]:
    meta = _load_session(session_id)
    if meta.get("finalized"):
        return {
            "ok": True,
            "session_id": session_id,
            "already": True,
            "counts": meta.get("counts"),
            "download_url": f"/api/sessions/{session_id}/download",
        }

    eligible: list[dict[str, Any]] = []
    for frame in meta.get("frames") or []:
        kept = [d for d in frame.get("detections") or [] if d.get("kept", True)]
        if not kept:
            continue
        eligible.append({**frame, "_kept": kept})

    if not eligible:
        raise HTTPException(400, "нет кадров с оставшимися боксами после валидации")

    root = OUTPUT_DIR / session_id
    staging = root / "staging"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(meta.get("seed", 42)))
    shuffled = list(eligible)
    rng.shuffle(shuffled)
    ratio = float(meta.get("val_ratio", 0.2))
    n_val = max(1, int(round(len(shuffled) * ratio))) if len(shuffled) > 1 else 0
    if n_val >= len(shuffled):
        n_val = len(shuffled) - 1
    val_stems = {f["stem"] for f in shuffled[:n_val]}
    counts = {"train": 0, "val": 0, "boxes": 0}

    for frame in shuffled:
        split = "val" if frame["stem"] in val_stems else "train"
        stem = frame["stem"]
        suffix = frame["suffix"]
        src_img = staging / f"{stem}{suffix}"
        if not src_img.is_file():
            continue
        w = int(frame.get("width") or 0)
        h = int(frame.get("height") or 0)
        lines: list[str] = []
        for det in frame["_kept"]:
            line = detection_yolo_line(det, w, h)
            if line:
                lines.append(line)
        if not lines:
            continue
        dest_img = root / "images" / split / f"{stem}{suffix}"
        dest_lbl = root / "labels" / split / f"{stem}.txt"
        shutil.copy2(src_img, dest_img)
        dest_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts[split] += 1
        counts["boxes"] += len(lines)

    if counts["train"] < 1:
        raise HTTPException(400, "после split пустой train")
    if counts["val"] < 1 and len(shuffled) > 1:
        raise HTTPException(400, "после split пустой val")

    _write_yaml(root / "data.yaml", root, CLASS_NAMES)
    meta["finalized"] = True
    meta["counts"] = counts
    _save_meta(session_id)
    shutil.rmtree(staging, ignore_errors=True)
    return {
        "ok": True,
        "session_id": session_id,
        "classes": CLASS_NAMES,
        "counts": counts,
        "stats": meta["stats"],
        "validation": _validation_stats(meta),
        "download_url": f"/api/sessions/{session_id}/download",
    }


@app.get("/api/sessions/{session_id}/download")
async def download_session(session_id: str) -> FileResponse:
    root = _session_dir(session_id)
    if not (root / "data.yaml").is_file():
        raise HTTPException(400, "сначала нажмите «Собрать YOLO»")

    zip_path = root / "dataset.zip"
    if zip_path.is_file():
        zip_path.unlink()
    parts = ("images/train", "images/val", "labels/train", "labels/val")
    jpeg = {".jpg", ".jpeg", ".png"}
    with zipfile.ZipFile(zip_path, "w", allowZip64=True) as zf:
        yaml_text = (root / "data.yaml").read_text(encoding="utf-8")
        portable = "\n".join(
            "path: ." if line.startswith("path:") else line for line in yaml_text.splitlines()
        )
        zf.writestr("data.yaml", portable + "\n", compress_type=zipfile.ZIP_DEFLATED)
        for rel in parts:
            folder = root / rel
            if not folder.is_dir():
                continue
            for file in folder.iterdir():
                if not file.is_file():
                    continue
                compress = zipfile.ZIP_STORED if file.suffix.lower() in jpeg else zipfile.ZIP_DEFLATED
                zf.write(file, f"{rel}/{file.name}", compress_type=compress)
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"yolo_{session_id}.zip",
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def run() -> None:
    import uvicorn

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run("server:app", host="0.0.0.0", port=UI_PORT, reload=False, log_level="info")


if __name__ == "__main__":
    run()
