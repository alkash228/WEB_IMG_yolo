"""YOLO class semantics and bbox intersection (boxes-only ST3 analogue)."""

from __future__ import annotations

from typing import Any

# Studio defaults (Stage/ST3/config.py, web/state.py)
POSITIVE_LABEL = "Helmet"
NEGATIVE_LABEL = "person_head"
DEFAULT_SUBJECT_PROMPT = "person head"
DEFAULT_PROBE_PROMPT = "helmet"
CLASS_NAMES = [POSITIVE_LABEL, NEGATIVE_LABEL]
CLASS_ID_POSITIVE = 0
CLASS_ID_NEGATIVE = 1

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def xyxy_to_yolo(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    if not bbox or len(bbox) < 4 or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    bw = x2 - x1
    bh = y2 - y1
    if bw < 1 or bh < 1:
        return None
    cx = min(1.0, max(0.0, (x1 + x2) / 2.0 / width))
    cy = min(1.0, max(0.0, (y1 + y2) / 2.0 / height))
    nw = min(1.0, max(1.0 / width, bw / width))
    nh = min(1.0, max(1.0 / height, bh / height))
    return cx, cy, nw, nh


def bbox_intersection_area(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def classify_subject_boxes(
    subjects: list[dict[str, Any]],
    probes: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], str, int]]:
    """Mirror ST3 classify_by_intersection: any probe overlap → positive (class 0)."""
    out: list[tuple[dict[str, Any], str, int]] = []
    for subject in subjects:
        box = subject.get("bbox_xyxy") or []
        if len(box) < 4:
            continue
        best_area = 0.0
        for probe in probes:
            pbox = probe.get("bbox_xyxy") or []
            if len(pbox) < 4:
                continue
            area = bbox_intersection_area(box, pbox)
            if area > best_area:
                best_area = area
        if best_area > 0:
            out.append((subject, POSITIVE_LABEL, CLASS_ID_POSITIVE))
        else:
            out.append((subject, NEGATIVE_LABEL, CLASS_ID_NEGATIVE))
    return out


def objects_for_phrase(objects: list[dict[str, Any]], phrase: str) -> list[dict[str, Any]]:
    phrase = phrase.strip()
    return [o for o in objects if str(o.get("prompt") or "").strip() == phrase]


def label_lines(
    classified: list[tuple[dict[str, Any], str, int]],
    width: int,
    height: int,
) -> list[str]:
    lines: list[str] = []
    for subject, _label, cid in classified:
        yolo = xyxy_to_yolo(list(subject.get("bbox_xyxy") or []), width, height)
        if yolo is None:
            continue
        lines.append(f"{cid} {yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}")
    return lines


def detection_yolo_line(det: dict[str, Any], width: int, height: int) -> str | None:
    """One kept detection → YOLO label line (subject box, class from review)."""
    if not det.get("kept", True):
        return None
    cid = int(det.get("class_id", CLASS_ID_NEGATIVE))
    if cid not in (CLASS_ID_POSITIVE, CLASS_ID_NEGATIVE):
        cid = CLASS_ID_NEGATIVE
    yolo = xyxy_to_yolo(list(det.get("bbox_xyxy") or []), width, height)
    if yolo is None:
        return None
    return f"{cid} {yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}"


def class_label(class_id: int) -> str:
    return POSITIVE_LABEL if int(class_id) == CLASS_ID_POSITIVE else NEGATIVE_LABEL
