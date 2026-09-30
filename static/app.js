(() => {
  "use strict";

  const STORAGE_KEY = "web_img_yolo_sam_url";
  const SESSION_KEY = "web_img_yolo_session_id";
  const DEFAULT_SAM_URL = "http://127.0.0.1:8080";
  const SAM_URL_RE = /^https?:\/\/.+/i;
  const IMAGE_EXT = /\.(jpe?g|png|webp|bmp)$/i;
  const CLASS_HELMET = 0;
  const COLOR_HELMET = "#e2b15a";
  const COLOR_HEAD = "#79c3d4";
  const COLOR_DROP = "#e25b45";

  const $ = (s) => document.querySelector(s);

  let files = [];
  let samRunning = false;
  let pollTimer = null;

  const V = {
    sessionId: null,
    frames: [],
    frameIdx: 0,
    frame: null,
    detId: null,
    validation: null,
    finalized: false,
  };

  function toast(msg, type = "info") {
    const el = document.createElement("div");
    el.className = `toast toast--${type === "error" ? "err" : type === "ok" ? "ok" : ""}`;
    el.textContent = msg;
    $("#toasts").appendChild(el);
    setTimeout(() => el.remove(), 4500);
  }

  function normalizeSamUrl(raw) {
    return (raw || "").trim().replace(/\/+$/, "");
  }

  function apiUrl() {
    const u = normalizeSamUrl($("#apiUrl").value);
    return u || DEFAULT_SAM_URL;
  }

  function validateSamUrl(showToast = false) {
    const u = normalizeSamUrl($("#apiUrl").value);
    const input = $("#apiUrl");
    if (!u) {
      input.classList.remove("input--invalid");
      return { ok: true, url: DEFAULT_SAM_URL };
    }
    if (!SAM_URL_RE.test(u)) {
      input.classList.add("input--invalid");
      if (showToast) {
        toast("Адрес SAM API должен начинаться с http:// или https://", "error");
      }
      return { ok: false, url: u };
    }
    input.classList.remove("input--invalid");
    return { ok: true, url: u };
  }

  function saveSettings() {
    const v = validateSamUrl(false);
    if (v.ok) localStorage.setItem(STORAGE_KEY, v.url);
  }

  function loadSettings() {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) $("#apiUrl").value = saved;
    validateSamUrl(false);
  }

  function samHeaders(extra = {}) {
    return { "X-SAM3-Base-URL": apiUrl(), ...extra };
  }

  function setSetupLocked(locked) {
    samRunning = locked;
    $("#btnRunSam").disabled = locked || files.length === 0;
    $("#folderInput").disabled = locked;
    document.querySelectorAll(".setup-fields input, #apiUrl, #subjectPrompt, #probePrompt").forEach((el) => {
      el.disabled = locked;
    });
  }

  function showView(name) {
    $("#viewSetup").classList.toggle("view--hidden", name !== "setup");
    $("#viewValidation").classList.toggle("view--hidden", name !== "validation");
  }

  async function checkHealth() {
    const v = validateSamUrl(true);
    if (!v.ok) {
      $("#healthStatus").textContent = "Неверный URL";
      $("#healthStatus").style.color = "var(--err)";
      return;
    }
    saveSettings();
    $("#healthStatus").textContent = "…";
    try {
      const r = await fetch("/proxy/health", { headers: samHeaders() });
      if (!r.ok) throw new Error(String(r.status));
      $("#healthStatus").textContent = "SAM доступен";
      $("#healthStatus").style.color = "var(--ok)";
    } catch {
      $("#healthStatus").textContent = "SAM недоступен";
      $("#healthStatus").style.color = "var(--err)";
    }
  }

  function filterImages(fileList) {
    return Array.from(fileList).filter((f) => IMAGE_EXT.test(f.name));
  }

  function onFolderPick(ev) {
    if (samRunning) return;
    files = filterImages(ev.target.files);
    files.sort((a, b) => a.webkitRelativePath.localeCompare(b.webkitRelativePath));
    $("#folderInfo").textContent = files.length
      ? `Файлов: ${files.length}`
      : "Нет подходящих изображений";
    $("#btnRunSam").disabled = files.length === 0;

    const row = $("#thumbRow");
    row.innerHTML = "";
    const max = Math.min(24, files.length);
    for (let i = 0; i < max; i++) {
      const img = document.createElement("img");
      img.alt = files[i].name;
      img.src = URL.createObjectURL(files[i]);
      img.onload = () => URL.revokeObjectURL(img.src);
      row.appendChild(img);
    }
  }

  function updateStats(stats) {
    if (!stats) return;
    $("#statPos").textContent = stats.positive_boxes ?? 0;
    $("#statNeg").textContent = stats.negative_boxes ?? 0;
    $("#statSkip").textContent = stats.skipped_no_heads ?? 0;
    $("#statErr").textContent = stats.errors ?? 0;
  }

  function applySamProgress(prog, stats) {
    if (!prog) return;
    const total = prog.total || 0;
    const done = prog.completed || 0;
    const pct = prog.percent ?? (total ? Math.round((100 * done) / total) : 0);
    $("#samProgressWrap").hidden = false;
    $("#progressBar").style.width = `${pct}%`;
    $("#progressText").textContent = `${done} / ${total} (${pct}%)`;
    const cur = prog.current_file || "";
    $("#progressFile").textContent = cur ? `Файл: ${cur}` : done >= total && total ? "Готово" : "";
    if (stats) updateStats(stats);
  }

  function startPoll(sessionId) {
    stopPoll();
    pollTimer = setInterval(async () => {
      if (!samRunning) return;
      try {
        const r = await fetch(`/api/sessions/${sessionId}`);
        if (!r.ok) return;
        const data = await r.json();
        applySamProgress(data.sam_progress, data.stats);
      } catch {
        /* ignore */
      }
    }, 350);
  }

  function stopPoll() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  async function runSam() {
    const v = validateSamUrl(true);
    if (!v.ok || samRunning || !files.length) return;
    saveSettings();
    setSetupLocked(true);
    $("#downloadLink").hidden = true;
    updateStats({ positive_boxes: 0, negative_boxes: 0, skipped_no_heads: 0, errors: 0 });
    applySamProgress({ total: files.length, completed: 0, percent: 0, current_file: "" }, null);

    const fd = new FormData();
    fd.append("val_ratio", $("#valRatio").value);
    fd.append("seed", $("#seed").value);
    fd.append("subject_prompt", $("#subjectPrompt").value);
    fd.append("probe_prompt", $("#probePrompt").value);
    fd.append("threshold", $("#threshold").value);
    fd.append("sam_base_url", apiUrl());

    let sessionId;
    try {
      const cr = await fetch("/api/sessions", { method: "POST", body: fd });
      if (!cr.ok) throw new Error(await cr.text());
      sessionId = (await cr.json()).session_id;
    } catch (e) {
      toast("Не удалось создать сессию: " + e.message, "error");
      setSetupLocked(false);
      return;
    }

    V.sessionId = sessionId;
    sessionStorage.setItem(SESSION_KEY, sessionId);

    try {
      const bf = new FormData();
      bf.append("total", String(files.length));
      await fetch(`/api/sessions/${sessionId}/sam/begin`, { method: "POST", body: bf });
    } catch (e) {
      toast("SAM begin: " + e.message, "error");
      setSetupLocked(false);
      return;
    }

    startPoll(sessionId);
    const total = files.length;
    let stats = { positive_boxes: 0, negative_boxes: 0, skipped_no_heads: 0, errors: 0 };

    for (let i = 0; i < total; i++) {
      const file = files[i];
      applySamProgress(
        {
          total,
          completed: i,
          percent: total ? Math.round((100 * i) / total) : 0,
          current_file: file.name,
        },
        stats
      );
      const body = new FormData();
      body.append("file", file, file.name);
      body.append("relative_path", file.webkitRelativePath || file.name);
      try {
        const pr = await fetch(`/api/sessions/${sessionId}/process`, { method: "POST", body });
        const data = await pr.json();
        if (data.stats) stats = data.stats;
        if (data.sam_progress) applySamProgress(data.sam_progress, stats);
        if (!data.ok && data.error) console.warn(file.name, data.error);
      } catch (e) {
        stats.errors = (stats.errors || 0) + 1;
        updateStats(stats);
      }
    }

    stopPoll();
    try {
      const done = await fetch(`/api/sessions/${sessionId}/sam/complete`, { method: "POST" });
      const fin = await done.json();
      applySamProgress(fin.sam_progress || { total, completed: total, percent: 100 }, stats);
      if (!fin.frame_count) {
        toast("Нет кадров с головами — валидация невозможна", "error");
        setSetupLocked(false);
        return;
      }
      toast(`SAM готов: ${fin.frame_count} кадров для проверки`, "ok");
      await enterValidation(sessionId);
    } catch (e) {
      toast("Завершение SAM: " + e.message, "error");
    }
    setSetupLocked(false);
  }

  function renderValStats() {
    const val = V.validation || {};
    $("#valStats").innerHTML = `
      <span class="pill-stat">кадров <b>${val.frames || 0}</b></span>
      <span class="pill-stat pill--helmet">Helmet <b>${val.kept_helmet || 0}</b></span>
      <span class="pill-stat pill--head">person_head <b>${val.kept_person_head || 0}</b></span>
      <span class="pill-stat">в датасете <b>${val.kept_total || 0}</b></span>
      <span class="pill-stat pill--drop">снято <b>${val.dropped || 0}</b></span>
    `;
  }

  async function loadFrameList() {
    const r = await fetch(`/api/sessions/${V.sessionId}/frames`);
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    V.frames = data.frames || [];
    V.validation = data.validation;
    renderValStats();
    paintFilmstrip();
  }

  function paintFilmstrip() {
    const strip = $("#filmstrip");
    strip.innerHTML = "";
    V.frames.forEach((f, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "film-thumb";
      if (i === V.frameIdx) btn.classList.add("is-on");
      if (f.has_dropped) btn.classList.add("has-drop");
      btn.title = f.source_name || f.stem;
      btn.innerHTML = `<span class="film-thumb__n">${i + 1}</span><span class="film-thumb__c">${f.kept_count}/${f.box_count}</span>`;
      btn.addEventListener("click", () => goFrame(i));
      strip.appendChild(btn);
    });
  }

  async function loadFrame(idx) {
    if (!V.frames.length) return;
    V.frameIdx = Math.max(0, Math.min(idx, V.frames.length - 1));
    const fi = V.frames[V.frameIdx].index;
    const r = await fetch(`/api/sessions/${V.sessionId}/frames/${fi}`);
    if (!r.ok) throw new Error(await r.text());
    V.frame = await r.json();
    V.validation = V.frame.validation;
    renderValStats();
    if (!V.detId || !V.frame.detections.some((d) => d.id === V.detId)) {
      V.detId = (V.frame.detections.find((d) => d.kept) || V.frame.detections[0])?.id || null;
    }
    const img = $("#frameImg");
    img.onload = () => scheduleOverlay();
    img.src = `${V.frame.image_url}?t=${Date.now()}`;
    $("#frameLabel").textContent = `${V.frameIdx + 1} / ${V.frames.length} · ${V.frame.source_name || ""}`;
    paintFilmstrip();
    paintDetList();
    scheduleOverlay();
  }

  function goFrame(i) {
    loadFrame(i).catch((e) => toast(e.message, "error"));
  }

  function updateDetButtons() {
    const det = V.frame?.detections.find((d) => d.id === V.detId);
    const drop = $("#btnDetDrop");
    if (drop) drop.textContent = det && !det.kept ? "Вернуть бокс" : "Удалить бокс";
  }

  function paintDetList() {
    const ul = $("#detList");
    ul.innerHTML = "";
    if (!V.frame) return;
    V.frame.detections.forEach((d) => {
      const li = document.createElement("li");
      li.className = "det-item";
      if (!d.kept) li.classList.add("is-drop");
      if (d.id === V.detId) li.classList.add("is-on");
      const cls = d.class_id === CLASS_HELMET ? "Helmet" : "person_head";
      li.innerHTML = `<span class="det-item__cls det-item__cls--${d.class_id}">${cls}</span>
        <span class="det-item__id">${d.id}</span>`;
      li.addEventListener("click", () => {
        V.detId = d.id;
        paintDetList();
        scheduleOverlay();
      });
      ul.appendChild(li);
    });
    updateDetButtons();
  }

  function frameViewBox() {
    const view = $("#frameView");
    const img = $("#frameImg");
    if (!view || !img.naturalWidth) return null;
    const vw = view.clientWidth;
    const vh = view.clientHeight;
    const iw = img.naturalWidth;
    const ih = img.naturalHeight;
    const scale = Math.min(vw / iw, vh / ih);
    const w = iw * scale;
    const h = ih * scale;
    const left = (vw - w) / 2;
    const top = (vh - h) / 2;
    return { scale, left, top, w, h, vw, vh, iw, ih };
  }

  function scheduleOverlay() {
    requestAnimationFrame(drawOverlay);
  }

  function drawOverlay() {
    const slot = frameViewBox();
    const canvas = $("#frameCanvas");
    const frame = V.frame;
    if (!slot || !frame) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(slot.vw * dpr));
    canvas.height = Math.max(1, Math.round(slot.vh * dpr));
    canvas.style.width = `${slot.vw}px`;
    canvas.style.height = `${slot.vh}px`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, slot.vw, slot.vh);
    frame.detections.forEach((d) => {
      const [x1, y1, x2, y2] = d.bbox_xyxy || [0, 0, 0, 0];
      const x = slot.left + x1 * slot.scale;
      const y = slot.top + y1 * slot.scale;
      const w = (x2 - x1) * slot.scale;
      const h = (y2 - y1) * slot.scale;
      const pos = d.class_id === CLASS_HELMET;
      ctx.lineWidth = d.id === V.detId ? 3 : 2;
      ctx.globalAlpha = d.kept ? 1 : 0.35;
      ctx.strokeStyle = d.kept ? (pos ? COLOR_HELMET : COLOR_HEAD) : COLOR_DROP;
      ctx.setLineDash(d.kept ? [] : [6, 5]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    });
  }

  async function patchDet(body) {
    if (!V.sessionId || !V.detId || V.finalized) return;
    const r = await fetch(`/api/sessions/${V.sessionId}/detections/${encodeURIComponent(V.detId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    V.validation = data.validation;
    const det = V.frame.detections.find((d) => d.id === V.detId);
    if (det && data.detection) Object.assign(det, data.detection);
    renderValStats();
    paintDetList();
    paintFilmstrip();
    scheduleOverlay();
  }

  function toggleKept() {
    const det = V.frame?.detections.find((d) => d.id === V.detId);
    if (!det) return;
    patchDet({ kept: !det.kept }).catch((e) => toast(e.message, "error"));
  }

  function toggleClass() {
    const det = V.frame?.detections.find((d) => d.id === V.detId);
    if (!det) return;
    const next = det.class_id === CLASS_HELMET ? 1 : CLASS_HELMET;
    patchDet({ class_id: next }).catch((e) => toast(e.message, "error"));
  }

  async function buildYolo() {
    if (!V.sessionId || V.finalized) return;
    $("#btnBuildYolo").disabled = true;
    try {
      const r = await fetch(`/api/sessions/${V.sessionId}/finalize`, { method: "POST" });
      if (!r.ok) throw new Error(await r.text());
      const fin = await r.json();
      V.finalized = true;
      $("#buildResult").textContent = `Train: ${fin.counts.train}, val: ${fin.counts.val}, боксов: ${fin.counts.boxes}`;
      const link = $("#downloadLink");
      link.href = fin.download_url;
      link.hidden = false;
      toast("YOLO датасет собран", "ok");
    } catch (e) {
      toast("Сборка: " + e.message, "error");
    }
    $("#btnBuildYolo").disabled = V.finalized;
  }

  async function enterValidation(sessionId) {
    V.sessionId = sessionId;
    V.frameIdx = 0;
    V.detId = null;
    V.finalized = false;
    $("#downloadLink").hidden = true;
    $("#buildResult").textContent = "";
    $("#btnBuildYolo").disabled = false;
    showView("validation");
    await loadFrameList();
    if (!V.frames.length) {
      toast("Нет кадров для валидации", "error");
      showView("setup");
      return;
    }
    await loadFrame(0);
  }

  async function tryResumeSession() {
    const sid = sessionStorage.getItem(SESSION_KEY);
    if (!sid) return;
    try {
      const r = await fetch(`/api/sessions/${sid}`);
      if (!r.ok) return;
      const data = await r.json();
      if (data.finalized) return;
      if ((data.validation?.frames || 0) > 0) {
        await enterValidation(sid);
      }
    } catch {
      /* ignore */
    }
  }

  $("#folderInput").addEventListener("change", onFolderPick);
  $("#btnHealth").addEventListener("click", checkHealth);
  $("#btnRunSam").addEventListener("click", runSam);
  $("#btnBackSetup").addEventListener("click", () => showView("setup"));
  $("#btnPrev").addEventListener("click", () => goFrame(V.frameIdx - 1));
  $("#btnNext").addEventListener("click", () => goFrame(V.frameIdx + 1));
  $("#btnBuildYolo").addEventListener("click", buildYolo);
  $("#btnDetDrop").addEventListener("click", toggleKept);
  $("#btnDetClass").addEventListener("click", toggleClass);

  $("#frameView").addEventListener("click", (ev) => {
    const slot = frameViewBox();
    if (!slot || !V.frame) return;
    const rect = $("#frameView").getBoundingClientRect();
    const cx = ev.clientX - rect.left;
    const cy = ev.clientY - rect.top;
    let hit = null;
    let best = Infinity;
    V.frame.detections.forEach((d) => {
      const [x1, y1, x2, y2] = d.bbox_xyxy || [];
      const x = slot.left + x1 * slot.scale;
      const y = slot.top + y1 * slot.scale;
      const w = (x2 - x1) * slot.scale;
      const h = (y2 - y1) * slot.scale;
      if (cx >= x && cx <= x + w && cy >= y && cy <= y + h) {
        const area = w * h;
        if (area < best) {
          best = area;
          hit = d.id;
        }
      }
    });
    if (hit) {
      V.detId = hit;
      paintDetList();
      scheduleOverlay();
    }
  });

  window.addEventListener("keydown", (ev) => {
    if ($("#viewValidation").classList.contains("view--hidden")) return;
    if (ev.key === "ArrowLeft") {
      ev.preventDefault();
      goFrame(V.frameIdx - 1);
    } else if (ev.key === "ArrowRight") {
      ev.preventDefault();
      goFrame(V.frameIdx + 1);
    } else if (ev.key === "Delete" || ev.key === "Backspace") {
      ev.preventDefault();
      toggleKept();
    } else if (ev.key === "c" || ev.key === "C") {
      ev.preventDefault();
      toggleClass();
    }
  });

  window.addEventListener("resize", scheduleOverlay);

  $("#apiUrl").addEventListener("input", () => validateSamUrl(false));
  $("#apiUrl").addEventListener("change", saveSettings);
  $("#apiUrl").addEventListener("blur", saveSettings);

  loadSettings();
  tryResumeSession();
})();
