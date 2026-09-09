const ANNOTATION_VIEW_DEFAULTS = Object.freeze({
  contrast: 1.25,
  brightness: 1.0,
  gamma: 1.0,
  invert: false,
});

const state = {
  jobs: [],
  selected: null,
  logOffset: null,
  autoScroll: true,
  cryosparcConnected: false,
  liveSummary: {
    mode: "all",
    count: 100,
  },
  annotation: {
    sessions: [],
    session: null,
    row: null,
    index: 0,
    image: null,
    polygons: [],
    currentPoints: [],
    mode: "polygon",
    action: "add",
    ellipseDrag: null,
    moveDrag: null,
    rotateDrag: null,
    selectedShapeIndex: null,
    selectedShapeIndices: [],
    hoverShapeIndex: null,
    previewShape: null,
    suppressNextClick: false,
    dirty: false,
    loading: false,
    view: { ...ANNOTATION_VIEW_DEFAULTS },
    viewByKey: {},
    adjustedImage: null,
  },
};

const ANNOTATION_TYPE_COLORS = {
  "": "#c9b7ff",
  1: "#D95F02",
  2: "#1B9E77",
  3: "#CC79A7",
  4: "#E6AB02",
};
const ANNOTATION_ERASE_COLOR = "#E05A5A";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || response.statusText);
  }
  return payload;
}

function formPayload(form) {
  const payload = {};
  for (const element of Array.from(form.elements)) {
    if (!element.name || element.disabled) continue;
    if (element.type === "checkbox") {
      payload[element.name] = element.checked;
      continue;
    }
    const value = element.value.trim();
    if (value) payload[element.name] = value;
  }
  return payload;
}

function setCryosparcStatus(message, tone = "") {
  const element = $("#cryosparcConnectStatus");
  if (!element) return;
  element.textContent = message || "";
  element.classList.toggle("ok", tone === "ok");
  element.classList.toggle("error", tone === "error");
}

function setCryosparcGate(connected, details = {}) {
  const gate = $("#cryosparcGate");
  const runForm = $("#cryosparcRunForm");
  const summary = $("#cryosparcConnectionSummary");
  if (!gate || !runForm) return;
  state.cryosparcConnected = Boolean(connected);
  gate.classList.toggle("is-locked", !connected);
  runForm.setAttribute("aria-hidden", connected ? "false" : "true");
  if (!connected) {
    summary.textContent = "Not connected";
    return;
  }
  const user = details.email ? ` as ${details.email}` : "";
  const version = details.server_version ? ` | ${details.server_version}` : "";
  summary.textContent = `Connected to ${details.display || details.host || "CryoSPARC"}${user}${version}`;
}

function syncCryosparcRunCredentials(payload) {
  for (const form of $$("form[data-uses-cryosparc-connection], #cryosparcRunForm")) {
    for (const name of ["cryosparc_base_url", "cryosparc_email", "cryosparc_password"]) {
      const target = form.elements[name];
      if (target) target.value = payload[name] || "";
    }
  }
}

async function connectCryosparc(form) {
  const payload = formPayload(form);
  const button = form.querySelector('button[type="submit"]');
  button.disabled = true;
  setCryosparcStatus("Checking connection...");
  try {
    const result = await api("/api/cryosparc/connect", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    syncCryosparcRunCredentials(payload);
    if (form.elements.cryosparc_password) form.elements.cryosparc_password.value = "";
    setCryosparcGate(true, result);
    setCryosparcStatus("Connected.", "ok");
    $("#cryosparcRunForm input[name='project']")?.focus();
  } catch (error) {
    setCryosparcStatus(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

function resetCryosparcConnection() {
  syncCryosparcRunCredentials({});
  const connectForm = $("#cryosparcConnectForm");
  if (connectForm?.elements.cryosparc_password) connectForm.elements.cryosparc_password.value = "";
  setCryosparcGate(false);
  setCryosparcStatus("");
  $("#cryosparcConnectForm input[name='cryosparc_base_url']")?.focus();
}

function updateAnnotationSourceFields() {
  const form = $("#annotationForm");
  if (!form) return;
  const mode = form.elements.source_mode?.value || "micrographs";
  $$("[data-source-field]", form).forEach((label) => {
    const visible = label.dataset.sourceField === mode;
    label.hidden = !visible;
    for (const element of Array.from(label.querySelectorAll("input, select"))) {
      element.disabled = !visible;
    }
  });
}

function updateTrainingMicrographSourceFields() {
  const form = $("#training form[data-kind='train']");
  if (!form) return;
  const mode = form.elements.mic_source_mode?.value || "manifest";
  $$("[data-train-mic-source]", form).forEach((label) => {
    const visible = label.dataset.trainMicSource === mode;
    label.hidden = !visible;
    for (const element of Array.from(label.querySelectorAll("input, select"))) {
      element.disabled = !visible;
    }
  });
}

function populateFineTuneFromAnnotationSession(summary) {
  const form = $("#training form[data-kind='train']");
  if (!form || !summary) return;
  if (summary.manifest_edited_path && form.elements.manifest) {
    form.elements.manifest.value = summary.manifest_edited_path;
  }
  const mode = form.elements.mic_source_mode;
  if (summary.source_mode === "cryosparc" && summary.cryosparc_project && summary.cryosparc_workspace && summary.micrographs) {
    if (mode) mode.value = "cryosparc";
    if (form.elements.cryosparc_project) form.elements.cryosparc_project.value = summary.cryosparc_project;
    if (form.elements.cryosparc_workspace) form.elements.cryosparc_workspace.value = summary.cryosparc_workspace;
    if (form.elements.cryosparc_micrographs) form.elements.cryosparc_micrographs.value = summary.micrographs;
  } else if (mode && !form.elements.mic_dir?.value) {
    mode.value = "manifest";
  }
  updateTrainingMicrographSourceFields();
}

function setAnnotationLaunchStatus(message, tone = "") {
  const element = $("#annotationLaunchStatus");
  if (!element) return;
  element.textContent = message || "";
  element.classList.toggle("ok", tone === "ok");
  element.classList.toggle("error", tone === "error");
}

async function launchAnnotationSession(form, payload) {
  if (payload.source_mode === "cryosparc" && !state.cryosparcConnected) {
    resetCryosparcConnection();
    setCryosparcStatus("Connect to CryoSPARC before creating an annotation session.", "error");
    document.querySelector('[data-tab="cryosparc"]').click();
    return;
  }
  setAnnotationLaunchStatus("Creating annotation session...");
  const session = await api("/api/annotation/sessions", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  state.annotation.session = session;
  state.annotation.index = annotationStartIndex(session);
  $("#annotationBrowser").hidden = false;
  setAnnotationLaunchStatus("Session ready.", "ok");
  rememberAnnotationSession(session);
  await loadAnnotationSessions();
  await loadAnnotationRow(state.annotation.index);
  form.querySelector("[name='manifest']")?.blur();
}

function annotationStartIndex(session) {
  const rowCount = Math.max(1, Number(session?.row_count || 1));
  const rawIndex = session?.first_unsaved_index ?? session?.next_index ?? 0;
  const index = Number(rawIndex);
  if (!Number.isFinite(index)) return 0;
  return Math.max(0, Math.min(Math.floor(index), rowCount - 1));
}

function annotationSessionLabel(session) {
  const title = session.title || session.id || "Annotation session";
  const saved = Number(session.saved_count || 0).toLocaleString();
  const total = Number(session.row_count || 0).toLocaleString();
  const updated = parseDate(session.updated_at);
  const stamp = updated
    ? updated.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })
    : "saved session";
  return `${title} | ${saved}/${total} saved | ${stamp}`;
}

async function loadAnnotationSessions() {
  const select = $("#annotationSessionSelect");
  if (!select) return;
  try {
    const payload = await api("/api/annotation/sessions");
    state.annotation.sessions = payload.sessions || [];
    renderAnnotationSessions();
  } catch (error) {
    state.annotation.sessions = [];
    renderAnnotationSessions(error.message);
  }
}

function renderAnnotationSessions(errorMessage = "") {
  const select = $("#annotationSessionSelect");
  const button = $("#annotationResumeSession");
  if (!select || !button) return;
  const sessions = state.annotation.sessions || [];
  if (!sessions.length) {
    select.innerHTML = `<option value="">No recent sessions</option>`;
    select.disabled = true;
    button.disabled = true;
    button.title = errorMessage || "Start a session to resume it later.";
    return;
  }
  const selectedId = state.annotation.session?.id || select.value || sessions[0].id;
  select.disabled = false;
  button.disabled = false;
  button.title = "Resume the selected annotation session.";
  select.innerHTML = sessions.map((session) => `
    <option value="${escapeHtml(session.id)}" ${session.id === selectedId ? "selected" : ""}>
      ${escapeHtml(annotationSessionLabel(session))}
    </option>
  `).join("");
}

function rememberAnnotationSession(session) {
  if (!session?.id) return;
  const rest = (state.annotation.sessions || []).filter((item) => item.id !== session.id);
  state.annotation.sessions = [session, ...rest];
  renderAnnotationSessions();
}

async function resumeAnnotationSession() {
  const sessionId = $("#annotationSessionSelect")?.value || "";
  if (!sessionId) return;
  if (state.annotation.dirty && !window.confirm("Discard unsaved edits for this row?")) return;
  setAnnotationLaunchStatus("Resuming annotation session...");
  try {
    const session = await api(`/api/annotation/sessions/${sessionId}`);
    state.annotation.session = session;
    state.annotation.index = annotationStartIndex(session);
    state.annotation.currentPoints = [];
    state.annotation.ellipseDrag = null;
    state.annotation.moveDrag = null;
    state.annotation.rotateDrag = null;
    state.annotation.selectedShapeIndex = null;
    state.annotation.selectedShapeIndices = [];
    state.annotation.hoverShapeIndex = null;
    state.annotation.previewShape = null;
    state.annotation.dirty = false;
    $("#annotationBrowser").hidden = false;
    rememberAnnotationSession(session);
    setAnnotationLaunchStatus("Session resumed.", "ok");
    await loadAnnotationRow(state.annotation.index);
  } catch (error) {
    setAnnotationLaunchStatus(error.message, "error");
  }
}

function annotationTypeId() {
  const value = $("#annotationType")?.value || "";
  return value ? Number(value) : null;
}

function annotationMode() {
  const mode = $("#annotationMode")?.value || state.annotation.mode || "polygon";
  return mode === "ellipse" ? "ellipse" : "polygon";
}

function annotationAction() {
  const action = $("#annotationAction")?.value || state.annotation.action || "add";
  return action === "erase" ? "erase" : "add";
}

function annotationTypeColor(typeId) {
  return ANNOTATION_TYPE_COLORS[typeId ?? ""] || ANNOTATION_TYPE_COLORS[""];
}

function annotationDrawColor(typeId = annotationTypeId(), mode = annotationAction()) {
  return mode === "erase" ? ANNOTATION_ERASE_COLOR : annotationTypeColor(typeId);
}

function syncAnnotationActionControls() {
  const type = $("#annotationType");
  if (!type) return;
  const erase = annotationAction() === "erase";
  type.disabled = erase;
  type.title = erase ? "Erase mode subtracts from the saved mask; region type is ignored." : "";
}

function normalizePolygon(polygon) {
  const points = Array.isArray(polygon?.points)
    ? polygon.points
        .filter((point) => Array.isArray(point) && point.length >= 2)
        .map((point) => [Number(point[0]), Number(point[1])])
        .filter((point) => Number.isFinite(point[0]) && Number.isFinite(point[1]))
    : [];
  const shape = polygon?.shape === "ellipse" ? "ellipse" : "polygon";
  const normalized = {
    points,
    type_id: polygon?.type_id == null ? null : Number(polygon.type_id),
    mode: polygon?.mode === "erase" ? "erase" : "add",
    shape,
    rotation: shape === "ellipse" && Number.isFinite(Number(polygon?.rotation)) ? Number(polygon.rotation) : 0,
  };
  const bounds = polygon?.bounds;
  if (bounds && typeof bounds === "object") {
    const x0 = Number(bounds.x0);
    const y0 = Number(bounds.y0);
    const x1 = Number(bounds.x1);
    const y1 = Number(bounds.y1);
    if ([x0, y0, x1, y1].every(Number.isFinite)) {
      normalized.bounds = { x0, y0, x1, y1 };
    }
  }
  return normalized;
}

function sourceLimits() {
  const canvas = $("#annotationCanvas");
  const row = state.annotation.row;
  return {
    width: Math.max(1, Number(row?.source_width || canvas?.width || 1)),
    height: Math.max(1, Number(row?.source_height || canvas?.height || 1)),
  };
}

function normalizedBounds(bounds) {
  if (!bounds || typeof bounds !== "object") return null;
  const limits = sourceLimits();
  const values = {
    x0: Number(bounds.x0),
    y0: Number(bounds.y0),
    x1: Number(bounds.x1),
    y1: Number(bounds.y1),
  };
  if (!Object.values(values).every(Number.isFinite)) return null;
  const clamp = (value, max) => Math.max(0, Math.min(value, max));
  const x0 = clamp(Math.min(values.x0, values.x1), limits.width - 1);
  const x1 = clamp(Math.max(values.x0, values.x1), limits.width - 1);
  const y0 = clamp(Math.min(values.y0, values.y1), limits.height - 1);
  const y1 = clamp(Math.max(values.y0, values.y1), limits.height - 1);
  if (x1 - x0 < 1 || y1 - y0 < 1) return null;
  return { x0, y0, x1, y1 };
}

function boundsFromPoints(points) {
  const usable = (points || []).filter((point) => Array.isArray(point) && point.length >= 2);
  if (!usable.length) return null;
  const xs = usable.map((point) => Number(point[0])).filter(Number.isFinite);
  const ys = usable.map((point) => Number(point[1])).filter(Number.isFinite);
  if (!xs.length || !ys.length) return null;
  return normalizedBounds({
    x0: Math.min(...xs),
    y0: Math.min(...ys),
    x1: Math.max(...xs),
    y1: Math.max(...ys),
  });
}

function ellipseBounds(polygon) {
  if (polygon?.shape !== "ellipse") return null;
  const bounds = normalizedBounds(polygon.bounds) || boundsFromPoints(polygon.points);
  if (bounds && !polygon.bounds) polygon.bounds = bounds;
  return bounds;
}

function ellipseCenter(bounds) {
  return [(bounds.x0 + bounds.x1) / 2, (bounds.y0 + bounds.y1) / 2];
}

function normalizeAngle(angle) {
  const fullTurn = Math.PI * 2;
  return ((Number(angle || 0) + Math.PI) % fullTurn + fullTurn) % fullTurn - Math.PI;
}

function rotateOffset(x, y, rotation) {
  const cos = Math.cos(rotation);
  const sin = Math.sin(rotation);
  return [x * cos - y * sin, x * sin + y * cos];
}

function annotationViewForKey(key) {
  return { ...ANNOTATION_VIEW_DEFAULTS, ...(state.annotation.viewByKey[key] || {}) };
}

function annotationViewSignature() {
  const view = state.annotation.view;
  return [
    state.annotation.row?.key || "",
    state.annotation.image?.src || "",
    $("#annotationCanvas")?.width || 0,
    $("#annotationCanvas")?.height || 0,
    view.contrast,
    view.brightness,
    view.gamma,
    view.invert ? 1 : 0,
  ].join(":");
}

function syncAnnotationViewControls() {
  const view = state.annotation.view;
  const contrast = Math.round(Number(view.contrast || 1) * 100);
  const brightness = Math.round(Number(view.brightness || 1) * 100);
  const gamma = Math.round(Number(view.gamma || 1) * 100);
  const contrastInput = $("#annotationContrast");
  const brightnessInput = $("#annotationBrightness");
  const gammaInput = $("#annotationGamma");
  const invertInput = $("#annotationInvert");
  const contrastValue = $("#annotationContrastValue");
  const brightnessValue = $("#annotationBrightnessValue");
  const gammaValue = $("#annotationGammaValue");
  if (contrastInput) contrastInput.value = String(contrast);
  if (brightnessInput) brightnessInput.value = String(brightness);
  if (gammaInput) gammaInput.value = String(gamma);
  if (invertInput) invertInput.checked = Boolean(view.invert);
  if (contrastValue) contrastValue.textContent = `${contrast}%`;
  if (brightnessValue) brightnessValue.textContent = `${brightness}%`;
  if (gammaValue) gammaValue.textContent = `${gamma}%`;
}

function updateAnnotationViewFromControls() {
  const view = {
    contrast: Number($("#annotationContrast")?.value || 125) / 100,
    brightness: Number($("#annotationBrightness")?.value || 100) / 100,
    gamma: Number($("#annotationGamma")?.value || 100) / 100,
    invert: Boolean($("#annotationInvert")?.checked),
  };
  state.annotation.view = view;
  if (state.annotation.row?.key) {
    state.annotation.viewByKey[state.annotation.row.key] = view;
  }
  state.annotation.adjustedImage = null;
  syncAnnotationViewControls();
  drawAnnotationCanvas();
}

function resetAnnotationView() {
  state.annotation.view = { ...ANNOTATION_VIEW_DEFAULTS };
  if (state.annotation.row?.key) {
    delete state.annotation.viewByKey[state.annotation.row.key];
  }
  state.annotation.adjustedImage = null;
  syncAnnotationViewControls();
  drawAnnotationCanvas();
}

function adjustedAnnotationImage() {
  const image = state.annotation.image;
  const canvas = $("#annotationCanvas");
  if (!image || !canvas) return null;
  const signature = annotationViewSignature();
  if (state.annotation.adjustedImage?.signature === signature) {
    return state.annotation.adjustedImage.canvas;
  }
  const output = document.createElement("canvas");
  output.width = canvas.width;
  output.height = canvas.height;
  const ctx = output.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(image, 0, 0, output.width, output.height);
  const pixels = ctx.getImageData(0, 0, output.width, output.height);
  const data = pixels.data;
  const view = state.annotation.view;
  const contrast = Number(view.contrast || 1);
  const brightness = Number(view.brightness || 1);
  const gamma = Math.max(0.05, Number(view.gamma || 1));
  const invGamma = 1 / gamma;
  for (let index = 0; index < data.length; index += 4) {
    let value = data[index] / 255;
    if (view.invert) value = 1 - value;
    value = ((value - 0.5) * contrast + 0.5) * brightness;
    value = Math.min(1, Math.max(0, value));
    value = Math.pow(value, invGamma);
    const byte = Math.round(Math.min(1, Math.max(0, value)) * 255);
    data[index] = byte;
    data[index + 1] = byte;
    data[index + 2] = byte;
  }
  ctx.putImageData(pixels, 0, 0);
  state.annotation.adjustedImage = { signature, canvas: output };
  return output;
}

async function loadAnnotationRow(index) {
  const session = state.annotation.session;
  if (!session) return;
  const maxIndex = Math.max(0, Number(session.row_count || 1) - 1);
  const nextIndex = Math.max(0, Math.min(Number(index) || 0, maxIndex));
  state.annotation.loading = true;
  updateAnnotationControls();
  setAnnotationRowStatus("Loading...");
  try {
    const row = await api(`/api/annotation/sessions/${session.id}/rows/${nextIndex}`);
    const image = new Image();
    const imageLoaded = new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error("Could not load annotation preview."));
    });
    image.src = `${row.image_url}?max_dim=1800&v=${Date.now()}`;
    await imageLoaded;
    state.annotation.row = row;
    state.annotation.index = nextIndex;
    state.annotation.image = image;
    state.annotation.polygons = Array.isArray(row.annotation?.polygons)
      ? row.annotation.polygons.map(normalizePolygon).filter((polygon) => polygon.points.length >= 3)
      : [];
    state.annotation.currentPoints = [];
    state.annotation.ellipseDrag = null;
    state.annotation.moveDrag = null;
    state.annotation.rotateDrag = null;
    state.annotation.selectedShapeIndex = null;
    state.annotation.selectedShapeIndices = [];
    state.annotation.hoverShapeIndex = null;
    state.annotation.previewShape = null;
    state.annotation.dirty = false;
    state.annotation.view = annotationViewForKey(row.key);
    state.annotation.adjustedImage = null;
    const canvas = $("#annotationCanvas");
    canvas.width = image.naturalWidth || image.width;
    canvas.height = image.naturalHeight || image.height;
    $("#annotationCanvasEmpty").hidden = true;
    $("#annotationJump").value = String(nextIndex + 1);
    $("#annotationJump").max = String(row.row_count || session.row_count || 1);
    syncAnnotationViewControls();
    renderAnnotationMeta();
    drawAnnotationCanvas();
  } catch (error) {
    setAnnotationLaunchStatus(error.message, "error");
    setAnnotationRowStatus("Failed");
  } finally {
    state.annotation.loading = false;
    updateAnnotationControls();
  }
}

async function navigateAnnotationRow(index) {
  if (state.annotation.dirty && !window.confirm("Discard unsaved edits for this row?")) return;
  await loadAnnotationRow(index);
}

function renderAnnotationMeta() {
  const session = state.annotation.session;
  const row = state.annotation.row;
  $("#annotationSessionTitle").textContent = session?.title || "Annotation session";
  if (!session || !row) {
    $("#annotationSessionMeta").textContent = "No image loaded";
    return;
  }
  const saved = Number(session.saved_count || 0).toLocaleString();
  const total = Number(row.row_count || session.row_count || 0).toLocaleString();
  const split = row.split ? ` | ${row.split}` : "";
  $("#annotationSessionMeta").textContent = `row ${row.index + 1}/${total} | saved ${saved}/${total}${split}`;
  $("#annotationRowKey").textContent = row.key || "-";
  $("#annotationRowPath").textContent = formatPathTail(row.micrograph_path || "");
  $("#annotationRowPath").title = row.micrograph_path || "";
  setAnnotationRowStatus(row.saved ? "Saved" : "Unsaved");
}

function setAnnotationRowStatus(message) {
  const element = $("#annotationSaveState");
  if (element) element.textContent = message || "";
}

function annotationRowStatusText() {
  if (state.annotation.dirty) return "Unsaved changes";
  return state.annotation.row?.saved ? "Saved" : "Unsaved";
}

function updateAnnotationControls() {
  const session = state.annotation.session;
  const row = state.annotation.row;
  const busy = state.annotation.loading;
  const maxIndex = Math.max(0, Number(session?.row_count || 1) - 1);
  const disabled = !session || busy;
  const hasTransientShape = Boolean(
    state.annotation.currentPoints.length ||
    state.annotation.previewShape ||
    state.annotation.ellipseDrag ||
    state.annotation.moveDrag ||
    state.annotation.rotateDrag,
  );
  const hasAnnotations = Boolean(state.annotation.polygons.length || hasTransientShape);
  $("#annotationPrev").disabled = disabled || state.annotation.index <= 0;
  $("#annotationNext").disabled = disabled || state.annotation.index >= maxIndex;
  $("#annotationJump").disabled = disabled;
  $("#annotationUndo").disabled = disabled || !hasAnnotations;
  $("#annotationDelete").disabled = disabled || !selectedAnnotationShapeIndices().length;
  $("#annotationClosePolygon").disabled = disabled || annotationMode() !== "polygon" || state.annotation.currentPoints.length < 3;
  $("#annotationClear").disabled = disabled || (!hasAnnotations && !row?.saved);
  $("#annotationSave").disabled = disabled || !row;
  $("#annotationSaveNext").disabled = disabled || !row;
  $("#annotationFinalize").disabled = disabled || !session;
}

function canvasPoint(event) {
  const canvas = $("#annotationCanvas");
  const rect = canvas.getBoundingClientRect();
  return [
    ((event.clientX - rect.left) * canvas.width) / Math.max(1, rect.width),
    ((event.clientY - rect.top) * canvas.height) / Math.max(1, rect.height),
  ];
}

function canvasSourcePoint(event) {
  const canvas = $("#annotationCanvas");
  const [canvasX, canvasY] = canvasPoint(event);
  const row = state.annotation.row;
  const sourceX = canvasX * Number(row.source_width || canvas.width) / Math.max(1, canvas.width);
  const sourceY = canvasY * Number(row.source_height || canvas.height) / Math.max(1, canvas.height);
  return [sourceX, sourceY];
}

function sourceToCanvas(point) {
  const canvas = $("#annotationCanvas");
  const row = state.annotation.row;
  return [
    Number(point[0]) * canvas.width / Math.max(1, Number(row?.source_width || canvas.width)),
    Number(point[1]) * canvas.height / Math.max(1, Number(row?.source_height || canvas.height)),
  ];
}

function pointInPolygon(point, sourcePoints) {
  if (!Array.isArray(sourcePoints) || sourcePoints.length < 3) return false;
  const x = Number(point[0]);
  const y = Number(point[1]);
  let inside = false;
  for (let i = 0, j = sourcePoints.length - 1; i < sourcePoints.length; j = i, i += 1) {
    const xi = Number(sourcePoints[i][0]);
    const yi = Number(sourcePoints[i][1]);
    const xj = Number(sourcePoints[j][0]);
    const yj = Number(sourcePoints[j][1]);
    if (![xi, yi, xj, yj].every(Number.isFinite)) continue;
    const crosses = ((yi > y) !== (yj > y)) && (x < ((xj - xi) * (y - yi)) / ((yj - yi) || 1e-9) + xi);
    if (crosses) inside = !inside;
  }
  return inside;
}

function shapeIndexAtEvent(event) {
  if (!$("#annotationOverlayToggle")?.checked) return null;
  const sourcePoint = canvasSourcePoint(event);
  for (let index = state.annotation.polygons.length - 1; index >= 0; index -= 1) {
    if (pointInPolygon(sourcePoint, state.annotation.polygons[index].points)) return index;
  }
  return null;
}

function orientation(a, b, c) {
  return (Number(b[1]) - Number(a[1])) * (Number(c[0]) - Number(b[0])) -
    (Number(b[0]) - Number(a[0])) * (Number(c[1]) - Number(b[1]));
}

function pointOnSegment(a, b, c) {
  return Math.min(Number(a[0]), Number(c[0])) <= Number(b[0]) + 1e-9 &&
    Number(b[0]) <= Math.max(Number(a[0]), Number(c[0])) + 1e-9 &&
    Math.min(Number(a[1]), Number(c[1])) <= Number(b[1]) + 1e-9 &&
    Number(b[1]) <= Math.max(Number(a[1]), Number(c[1])) + 1e-9;
}

function segmentsIntersect(a, b, c, d) {
  const o1 = orientation(a, b, c);
  const o2 = orientation(a, b, d);
  const o3 = orientation(c, d, a);
  const o4 = orientation(c, d, b);
  if (Math.abs(o1) < 1e-9 && pointOnSegment(a, c, b)) return true;
  if (Math.abs(o2) < 1e-9 && pointOnSegment(a, d, b)) return true;
  if (Math.abs(o3) < 1e-9 && pointOnSegment(c, a, d)) return true;
  if (Math.abs(o4) < 1e-9 && pointOnSegment(c, b, d)) return true;
  return (o1 > 0) !== (o2 > 0) && (o3 > 0) !== (o4 > 0);
}

function polygonsOverlap(first, second) {
  const a = first?.points || [];
  const b = second?.points || [];
  if (a.length < 3 || b.length < 3) return false;
  if (a.some((point) => pointInPolygon(point, b))) return true;
  if (b.some((point) => pointInPolygon(point, a))) return true;
  for (let i = 0; i < a.length; i += 1) {
    const a0 = a[i];
    const a1 = a[(i + 1) % a.length];
    for (let j = 0; j < b.length; j += 1) {
      if (segmentsIntersect(a0, a1, b[j], b[(j + 1) % b.length])) return true;
    }
  }
  return false;
}

function overlappingAddMaskGroup(startIndex) {
  const start = state.annotation.polygons[startIndex];
  if (!start || start.mode === "erase") return [startIndex];
  const selected = new Set([startIndex]);
  const queue = [startIndex];
  while (queue.length) {
    const currentIndex = queue.shift();
    const current = state.annotation.polygons[currentIndex];
    for (let index = 0; index < state.annotation.polygons.length; index += 1) {
      if (selected.has(index)) continue;
      const candidate = state.annotation.polygons[index];
      if (!candidate || candidate.mode === "erase") continue;
      if (polygonsOverlap(current, candidate)) {
        selected.add(index);
        queue.push(index);
      }
    }
  }
  return [...selected].sort((a, b) => a - b);
}

function selectedAnnotationShapeIndices() {
  const indices = Array.isArray(state.annotation.selectedShapeIndices)
    ? state.annotation.selectedShapeIndices
    : [];
  const valid = indices.filter((index) => Number.isInteger(index) && state.annotation.polygons[index]);
  if (valid.length) return [...new Set(valid)].sort((a, b) => a - b);
  const index = state.annotation.selectedShapeIndex;
  return Number.isInteger(index) && state.annotation.polygons[index] ? [index] : [];
}

function setSelectedAnnotationShapes(indices, message = "") {
  const valid = [...new Set((indices || []).filter((index) => Number.isInteger(index) && state.annotation.polygons[index]))]
    .sort((a, b) => a - b);
  state.annotation.selectedShapeIndices = valid;
  state.annotation.selectedShapeIndex = valid.length ? valid[valid.length - 1] : null;
  state.annotation.hoverShapeIndex = state.annotation.selectedShapeIndex;
  if (message) {
    setAnnotationRowStatus(message);
  } else if (valid.length > 1) {
    setAnnotationRowStatus(`Selected ${valid.length} overlapping masks; Delete removes group`);
  } else if (valid.length === 1) {
    setAnnotationRowStatus("Selected mask; Delete removes it");
  } else {
    setAnnotationRowStatus(annotationRowStatusText());
  }
  drawAnnotationCanvas();
  updateAnnotationControls();
  return valid.length > 0;
}

function clearSelectedAnnotationShapes() {
  state.annotation.selectedShapeIndex = null;
  state.annotation.selectedShapeIndices = [];
  state.annotation.hoverShapeIndex = null;
}

function annotationShapeSelected(index) {
  return selectedAnnotationShapeIndices().includes(index);
}

function selectAnnotationShape(index) {
  if (index == null || !state.annotation.polygons[index]) return false;
  return setSelectedAnnotationShapes(overlappingAddMaskGroup(index));
}

function selectAnnotationShapeAtEvent(event) {
  const index = shapeIndexAtEvent(event);
  if (index == null) {
    if (selectedAnnotationShapeIndices().length) {
      clearSelectedAnnotationShapes();
      setAnnotationRowStatus(annotationRowStatusText());
      drawAnnotationCanvas();
      updateAnnotationControls();
    }
    return false;
  }
  return selectAnnotationShape(index);
}

function deleteSelectedAnnotationShape() {
  const indices = selectedAnnotationShapeIndices();
  if (!indices.length) return false;
  for (const index of [...indices].sort((a, b) => b - a)) {
    state.annotation.polygons.splice(index, 1);
  }
  clearSelectedAnnotationShapes();
  state.annotation.currentPoints = [];
  state.annotation.ellipseDrag = null;
  state.annotation.moveDrag = null;
  state.annotation.rotateDrag = null;
  state.annotation.previewShape = null;
  state.annotation.dirty = true;
  setAnnotationRowStatus(indices.length > 1 ? "Deleted overlapping masks; Save to apply" : "Deleted mask; Save to apply");
  drawAnnotationCanvas();
  updateAnnotationControls();
  return true;
}

function drawAnnotationCanvas() {
  const canvas = $("#annotationCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.annotation.image) return;
  const adjusted = adjustedAnnotationImage() || state.annotation.image;
  ctx.drawImage(adjusted, 0, 0, canvas.width, canvas.height);
  if ($("#annotationOverlayToggle")?.checked) {
    drawAnnotationOverlay(ctx);
    for (const [index, polygon] of state.annotation.polygons.entries()) {
      drawShapeOutline(ctx, polygon, index);
      drawEllipseHandle(ctx, polygon, index);
    }
    if (state.annotation.currentPoints.length) {
      drawCurrentPath(ctx, state.annotation.currentPoints, annotationDrawColor());
    }
    if (state.annotation.previewShape?.bounds) {
      drawEllipsePreview(ctx, state.annotation.previewShape);
    }
  }
}

function drawAnnotationOverlay(ctx) {
  const canvas = $("#annotationCanvas");
  if (!canvas) return;
  const overlay = document.createElement("canvas");
  overlay.width = canvas.width;
  overlay.height = canvas.height;
  const overlayCtx = overlay.getContext("2d");
  for (const polygon of state.annotation.polygons) {
    if (!Array.isArray(polygon.points) || polygon.points.length < 3) continue;
    overlayCtx.save();
    overlayCtx.globalCompositeOperation = polygon.mode === "erase" ? "destination-out" : "source-over";
    overlayCtx.globalAlpha = 1;
    overlayCtx.fillStyle = annotationTypeColor(polygon.type_id);
    if (buildShapePath(overlayCtx, polygon.points)) overlayCtx.fill();
    overlayCtx.restore();
  }
  ctx.save();
  ctx.globalAlpha = 0.25;
  ctx.drawImage(overlay, 0, 0);
  ctx.restore();
}

function buildShapePath(ctx, sourcePoints) {
  const points = sourcePoints.map(sourceToCanvas);
  if (points.length < 2) return false;
  ctx.beginPath();
  points.forEach(([x, y], index) => {
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  return true;
}

function drawShapeOutline(ctx, polygon, index) {
  const erase = polygon.mode === "erase";
  const selected = annotationShapeSelected(index);
  const hovered = state.annotation.hoverShapeIndex === index;
  const color = annotationDrawColor(polygon.type_id, polygon.mode);
  ctx.save();
  if (!buildShapePath(ctx, polygon.points)) {
    ctx.restore();
    return;
  }
  ctx.strokeStyle = "#f8fbff";
  ctx.lineWidth = selected || hovered ? 2.5 : 1.5;
  ctx.stroke();
  ctx.strokeStyle = selected ? "#4FB8D8" : color;
  ctx.lineWidth = selected || hovered ? 4 : 3;
  if (erase) ctx.setLineDash([8, 5]);
  ctx.stroke();
  ctx.restore();
}

function drawCurrentPath(ctx, sourcePoints, color) {
  const points = sourcePoints.map(sourceToCanvas);
  ctx.save();
  ctx.beginPath();
  points.forEach(([x, y], index) => {
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.setLineDash([6, 5]);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = color;
  for (const [x, y] of points) {
    ctx.beginPath();
    ctx.arc(x, y, 3.2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawEllipseHandle(ctx, polygon, index) {
  if (annotationMode() !== "ellipse") return;
  const bounds = ellipseBounds(polygon);
  if (!bounds) return;
  const [x, y] = sourceToCanvas(ellipseCenter(bounds));
  const selected = annotationShapeSelected(index);
  const hovered = state.annotation.hoverShapeIndex === index;
  const color = annotationDrawColor(polygon.type_id, polygon.mode);
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, selected || hovered ? 7 : 5, 0, Math.PI * 2);
  ctx.fillStyle = "#f8fbff";
  ctx.globalAlpha = 0.94;
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.lineWidth = selected || hovered ? 3 : 2;
  ctx.strokeStyle = color;
  ctx.stroke();
  if (selected) {
    ctx.beginPath();
    ctx.arc(x, y, 11, 0, Math.PI * 2);
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "#f8fbff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  ctx.restore();
  if (selected || hovered) {
    drawEllipseCornerHandles(ctx, polygon);
  }
}

function ellipseCornerHandles(polygon) {
  const bounds = ellipseBounds(polygon);
  if (!bounds) return [];
  const [cx, cy] = ellipseCenter(bounds);
  const rx = (bounds.x1 - bounds.x0) / 2;
  const ry = (bounds.y1 - bounds.y0) / 2;
  const rotation = normalizeAngle(polygon.rotation || 0);
  return [
    { corner: "nw", local: [-rx, -ry] },
    { corner: "ne", local: [rx, -ry] },
    { corner: "se", local: [rx, ry] },
    { corner: "sw", local: [-rx, ry] },
  ].map((handle) => {
    const [dx, dy] = rotateOffset(handle.local[0], handle.local[1], rotation);
    return {
      ...handle,
      source: [cx + dx, cy + dy],
      localAngle: Math.atan2(handle.local[1], handle.local[0]),
    };
  });
}

function drawEllipseCornerHandles(ctx, polygon) {
  const handles = ellipseCornerHandles(polygon);
  if (!handles.length) return;
  const color = annotationDrawColor(polygon.type_id, polygon.mode);
  ctx.save();
  ctx.fillStyle = "#f8fbff";
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  for (const handle of handles) {
    const [x, y] = sourceToCanvas(handle.source);
    ctx.beginPath();
    ctx.rect(x - 4.5, y - 4.5, 9, 9);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function sourceBoundsToCanvas(bounds) {
  const [x0, y0] = sourceToCanvas([bounds.x0, bounds.y0]);
  const [x1, y1] = sourceToCanvas([bounds.x1, bounds.y1]);
  return {
    x: Math.min(x0, x1),
    y: Math.min(y0, y1),
    width: Math.abs(x1 - x0),
    height: Math.abs(y1 - y0),
  };
}

function normalizedSourceBounds(start, end) {
  const row = state.annotation.row;
  const maxX = Math.max(0, Number(row?.source_width || 1) - 1);
  const maxY = Math.max(0, Number(row?.source_height || 1) - 1);
  const clamp = (value, max) => Math.max(0, Math.min(Number(value) || 0, max));
  const x0 = clamp(Math.min(start[0], end[0]), maxX);
  const x1 = clamp(Math.max(start[0], end[0]), maxX);
  const y0 = clamp(Math.min(start[1], end[1]), maxY);
  const y1 = clamp(Math.max(start[1], end[1]), maxY);
  return { x0, y0, x1, y1 };
}

function ellipsePointsFromBounds(bounds, count = 72, rotation = 0) {
  const cx = (bounds.x0 + bounds.x1) / 2;
  const cy = (bounds.y0 + bounds.y1) / 2;
  const rx = Math.abs(bounds.x1 - bounds.x0) / 2;
  const ry = Math.abs(bounds.y1 - bounds.y0) / 2;
  const points = [];
  for (let index = 0; index < count; index += 1) {
    const theta = (Math.PI * 2 * index) / count;
    const [dx, dy] = rotateOffset(Math.cos(theta) * rx, Math.sin(theta) * ry, rotation);
    points.push([cx + dx, cy + dy]);
  }
  return points;
}

function translatedEllipseBounds(bounds, dx, dy) {
  const limits = sourceLimits();
  const width = bounds.x1 - bounds.x0;
  const height = bounds.y1 - bounds.y0;
  const maxX0 = Math.max(0, limits.width - 1 - width);
  const maxY0 = Math.max(0, limits.height - 1 - height);
  const x0 = Math.max(0, Math.min(bounds.x0 + dx, maxX0));
  const y0 = Math.max(0, Math.min(bounds.y0 + dy, maxY0));
  return { x0, y0, x1: x0 + width, y1: y0 + height };
}

function ellipseRotationHandleAtEvent(event) {
  if (annotationMode() !== "ellipse") return null;
  if (!$("#annotationOverlayToggle")?.checked) return null;
  const [x, y] = canvasPoint(event);
  let best = null;
  let bestDistance = Infinity;
  for (let index = state.annotation.polygons.length - 1; index >= 0; index -= 1) {
    const polygon = state.annotation.polygons[index];
    if (polygon?.shape !== "ellipse") continue;
    for (const handle of ellipseCornerHandles(polygon)) {
      const [hx, hy] = sourceToCanvas(handle.source);
      const distance = Math.hypot(x - hx, y - hy);
      if (distance <= 12 && distance < bestDistance) {
        best = { index, handle };
        bestDistance = distance;
      }
    }
  }
  return best;
}

function ellipseHandleIndexAtEvent(event) {
  if (annotationMode() !== "ellipse") return null;
  if (!$("#annotationOverlayToggle")?.checked) return null;
  const [x, y] = canvasPoint(event);
  let bestIndex = null;
  let bestDistance = Infinity;
  for (let index = state.annotation.polygons.length - 1; index >= 0; index -= 1) {
    const polygon = state.annotation.polygons[index];
    const bounds = ellipseBounds(polygon);
    if (!bounds) continue;
    const [cx, cy] = sourceToCanvas(ellipseCenter(bounds));
    const distance = Math.hypot(x - cx, y - cy);
    if (distance <= 14 && distance < bestDistance) {
      bestIndex = index;
      bestDistance = distance;
    }
  }
  return bestIndex;
}

function updateAnnotationCanvasCursor(event) {
  const canvas = $("#annotationCanvas");
  if (!canvas) return;
  if (!state.annotation.row || state.annotation.loading) {
    state.annotation.hoverShapeIndex = null;
    canvas.style.cursor = "default";
    return;
  }
  if (annotationMode() === "ellipse") {
    const previous = state.annotation.hoverShapeIndex;
    const rotationHandle = ellipseRotationHandleAtEvent(event);
    const handleIndex = rotationHandle == null ? ellipseHandleIndexAtEvent(event) : null;
    const shapeIndex = rotationHandle?.index ?? handleIndex ?? shapeIndexAtEvent(event);
    state.annotation.hoverShapeIndex = shapeIndex;
    canvas.style.cursor = rotationHandle != null ? "grab" : (handleIndex != null ? "move" : (shapeIndex == null ? "crosshair" : "pointer"));
    if (previous !== shapeIndex) drawAnnotationCanvas();
    return;
  }
  const previous = state.annotation.hoverShapeIndex;
  const shapeIndex = state.annotation.currentPoints.length ? null : shapeIndexAtEvent(event);
  state.annotation.hoverShapeIndex = shapeIndex;
  canvas.style.cursor = shapeIndex == null ? "crosshair" : "pointer";
  if (previous !== shapeIndex) {
    drawAnnotationCanvas();
  }
}

function leaveAnnotationCanvas() {
  const canvas = $("#annotationCanvas");
  if (state.annotation.moveDrag || state.annotation.rotateDrag || state.annotation.ellipseDrag) return;
  if (state.annotation.hoverShapeIndex != null) {
    state.annotation.hoverShapeIndex = null;
    drawAnnotationCanvas();
  }
  if (canvas) canvas.style.cursor = state.annotation.row ? "crosshair" : "default";
}

function continueAnnotationDrag(event) {
  if (!state.annotation.moveDrag && !state.annotation.rotateDrag && !state.annotation.ellipseDrag) return;
  previewAnnotationEllipse(event);
}

function beginRotateAnnotationEllipse(event, index, handle, start) {
  const polygon = state.annotation.polygons[index];
  const bounds = ellipseBounds(polygon);
  if (!bounds) return false;
  const center = ellipseCenter(bounds);
  state.annotation.currentPoints = [];
  state.annotation.ellipseDrag = null;
  state.annotation.moveDrag = null;
  state.annotation.previewShape = null;
  state.annotation.rotateDrag = {
    index,
    start,
    center,
    handleLocalAngle: handle.localAngle,
    originalRotation: normalizeAngle(polygon.rotation || 0),
    originalPoints: (polygon.points || []).map((point) => [Number(point[0]), Number(point[1])]),
    pointCount: Math.max(16, Number(polygon.points?.length || 72)),
    wasDirty: state.annotation.dirty,
    moved: false,
  };
  setSelectedAnnotationShapes([index], "Selected ellipse; drag corner to rotate");
  return true;
}

function rotateAnnotationEllipse(event) {
  const drag = state.annotation.rotateDrag;
  if (!drag) return false;
  event.preventDefault();
  const polygon = state.annotation.polygons[drag.index];
  if (!polygon) return false;
  const bounds = ellipseBounds(polygon);
  if (!bounds) return false;
  const current = canvasSourcePoint(event);
  const angle = Math.atan2(current[1] - drag.center[1], current[0] - drag.center[0]);
  const rotation = normalizeAngle(angle - drag.handleLocalAngle);
  polygon.rotation = rotation;
  polygon.points = ellipsePointsFromBounds(bounds, drag.pointCount, rotation);
  const delta = Math.abs(normalizeAngle(rotation - drag.originalRotation));
  if (!drag.moved && delta > 0.01) {
    drag.moved = true;
    state.annotation.dirty = true;
    setAnnotationRowStatus("Unsaved changes");
  }
  drawAnnotationCanvas();
  updateAnnotationControls();
  return true;
}

function finishRotateAnnotationEllipse(event) {
  if (!state.annotation.rotateDrag) return false;
  rotateAnnotationEllipse(event);
  state.annotation.rotateDrag = null;
  drawAnnotationCanvas();
  updateAnnotationControls();
  return true;
}

function restoreRotateAnnotationEllipse() {
  const drag = state.annotation.rotateDrag;
  if (!drag) return false;
  const polygon = state.annotation.polygons[drag.index];
  if (polygon) {
    polygon.rotation = drag.originalRotation;
    polygon.points = drag.originalPoints;
  }
  state.annotation.dirty = Boolean(drag.wasDirty);
  state.annotation.rotateDrag = null;
  setAnnotationRowStatus(annotationRowStatusText());
  drawAnnotationCanvas();
  updateAnnotationControls();
  return true;
}

function beginMoveAnnotationEllipse(event, index, start) {
  const polygon = state.annotation.polygons[index];
  const bounds = ellipseBounds(polygon);
  if (!bounds) return false;
  state.annotation.currentPoints = [];
  state.annotation.ellipseDrag = null;
  state.annotation.previewShape = null;
  state.annotation.moveDrag = {
    index,
    start,
    originalBounds: { ...bounds },
    originalRotation: normalizeAngle(polygon.rotation || 0),
    pointCount: Math.max(16, Number(polygon.points?.length || 72)),
    wasDirty: state.annotation.dirty,
    moved: false,
  };
  setSelectedAnnotationShapes([index], "Selected ellipse; drag center to move");
  return true;
}

function moveAnnotationEllipse(event) {
  const drag = state.annotation.moveDrag;
  if (!drag) return false;
  event.preventDefault();
  const polygon = state.annotation.polygons[drag.index];
  if (!polygon) return false;
  const current = canvasSourcePoint(event);
  const dx = current[0] - drag.start[0];
  const dy = current[1] - drag.start[1];
  const bounds = translatedEllipseBounds(drag.originalBounds, dx, dy);
  const rotation = normalizeAngle(polygon.rotation || 0);
  polygon.bounds = bounds;
  polygon.rotation = rotation;
  polygon.points = ellipsePointsFromBounds(bounds, drag.pointCount, rotation);
  if (!drag.moved && Math.hypot(dx, dy) > 0.5) {
    drag.moved = true;
    state.annotation.dirty = true;
    setAnnotationRowStatus("Unsaved changes");
  }
  drawAnnotationCanvas();
  updateAnnotationControls();
  return true;
}

function finishMoveAnnotationEllipse(event) {
  if (!state.annotation.moveDrag) return false;
  moveAnnotationEllipse(event);
  state.annotation.moveDrag = null;
  drawAnnotationCanvas();
  updateAnnotationControls();
  return true;
}

function restoreMoveAnnotationEllipse() {
  const drag = state.annotation.moveDrag;
  if (!drag) return false;
  const polygon = state.annotation.polygons[drag.index];
  if (polygon) {
    polygon.bounds = { ...drag.originalBounds };
    polygon.rotation = drag.originalRotation;
    polygon.points = ellipsePointsFromBounds(drag.originalBounds, drag.pointCount, drag.originalRotation);
  }
  state.annotation.dirty = Boolean(drag.wasDirty);
  state.annotation.moveDrag = null;
  setAnnotationRowStatus(annotationRowStatusText());
  drawAnnotationCanvas();
  updateAnnotationControls();
  return true;
}

function drawEllipsePreview(ctx, shape) {
  const bounds = sourceBoundsToCanvas(shape.bounds);
  const color = annotationDrawColor(shape.type_id, shape.mode);
  if (bounds.width < 2 || bounds.height < 2) return;
  ctx.save();
  ctx.beginPath();
  ctx.ellipse(
    bounds.x + bounds.width / 2,
    bounds.y + bounds.height / 2,
    bounds.width / 2,
    bounds.height / 2,
    normalizeAngle(shape.rotation || 0),
    0,
    Math.PI * 2,
  );
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.18;
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "#f8fbff";
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  if (shape.mode === "erase") ctx.setLineDash([8, 5]);
  ctx.stroke();
  ctx.restore();
}

function addAnnotationPoint(event) {
  if (!state.annotation.row || state.annotation.loading) return;
  if (state.annotation.suppressNextClick) {
    state.annotation.suppressNextClick = false;
    return;
  }
  if (annotationMode() !== "polygon") {
    selectAnnotationShapeAtEvent(event);
    return;
  }
  if (!state.annotation.currentPoints.length && selectAnnotationShapeAtEvent(event)) return;
  clearSelectedAnnotationShapes();
  state.annotation.currentPoints.push(canvasSourcePoint(event));
  state.annotation.dirty = true;
  setAnnotationRowStatus("Unsaved changes");
  drawAnnotationCanvas();
  updateAnnotationControls();
}

function beginAnnotationEllipse(event) {
  if (annotationMode() !== "ellipse" || !state.annotation.row || state.annotation.loading) return;
  if (event.button !== 0) return;
  event.preventDefault();
  const start = canvasSourcePoint(event);
  const rotationHandle = ellipseRotationHandleAtEvent(event);
  if (rotationHandle && beginRotateAnnotationEllipse(event, rotationHandle.index, rotationHandle.handle, start)) {
    state.annotation.suppressNextClick = true;
    return;
  }
  const hitIndex = ellipseHandleIndexAtEvent(event);
  if (hitIndex != null && beginMoveAnnotationEllipse(event, hitIndex, start)) {
    state.annotation.suppressNextClick = true;
    return;
  }
  if (selectAnnotationShapeAtEvent(event)) {
    state.annotation.suppressNextClick = true;
    return;
  }
  clearSelectedAnnotationShapes();
  state.annotation.currentPoints = [];
  state.annotation.ellipseDrag = { start, current: start };
  state.annotation.previewShape = {
    bounds: normalizedSourceBounds(start, start),
    type_id: annotationAction() === "erase" ? null : annotationTypeId(),
    mode: annotationAction(),
    shape: "ellipse",
  };
  drawAnnotationCanvas();
  updateAnnotationControls();
}

function previewAnnotationEllipse(event) {
  if (state.annotation.rotateDrag) {
    rotateAnnotationEllipse(event);
    return;
  }
  if (state.annotation.moveDrag) {
    moveAnnotationEllipse(event);
    return;
  }
  if (!state.annotation.ellipseDrag) {
    updateAnnotationCanvasCursor(event);
    return;
  }
  event.preventDefault();
  const current = canvasSourcePoint(event);
  state.annotation.ellipseDrag.current = current;
  state.annotation.previewShape = {
    bounds: normalizedSourceBounds(state.annotation.ellipseDrag.start, current),
    type_id: annotationAction() === "erase" ? null : annotationTypeId(),
    mode: annotationAction(),
    shape: "ellipse",
  };
  drawAnnotationCanvas();
  updateAnnotationControls();
}

function finishAnnotationEllipse(event) {
  if (state.annotation.rotateDrag) {
    finishRotateAnnotationEllipse(event);
    state.annotation.suppressNextClick = true;
    return;
  }
  if (state.annotation.moveDrag) {
    finishMoveAnnotationEllipse(event);
    state.annotation.suppressNextClick = true;
    return;
  }
  if (!state.annotation.ellipseDrag) return;
  event.preventDefault();
  const end = canvasSourcePoint(event);
  const bounds = normalizedSourceBounds(state.annotation.ellipseDrag.start, end);
  state.annotation.ellipseDrag = null;
  state.annotation.previewShape = null;
  if (bounds.x1 - bounds.x0 >= 3 && bounds.y1 - bounds.y0 >= 3) {
    state.annotation.polygons.push({
      points: ellipsePointsFromBounds(bounds),
      type_id: annotationAction() === "erase" ? null : annotationTypeId(),
      mode: annotationAction(),
      shape: "ellipse",
      bounds,
      rotation: 0,
    });
    selectAnnotationShape(state.annotation.polygons.length - 1);
    state.annotation.dirty = true;
    setAnnotationRowStatus("Unsaved changes");
  }
  state.annotation.suppressNextClick = true;
  drawAnnotationCanvas();
  updateAnnotationControls();
}

function cancelTransientAnnotationShape() {
  if (restoreRotateAnnotationEllipse()) return;
  if (restoreMoveAnnotationEllipse()) return;
  state.annotation.currentPoints = [];
  state.annotation.ellipseDrag = null;
  state.annotation.moveDrag = null;
  state.annotation.rotateDrag = null;
  clearSelectedAnnotationShapes();
  state.annotation.previewShape = null;
  drawAnnotationCanvas();
  updateAnnotationControls();
}

function setAnnotationMode(mode) {
  state.annotation.mode = mode === "ellipse" ? "ellipse" : "polygon";
  const control = $("#annotationMode");
  if (control) control.value = state.annotation.mode;
  cancelTransientAnnotationShape();
}

function setAnnotationAction(action) {
  state.annotation.action = action === "erase" ? "erase" : "add";
  const control = $("#annotationAction");
  if (control) control.value = state.annotation.action;
  syncAnnotationActionControls();
  cancelTransientAnnotationShape();
}

function closeAnnotationPolygon() {
  if (annotationMode() !== "polygon") return false;
  if (state.annotation.currentPoints.length < 3) return false;
  state.annotation.polygons.push({
    points: state.annotation.currentPoints,
    type_id: annotationAction() === "erase" ? null : annotationTypeId(),
    mode: annotationAction(),
    shape: "polygon",
  });
  selectAnnotationShape(state.annotation.polygons.length - 1);
  state.annotation.currentPoints = [];
  state.annotation.dirty = true;
  setAnnotationRowStatus("Unsaved changes");
  drawAnnotationCanvas();
  updateAnnotationControls();
  return true;
}

function undoAnnotationEdit() {
  if (!state.annotation.row) return;
  if (state.annotation.moveDrag) {
    restoreMoveAnnotationEllipse();
    return;
  }
  if (state.annotation.ellipseDrag || state.annotation.previewShape) {
    state.annotation.ellipseDrag = null;
    state.annotation.previewShape = null;
  } else if (state.annotation.currentPoints.length) {
    state.annotation.currentPoints.pop();
  } else if (state.annotation.polygons.length) {
    state.annotation.polygons.pop();
    clearSelectedAnnotationShapes();
  } else {
    return;
  }
  state.annotation.dirty = true;
  setAnnotationRowStatus("Unsaved changes");
  drawAnnotationCanvas();
  updateAnnotationControls();
}

function clearAnnotationRow() {
  if (!state.annotation.row) return;
  if (state.annotation.row.saved && !window.confirm("Clear saved annotations for this row?")) return;
  state.annotation.polygons = [];
  state.annotation.currentPoints = [];
  state.annotation.ellipseDrag = null;
  state.annotation.moveDrag = null;
  state.annotation.rotateDrag = null;
  clearSelectedAnnotationShapes();
  state.annotation.previewShape = null;
  state.annotation.dirty = true;
  setAnnotationRowStatus("Unsaved changes");
  drawAnnotationCanvas();
  updateAnnotationControls();
}

async function saveAnnotationRow() {
  const session = state.annotation.session;
  const row = state.annotation.row;
  if (!session || !row) return null;
  if (state.annotation.currentPoints.length >= 3) closeAnnotationPolygon();
  const payload = { polygons: state.annotation.polygons };
  setAnnotationRowStatus("Saving...");
  const summary = await api(`/api/annotation/sessions/${session.id}/rows/${row.index}`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  state.annotation.session = summary;
  rememberAnnotationSession(summary);
  state.annotation.row.saved = true;
  state.annotation.dirty = false;
  renderAnnotationMeta();
  setAnnotationRowStatus("Saved");
  updateAnnotationControls();
  return summary;
}

async function finalizeAnnotationSession() {
  const session = state.annotation.session;
  if (!session) return;
  if (state.annotation.dirty) {
    await saveAnnotationRow();
  }
  const summary = await api(`/api/annotation/sessions/${session.id}/finalize`, {
    method: "POST",
    body: "{}",
  });
  state.annotation.session = summary;
  rememberAnnotationSession(summary);
  renderAnnotationMeta();
  setAnnotationLaunchStatus(`Exported ${formatPathTail(summary.manifest_edited_path)}.`, "ok");
  populateFineTuneFromAnnotationSession(summary);
}

function statusText(status) {
  const labels = {
    queued: "queued",
    running: "running",
    succeeded: "done",
    failed: "failed",
    canceled: "canceled",
    unknown: "unknown",
  };
  return labels[status] || status || "unknown";
}

function classToken(value) {
  return String(value || "unknown").toLowerCase().replace(/[^a-z0-9_-]/g, "-");
}

function formatPathTail(path) {
  const text = String(path || "");
  const parts = text.split("/").filter(Boolean);
  if (parts.length <= 2) return text;
  return `.../${parts.slice(-2).join("/")}`;
}

function parseDate(value) {
  const date = value ? new Date(value) : null;
  return date && Number.isFinite(date.getTime()) ? date : null;
}

function durationText(start, end) {
  const startDate = parseDate(start);
  if (!startDate) return "pending";
  const endDate = parseDate(end) || new Date();
  const seconds = Math.max(0, Math.floor((endDate.getTime() - startDate.getTime()) / 1000));
  return formatDurationSeconds(seconds);
}

function durationSeconds(start, end) {
  const startDate = parseDate(start);
  if (!startDate) return null;
  const endDate = parseDate(end) || new Date();
  const seconds = Math.max(0, (endDate.getTime() - startDate.getTime()) / 1000);
  return Number.isFinite(seconds) ? seconds : null;
}

function formatDurationSeconds(seconds) {
  const totalSeconds = Math.max(0, Math.round(Number(seconds) || 0));
  const minutes = Math.floor(totalSeconds / 60);
  const hours = Math.floor(minutes / 60);
  if (hours) return `${hours}h ${minutes % 60}m`;
  if (minutes) return `${minutes}m ${totalSeconds % 60}s`;
  return `${totalSeconds}s`;
}

function parseDurationSeconds(text) {
  const value = String(text || "").trim();
  if (!value) return null;
  const hourMatch = value.match(/(?:(\d+(?:\.\d+)?)h\s*)?(?:(\d+(?:\.\d+)?)m\s*)?(?:(\d+(?:\.\d+)?)s)?/i);
  if (!hourMatch || !hourMatch[0].trim()) return null;
  const hours = Number(hourMatch[1] || 0);
  const minutes = Number(hourMatch[2] || 0);
  const seconds = Number(hourMatch[3] || 0);
  const total = hours * 3600 + minutes * 60 + seconds;
  return Number.isFinite(total) ? total : null;
}

function formatThroughputSeconds(seconds) {
  const numeric = Number(seconds);
  if (!Number.isFinite(numeric) || numeric <= 0) return null;
  return numeric < 10 ? `${numeric.toFixed(2)}s` : `${numeric.toFixed(1)}s`;
}

function optionValue(job, option) {
  for (const step of job?.steps || []) {
    const argv = step.argv || [];
    const prefix = `${option}=`;
    for (let index = 0; index < argv.length; index += 1) {
      const token = String(argv[index]);
      if (token.startsWith(prefix)) return token.slice(prefix.length);
      if (token === option && index + 1 < argv.length) return String(argv[index + 1]);
    }
  }
  return null;
}

function progressSamples(logText) {
  const samples = [];
  for (const line of String(logText || "").split("\n")) {
    const progress = line.match(/progress:\s*(\d+)\/(\d+)\s+complete;/i);
    if (!progress) continue;
    const elapsedNumeric = line.match(/elapsed_s=([0-9]+(?:\.[0-9]+)?)/i);
    const lastNumeric = line.match(/last_s=([0-9]+(?:\.[0-9]+)?)/i);
    const elapsedText = line.match(/elapsed=([^;]+)/i);
    const lastText = line.match(/last=([^;]+)/i);
    const elapsedSeconds = elapsedNumeric
      ? Number(elapsedNumeric[1])
      : parseDurationSeconds(elapsedText?.[1]);
    const lastSeconds = lastNumeric
      ? Number(lastNumeric[1])
      : parseDurationSeconds(lastText?.[1]);
    samples.push({
      completed: Number(progress[1]),
      total: Number(progress[2]),
      elapsedSeconds,
      lastSeconds,
    });
  }
  return samples;
}

function finalInferenceCompletedCount(logText) {
  const matches = Array.from(String(logText || "").matchAll(/^Finished inference on\s+(\d+)\s+micrograph/gim));
  if (!matches.length) return null;
  const completed = Number(matches.at(-1)[1]);
  return Number.isFinite(completed) && completed > 0 ? completed : null;
}

function totalMicrographCountFromLog(logText) {
  const text = String(logText || "");
  const multiGpuMatches = Array.from(text.matchAll(/Using\s+\d+\s+GPUs?\s+for\s+(\d+)\s+micrograph/gim));
  if (multiGpuMatches.length) {
    const total = Number(multiGpuMatches.at(-1)[1]);
    if (Number.isFinite(total) && total > 0) return total;
  }
  return finalInferenceCompletedCount(text);
}

function bracketMicrographProgress(job, logText) {
  const text = String(logText || "");
  const matches = Array.from(text.matchAll(/^\[(\d+)\/(\d+)\]\s+(.+)$/gm));
  if (!matches.length) return null;
  const finalCompleted = finalInferenceCompletedCount(text);
  let shardTotal = 0;
  const startedImages = new Set();
  for (const match of matches) {
    shardTotal = Math.max(shardTotal, Number(match[2]) || 0);
    const imageLabel = String(match[3] || "").split(":")[0].trim();
    startedImages.add(imageLabel || `${match[1]}:${match.index}`);
  }
  const total = totalMicrographCountFromLog(text) || shardTotal;
  if (finalCompleted) {
    return { completed: finalCompleted, total: total || finalCompleted, elapsedSeconds: null };
  }
  const activeWorkers = Math.max(1, requestedGpuCount(job, text) || 1);
  return {
    completed: Math.max(0, startedImages.size - activeWorkers),
    total,
    elapsedSeconds: null,
  };
}

function latestMicrographProgress(job, logText, liveSummary = null) {
  const totalFromLog = totalMicrographCountFromLog(logText);
  const finalCompleted = finalInferenceCompletedCount(logText);
  if (finalCompleted) {
    return { completed: finalCompleted, total: totalFromLog || finalCompleted, elapsedSeconds: null };
  }
  if (liveSummary?.available && Number.isFinite(Number(liveSummary.n_images_total)) && Number(liveSummary.n_images_total) > 0) {
    return {
      completed: Number(liveSummary.n_images_total),
      total: totalFromLog || Number(liveSummary.n_images_total),
      elapsedSeconds: null,
    };
  }
  const samples = progressSamples(logText);
  if (samples.length) {
    const latest = samples.at(-1);
    return { ...latest, total: totalFromLog || latest.total };
  }
  return bracketMicrographProgress(job, logText);
}

function requestedGpuCount(job, logText) {
  const multiGpu = String(logText || "").match(/Using\s+(\d+)\s+GPUs?\s+for\s+\d+\s+micrograph/i);
  if (multiGpu) {
    const count = Number(multiGpu[1]);
    if (Number.isFinite(count) && count > 0) return count;
  }
  const parallel = String(logText || "").match(/Using DataParallel across\s+(\d+)\s+CUDA GPUs/i);
  if (parallel) {
    const count = Number(parallel[1]);
    if (Number.isFinite(count) && count > 0) return count;
  }
  const requested = Number(optionValue(job, "--num-gpus"));
  if (Number.isFinite(requested) && requested > 0) return requested;
  if (/Using device=cuda/i.test(logText) || optionValue(job, "--device") === "cuda") return 1;
  return 0;
}

function rollingMicrographSeconds(samples) {
  if (!Array.isArray(samples) || !samples.length) return null;
  const latest = samples.at(-1);
  if (samples.length >= 2) {
    const start = samples[Math.max(0, samples.length - 6)];
    const imageCount = Number(latest.completed) - Number(start.completed);
    const elapsed = Number(latest.elapsedSeconds) - Number(start.elapsedSeconds);
    if (Number.isFinite(imageCount) && imageCount > 0 && Number.isFinite(elapsed) && elapsed > 0) {
      return { seconds: elapsed / imageCount, images: imageCount };
    }
  }
  if (Number.isFinite(latest.lastSeconds) && latest.lastSeconds > 0) {
    return { seconds: latest.lastSeconds, images: 1 };
  }
  return null;
}

function timePerMicrographMetric(job, logText, liveSummary = null) {
  const samples = progressSamples(logText);
  const progress = latestMicrographProgress(job, logText, liveSummary);
  if (!progress || !Number.isFinite(progress.completed) || progress.completed <= 0) return null;
  const elapsed = progress.elapsedSeconds ?? durationSeconds(job.started_at || job.created_at, job.ended_at);
  if (!Number.isFinite(elapsed) || elapsed <= 0) return null;
  const seconds = elapsed / progress.completed;
  const average = formatThroughputSeconds(seconds);
  if (!average) return null;
  const recent = rollingMicrographSeconds(samples);
  const pieces = [`avg ${average}`];
  if (recent) {
    const recentText = formatThroughputSeconds(recent.seconds);
    if (recentText) pieces.push(`last ${recent.images} ${recentText}`);
  }
  const suffix = progress.total ? ` (${progress.completed}/${progress.total})` : "";
  return { label: "~time/micrograph", value: `${pieces.join(" | ")}${suffix}` };
}

function jobSubline(job) {
  const runtime = durationText(job.started_at || job.created_at, job.ended_at);
  const status = statusText(job.status);
  if (job.status === "succeeded") return `done | ${runtime}`;
  if (job.status === "failed") return `failed | code ${job.returncode ?? "unknown"}`;
  if (job.status === "canceled") return `canceled | ${runtime}`;
  if (job.status === "running") return `running | ${runtime}`;
  if (job.status === "queued") return "queued | waiting";
  return `${status} | ${job.created_at || "no timestamp"}`;
}

function jobProgress(job) {
  const raw = job.progress_fraction ?? job.progress;
  const numeric = Number(raw);
  if (!Number.isFinite(numeric)) return null;
  const fraction = numeric > 1 ? numeric / 100 : numeric;
  return Math.max(0, Math.min(100, Math.round(fraction * 100)));
}

function renderLiveIndicator() {
  const indicator = $("#liveIndicator");
  const label = $("#liveRunName");
  const active = state.jobs.find((job) => ["queued", "running"].includes(job.status));
  indicator.hidden = !active;
  if (active) {
    label.textContent = active.title || active.kind || active.id;
    label.title = label.textContent;
  }
}

function renderJobs() {
  const list = $("#jobsList");
  if (!state.jobs.length) {
    list.classList.add("empty");
    list.innerHTML = `
      <div class="empty-state">
        <h3>Start your first run</h3>
        <p>Point cryoFILTER at a micrograph directory to begin.</p>
        <button class="ghost" type="button" data-action="launch-run">Launch run</button>
      </div>
    `;
    list.querySelector("[data-action='launch-run']").addEventListener("click", openPrimaryLaunchTab);
    return;
  }
  list.classList.remove("empty");
  list.innerHTML = state.jobs.map((job) => `
    <button class="job-row ${state.selected === job.id ? "active" : ""}" data-job="${job.id}">
      <span class="job-main">
        <span class="status-dot ${classToken(job.status)}" aria-hidden="true"></span>
        <span class="job-title">${escapeHtml(job.title || job.kind || job.id)}</span>
        <span class="job-status">${escapeHtml(statusText(job.status))}</span>
      </span>
      <span class="job-meta">${escapeHtml(jobSubline(job))}</span>
      ${["queued", "running"].includes(job.status) ? renderProgress(job) : ""}
    </button>
  `).join("");
  $$(".job-row").forEach((card) => {
    card.addEventListener("click", () => selectJob(card.dataset.job));
  });
}

function renderProgress(job) {
  const progress = jobProgress(job);
  if (progress === null) {
    return `<span class="progress-bar ${classToken(job.status)}" aria-label="${escapeHtml(statusText(job.status))}"><span></span></span>`;
  }
  return `<span class="progress-bar ${classToken(job.status)}" aria-label="${progress}% complete"><span style="width: ${progress}%"></span></span>`;
}

async function loadStatus() {
  const status = await api("/api/status");
  $("#workDir").textContent = formatPathTail(status.work_dir);
  $("#workDir").title = status.work_dir || "Local app";
  $("#statusLine").textContent = `python ${status.python_version || status.python || "unknown"} | node ${status.node || "unknown"} | gpu ${status.gpu || "not detected"}`;
}

async function loadJobs() {
  const payload = await api("/api/jobs");
  state.jobs = payload.jobs || [];
  if (!state.jobs.length) state.selected = null;
  if (state.selected && !state.jobs.some((job) => job.id === state.selected)) state.selected = null;
  if (!state.selected && state.jobs[0]) state.selected = state.jobs[0].id;
  renderJobs();
  renderLiveIndicator();
  await renderSelected();
}

async function selectJob(id) {
  state.selected = id;
  state.logOffset = null;
  state.autoScroll = true;
  renderJobs();
  await renderSelected();
}

async function renderSelected() {
  const details = $(".details");
  const cancelButton = $("#cancelButton");
  const logEl = $("#jobLog");
  const artifacts = $("#artifactGrid");
  const metrics = $("#metricsGrid");
  const liveSummary = $("#liveSummary");
  if (!state.selected) {
    details.classList.add("empty");
    $("#selectedTitle").textContent = "No run selected";
    $("#selectedMeta").textContent = "Launch a workflow to see live output.";
    cancelButton.hidden = true;
    artifacts.hidden = true;
    metrics.hidden = true;
    liveSummary.hidden = true;
    logEl.hidden = true;
    logEl.textContent = "";
    artifacts.innerHTML = "";
    metrics.innerHTML = "";
    $("#liveSummaryCharts").innerHTML = "";
    return;
  }
  const job = await api(`/api/jobs/${state.selected}`);
  details.classList.remove("empty");
  $("#selectedTitle").textContent = job.title || job.kind || job.id;
  $("#selectedMeta").textContent = jobSubline(job);
  cancelButton.hidden = !["queued", "running"].includes(job.status);
  const log = await api(`/api/jobs/${state.selected}/log?tail=256000`);
  const isLive = ["queued", "running"].includes(job.status);
  logEl.hidden = false;
  const shouldScroll = state.autoScroll || isNearBottom(logEl);
  logEl.innerHTML = formatLog(log.text || "", isLive);
  if (shouldScroll) {
    logEl.scrollTop = logEl.scrollHeight;
  }
  const artifactCount = await renderArtifacts();
  const liveSummary = await renderLiveSummary(job);
  renderMetrics(job, log.text || "", artifactCount, liveSummary);
}

async function renderArtifacts() {
  const grid = $("#artifactGrid");
  const payload = await api(`/api/jobs/${state.selected}/artifacts`);
  const artifacts = payload.artifacts || [];
  if (!artifacts.length) {
    grid.innerHTML = "";
    grid.hidden = true;
    return 0;
  }
  grid.hidden = false;
  const images = artifacts.filter((artifact) => [".png", ".jpg", ".jpeg"].includes(artifact.suffix)).slice(0, 16);
  const other = artifacts.filter((artifact) => ![".png", ".jpg", ".jpeg"].includes(artifact.suffix)).slice(0, 16);
  grid.innerHTML = [...images, ...other].map((artifact) => {
    const name = escapeHtml(artifact.relative_path || artifact.name);
    if ([".png", ".jpg", ".jpeg"].includes(artifact.suffix)) {
      const imageUrl = `${artifact.url}?v=${encodeURIComponent(String(artifact.mtime || ""))}`;
      return `<a class="artifact image" href="${escapeHtml(artifact.url)}" target="_blank" aria-label="${name}"><img src="${escapeHtml(imageUrl)}" alt="" loading="lazy"><span>${name}</span></a>`;
    }
    return `<a class="artifact" href="${escapeHtml(artifact.url)}" target="_blank"><span>${name}</span></a>`;
  }).join("");
  return artifacts.length;
}

function renderMetrics(job, logText, artifactCount, liveSummary = null) {
  const metrics = $("#metricsGrid");
  const rejected = parseRejectedMetric(logText);
  const timePerMic = timePerMicrographMetric(job, logText, liveSummary);
  const rows = [
    { label: "status", value: statusText(job.status) },
    { label: "runtime", value: durationText(job.started_at || job.created_at, job.ended_at) },
    { label: "artifacts", value: String(artifactCount) },
  ];
  if (timePerMic) rows.push(timePerMic);
  if (rejected) rows.push(rejected);
  metrics.hidden = false;
  metrics.innerHTML = rows.map((row) => `
    <div class="metric ${row.mask ? "metric-mask" : ""}">
      <span>${escapeHtml(row.label)}</span>
      <strong>${escapeHtml(row.value)}</strong>
    </div>
  `).join("");
}

function syncLiveSummaryControls() {
  const mode = $("#liveSummaryMode");
  const count = $("#liveSummaryCount");
  const countWrap = $("#liveSummaryCountWrap");
  if (mode) mode.value = state.liveSummary.mode || "all";
  if (count) count.value = String(state.liveSummary.count || 100);
  if (countWrap) countWrap.hidden = (state.liveSummary.mode || "all") === "all";
}

function liveSummaryJobKind(job) {
  return ["infer", "cryosparc_predict", "type"].includes(job?.kind || "");
}

async function renderLiveSummary(job) {
  const panel = $("#liveSummary");
  const charts = $("#liveSummaryCharts");
  const meta = $("#liveSummaryMeta");
  if (!panel || !charts || !meta) return null;
  if (!state.selected || !liveSummaryJobKind(job)) {
    panel.hidden = true;
    charts.innerHTML = "";
    return null;
  }
  syncLiveSummaryControls();
  panel.hidden = false;
  try {
    const params = new URLSearchParams({
      mode: state.liveSummary.mode || "all",
      count: String(state.liveSummary.count || 100),
    });
    const summary = await api(`/api/jobs/${state.selected}/live-summary?${params.toString()}`);
    if (!summary.available) {
      meta.textContent = summary.message || "Waiting for summary data.";
      charts.innerHTML = [
        renderPieBlock("Clean vs contamination", [], "0%", "Waiting for image rows"),
        renderPieBlock("Type breakdown", [], "0%", "Waiting for typing"),
      ].join("");
      return summary;
    }
    meta.textContent = liveSummaryMeta(summary);
    const contaminationPct = percent(summary.contamination_fraction || 0);
    const totalSegments = [
      { label: "clean", value: Number(summary.clean_pixels || 0), color: "#3FBF9B" },
      { label: "contamination", value: Number(summary.contaminated_pixels || 0), color: "#E0559B" },
    ];
    const typeSegments = (summary.types || [])
      .filter((item) => Number(item.area_px || 0) > 0)
      .map((item) => ({
        label: item.label || "type",
        value: Number(item.area_px || 0),
        color: safeColor(item.color),
      }));
    const typeTotal = typeSegments.reduce((sum, item) => sum + item.value, 0);
    const largestType = typeSegments.reduce((best, item) => item.value > (best?.value || 0) ? item : best, null);
    const typeCenter = largestType ? percent(largestType.value / Math.max(typeTotal, 1)) : "0%";
    const typeSubline = largestType
      ? `${largestType.label} leads | ${formatCount(typeTotal)} px typed`
      : "Typing breakdown not available yet";
    charts.innerHTML = [
      renderPieBlock(
        "Clean vs contamination",
        totalSegments,
        contaminationPct,
        `${formatCount(summary.contaminated_pixels || 0)} px contaminated`,
      ),
      renderPieBlock("Type breakdown", typeSegments, typeCenter, typeSubline),
    ].join("");
    return summary;
  } catch (error) {
    meta.textContent = error.message;
    charts.innerHTML = "";
    return null;
  }
}

function liveSummaryMeta(summary) {
  const mode = summary.mode || "all";
  const range = mode === "all"
    ? "whole data set"
    : `${mode === "last" ? "last" : "first"} ${Number(summary.n_images || 0).toLocaleString()} images`;
  const sourceLabels = {
    typing: "typing",
    multi_gpu_workers: "multi-GPU inference",
    inference: "inference",
  };
  const source = sourceLabels[summary.source] || "inference";
  return `${Number(summary.n_images || 0).toLocaleString()}/${Number(summary.n_images_total || 0).toLocaleString()} images | ${range} | ${source}`;
}

function renderPieBlock(title, segments, center, subline) {
  const total = segments.reduce((sum, segment) => sum + Math.max(0, Number(segment.value || 0)), 0);
  const legend = segments.length
    ? segments.map((segment) => renderLegendRow(segment, total)).join("")
    : `<div class="live-legend-row"><span class="live-swatch"></span><span>waiting</span><strong>0%</strong></div>`;
  return `
    <div class="live-chart">
      <div class="live-pie" style="--pie-slices: ${escapeHtml(pieSlices(segments))}"><span>${escapeHtml(center)}</span></div>
      <div class="live-chart-copy">
        <strong>${escapeHtml(title)}</strong>
        <span>${escapeHtml(subline)}</span>
        <div class="live-legend">${legend}</div>
      </div>
    </div>
  `;
}

function renderLegendRow(segment, total) {
  const fraction = total > 0 ? Number(segment.value || 0) / total : 0;
  return `
    <div class="live-legend-row">
      <span class="live-swatch" style="--swatch: ${escapeHtml(safeColor(segment.color))}"></span>
      <span>${escapeHtml(segment.label)}</span>
      <strong>${escapeHtml(percent(fraction))}</strong>
    </div>
  `;
}

function pieSlices(segments) {
  const usable = segments
    .map((segment) => ({
      value: Math.max(0, Number(segment.value || 0)),
      color: safeColor(segment.color),
    }))
    .filter((segment) => segment.value > 0);
  const total = usable.reduce((sum, segment) => sum + segment.value, 0);
  if (total <= 0) return "var(--line-2) 0deg 360deg";
  let cursor = 0;
  return usable.map((segment, index) => {
    const start = cursor;
    const end = index === usable.length - 1 ? 360 : cursor + (segment.value / total) * 360;
    cursor = end;
    return `${segment.color} ${start.toFixed(2)}deg ${end.toFixed(2)}deg`;
  }).join(", ");
}

function safeColor(color) {
  const text = String(color || "");
  return /^#[0-9a-fA-F]{6}$/.test(text) ? text : "#8DA0AE";
}

function percent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "0%";
  return `${(Math.max(0, Math.min(1, numeric)) * 100).toFixed(1)}%`;
}

function formatCount(value) {
  const numeric = Number(value || 0);
  if (!Number.isFinite(numeric)) return "0";
  return Math.round(numeric).toLocaleString();
}

function parseRejectedMetric(text) {
  const keptRemoved = text.match(/Particle filtering:\s*kept\s*(\d[\d,]*)\/(\d[\d,]*)\s*\((\d[\d,]*)\s+removed\)/i);
  if (keptRemoved) {
    const total = Number(keptRemoved[2].replaceAll(",", ""));
    const removed = Number(keptRemoved[3].replaceAll(",", ""));
    const fraction = total ? ` (${((removed / total) * 100).toFixed(2)}%)` : "";
    return { label: "removed", value: `${removed.toLocaleString()}${fraction}`, mask: true };
  }
  const pair = text.match(/(\d[\d,]*)\s+accepted[\s,;]+(\d[\d,]*)\s+rejected/i);
  if (pair) {
    const accepted = Number(pair[1].replaceAll(",", ""));
    const rejected = Number(pair[2].replaceAll(",", ""));
    const total = accepted + rejected;
    const fraction = total ? ` (${((rejected / total) * 100).toFixed(2)}%)` : "";
    return { label: "rejected", value: `${rejected.toLocaleString()}${fraction}`, mask: true };
  }
  const direct = text.match(/rejected(?: particles| picks| micrographs)?:\s*(\d[\d,]*)/i);
  if (!direct) return null;
  return { label: "rejected", value: Number(direct[1].replaceAll(",", "")).toLocaleString(), mask: true };
}

function isNearBottom(element) {
  return element.scrollHeight - element.scrollTop - element.clientHeight < 24;
}

function formatLog(text, live) {
  const cleaned = text.replace(/\s+$/, "");
  const lines = cleaned ? cleaned.split("\n") : [];
  const body = lines.map(formatLogLine).join("\n");
  const cursor = live ? `${body ? "\n" : ""}<span class="log-cursor" aria-hidden="true"></span>` : "";
  return body + cursor;
}

function formatLogLine(line) {
  const error = /\b(error|failed|traceback|oom|exception)\b/i.test(line);
  let html = escapeHtml(line);
  html = html.replace(/^(\[?\d{4}-\d{2}-\d{2}[^\s\]]*\]?)/, '<span class="log-time">$1</span>');
  html = html.replace(/\b(\d[\d,]*(?:\.\d+)?%?)(?=\s*(?:flagged|rejected|contaminated|contamination))/gi, '<span class="log-mask">$1</span>');
  html = html.replace(/\b(flagged|rejected|contaminated|contamination)\b/gi, '<span class="log-mask">$1</span>');
  return `<span class="log-line ${error ? "error" : ""}">${html}</span>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function launchJob(form) {
  const kind = form.dataset.kind;
  const payload = formPayload(form);
  if (kind === "annotation" && (payload.editor_mode || "browser") !== "desktop") {
    await launchAnnotationSession(form, payload);
    return;
  }
  if (
    (
      kind === "cryosparc_predict" ||
      (kind === "annotation" && payload.source_mode === "cryosparc") ||
      (kind === "train" && payload.mic_source_mode === "cryosparc")
    ) &&
    !state.cryosparcConnected
  ) {
    resetCryosparcConnection();
    setCryosparcStatus("Connect to CryoSPARC before launching.", "error");
    return;
  }
  const job = await api("/api/jobs", {
    method: "POST",
    body: JSON.stringify({ kind, payload }),
  });
  state.selected = job.id;
  state.logOffset = null;
  await loadJobs();
  document.querySelector('[data-tab="monitor"]').click();
}

function openPrimaryLaunchTab() {
  document.querySelector('[data-tab="cryosparc"]').click();
}

function bindTabs() {
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".tab").forEach((item) => item.classList.remove("active"));
      $$(".panel").forEach((panel) => panel.classList.remove("active"));
      tab.classList.add("active");
      $(`#${tab.dataset.tab}`).classList.add("active");
    });
  });
}

function bindForms() {
  $$("form[data-kind]").forEach((form) => {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const button = form.querySelector('button[type="submit"]');
      button.disabled = true;
      try {
        await launchJob(form);
      } catch (error) {
        alert(error.message);
      } finally {
        button.disabled = false;
      }
    });
  });
}

function bindControls() {
  $("#cryosparcConnectForm")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    await connectCryosparc(event.currentTarget);
  });
  $$("[data-action='change-cryosparc']").forEach((button) => {
    button.addEventListener("click", resetCryosparcConnection);
  });
  $("#liveSummaryMode")?.addEventListener("change", (event) => {
    state.liveSummary.mode = event.currentTarget.value || "all";
    syncLiveSummaryControls();
    const job = state.jobs.find((item) => item.id === state.selected);
    if (job) renderLiveSummary(job);
  });
  $("#liveSummaryCount")?.addEventListener("change", (event) => {
    const count = Number(event.currentTarget.value || 100);
    state.liveSummary.count = Number.isFinite(count) ? Math.max(1, Math.min(Math.round(count), 10000)) : 100;
    syncLiveSummaryControls();
    const job = state.jobs.find((item) => item.id === state.selected);
    if (job) renderLiveSummary(job);
  });
  $("#annotationSourceMode")?.addEventListener("change", updateAnnotationSourceFields);
  $("#trainMicrographSourceMode")?.addEventListener("change", updateTrainingMicrographSourceFields);
  $("#annotationRefreshSessions")?.addEventListener("click", () => loadAnnotationSessions());
  $("#annotationResumeSession")?.addEventListener("click", () => resumeAnnotationSession());
  $("#annotationMode")?.addEventListener("change", (event) => setAnnotationMode(event.currentTarget.value));
  $("#annotationAction")?.addEventListener("change", (event) => setAnnotationAction(event.currentTarget.value));
  $("#annotationCanvas")?.addEventListener("click", addAnnotationPoint);
  $("#annotationCanvas")?.addEventListener("mousedown", beginAnnotationEllipse);
  $("#annotationCanvas")?.addEventListener("mousemove", previewAnnotationEllipse);
  $("#annotationCanvas")?.addEventListener("mouseup", finishAnnotationEllipse);
  $("#annotationCanvas")?.addEventListener("mouseleave", leaveAnnotationCanvas);
  window.addEventListener("mousemove", continueAnnotationDrag);
  window.addEventListener("mouseup", finishAnnotationEllipse);
  $("#annotationCanvas")?.addEventListener("dblclick", (event) => {
    event.preventDefault();
    closeAnnotationPolygon();
  });
  $("#annotationOverlayToggle")?.addEventListener("change", drawAnnotationCanvas);
  $("#annotationType")?.addEventListener("change", drawAnnotationCanvas);
  for (const selector of ["#annotationContrast", "#annotationBrightness", "#annotationGamma"]) {
    $(selector)?.addEventListener("input", updateAnnotationViewFromControls);
  }
  $("#annotationInvert")?.addEventListener("change", updateAnnotationViewFromControls);
  $("#annotationResetView")?.addEventListener("click", resetAnnotationView);
  $("#annotationPrev")?.addEventListener("click", () => navigateAnnotationRow(state.annotation.index - 1));
  $("#annotationNext")?.addEventListener("click", () => navigateAnnotationRow(state.annotation.index + 1));
  $("#annotationJump")?.addEventListener("change", (event) => {
    navigateAnnotationRow(Number(event.currentTarget.value || 1) - 1);
  });
  $("#annotationUndo")?.addEventListener("click", undoAnnotationEdit);
  $("#annotationDelete")?.addEventListener("click", deleteSelectedAnnotationShape);
  $("#annotationClosePolygon")?.addEventListener("click", closeAnnotationPolygon);
  $("#annotationClear")?.addEventListener("click", clearAnnotationRow);
  $("#annotationSave")?.addEventListener("click", () => saveAnnotationRow().catch((error) => setAnnotationRowStatus(error.message)));
  $("#annotationSaveNext")?.addEventListener("click", async () => {
    try {
      await saveAnnotationRow();
      if (state.annotation.index + 1 < Number(state.annotation.session?.row_count || 0)) {
        await loadAnnotationRow(state.annotation.index + 1);
      }
    } catch (error) {
      setAnnotationRowStatus(error.message);
    }
  });
  $("#annotationFinalize")?.addEventListener("click", () => {
    finalizeAnnotationSession().catch((error) => setAnnotationLaunchStatus(error.message, "error"));
  });
  window.addEventListener("keydown", (event) => {
    if (!$("#annotation")?.classList.contains("active")) return;
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName || "")) return;
    if (event.key === "Enter") {
      event.preventDefault();
      closeAnnotationPolygon();
    } else if (event.key === "Delete") {
      event.preventDefault();
      deleteSelectedAnnotationShape();
    } else if (event.key === "Backspace") {
      event.preventDefault();
      if (!deleteSelectedAnnotationShape()) undoAnnotationEdit();
    } else if (event.key === "Escape") {
      cancelTransientAnnotationShape();
    } else if (event.key === "ArrowLeft") {
      navigateAnnotationRow(state.annotation.index - 1);
    } else if (event.key === "ArrowRight") {
      navigateAnnotationRow(state.annotation.index + 1);
    }
  });
  $("#refreshButton").addEventListener("click", loadJobs);
  $("#launchRunButton").addEventListener("click", openPrimaryLaunchTab);
  $("#jobLog").addEventListener("scroll", () => {
    state.autoScroll = isNearBottom($("#jobLog"));
  });
  $("#cancelButton").addEventListener("click", async () => {
    if (!state.selected) return;
    await api(`/api/jobs/${state.selected}/cancel`, { method: "POST", body: "{}" });
    await renderSelected();
  });
}

async function boot() {
  bindTabs();
  bindForms();
  bindControls();
  updateAnnotationSourceFields();
  updateTrainingMicrographSourceFields();
  syncAnnotationActionControls();
  await loadAnnotationSessions();
  await loadStatus();
  await loadJobs();
  setInterval(loadJobs, 3000);
}

boot().catch((error) => {
  $("#statusLine").textContent = error.message;
});
