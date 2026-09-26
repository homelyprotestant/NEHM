const form = document.querySelector("#upload-form");
const imageInput = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const dropCopy = document.querySelector("#drop-copy");
const uploadPreview = document.querySelector("#upload-preview");
const resultPreview = document.querySelector("#result-preview");
const submitButton = document.querySelector("#submit-button");
const formError = document.querySelector("#form-error");
const progressPanel = document.querySelector("#progress-panel");
const progressBar = document.querySelector("#progress-bar");
const progressTrack = document.querySelector("#progress-track");
const progressPercent = document.querySelector("#progress-percent");
const progressTitle = document.querySelector("#progress-title");
const progressDetail = document.querySelector("#progress-detail");
const cancelButton = document.querySelector("#cancel-job");
const results = document.querySelector("#results");
const runtimeStatus = document.querySelector("#runtime-status");

const stageOrder = ["validation", "embedding", "neighbors", "atoms", "vision", "synthesis"];
let selectedFile = null;
let previewUrl = null;
let activeJob = null;
let eventSource = null;
let pollTimer = null;

function setError(message = "") {
  formError.textContent = message;
  formError.hidden = !message;
}

function setSelectedFile(file) {
  if (!file) return;
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  uploadPreview.src = previewUrl;
  resultPreview.src = previewUrl;
  uploadPreview.hidden = false;
  dropCopy.hidden = true;
  setError("");
}

imageInput.addEventListener("change", () => setSelectedFile(imageInput.files[0]));

["dragenter", "dragover"].forEach((name) => {
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((name) => {
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) setSelectedFile(file);
});

function setProgress(event) {
  const percent = Math.max(0, Math.min(100, Number(event.percent || 0)));
  progressBar.style.width = `${percent}%`;
  progressPercent.textContent = `${percent}%`;
  progressTrack.setAttribute("aria-valuenow", String(percent));
  progressTitle.textContent =
    event.stage === "complete"
      ? "Interpretation complete"
      : event.stage === "failed"
        ? "Interpretation failed"
        : "Analyzing your microscopy image";
  progressDetail.textContent = event.message || "Working…";

  const activeIndex = stageOrder.indexOf(event.stage);
  document.querySelectorAll("#progress-steps li").forEach((item, index) => {
    item.classList.toggle("complete", activeIndex > index || event.stage === "complete");
    item.classList.toggle("active", activeIndex === index);
  });
}

function stopMonitoring() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  if (pollTimer) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch (_) {
      // Keep HTTP status.
    }
    throw new Error(message);
  }
  return response.json();
}

async function pollStatus() {
  if (!activeJob) return;
  try {
    const status = await fetchJson(activeJob.status_url);
    setProgress(status);
    if (status.status === "complete") {
      await loadResult();
      return;
    }
    if (status.status === "failed" || status.status === "cancelled") {
      finishWithError(status.error || status.message);
      return;
    }
    pollTimer = window.setTimeout(pollStatus, 1500);
  } catch (error) {
    finishWithError(error.message);
  }
}

function monitorJob() {
  eventSource = new EventSource(activeJob.events_url);
  eventSource.onmessage = async (message) => {
    const event = JSON.parse(message.data);
    setProgress(event);
    if (event.status === "complete") {
      stopMonitoring();
      await loadResult();
    } else if (event.status === "failed" || event.status === "cancelled") {
      stopMonitoring();
      try {
        const status = await fetchJson(activeJob.status_url);
        finishWithError(status.error || status.message);
      } catch (error) {
        finishWithError(error.message);
      }
    }
  };
  eventSource.onerror = () => {
    stopMonitoring();
    pollStatus();
  };
}

function addDetail(container, title, value) {
  if (value === undefined || value === null || value === "") return;
  const item = document.createElement("div");
  item.className = "detail-item";
  const heading = document.createElement("strong");
  heading.textContent = title;
  const content = document.createElement("span");
  content.textContent = Array.isArray(value) ? value.join("; ") : String(value);
  item.append(heading, content);
  container.append(item);
}

function renderVision(vision) {
  const container = document.querySelector("#vision-observations");
  container.replaceChildren();
  const morphology = vision.morphology || {};
  addDetail(container, "Morphology", morphology.primary);
  addDetail(container, "Morphology details", morphology.notes);
  addDetail(container, "Color and tone", vision.color_and_tone);
  addDetail(container, "Size and distribution", vision.size_and_distribution);
  addDetail(container, "Optical cues", vision.optical_cues);
  const estimate = vision.magnification_estimate || {};
  addDetail(container, "Vision magnification estimate", estimate.estimate);
  addDetail(container, "Estimate basis", estimate.basis);
}

function renderAtoms(payload) {
  const container = document.querySelector("#atom-contributions");
  container.replaceChildren();
  const labels = new Map(
    (payload.weighted_atoms || []).map((atom) => [Number(atom.atom_id), atom])
  );
  const contributions = payload.atom_contributions || [];
  if (!contributions.length) {
    addDetail(container, "Atom evidence", "No atom contribution was selected for the final prose.");
    return;
  }
  contributions.forEach((entry) => {
    const atom = labels.get(Number(entry.atom_id)) || {};
    const title = `Atom ${entry.atom_id}: ${atom.label || "Dictionary feature"}`;
    const percent = Number(atom.importance_percent || entry.importance_percent || 0);
    addDetail(container, `${title} (${percent.toFixed(1)}%)`, entry.contribution);
  });
}

function renderNeighbors(neighbors) {
  const body = document.querySelector("#neighbors-body");
  body.replaceChildren();
  document.querySelector("#neighbor-count").textContent = String(neighbors.length);
  neighbors.forEach((neighbor, index) => {
    const row = document.createElement("tr");
    const values = [
      index + 1,
      neighbor.specimen || "—",
      neighbor.chemical_formula || "—",
      neighbor.illumination || "—",
      neighbor.magnification || "—",
      Number(neighbor.d2).toFixed(3),
    ];
    values.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.append(cell);
    });
    body.append(row);
  });
}

function renderResult(payload) {
  document.querySelector("#microscopy-description").textContent =
    payload.microscopy_description || "Description unavailable.";
  document.querySelector("#established-type").textContent =
    payload.established_type || "—";
  const context = payload.upload_inference?.context_inference || {};
  document.querySelector("#illumination").textContent =
    payload.target?.inferred_illumination_modality || context.illumination || "—";
  document.querySelector("#magnification").textContent =
    payload.magnification_estimate || context.magnification || "—";
  document.querySelector("#formula").textContent =
    payload.suggested_chemical_formula || "—";
  renderNeighbors(payload.faiss_neighbors || []);
  renderVision(payload.direct_visual_observations || {});
  renderAtoms(payload);
  document.querySelector("#fallback-note").hidden =
    !payload.pipeline?.synthesis_fallback_used;
  document.querySelector("#download-link").href = activeJob.download_url;
  results.hidden = false;
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function loadResult() {
  try {
    const payload = await fetchJson(activeJob.result_url);
    renderResult(payload);
    progressPanel.hidden = true;
    submitButton.disabled = false;
    cancelButton.hidden = true;
  } catch (error) {
    finishWithError(error.message);
  }
}

function finishWithError(message) {
  stopMonitoring();
  setError(message || "The interpretation did not complete.");
  progressPanel.hidden = true;
  submitButton.disabled = false;
  cancelButton.hidden = true;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFile) {
    setError("Choose a microscopy image first.");
    return;
  }
  stopMonitoring();
  setError("");
  results.hidden = true;
  submitButton.disabled = true;
  cancelButton.hidden = false;
  progressPanel.hidden = false;
  setProgress({ stage: "validation", percent: 2, message: "Uploading and validating image…" });

  const data = new FormData();
  data.append("image", selectedFile);
  data.append("coordinator", "openai");
  try {
    activeJob = await fetchJson("/api/jobs", { method: "POST", body: data });
    monitorJob();
  } catch (error) {
    finishWithError(error.message);
  }
});

cancelButton.addEventListener("click", async () => {
  if (!activeJob) return;
  cancelButton.disabled = true;
  try {
    await fetchJson(activeJob.cancel_url, { method: "POST" });
    finishWithError("Interpretation cancelled.");
  } catch (error) {
    setError(error.message);
  } finally {
    cancelButton.disabled = false;
  }
});

async function checkHealth() {
  try {
    const health = await fetchJson("/api/health");
    runtimeStatus.classList.add("ready");
    runtimeStatus.innerHTML = "";
    const dot = document.createElement("i");
    dot.setAttribute("aria-hidden", "true");
    runtimeStatus.append(dot, ` Ready on ${health.runtime.device}`);
  } catch (error) {
    runtimeStatus.textContent = "Runtime unavailable";
    setError(`PLM runtime is unavailable: ${error.message}`);
  }
}

checkHealth();

