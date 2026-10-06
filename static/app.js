"use strict";

const CARDS_PER_PAGE = 60;
const MAX_TIMELINE_MARKERS = 500;

const elements = {
  dropZone: document.getElementById("dropZone"),
  fileInput: document.getElementById("fileInput"),
  selectedFile: document.getElementById("selectedFile"),
  sampleFps: document.getElementById("sampleFps"),
  recognizer: document.getElementById("recognizer"),
  analyzeButton: document.getElementById("analyzeButton"),
  localPath: document.getElementById("localPath"),
  analyzePathButton: document.getElementById("analyzePathButton"),
  progressPanel: document.getElementById("progressPanel"),
  progressTrack: document.getElementById("progressTrack"),
  progressBar: document.getElementById("progressBar"),
  progressStatus: document.getElementById("progressStatus"),
  progressPercent: document.getElementById("progressPercent"),
  errorBox: document.getElementById("errorBox"),
  resultSection: document.getElementById("resultSection"),
  totalFrames: document.getElementById("totalFrames"),
  maxConfidence: document.getElementById("maxConfidence"),
  resultRecognizer: document.getElementById("resultRecognizer"),
  videoStage: document.getElementById("videoStage"),
  video: document.getElementById("video"),
  overlay: document.getElementById("overlay"),
  videoError: document.getElementById("videoError"),
  ratePreset: document.getElementById("ratePreset"),
  customRate: document.getElementById("customRate"),
  applyCustomRate: document.getElementById("applyCustomRate"),
  currentRate: document.getElementById("currentRate"),
  timelineTrack: document.getElementById("timelineTrack"),
  timelineMarkers: document.getElementById("timelineMarkers"),
  timelinePlayhead: document.getElementById("timelinePlayhead"),
  deleteTimelineFrame: document.getElementById("deleteTimelineFrame"),
  durationLabel: document.getElementById("durationLabel"),
  threshold: document.getElementById("threshold"),
  thresholdValue: document.getElementById("thresholdValue"),
  minConsecutive: document.getElementById("minConsecutive"),
  catRuleText: document.getElementById("catRuleText"),
  filterMode: document.getElementById("filterMode"),
  visibleCount: document.getElementById("visibleCount"),
  emptyResults: document.getElementById("emptyResults"),
  keyframesContainer: document.getElementById("keyframesContainer"),
  pagination: document.getElementById("pagination"),
  previousPage: document.getElementById("previousPage"),
  nextPage: document.getElementById("nextPage"),
  pageInfo: document.getElementById("pageInfo"),
};

let selectedFile = null;
let busy = false;
let pollVersion = 0;
let currentJob = null;
let currentDuration = 0;
let currentPage = 1;
let currentPlaybackRate = 1;
let confidenceThreshold = Number(elements.threshold.value);
let minConsecutiveFrames = 1;
let qualifiedFrameUrls = new Set();
let cardElements = new Map();
let activeIndex = -1;
let selectedTimelineIndex = -1;
let playbackRaf = 0;
let renderRaf = 0;

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function formatTime(value) {
  const totalTenths = Math.max(0, Math.round((Number(value) || 0) * 10));
  const minutes = Math.floor(totalTenths / 600);
  const seconds = Math.floor((totalTenths % 600) / 10);
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${totalTenths % 10}`;
}

function formatPercent(value) {
  return `${(clamp(Number(value) || 0, 0, 1) * 100).toFixed(1)}%`;
}

function formatBytes(value) {
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const unit = Math.min(units.length - 1, Math.floor(Math.log(value) / Math.log(1024)));
  return `${(value / 1024 ** unit).toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function setProgress(value, message) {
  const fraction = clamp(Number(value) || 0, 0, 1);
  const percent = Math.round(fraction * 100);
  elements.progressBar.style.width = `${percent}%`;
  elements.progressTrack.setAttribute("aria-valuenow", String(percent));
  elements.progressPercent.textContent = `${percent}%`;
  if (message) elements.progressStatus.textContent = message;
}

function clearError() {
  elements.errorBox.hidden = true;
  elements.errorBox.textContent = "";
}

function showError(message) {
  elements.errorBox.textContent = message;
  elements.errorBox.hidden = false;
}

function setBusy(value) {
  busy = value;
  elements.fileInput.disabled = value;
  elements.sampleFps.disabled = value;
  elements.recognizer.disabled = value;
  elements.localPath.disabled = value;
  elements.dropZone.classList.toggle("disabled", value);
  elements.dropZone.setAttribute("aria-disabled", String(value));
  elements.analyzeButton.textContent = value ? "正在处理…" : "上传并开始分析";
  elements.analyzeButton.disabled = value || !selectedFile;
  elements.analyzePathButton.textContent = value ? "正在处理…" : "从路径开始分析";
  elements.analyzePathButton.disabled = value || !elements.localPath.value.trim();
}

function selectFile(file) {
  clearError();
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".dav")) {
    selectedFile = null;
    elements.fileInput.value = "";
    elements.selectedFile.textContent = "尚未选择文件";
    showError("文件格式不支持，请选择扩展名为 .dav 的视频。");
  } else if (file.size === 0) {
    selectedFile = null;
    elements.selectedFile.textContent = "尚未选择文件";
    showError("所选文件为空，请重新选择。");
  } else {
    selectedFile = file;
    elements.selectedFile.textContent = `${file.name} · ${formatBytes(file.size)}`;
  }
  setBusy(busy);
}

function responseError(data, fallback) {
  if (data && typeof data.detail === "string") return data.detail;
  if (data && Array.isArray(data.detail)) {
    return data.detail.map((item) => item.msg || String(item)).join("；");
  }
  return fallback;
}

function uploadVideo(file, sampleFps, token) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/jobs");
    request.upload.addEventListener("progress", (event) => {
      if (token !== pollVersion || !event.lengthComputable) return;
      const fraction = event.loaded / event.total;
      setProgress(fraction * 0.45, `正在上传视频（${Math.round(fraction * 100)}%）`);
    });
    request.upload.addEventListener("load", () => {
      if (token === pollVersion) setProgress(0.47, "上传完成，等待服务器响应");
    });
    request.addEventListener("load", () => {
      let data = null;
      try {
        data = JSON.parse(request.responseText);
      } catch (_error) {
        reject(new Error("服务器返回了无法识别的响应"));
        return;
      }
      if (request.status >= 200 && request.status < 300) {
        resolve(data);
      } else {
        reject(new Error(responseError(data, `上传失败（HTTP ${request.status}）`)));
      }
    });
    request.addEventListener("error", () => reject(new Error("上传时网络连接中断，请确认本地服务仍在运行")));
    request.addEventListener("abort", () => reject(new Error("上传已取消")));
    const form = new FormData();
    form.append("file", file);
    form.append("sample_fps", String(sampleFps));
    form.append("recognizer", elements.recognizer.value);
    request.send(form);
  });
}

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function pollJob(jobId, token) {
  let networkFailures = 0;
  while (token === pollVersion) {
    let response;
    try {
      response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, { cache: "no-store" });
      networkFailures = 0;
    } catch (_error) {
      networkFailures += 1;
      if (networkFailures >= 5) throw new Error("多次查询任务状态失败，请检查本地服务后重试");
      elements.progressStatus.textContent = "状态查询中断，正在重试…";
      await sleep(1000);
      continue;
    }
    let job;
    try {
      job = await response.json();
    } catch (_error) {
      throw new Error("服务器返回的任务状态无法解析");
    }
    if (!response.ok) throw new Error(responseError(job, `查询任务失败（HTTP ${response.status}）`));
    const jobProgress = clamp(Number(job.progress) || 0, 0, 1);
    setProgress(0.5 + jobProgress * 0.5, job.message || "正在处理");
    if (job.status === "ready") return job;
    if (job.status === "error") throw new Error(job.message || "分析失败");
    if (job.status !== "queued" && job.status !== "processing") {
      throw new Error(`服务器返回了未知任务状态：${job.status}`);
    }
    await sleep(800);
  }
  throw new Error("任务已被新的分析请求取代");
}

async function startAnalysis() {
  clearError();
  if (!selectedFile) {
    showError("请先选择一个 DAV 视频。");
    return;
  }
  const sampleFps = Number(elements.sampleFps.value);
  if (!Number.isFinite(sampleFps) || sampleFps < 0.5 || sampleFps > 5) {
    showError("采样 FPS 必须在 0.5 到 5 之间。");
    return;
  }
  const token = ++pollVersion;
  setBusy(true);
  elements.progressPanel.hidden = false;
  setProgress(0, "准备上传");
  try {
    const created = await uploadVideo(selectedFile, sampleFps, token);
    if (!created.jobId) throw new Error("服务器未返回任务编号");
    const url = new URL(window.location.href);
    url.searchParams.set("job", created.jobId);
    window.history.replaceState(null, "", url);
    setProgress(0.5, "上传完成，等待分析");
    const result = await pollJob(created.jobId, token);
    if (token !== pollVersion) return;
    setProgress(1, "分析完成");
    showResults(result);
  } catch (error) {
    if (token === pollVersion) {
      elements.progressStatus.textContent = "处理未完成";
      showError(error instanceof Error ? error.message : String(error));
    }
  } finally {
    if (token === pollVersion) setBusy(false);
  }
}

async function startPathAnalysis() {
  clearError();
  const localPath = elements.localPath.value.trim().replace(/^(['"])(.*)\1$/, "$2").trim();
  if (!localPath) {
    showError("请输入 DAV 文件的绝对路径。");
    return;
  }
  if (!localPath.toLowerCase().endsWith(".dav")) {
    showError("文件路径必须以 .dav 结尾。");
    return;
  }
  const sampleFps = Number(elements.sampleFps.value);
  if (!Number.isFinite(sampleFps) || sampleFps < 0.5 || sampleFps > 5) {
    showError("采样 FPS 必须在 0.5 到 5 之间。");
    return;
  }
  const token = ++pollVersion;
  setBusy(true);
  elements.progressPanel.hidden = false;
  setProgress(0, "正在连接本机文件");
  try {
    const response = await fetch("/api/jobs/local", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        path: localPath,
        sample_fps: sampleFps,
        recognizer: elements.recognizer.value,
      }),
    });
    let created;
    try {
      created = await response.json();
    } catch (_error) {
      throw new Error("服务器返回了无法识别的响应");
    }
    if (!response.ok) {
      throw new Error(responseError(created, `读取文件失败（HTTP ${response.status}）`));
    }
    if (!created.jobId) throw new Error("服务器未返回任务编号");
    const url = new URL(window.location.href);
    url.searchParams.set("job", created.jobId);
    window.history.replaceState(null, "", url);
    setProgress(0.5, "已连接本机文件，等待分析");
    const result = await pollJob(created.jobId, token);
    if (token !== pollVersion) return;
    setProgress(1, "分析完成");
    showResults(result);
  } catch (error) {
    if (token === pollVersion) {
      elements.progressStatus.textContent = "处理未完成";
      showError(error instanceof Error ? error.message : String(error));
    }
  } finally {
    if (token === pollVersion) setBusy(false);
  }
}

async function resumeJob(jobId) {
  clearError();
  const token = ++pollVersion;
  setBusy(true);
  elements.progressPanel.hidden = false;
  setProgress(0.5, "正在恢复已有的分析任务");
  try {
    const result = await pollJob(jobId, token);
    if (token !== pollVersion) return;
    setProgress(1, "分析完成");
    showResults(result);
  } catch (error) {
    if (token === pollVersion) {
      elements.progressStatus.textContent = "任务恢复失败";
      showError(error instanceof Error ? error.message : String(error));
    }
  } finally {
    if (token === pollVersion) setBusy(false);
  }
}

function frameCatConfidence(frame) {
  if (!Array.isArray(frame.boxes)) return 0;
  return frame.boxes.reduce(
    (maximum, box) => Math.max(maximum, Number(box.catConfidence) || 0),
    0,
  );
}

function rebuildCatQualification() {
  qualifiedFrameUrls = new Set();
  if (!currentJob) return;
  const sampleFps = Number(currentJob.sampleFps) || 2;
  const maximumGap = Math.max(0.75, 1.5 / sampleFps);
  let window = [];
  let scoreSum = 0;
  for (const frame of currentJob.keyframes) {
    const previous = window[window.length - 1];
    if (previous && frame.time - previous.frame.time > maximumGap) {
      window = [];
      scoreSum = 0;
    }
    const score = frameCatConfidence(frame);
    window.push({ frame, score });
    scoreSum += score;
    if (window.length > minConsecutiveFrames) {
      const removed = window.shift();
      scoreSum -= removed.score;
    }
    if (window.length === minConsecutiveFrames && scoreSum / minConsecutiveFrames >= confidenceThreshold) {
      for (const item of window) qualifiedFrameUrls.add(item.frame.imageUrl);
    }
  }
}

function isCatFrame(frame) {
  return qualifiedFrameUrls.has(frame.imageUrl);
}

function visibleFrames() {
  if (!currentJob) return [];
  const onlyCats = elements.filterMode.value === "cat";
  const visible = [];
  currentJob.keyframes.forEach((frame, index) => {
    if (!onlyCats || isCatFrame(frame)) visible.push({ frame, index });
  });
  return visible;
}

function sampleEvenly(items, limit) {
  if (items.length <= limit) return items;
  if (limit <= 0) return [];
  if (limit === 1) return [items[0]];
  const sampled = [];
  let previousPosition = -1;
  for (let index = 0; index < limit; index += 1) {
    const position = Math.round(index * (items.length - 1) / (limit - 1));
    if (position !== previousPosition) sampled.push(items[position]);
    previousPosition = position;
  }
  return sampled;
}

function markerSample(visible) {
  const selected = sampleEvenly(
    visible.filter(({ frame }) => isCatFrame(frame)),
    MAX_TIMELINE_MARKERS,
  );
  const selectedItems = new Set(selected);
  if (selected.length < MAX_TIMELINE_MARKERS) {
    const remaining = visible.filter((item) => !selectedItems.has(item));
    selected.push(...sampleEvenly(remaining, MAX_TIMELINE_MARKERS - selected.length));
  }
  return selected.sort((first, second) => first.frame.time - second.frame.time);
}

function updateTimelineSelection() {
  if (!currentJob || !currentJob.keyframes[selectedTimelineIndex]) {
    selectedTimelineIndex = -1;
  }
  elements.timelineMarkers.querySelectorAll(".timeline-marker").forEach((marker) => {
    marker.classList.toggle("selected", Number(marker.dataset.index) === selectedTimelineIndex);
  });
  elements.deleteTimelineFrame.disabled = selectedTimelineIndex < 0;
  elements.deleteTimelineFrame.title = selectedTimelineIndex >= 0
    ? `删除 ${formatTime(currentJob.keyframes[selectedTimelineIndex].time)} 的假阳性关键帧`
    : "先点击红色猫标记进行选择";
}

function selectTimelineFrame(index) {
  selectedTimelineIndex = index;
  updateTimelineSelection();
}

function renderMarkers(visible) {
  const fragment = document.createDocumentFragment();
  const catFrames = visible.filter(({ frame }) => isCatFrame(frame));
  for (const { frame, index } of markerSample(catFrames)) {
    const marker = document.createElement("button");
    marker.type = "button";
    marker.className = "timeline-marker cat";
    const position = currentDuration > 0 ? clamp(frame.time / currentDuration, 0, 1) : 0;
    marker.style.left = `${position * 100}%`;
    marker.title = `${formatTime(frame.time)} · 疑似猫`;
    marker.setAttribute("aria-label", `跳转到 ${formatTime(frame.time)}`);
    marker.addEventListener("click", (event) => {
      event.stopPropagation();
      selectTimelineFrame(index);
      seekTo(frame.time);
    });
    marker.dataset.index = String(index);
    fragment.appendChild(marker);
  }
  elements.timelineMarkers.replaceChildren(fragment);
  updateTimelineSelection();
}

function renderCards(visible) {
  const fragment = document.createDocumentFragment();
  cardElements = new Map();
  for (const { frame, index } of visible) {
    const card = document.createElement("article");
    card.className = `frame-card${isCatFrame(frame) ? " cat-frame" : ""}`;
    card.dataset.index = String(index);
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-label", `播放 ${formatTime(frame.time)} 的关键帧`);

    const image = document.createElement("img");
    image.src = frame.imageUrl;
    image.alt = `${formatTime(frame.time)} 的检测关键帧`;
    image.loading = "lazy";
    card.appendChild(image);

    const body = document.createElement("div");
    body.className = "frame-card-body";
    const heading = document.createElement("div");
    heading.className = "frame-card-heading";
    const time = document.createElement("strong");
    time.textContent = formatTime(frame.time);
    const badge = document.createElement("span");
    badge.className = `frame-badge${isCatFrame(frame) ? " cat" : ""}`;
    badge.textContent = isCatFrame(frame) ? "疑似猫" : "普通移动";
    const actions = document.createElement("div");
    actions.className = "frame-actions";
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "delete-frame";
    deleteButton.textContent = "删除";
    deleteButton.title = "删除这个假阳性关键帧";
    deleteButton.setAttribute("aria-label", `删除 ${formatTime(frame.time)} 的假阳性关键帧`);
    actions.append(badge, deleteButton);
    heading.append(time, actions);
    body.appendChild(heading);

    const objects = document.createElement("ul");
    objects.className = "object-list";
    frame.boxes.forEach((box, boxIndex) => {
      const item = document.createElement("li");
      const label = document.createElement("span");
      label.className = "object-label";
      label.textContent = `对象 ${boxIndex + 1} · ${box.topLabel}`;
      label.title = box.topLabel;
      const confidence = document.createElement("strong");
      const isCat = Number(box.catConfidence) >= confidenceThreshold;
      confidence.className = isCat ? "confidence cat" : "confidence";
      confidence.textContent = `猫 ${formatPercent(box.catConfidence)}`;
      item.append(label, confidence);
      objects.appendChild(item);
    });
    body.appendChild(objects);
    card.appendChild(body);
    cardElements.set(index, card);
    fragment.appendChild(card);
  }
  elements.keyframesContainer.replaceChildren(fragment);
}

async function deleteKeyframe(index, button) {
  if (!currentJob) return;
  const job = currentJob;
  const frame = job.keyframes[index];
  if (!frame) return;
  const approved = window.confirm(
    `删除 ${formatTime(frame.time)} 的假阳性关键帧？\n只会删除检测结果，不会影响视频。`,
  );
  if (!approved) return;
  const filename = frame.imageUrl.split("/").pop();
  button.disabled = true;
  try {
    const response = await fetch(
      `/api/jobs/${encodeURIComponent(job.jobId)}/keyframes/${encodeURIComponent(filename)}`,
      { method: "DELETE" },
    );
    const data = await response.json().catch(() => null);
    if (!response.ok) throw new Error(responseError(data, `删除失败（HTTP ${response.status}）`));
    if (currentJob !== job) return;
    job.keyframes = job.keyframes.filter((candidate) => candidate.imageUrl !== frame.imageUrl);
    elements.totalFrames.textContent = String(job.keyframes.length);
    let maximum = 0;
    for (const keyframe of job.keyframes) {
      for (const box of keyframe.boxes) {
        maximum = Math.max(maximum, Number(box.catConfidence) || 0);
      }
    }
    elements.maxConfidence.textContent = job.keyframes.length ? formatPercent(maximum) : "—";
    renderResults();
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
  } finally {
    if (button.isConnected && button.classList.contains("delete-frame")) {
      button.disabled = false;
    }
    updateTimelineSelection();
  }
}

function renderResults() {
  if (!currentJob) return;
  selectTimelineFrame(-1);
  rebuildCatQualification();
  elements.catRuleText.textContent = minConsecutiveFrames > 1
    ? `连续 ${minConsecutiveFrames} 帧平均 ≥${formatPercent(confidenceThreshold)}`
    : `单帧 ≥${formatPercent(confidenceThreshold)}`;
  const visible = visibleFrames();
  const pageCount = Math.max(1, Math.ceil(visible.length / CARDS_PER_PAGE));
  currentPage = clamp(Math.round(currentPage), 1, pageCount);
  const start = (currentPage - 1) * CARDS_PER_PAGE;
  const end = Math.min(visible.length, start + CARDS_PER_PAGE);
  renderMarkers(visible);
  renderCards(visible.slice(start, end));
  elements.visibleCount.textContent = visible.length
    ? `筛选出 ${visible.length} / ${currentJob.keyframes.length} 个，当前 ${start + 1}–${end}`
    : `显示 0 / ${currentJob.keyframes.length} 个`;
  elements.emptyResults.hidden = visible.length > 0;
  elements.emptyResults.textContent = currentJob.keyframes.length === 0
    ? "没有检测到明显移动。可以尝试提高采样 FPS，或确认摄像头画面保持稳定。"
    : "当前筛选条件下没有关键帧，可以降低阈值、减少连续帧要求或查看全部移动。";
  elements.pagination.hidden = visible.length <= CARDS_PER_PAGE;
  elements.previousPage.disabled = currentPage <= 1;
  elements.nextPage.disabled = currentPage >= pageCount;
  elements.pageInfo.textContent = `${currentPage} / ${pageCount} 页`;
  activeIndex = -1;
  updatePlayback();
}

function showResults(job) {
  if (!Array.isArray(job.keyframes) || !job.videoUrl) {
    showError("分析结果缺少视频或关键帧数据。");
    return;
  }
  currentJob = job;
  currentDuration = Number(job.duration) || 0;
  currentPage = 1;
  const recognizer = job.recognizer === "catfinder" ? "catfinder" : "clip";
  elements.recognizer.value = recognizer;
  elements.resultRecognizer.textContent = recognizer === "catfinder" ? "CatFinder" : "CLIP";
  elements.totalFrames.textContent = String(job.keyframes.length);
  let maximum = 0;
  for (const frame of job.keyframes) {
    for (const box of frame.boxes) maximum = Math.max(maximum, Number(box.catConfidence) || 0);
  }
  elements.maxConfidence.textContent = job.keyframes.length ? formatPercent(maximum) : "—";
  elements.durationLabel.textContent = formatTime(currentDuration);
  elements.videoError.hidden = true;
  elements.video.src = job.videoUrl;
  elements.video.load();
  elements.video.defaultPlaybackRate = currentPlaybackRate;
  elements.video.playbackRate = currentPlaybackRate;
  elements.resultSection.hidden = false;
  renderResults();
  elements.resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function nearestKeyframeIndex(time) {
  if (!currentJob || currentJob.keyframes.length === 0) return -1;
  const frames = currentJob.keyframes;
  let low = 0;
  let high = frames.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (frames[middle].time < time) low = middle + 1;
    else high = middle;
  }
  if (low === 0) return 0;
  if (low >= frames.length) return frames.length - 1;
  return Math.abs(frames[low].time - time) < Math.abs(frames[low - 1].time - time)
    ? low
    : low - 1;
}

function drawOverlay(nearestIndex) {
  const canvas = elements.overlay;
  const stageRect = elements.videoStage.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const pixelWidth = Math.max(1, Math.round(stageRect.width * ratio));
  const pixelHeight = Math.max(1, Math.round(stageRect.height * ratio));
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, stageRect.width, stageRect.height);
  if (!currentJob || nearestIndex < 0 || !elements.video.videoWidth || !elements.video.videoHeight) return;

  const frame = currentJob.keyframes[nearestIndex];
  const sampleFps = Number(currentJob.sampleFps) || 2;
  const overlayWindow = Math.max(0.75, 1 / sampleFps);
  if (Math.abs(frame.time - elements.video.currentTime) > overlayWindow) return;

  const videoRect = elements.video.getBoundingClientRect();
  const scale = Math.min(
    videoRect.width / elements.video.videoWidth,
    videoRect.height / elements.video.videoHeight,
  );
  if (!Number.isFinite(scale) || scale <= 0) return;
  const displayWidth = elements.video.videoWidth * scale;
  const displayHeight = elements.video.videoHeight * scale;
  const originX = videoRect.left - stageRect.left + (videoRect.width - displayWidth) / 2;
  const originY = videoRect.top - stageRect.top + (videoRect.height - displayHeight) / 2;

  for (const box of frame.boxes) {
    const x = originX + Number(box.x) * displayWidth;
    const y = originY + Number(box.y) * displayHeight;
    const width = Number(box.w) * displayWidth;
    const height = Number(box.h) * displayHeight;
    const isCat = Number(box.catConfidence) >= confidenceThreshold;
    const color = isCat ? "#ed536c" : "#19a7ce";
    context.lineWidth = 2;
    context.strokeStyle = color;
    context.fillStyle = isCat ? "rgba(237, 83, 108, 0.10)" : "rgba(25, 167, 206, 0.08)";
    context.fillRect(x, y, width, height);
    context.strokeRect(x, y, width, height);
    if (isCat) {
      const text = `猫 ${Math.round(Number(box.catConfidence) * 100)}%`;
      context.font = "600 12px system-ui, sans-serif";
      const textWidth = context.measureText(text).width;
      const labelY = y >= 25 ? y - 23 : y + 3;
      context.fillStyle = color;
      context.fillRect(x, labelY, textWidth + 12, 20);
      context.fillStyle = "#ffffff";
      context.fillText(text, x + 6, labelY + 14);
    }
  }
}

function updatePlayback() {
  const time = elements.video.currentTime || 0;
  const nextIndex = nearestKeyframeIndex(time);
  if (nextIndex !== activeIndex) {
    cardElements.get(activeIndex)?.classList.remove("active");
    activeIndex = nextIndex;
    cardElements.get(activeIndex)?.classList.add("active");
  }
  const position = currentDuration > 0 ? clamp(time / currentDuration, 0, 1) : 0;
  elements.timelinePlayhead.style.left = `${position * 100}%`;
  elements.timelineTrack.setAttribute("aria-valuenow", String(Math.round(position * 100)));
  drawOverlay(nextIndex);
}

function seekTo(time) {
  const seekAndPlay = () => {
    const maximum = currentDuration > 0 ? Math.max(0, currentDuration - 0.05) : time;
    elements.video.currentTime = clamp(Number(time) || 0, 0, maximum);
    const playRequest = elements.video.play();
    if (playRequest) playRequest.catch(() => {});
  };
  if (elements.video.readyState >= 1) {
    seekAndPlay();
  } else {
    elements.video.addEventListener("loadedmetadata", seekAndPlay, { once: true });
  }
}

function startPlaybackLoop() {
  if (playbackRaf) return;
  const tick = () => {
    playbackRaf = 0;
    updatePlayback();
    if (!elements.video.paused && !elements.video.ended) {
      playbackRaf = requestAnimationFrame(tick);
    }
  };
  playbackRaf = requestAnimationFrame(tick);
}

function stopPlaybackLoop() {
  if (playbackRaf) cancelAnimationFrame(playbackRaf);
  playbackRaf = 0;
  updatePlayback();
}

function applyPlaybackRate(value, custom = false) {
  const rate = Number(value);
  if (!Number.isFinite(rate) || rate < 0.1 || rate > 16) {
    elements.customRate.setCustomValidity("请输入 0.1 到 16 之间的播放速率");
    elements.customRate.reportValidity();
    return;
  }
  elements.customRate.setCustomValidity("");
  currentPlaybackRate = rate;
  elements.video.playbackRate = rate;
  elements.video.defaultPlaybackRate = rate;
  elements.currentRate.textContent = `当前 ${Number(rate.toFixed(2))}×`;
  if (custom) elements.ratePreset.value = "custom";
}

elements.ratePreset.addEventListener("change", () => {
  if (elements.ratePreset.value === "custom") {
    elements.customRate.focus();
    elements.customRate.select();
    return;
  }
  elements.customRate.value = "";
  applyPlaybackRate(elements.ratePreset.value);
});
elements.applyCustomRate.addEventListener("click", () => {
  applyPlaybackRate(elements.customRate.value, true);
});
elements.customRate.addEventListener("input", () => {
  elements.customRate.setCustomValidity("");
});
elements.customRate.addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  event.preventDefault();
  applyPlaybackRate(elements.customRate.value, true);
});

elements.dropZone.addEventListener("click", (event) => {
  if (!busy && event.target !== elements.fileInput) elements.fileInput.click();
});
elements.dropZone.addEventListener("keydown", (event) => {
  if (!busy && (event.key === "Enter" || event.key === " ")) {
    event.preventDefault();
    elements.fileInput.click();
  }
});
elements.dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  if (!busy) elements.dropZone.classList.add("dragging");
});
elements.dropZone.addEventListener("dragleave", () => elements.dropZone.classList.remove("dragging"));
elements.dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.dropZone.classList.remove("dragging");
  if (!busy) selectFile(event.dataTransfer?.files[0]);
});
elements.fileInput.addEventListener("change", () => selectFile(elements.fileInput.files[0]));
elements.analyzeButton.addEventListener("click", startAnalysis);
elements.analyzePathButton.addEventListener("click", startPathAnalysis);
elements.localPath.addEventListener("input", () => {
  if (!busy) elements.analyzePathButton.disabled = !elements.localPath.value.trim();
});
elements.localPath.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !busy && elements.localPath.value.trim()) {
    event.preventDefault();
    startPathAnalysis();
  }
});

elements.threshold.addEventListener("input", () => {
  confidenceThreshold = Number(elements.threshold.value);
  currentPage = 1;
  elements.thresholdValue.textContent = formatPercent(confidenceThreshold);
  if (!renderRaf) {
    renderRaf = requestAnimationFrame(() => {
      renderRaf = 0;
      renderResults();
    });
  }
});
elements.minConsecutive.addEventListener("change", () => {
  const value = Math.round(Number(elements.minConsecutive.value));
  minConsecutiveFrames = Number.isFinite(value) ? clamp(value, 1, 50) : 1;
  elements.minConsecutive.value = String(minConsecutiveFrames);
  currentPage = 1;
  renderResults();
});
elements.filterMode.addEventListener("change", () => {
  currentPage = 1;
  renderResults();
});
elements.previousPage.addEventListener("click", () => {
  if (currentPage <= 1) return;
  currentPage -= 1;
  renderResults();
  elements.keyframesContainer.scrollIntoView({ behavior: "smooth", block: "start" });
});
elements.nextPage.addEventListener("click", () => {
  const visible = visibleFrames();
  if (currentPage >= Math.ceil(visible.length / CARDS_PER_PAGE)) return;
  currentPage += 1;
  renderResults();
  elements.keyframesContainer.scrollIntoView({ behavior: "smooth", block: "start" });
});

elements.deleteTimelineFrame.addEventListener("click", () => {
  if (selectedTimelineIndex < 0) return;
  deleteKeyframe(selectedTimelineIndex, elements.deleteTimelineFrame);
});

elements.keyframesContainer.addEventListener("click", (event) => {
  const deleteButton = event.target.closest(".delete-frame");
  const card = event.target.closest(".frame-card");
  if (!card || !currentJob) return;
  if (deleteButton) {
    event.preventDefault();
    event.stopPropagation();
    deleteKeyframe(Number(card.dataset.index), deleteButton);
    return;
  }
  selectTimelineFrame(-1);
  const index = Number(card.dataset.index);
  if (currentJob.keyframes[index]) seekTo(currentJob.keyframes[index].time);
});
elements.keyframesContainer.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  if (event.target.closest(".delete-frame")) return;
  const card = event.target.closest(".frame-card");
  if (!card || !currentJob) return;
  event.preventDefault();
  selectTimelineFrame(-1);
  const index = Number(card.dataset.index);
  if (currentJob.keyframes[index]) seekTo(currentJob.keyframes[index].time);
});

elements.timelineTrack.addEventListener("click", (event) => {
  if (currentDuration <= 0) return;
  selectTimelineFrame(-1);
  const rect = elements.timelineTrack.getBoundingClientRect();
  const fraction = clamp((event.clientX - rect.left) / rect.width, 0, 1);
  seekTo(fraction * currentDuration);
});
elements.timelineTrack.addEventListener("keydown", (event) => {
  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
  event.preventDefault();
  const offset = event.key === "ArrowLeft" ? -1 : 1;
  seekTo(Math.max(0, elements.video.currentTime + offset));
});

elements.video.addEventListener("loadedmetadata", () => {
  if (Number.isFinite(elements.video.duration) && elements.video.duration > 0) {
    currentDuration = elements.video.duration;
    elements.durationLabel.textContent = formatTime(currentDuration);
    renderMarkers(visibleFrames());
  }
  updatePlayback();
});
elements.video.addEventListener("timeupdate", updatePlayback);
elements.video.addEventListener("seeked", updatePlayback);
elements.video.addEventListener("play", startPlaybackLoop);
elements.video.addEventListener("pause", stopPlaybackLoop);
elements.video.addEventListener("ended", stopPlaybackLoop);
elements.video.addEventListener("error", () => {
  elements.videoError.hidden = false;
});
elements.video.addEventListener("loadeddata", () => {
  elements.videoError.hidden = true;
  updatePlayback();
});
window.addEventListener("resize", updatePlayback);
if ("ResizeObserver" in window) {
  new ResizeObserver(updatePlayback).observe(elements.videoStage);
}

const requestedJobId = new URLSearchParams(window.location.search).get("job");
if (requestedJobId) {
  if (/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(requestedJobId)) {
    resumeJob(requestedJobId);
  } else {
    showError("链接中的任务编号无效。");
  }
}
