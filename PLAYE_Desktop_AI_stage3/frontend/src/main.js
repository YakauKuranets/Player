import { Orchestrator } from "./orchestrator.js";
import { formatTime, hashFileStream } from "./utils.js";
import { createPlaylistBlueprint } from "./blueprints/playlist.js";
import { createPlayerBlueprint } from "./blueprints/player.js";
import { createScreenshotBlueprint } from "./blueprints/screenshot.js";
import { createClipBlueprint } from "./blueprints/clip.js";
import { createQualityBlueprint } from "./blueprints/quality.js";
import { createMotionBlueprint } from "./blueprints/motion.js";
import { createForensicBlueprint } from "./blueprints/forensic.js";
import { createTimelineBlueprint } from "./blueprints/timeline.js";
import { createAiBlueprint } from "./blueprints/ai.js";
import { createHypothesisBlueprint } from "./blueprints/hypothesis.js";
import { createPhotoBlueprint } from "./blueprints/photo.js";
import { createCompareBlueprint } from "./blueprints/compare.js";
import { createApiClient } from "./api-client.js";

const elements = {
  fileInput: document.getElementById("file-input"),
  playlist: document.getElementById("playlist"),
  video: document.getElementById("video"),
  viewerSurface: document.getElementById("viewer-surface"),
  speedInput: document.getElementById("speed"),
  speedValue: document.getElementById("speed-value"),
  frameBack: document.getElementById("frame-back"),
  frameForward: document.getElementById("frame-forward"),
  screenshotButton: document.getElementById("screenshot"),
  captureCanvas: document.getElementById("capture"),
  markInButton: document.getElementById("mark-in"),
  markOutButton: document.getElementById("mark-out"),
  clipInValue: document.getElementById("clip-in"),
  clipOutValue: document.getElementById("clip-out"),
  exportClipButton: document.getElementById("export-clip"),
  enhanceInput: document.getElementById("enhance"),
  exposureInput: document.getElementById("exposure"),
  temperatureInput: document.getElementById("temperature"),
  denoiseInput: document.getElementById("denoise"),
  temporalDenoiseToggle: document.getElementById("temporal-denoise"),
  temporalWindowInput: document.getElementById("temporal-window"),
  denoiseProfile: document.getElementById("denoise-profile"),
  clarityInput: document.getElementById("clarity"),
  sharpnessInput: document.getElementById("sharpness"),
  lowlightBoostToggle: document.getElementById("lowlight-boost"),
  upscaleToggle: document.getElementById("upscale"),
  upscaleFactor: document.getElementById("upscale-factor"),
  grayscaleToggle: document.getElementById("grayscale"),
  bypassFiltersToggle: document.getElementById("bypass-filters"),
  resetFiltersButton: document.getElementById("reset-filters"),
  stabilizationToggle: document.getElementById("stabilization-toggle"),
  stabilizationAutoToggle: document.getElementById("stabilization-auto-toggle"),
  stabilizationStrength: document.getElementById("stabilization-strength"),
  stabilizationSmoothing: document.getElementById("stabilization-smoothing"),
  stabilizationOffsetX: document.getElementById("stabilization-offset-x"),
  stabilizationOffsetY: document.getElementById("stabilization-offset-y"),
  stabilizationProfileLight: document.getElementById("stabilization-profile-light"),
  stabilizationProfileMedium: document.getElementById("stabilization-profile-medium"),
  stabilizationProfileStrong: document.getElementById("stabilization-profile-strong"),
  presetLowlightButton: document.getElementById("preset-lowlight"),
  presetNightButton: document.getElementById("preset-night"),
  presetDetailButton: document.getElementById("preset-detail"),
  presetUltraLowlightButton: document.getElementById("preset-ultra-lowlight"),
  motionStart: document.getElementById("motion-start"),
  motionStop: document.getElementById("motion-stop"),
  motionIndicator: document.getElementById("motion-indicator"),
  motionMarkerToggle: document.getElementById("motion-marker-toggle"),
  motionSensitivity: document.getElementById("motion-sensitivity"),
  motionSensitivityValue: document.getElementById("motion-sensitivity-value"),
  motionCooldown: document.getElementById("motion-cooldown"),
  motionCooldownValue: document.getElementById("motion-cooldown-value"),
  temporalPreview: document.getElementById("temporal-preview"),
  aiOverlay: document.getElementById("ai-overlay"),
  aiFaceDetectButton: document.getElementById("ai-face-detect"),
  aiObjectDetectButton: document.getElementById("ai-object-detect"),
  aiProviderSelect: document.getElementById("ai-provider-select"),
  aiCapabilityCheckButton: document.getElementById("ai-capability-check"),
  aiCapabilityStatus: document.getElementById("ai-capability-status"),
  aiTrackStartButton: document.getElementById("ai-track-start"),
  aiTrackStopButton: document.getElementById("ai-track-stop"),
  aiSrFactor: document.getElementById("ai-sr-factor"),
  aiSrApplyButton: document.getElementById("ai-sr-apply"),
  aiSrResetButton: document.getElementById("ai-sr-reset"),
  aiSceneThreshold: document.getElementById("ai-scene-threshold"),
  aiScenesDetectButton: document.getElementById("ai-scenes-detect"),
  aiScenesClearButton: document.getElementById("ai-scenes-clear"),
  aiFaceMarkerToggle: document.getElementById("ai-face-marker-toggle"),
  aiObjectMarkerToggle: document.getElementById("ai-object-marker-toggle"),
  aiStatus: document.getElementById("ai-status"),
  aiFaceList: document.getElementById("ai-face-list"),
  aiObjectList: document.getElementById("ai-object-list"),
  aiSceneList: document.getElementById("ai-scene-list"),
  modelsCheckButton: document.getElementById("models-check"),
  modelsUpdateButton: document.getElementById("models-update"),
  modelsOpenFolderButton: document.getElementById("models-open-folder"),
  modelsStatus: document.getElementById("models-status"),
  hypothesisGenerateButton: document.getElementById("hypothesis-generate"),
  hypothesisExportButton: document.getElementById("hypothesis-export"),
  hypothesisStatus: document.getElementById("hypothesis-status"),
  hypothesisList: document.getElementById("hypothesis-list"),
  photoSourceInput: document.getElementById("photo-source-input"),
  photoBlendButton: document.getElementById("photo-blend"),
  photoDownloadButton: document.getElementById("photo-download"),
  photoStatus: document.getElementById("photo-status"),
  photoCanvas: document.getElementById("photo-canvas"),
  compareLeftInput: document.getElementById("compare-left-input"),
  compareRightInput: document.getElementById("compare-right-input"),
  compareSplitInput: document.getElementById("compare-split"),
  compareSplitValue: document.getElementById("compare-split-value"),
  compareRenderButton: document.getElementById("compare-render"),
  compareStatus: document.getElementById("compare-status"),
  compareCanvas: document.getElementById("compare-canvas"),
  timeline: document.getElementById("timeline"),
  timelineMarkers: document.getElementById("timeline-markers"),
  timelineCurrent: document.getElementById("timeline-current"),
  timelineDuration: document.getElementById("timeline-duration"),
  timelineZoomIn: document.getElementById("timeline-zoom-in"),
  timelineZoomOut: document.getElementById("timeline-zoom-out"),
  timelineZoomValue: document.getElementById("timeline-zoom-value"),
  caseId: document.getElementById("case-id"),
  caseOwner: document.getElementById("case-owner"),
  caseStatus: document.getElementById("case-status"),
  caseTags: document.getElementById("case-tags"),
  caseSummary: document.getElementById("case-summary"),
  logEntryButton: document.getElementById("log-entry"),
  exportLogButton: document.getElementById("export-log"),
  exportReportButton: document.getElementById("export-report"),
  previewReportButton: document.getElementById("preview-report"),
  exportFfmpegJobButton: document.getElementById("export-ffmpeg-job"),
  downloadFfmpegJobButton: document.getElementById("download-ffmpeg-job"),
  queueFfmpegJobButton: document.getElementById("queue-ffmpeg-job"),
  pipelinePauseButton: document.getElementById("pipeline-pause"),
  pipelineResumeButton: document.getElementById("pipeline-resume"),
  pipelineRetryFailedButton: document.getElementById("pipeline-retry-failed"),
  pipelineClearTerminalButton: document.getElementById("pipeline-clear-terminal"),
  ffmpegJobPreview: document.getElementById("ffmpeg-job-preview"),
  pipelineStatus: document.getElementById("pipeline-status"),
  pipelineProgress: document.getElementById("pipeline-progress"),
  pipelineProgressLabel: document.getElementById("pipeline-progress-label"),
  pipelineQueue: document.getElementById("pipeline-queue"),
  pipelineErrors: document.getElementById("pipeline-errors"),
  logList: document.getElementById("log-list"),
  caseLibrary: document.getElementById("case-library"),
  caseSearch: document.getElementById("case-search"),
  caseClearSearchButton: document.getElementById("case-clear-search"),
  caseCount: document.getElementById("case-count"),
  caseFilesCount: document.getElementById("case-files-count"),
  caseMarkersCount: document.getElementById("case-markers-count"),
  caseLogsCount: document.getElementById("case-logs-count"),
  caseSaveButton: document.getElementById("case-save"),
  caseLoadButton: document.getElementById("case-load"),
  caseDeleteButton: document.getElementById("case-delete"),
  caseExportLibraryButton: document.getElementById("case-export-library"),
  caseImportLibraryButton: document.getElementById("case-import-library"),
  caseImportInput: document.getElementById("case-import-input"),
  addMarkerButton: document.getElementById("add-marker"),
  exportMarkersButton: document.getElementById("export-markers"),
  markerList: document.getElementById("marker-list"),
  markerType: document.getElementById("marker-type"),
  markerNote: document.getElementById("marker-note"),
  apiBaseUrl: document.getElementById("api-base-url"),
  apiToken: document.getElementById("api-token"),
  apiConnectButton: document.getElementById("api-connect"),
  apiCancelCurrentButton: document.getElementById("api-cancel-current"),
  apiConnectionStatus: document.getElementById("api-connection-status"),
  apiOperation: document.getElementById("api-operation"),
  apiPreset: document.getElementById("api-preset"),
  apiApplyPresetButton: document.getElementById("api-apply-preset"),
  apiUpscaleFactor: document.getElementById("api-upscale-factor"),
  apiDenoiseLevel: document.getElementById("api-denoise-level"),
};

const state = {
  zoomLevel: 1,
  clipIn: null,
  clipOut: null,
  motionDetectionActive: false,
  motionCanvas: null,
  motionContext: null,
  previousFrameData: null,
  motionLoopId: null,
  motionLastMarkerTime: null,
  stabilizationCanvas: null,
  stabilizationContext: null,
  stabilizationPrevFrame: null,
  stabilizationLoopId: null,
  stabilizationAutoOffset: { x: 0, y: 0 },
  temporalFrames: [],
  aiSuperResolutionFactor: 1,
  aiSuperResolutionActive: false,
  aiProvider: "mock",
  aiRuntimeInfo: {
    provider: "mock",
    modelVersion: "demo-mock-1.0.0",
  },
  aiCapabilities: null,
  timelineZoom: 1,
  timelineWindow: { start: 0, end: 0, duration: 0 },
  logEntries: [],
  markers: [],
  importedFiles: [],
  caseLibrary: [],
  caseMeta: {
    status: "active",
    tags: [],
    summary: "",
  },
  pipelineJobs: [],
  pipelineNextJobId: 1,
  pipelineProcessing: false,
  pipelinePaused: false,
  pipelineErrors: [],
  pipelineMaxRetries: 2,
  hypothesisClips: [],
  backendApi: {
    enabled: false,
    baseUrl: "http://127.0.0.1:8000/api",
    token: "",
    client: null,
    operation: "detect_objects",
    preset: "balanced",
    upscaleFactor: 2,
    denoiseLevel: "medium",
  },
};

const createLogItem = ({ timestamp, caseId, owner, action, message }) => {
  const item = document.createElement("li");
  item.textContent = `${timestamp} | ${caseId || "Без ID"} | ${
    owner || "Не указан"
  } | ${action} | ${message}`;
  return item;
};

const createMarkerItem = ({ timestamp, timecode, note, type }) => {
  const item = document.createElement("li");
  item.textContent = `${timestamp} | ${timecode} | ${type} | ${note}`;
  return item;
};

const createPipelineItem = (job) => {
  const item = document.createElement("li");
  const details = [
    `job-${job.id}`,
    `status: ${job.status}`,
    `stage: ${job.stage}`,
    `source: ${job.hasSource ? "yes" : "no"}`,
    `op: ${job.operation || "detect_objects"}`,
    `preset: ${job.preset || "balanced"}`,
    `params: ${JSON.stringify(job.params || {})}`,
    `progress: ${job.progress ?? 0}%`,
    `backend: ${job.backendStatus || "n/a"}`,
  ];
  if (job.error) {
    details.push(`error: ${job.error}`);
  }
  item.textContent = details.join(" | ");
  return item;
};



const blobToBase64 = (blob) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = String(reader.result || "");
      const base64 = result.includes(",") ? result.split(",")[1] : result;
      resolve(base64);
    };
    reader.onerror = () => reject(new Error("Не удалось сериализовать кадр"));
    reader.readAsDataURL(blob);
  });

const captureCurrentFrameBase64 = async () => {
  if (!elements.video.videoWidth || !elements.video.videoHeight) {
    return null;
  }
  const canvas = document.createElement("canvas");
  canvas.width = elements.video.videoWidth;
  canvas.height = elements.video.videoHeight;
  const context = canvas.getContext("2d");
  context.drawImage(elements.video, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) return null;
  return blobToBase64(blob);
};


const applyBackendPresetToControls = (preset) => {
  const normalizedPreset = String(preset || "balanced").toLowerCase();
  const presetMap = {
    forensic_safe: { upscaleFactor: "2", denoiseLevel: "light" },
    balanced: { upscaleFactor: "4", denoiseLevel: "medium" },
    presentation: { upscaleFactor: "8", denoiseLevel: "heavy" },
  };
  const selected = presetMap[normalizedPreset] || presetMap.balanced;
  if (elements.apiUpscaleFactor) {
    elements.apiUpscaleFactor.value = selected.upscaleFactor;
  }
  if (elements.apiDenoiseLevel) {
    elements.apiDenoiseLevel.value = selected.denoiseLevel;
  }
};

const buildBackendParamsByOperation = (operation) => {
  if (operation === "upscale") {
    const factor = Number.parseInt(elements.apiUpscaleFactor?.value || "2", 10);
    return {
      preset: state.backendApi.preset,
      factor: Number.isFinite(factor) ? factor : 2,
    };
  }
  if (operation === "denoise") {
    const level = elements.apiDenoiseLevel?.value || "light";
    return { preset: state.backendApi.preset, level };
  }
  if (operation === "detect_objects") {
    const sceneThreshold = Number.parseFloat(elements.aiSceneThreshold?.value || "28");
    const temporalWindow = Number.parseInt(elements.temporalWindowInput?.value || "3", 10);
    return {
      preset: state.backendApi.preset,
      scene_threshold: Number.isFinite(sceneThreshold) ? sceneThreshold : 28,
      temporal_window: Number.isFinite(temporalWindow) ? temporalWindow : 3,
    };
  }
  return { preset: state.backendApi.preset };
};


const findActiveBackendJob = () =>
  [...state.pipelineJobs]
    .reverse()
    .find(
      (job) =>
        job.backendTaskId &&
        ["pending", "running"].includes(job.status) &&
        !["done", "failed", "canceled", "cancel-unsupported"].includes(job.backendStatus)
    );

const createPipelineErrorItem = (entry) => {
  const item = document.createElement("li");
  item.textContent = `${entry.timestamp} | job-${entry.jobId} | attempt ${entry.attempt} | ${entry.error}`;
  return item;
};

const actions = {
  formatTime,
  updateSpeed: () => {
    const speed = Number.parseFloat(elements.speedInput.value);
    elements.video.playbackRate = speed;
    elements.speedValue.textContent = `${speed.toFixed(2)}x`;
  },
  updateZoom: () => {
    const upscaleFactor = elements.upscaleToggle.checked
      ? Number.parseFloat(elements.upscaleFactor.value) || 2
      : 1;
    const stabilizationStrength = Number.parseFloat(
      elements.stabilizationStrength.value
    );
    const stabilizationFactor = elements.stabilizationToggle.checked
      ? 1 + stabilizationStrength / 100
      : 1;
    const offsetX = Number.parseFloat(elements.stabilizationOffsetX.value);
    const offsetY = Number.parseFloat(elements.stabilizationOffsetY.value);
    const aiSrFactor = state.aiSuperResolutionActive
      ? state.aiSuperResolutionFactor || 1
      : 1;
    const scale = state.zoomLevel * upscaleFactor * stabilizationFactor * aiSrFactor;
    elements.video.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;
  },
  resetZoom: () => {
    state.zoomLevel = 1;
    actions.updateZoom();
  },
  buildFfmpegJobDraft: (stage = "3.1.2") => {
    const now = new Date().toISOString();
    const toNumber = (value, fallback = 0) => {
      const parsed = Number.parseFloat(value);
      return Number.isFinite(parsed) ? parsed : fallback;
    };

    const selectedFile = state.importedFiles[0] || null;
    const tags = elements.caseTags.value
      .split(",")
      .map((tag) => tag.trim())
      .filter(Boolean);

    return {
      schema: "forensic.ffmpeg-job.v1",
      stage,
      createdAt: now,
      case: {
        id: elements.caseId.value.trim() || null,
        owner: elements.caseOwner.value.trim() || null,
        status: elements.caseStatus.value,
        tags,
      },
      source: selectedFile
        ? {
            name: selectedFile.name,
            type: selectedFile.type || null,
            size: selectedFile.size || null,
            sha256: selectedFile.hash || null,
          }
        : null,
      playback: {
        speed: toNumber(elements.speedInput.value, 1),
        clipIn: state.clipIn,
        clipOut: state.clipOut,
      },
      processing: {
        enhance: toNumber(elements.enhanceInput.value),
        exposure: toNumber(elements.exposureInput.value),
        temperature: toNumber(elements.temperatureInput.value),
        denoise: toNumber(elements.denoiseInput.value),
        denoiseProfile: elements.denoiseProfile.value,
        temporalDenoise: {
          enabled: elements.temporalDenoiseToggle.checked,
          window: toNumber(elements.temporalWindowInput.value, 2),
        },
        clarity: toNumber(elements.clarityInput.value),
        sharpness: toNumber(elements.sharpnessInput.value),
        lowlightBoost: elements.lowlightBoostToggle.checked,
        grayscale: elements.grayscaleToggle.checked,
        bypassFilters: elements.bypassFiltersToggle.checked,
        upscale: {
          enabled: elements.upscaleToggle.checked,
          factor: toNumber(elements.upscaleFactor.value, 1),
        },
        stabilization: {
          enabled: elements.stabilizationToggle.checked,
          auto: elements.stabilizationAutoToggle.checked,
          strength: toNumber(elements.stabilizationStrength.value),
          smoothing: toNumber(elements.stabilizationSmoothing.value),
          offsetX: toNumber(elements.stabilizationOffsetX.value),
          offsetY: toNumber(elements.stabilizationOffsetY.value),
        },
      },
      ai: {
        provider: state.aiProvider || elements.aiProviderSelect.value || "mock",
        modelVersion: state.aiRuntimeInfo?.modelVersion || "unknown",
        capabilities: state.aiCapabilities,
        faceMarkerAuto: elements.aiFaceMarkerToggle.checked,
        objectMarkerAuto: elements.aiObjectMarkerToggle.checked,
        superResolution: {
          enabled: state.aiSuperResolutionActive,
          factor: state.aiSuperResolutionFactor || 1,
        },
        sceneThreshold: toNumber(elements.aiSceneThreshold.value, 15),
      },
      timeline: {
        zoom: state.timelineZoom,
        markers: state.markers.length,
      },
      audit: {
        entries: state.logEntries.length,
      },
    };
  },
  downloadJson: (payload, namePrefix) => {
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${namePrefix}-${Date.now()}.json`;
    link.click();
  },
  recordLog: (action, message, meta = {}) => {
    const entry = {
      timestamp: new Date().toISOString(),
      caseId: elements.caseId.value.trim(),
      owner: elements.caseOwner.value.trim(),
      status: elements.caseStatus.value,
      tags: elements.caseTags.value
        .split(",")
        .map((tag) => tag.trim())
        .filter(Boolean),
      action,
      message,
      meta,
    };
    state.logEntries.unshift(entry);
    const item = createLogItem(entry);
    elements.logList.prepend(item);
    actions.updateCaseSummary();
    return entry;
  },
  renderLogEntries: () => {
    elements.logList.innerHTML = "";
    state.logEntries.forEach((entry) => {
      const item = createLogItem(entry);
      elements.logList.appendChild(item);
    });
  },
  renderMarkers: () => {
    elements.markerList.innerHTML = "";
    state.markers.forEach((marker) => {
      const item = createMarkerItem(marker);
      elements.markerList.appendChild(item);
    });
  },
  appendMarkerEntry: (entry) => {
    const item = createMarkerItem(entry);
    elements.markerList.prepend(item);
    actions.updateCaseSummary();
  },
  updateCaseSummary: () => {
    elements.caseFilesCount.textContent = state.importedFiles.length.toString();
    elements.caseMarkersCount.textContent = state.markers.length.toString();
    elements.caseLogsCount.textContent = state.logEntries.length.toString();
  },
  renderPipelineErrors: () => {
    if (!elements.pipelineErrors) return;
    elements.pipelineErrors.innerHTML = "";
    if (!state.pipelineErrors.length) {
      const empty = document.createElement("li");
      empty.textContent = "Ошибки не зафиксированы.";
      elements.pipelineErrors.appendChild(empty);
      return;
    }
    state.pipelineErrors.forEach((entry) => {
      elements.pipelineErrors.appendChild(createPipelineErrorItem(entry));
    });
  },
  appendPipelineError: (job, errorCode) => {
    const errorEntry = {
      timestamp: new Date().toISOString(),
      jobId: job.id,
      attempt: job.attempt,
      error: errorCode,
    };
    state.pipelineErrors.unshift(errorEntry);
    state.pipelineErrors = state.pipelineErrors.slice(0, 30);
    actions.renderPipelineErrors();
    return errorEntry;
  },
  renderPipelineJobs: () => {
    if (!elements.pipelineQueue || !elements.pipelineStatus) return;
    elements.pipelineQueue.innerHTML = "";
    const jobs = [...state.pipelineJobs].reverse();

    if (!jobs.length) {
      const empty = document.createElement("li");
      empty.textContent = "Очередь pipeline пуста.";
      elements.pipelineQueue.appendChild(empty);
      elements.pipelineStatus.textContent = "Pipeline: задач нет.";
      if (elements.pipelineProgress) {
        elements.pipelineProgress.value = 0;
      }
      if (elements.pipelineProgressLabel) {
        elements.pipelineProgressLabel.textContent = "0%";
      }
      actions.renderPipelineErrors();
      return;
    }

    jobs.forEach((job) => {
      elements.pipelineQueue.appendChild(createPipelineItem(job));
    });

    const pending = state.pipelineJobs.filter((job) => job.status === "pending").length;
    const runningJobs = state.pipelineJobs.filter((job) => job.status === "running");
    const done = state.pipelineJobs.filter((job) => job.status === "done").length;
    const failed = state.pipelineJobs.filter((job) => job.status === "failed").length;
    const canceled = state.pipelineJobs.filter((job) => job.status === "canceled").length;
    const running = runningJobs.length;

    const activeProgress = runningJobs.length ? runningJobs[0].progress ?? 0 : 0;
    if (elements.pipelineProgress) {
      elements.pipelineProgress.value = activeProgress;
    }
    if (elements.pipelineProgressLabel) {
      elements.pipelineProgressLabel.textContent = `${activeProgress}%`;
    }

    const pausedSuffix = state.pipelinePaused ? " [paused]" : "";
    elements.pipelineStatus.textContent = `Pipeline: pending ${pending}, running ${running}, done ${done}, failed ${failed}, canceled ${canceled}.${pausedSuffix}`;
    actions.renderPipelineErrors();
  },
  processNextPipelineJob: async () => {
    if (state.pipelinePaused) {
      actions.renderPipelineJobs();
      return;
    }
    if (state.pipelineProcessing) return;
    const nextJob = state.pipelineJobs.find((job) => job.status === "pending");
    if (!nextJob) {
      actions.renderPipelineJobs();
      return;
    }
    state.pipelineProcessing = true;
    nextJob.status = "running";
    nextJob.startedAt = new Date().toISOString();
    nextJob.progress = 5;
    actions.renderPipelineJobs();
    actions.recordLog("pipeline-job-running", `Запущена обработка job-${nextJob.id}`, {
      jobId: nextJob.id,
      stage: nextJob.stage,
      attempt: nextJob.attempt,
      maxRetries: nextJob.maxRetries,
      backendEnabled: state.backendApi.enabled,
    });

    const finishAndContinue = () => {
      state.pipelineProcessing = false;
      actions.renderPipelineJobs();
      actions.processNextPipelineJob();
    };

    const markFailed = (errorCode) => {
      nextJob.error = errorCode;
      nextJob.backendStatus = "failed";
      const errorEntry = actions.appendPipelineError(nextJob, nextJob.error);
      const canRetry = nextJob.attempt <= nextJob.maxRetries;
      if (canRetry) {
        nextJob.status = "pending";
        nextJob.progress = 0;
        nextJob.lastErrorAt = errorEntry.timestamp;
        nextJob.attempt += 1;
        actions.recordLog("pipeline-job-retry", `Pipeline job-${nextJob.id} повторно поставлен в очередь`, {
          jobId: nextJob.id,
          error: nextJob.error,
          nextAttempt: nextJob.attempt,
          maxRetries: nextJob.maxRetries,
        });
      } else {
        nextJob.status = "failed";
        nextJob.progress = 100;
        nextJob.finishedAt = new Date().toISOString();
        actions.recordLog("pipeline-job-failed", `Pipeline job-${nextJob.id} завершен с ошибкой`, {
          jobId: nextJob.id,
          error: nextJob.error,
          attemptsUsed: nextJob.attempt,
          maxRetries: nextJob.maxRetries,
        });
      }
    };

    if (state.backendApi.enabled && state.backendApi.client) {
      try {
        const frameBase64 = await captureCurrentFrameBase64();
        if (!frameBase64) {
          markFailed("source-missing");
          finishAndContinue();
          return;
        }

        const submit = await state.backendApi.client.submitJob({
          operation: nextJob.operation || "detect_objects",
          image_base64: frameBase64,
          params: nextJob.params || buildBackendParamsByOperation(nextJob.operation || "detect_objects"),
        });
        const taskId = submit?.result?.task_id;
        if (!taskId) {
          throw new Error("task_id-missing");
        }

        nextJob.backendTaskId = taskId;
        nextJob.backendStatus = "queued";
        nextJob.progress = 20;
        actions.renderPipelineJobs();

        const polled = await state.backendApi.client.pollJobUntilFinal(taskId, {
          maxAttempts: 25,
          intervalMs: 600,
          onProgress: ({ status, progress, payload }) => {
            nextJob.backendStatus = status;
            nextJob.progress = Number.isFinite(progress) ? progress : nextJob.progress;
            if (["pending", "queued", "running", "started", "progress", "retry"].includes(status)) {
              nextJob.status = "running";
            }
            if (status === "canceled") {
              nextJob.status = "canceled";
            }
            if (payload?.meta) {
              nextJob.backendMeta = payload.meta;
            }
            actions.renderPipelineJobs();
          },
        });

        if (polled.final === "success") {
          nextJob.status = "done";
          nextJob.backendStatus = "done";
          nextJob.progress = 100;
          nextJob.finishedAt = new Date().toISOString();
          actions.recordLog("pipeline-job-done", `Pipeline job-${nextJob.id} завершен через backend`, {
            jobId: nextJob.id,
            taskId,
            backendStatus: nextJob.backendStatus,
            backendMeta: nextJob.backendMeta || null,
          });
          finishAndContinue();
          return;
        }

        if (polled.final === "failure") {
          if (String(polled?.payload?.status || "") === "canceled") {
            nextJob.status = "canceled";
            nextJob.backendStatus = "canceled";
            nextJob.error = polled?.payload?.error || "canceled-by-operator";
            nextJob.progress = 100;
            nextJob.finishedAt = new Date().toISOString();
            actions.recordLog("pipeline-job-canceled", `Pipeline job-${nextJob.id} отменен`, {
              jobId: nextJob.id,
              taskId,
              error: nextJob.error,
            });
          } else {
            markFailed(polled?.payload?.error || "backend-failed");
          }
          finishAndContinue();
          return;
        }

        markFailed("backend-timeout");
      } catch (error) {
        markFailed(error?.message || "backend-submit-error");
      }
      finishAndContinue();
      return;
    }

    const progressTimer = window.setInterval(() => {
      if (nextJob.status !== "running") {
        window.clearInterval(progressTimer);
        return;
      }
      const nextValue = Math.min(95, (nextJob.progress ?? 0) + 15);
      nextJob.progress = nextValue;
      actions.renderPipelineJobs();
    }, 180);

    window.setTimeout(() => {
      window.clearInterval(progressTimer);
      if (nextJob.hasSource) {
        nextJob.status = "done";
        nextJob.backendStatus = "mock-done";
        nextJob.progress = 100;
        nextJob.finishedAt = new Date().toISOString();
        actions.recordLog("pipeline-job-done", `Pipeline job-${nextJob.id} завершен (mock)`, {
          jobId: nextJob.id,
        });
      } else {
        markFailed("source-missing");
      }
      finishAndContinue();
    }, 1200);
  },
  enqueuePipelineJob: (jobPayload, stage = "3.3.2") => {
    const job = {
      id: state.pipelineNextJobId,
      stage,
      createdAt: new Date().toISOString(),
      status: "pending",
      progress: 0,
      attempt: 1,
      maxRetries: state.pipelineMaxRetries,
      hasSource: Boolean(jobPayload?.source),
      operation: state.backendApi.operation,
      preset: state.backendApi.preset,
      backendStatus: "created",
      params: buildBackendParamsByOperation(state.backendApi.operation),
    };
    state.pipelineNextJobId += 1;
    state.pipelineJobs.push(job);
    actions.renderPipelineJobs();
    actions.recordLog("pipeline-job-enqueue", `Добавлена pipeline job-${job.id}`, {
      jobId: job.id,
      stage,
      hasSource: job.hasSource,
      operation: job.operation,
      preset: job.preset,
      params: job.params,
    });
    actions.processNextPipelineJob();
    return job;
  },
  pausePipelineQueue: () => {
    state.pipelinePaused = true;
    actions.renderPipelineJobs();
    actions.recordLog("pipeline-paused", "Очередь pipeline поставлена на паузу", {
      pending: state.pipelineJobs.filter((job) => job.status === "pending").length,
      running: state.pipelineJobs.filter((job) => job.status === "running").length,
    });
  },
  resumePipelineQueue: () => {
    state.pipelinePaused = false;
    actions.renderPipelineJobs();
    actions.recordLog("pipeline-resumed", "Очередь pipeline возобновлена", {
      pending: state.pipelineJobs.filter((job) => job.status === "pending").length,
    });
    actions.processNextPipelineJob();
  },
  retryFailedPipelineJobs: () => {
    const failedJobs = state.pipelineJobs.filter((job) => job.status === "failed");
    failedJobs.forEach((job) => {
      job.status = "pending";
      job.error = null;
      job.progress = 0;
      job.backendStatus = "retry-manual";
      job.attempt += 1;
    });
    actions.renderPipelineJobs();
    actions.recordLog("pipeline-retry-failed", "Повтор failed задач очереди", {
      retried: failedJobs.length,
    });
    actions.processNextPipelineJob();
  },
  clearTerminalPipelineJobs: () => {
    const before = state.pipelineJobs.length;
    state.pipelineJobs = state.pipelineJobs.filter(
      (job) => !["done", "failed", "canceled"].includes(job.status)
    );
    const removed = before - state.pipelineJobs.length;
    actions.renderPipelineJobs();
    actions.recordLog("pipeline-clear-terminal", "Очистка завершенных задач очереди", {
      removed,
      remaining: state.pipelineJobs.length,
    });
  },
  refreshCaseLibraryOptions: (query = "") => {
    const normalizedQuery = query.trim().toLowerCase();
    elements.caseLibrary.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Выберите дело";
    elements.caseLibrary.appendChild(placeholder);
    const filtered = state.caseLibrary.filter((caseItem) => {
      if (!normalizedQuery) return true;
      const haystack = [
        caseItem.caseId,
        caseItem.owner,
        caseItem.status,
        caseItem.summary,
        ...(caseItem.tags || []),
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return haystack.includes(normalizedQuery);
    });
    filtered.forEach((caseItem) => {
      const option = document.createElement("option");
      option.value = caseItem.id;
      option.textContent = `${caseItem.caseId || "Без ID"} · ${
        caseItem.owner || "Не указан"
      } · ${new Date(caseItem.updatedAt).toLocaleString("ru-RU")}`;
      elements.caseLibrary.appendChild(option);
    });
    elements.caseCount.textContent = `Найдено дел: ${filtered.length}`;
  },
  loadCaseLibrary: () => {
    const stored = localStorage.getItem("forensicCaseLibrary");
    state.caseLibrary = stored ? JSON.parse(stored) : [];
    state.caseLibrary.sort(
      (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    );
    actions.refreshCaseLibraryOptions(elements.caseSearch.value);
    actions.updateCaseSummary();
  },
  saveCaseLibrary: () => {
    localStorage.setItem(
      "forensicCaseLibrary",
      JSON.stringify(state.caseLibrary)
    );
  },
  saveCurrentCase: () => {
    const selectedId = elements.caseLibrary.value;
    const existingIndex = state.caseLibrary.findIndex(
      (caseItem) => caseItem.id === selectedId
    );
    const now = new Date().toISOString();
    const tags = elements.caseTags.value
      .split(",")
      .map((tag) => tag.trim())
      .filter(Boolean);
    const caseItem = {
      id:
        existingIndex >= 0
          ? state.caseLibrary[existingIndex].id
          : crypto.randomUUID?.() || `case-${Date.now()}`,
      caseId: elements.caseId.value.trim(),
      owner: elements.caseOwner.value.trim(),
      status: elements.caseStatus.value,
      tags,
      summary: elements.caseSummary.value.trim(),
      createdAt:
        existingIndex >= 0
          ? state.caseLibrary[existingIndex].createdAt
          : now,
      updatedAt: now,
      logEntries: state.logEntries,
      markers: state.markers,
      importedFiles: state.importedFiles,
    };
    if (existingIndex >= 0) {
      state.caseLibrary[existingIndex] = caseItem;
    } else {
      state.caseLibrary.unshift(caseItem);
    }
    actions.saveCaseLibrary();
    actions.loadCaseLibrary();
    elements.caseLibrary.value = caseItem.id;
    elements.caseSearch.value = "";
    actions.recordLog("case-save", "Дело сохранено в библиотеку", {
      id: caseItem.id,
    });
  },
  loadCaseFromLibrary: () => {
    const selectedId = elements.caseLibrary.value;
    if (!selectedId) return;
    const caseItem = state.caseLibrary.find(
      (item) => item.id === selectedId
    );
    if (!caseItem) return;
    elements.caseId.value = caseItem.caseId || "";
    elements.caseOwner.value = caseItem.owner || "";
    elements.caseStatus.value = caseItem.status || "active";
    elements.caseTags.value = (caseItem.tags || []).join(", ");
    elements.caseSummary.value = caseItem.summary || "";
    state.logEntries = caseItem.logEntries || [];
    state.markers = caseItem.markers || [];
    state.importedFiles = caseItem.importedFiles || [];
    actions.renderLogEntries();
    actions.renderMarkers();
    actions.updateCaseSummary();
    if (actions.refreshTimeline) {
      actions.refreshTimeline();
    }
    actions.recordLog("case-load", "Дело загружено из библиотеки", {
      id: caseItem.id,
    });
  },
  deleteCaseFromLibrary: () => {
    const selectedId = elements.caseLibrary.value;
    if (!selectedId) return;
    state.caseLibrary = state.caseLibrary.filter(
      (item) => item.id !== selectedId
    );
    actions.saveCaseLibrary();
    actions.loadCaseLibrary();
    actions.recordLog("case-delete", "Дело удалено из библиотеки", {
      id: selectedId,
    });
  },
  hashFile: async (file) => hashFileStream(file),
};

const orchestrator = new Orchestrator({ elements, state, actions });

orchestrator.register(createPlaylistBlueprint());
orchestrator.register(createPlayerBlueprint());
orchestrator.register(createScreenshotBlueprint());
orchestrator.register(createClipBlueprint());
orchestrator.register(createQualityBlueprint());
orchestrator.register(createMotionBlueprint());
orchestrator.register(createForensicBlueprint());
orchestrator.register(createTimelineBlueprint());
orchestrator.register(createAiBlueprint());
orchestrator.register(createHypothesisBlueprint());
orchestrator.register(createPhotoBlueprint());
orchestrator.register(createCompareBlueprint());

orchestrator.start();
initControlTabs();
initBackendApiPanel();


function initControlTabs() {
  const buttons = Array.from(document.querySelectorAll(".tab-button"));
  const panels = Array.from(document.querySelectorAll(".tab-panel"));
  if (!buttons.length || !panels.length) {
    return;
  }

  const activateTab = (target) => {
    buttons.forEach((button) => {
      button.classList.toggle("active", button.dataset.tabTarget === target);
    });

    panels.forEach((panel) => {
      panel.classList.toggle("active", panel.classList.contains(target));
    });
  };

  buttons.forEach((button) => {
    button.addEventListener("click", () => activateTab(button.dataset.tabTarget));
  });

  const defaultTab = buttons.find((button) => button.classList.contains("active"))?.dataset.tabTarget || "tab-basic";
  activateTab(defaultTab);
}


function initBackendApiPanel() {
  if (!elements.apiBaseUrl || !elements.apiConnectButton || !elements.apiConnectionStatus) return;

  elements.apiBaseUrl.value = state.backendApi.baseUrl;
  elements.apiToken.value = state.backendApi.token;
  if (elements.apiPreset) {
    elements.apiPreset.value = state.backendApi.preset;
  }
  applyBackendPresetToControls(state.backendApi.preset);

  const setStatus = (message) => {
    elements.apiConnectionStatus.textContent = message;
  };

  elements.apiApplyPresetButton?.addEventListener("click", () => {
    const preset = elements.apiPreset?.value || "balanced";
    state.backendApi.preset = preset;
    applyBackendPresetToControls(preset);
    state.backendApi.upscaleFactor = Number.parseInt(elements.apiUpscaleFactor?.value || "2", 10) || 2;
    state.backendApi.denoiseLevel = elements.apiDenoiseLevel?.value || "light";
    setStatus(`Preset применен: ${preset}.`);
  });

  elements.apiCancelCurrentButton?.addEventListener("click", async () => {
    if (!state.backendApi.enabled || !state.backendApi.client) {
      setStatus("Backend queue не подключен.");
      return;
    }

    const activeJob = findActiveBackendJob();
    if (!activeJob || !activeJob.backendTaskId) {
      setStatus("Нет активной backend job для отмены.");
      return;
    }

    try {
      const response = await state.backendApi.client.cancelJob(activeJob.backendTaskId);
      const status = response?.result?.status || "canceled";
      activeJob.backendStatus = status;
      if (status === "canceled") {
        activeJob.status = "canceled";
        activeJob.error = "canceled-by-operator";
        activeJob.progress = 100;
        activeJob.finishedAt = new Date().toISOString();
      }
      actions.renderPipelineJobs();
      setStatus(`Cancel response: ${status} (job-${activeJob.id}).`);
      actions.recordLog("pipeline-job-cancel", `Отмена pipeline job-${activeJob.id}`, {
        jobId: activeJob.id,
        taskId: activeJob.backendTaskId,
        cancelStatus: status,
      });
    } catch (error) {
      setStatus(`Не удалось отменить job: ${error?.message || "unknown"}`);
    }
  });

  elements.apiConnectButton.addEventListener("click", async () => {
    const baseUrl = elements.apiBaseUrl.value.trim() || state.backendApi.baseUrl;
    const token = elements.apiToken.value.trim();

    state.backendApi.baseUrl = baseUrl;
    state.backendApi.token = token;
    state.backendApi.client = createApiClient({ baseUrl, token });

    try {
      await state.backendApi.client.ping();
      state.backendApi.enabled = true;
      state.backendApi.operation = elements.apiOperation?.value || state.backendApi.operation;
      state.backendApi.preset = elements.apiPreset?.value || state.backendApi.preset;
      setStatus(`Backend queue: подключен (${state.backendApi.operation}, preset: ${state.backendApi.preset}).`);
      state.backendApi.upscaleFactor = Number.parseInt(elements.apiUpscaleFactor?.value || "2", 10) || 2;
      state.backendApi.denoiseLevel = elements.apiDenoiseLevel?.value || "light";
      actions.recordLog("backend-connect", "Подключение к backend queue успешно", {
        baseUrl,
        operation: state.backendApi.operation,
        preset: state.backendApi.preset,
        upscaleFactor: state.backendApi.upscaleFactor,
        denoiseLevel: state.backendApi.denoiseLevel,
      });
    } catch (_error) {
      state.backendApi.enabled = false;
      setStatus("Backend queue: не удалось подключиться, работает mock fallback.");
    }
  });
}

// -----------------------------
// Models panel (Desktop only)
// -----------------------------

async function initModelsPanel() {
  if (!window.electronAPI || !elements.modelsStatus) {
    return;
  }

  const setStatus = (text) => {
    elements.modelsStatus.textContent = text;
  };

  try {
    const modelsDir = await window.electronAPI.getModelsPath();
    setStatus(`Папка моделей: ${modelsDir}`);
  } catch (_e) {
    setStatus('Модели: не удалось получить путь.');
  }

  const unsubscribe = window.electronAPI.onDownloadProgress((data) => {
    if (!data || !data.modelName) return;
    setStatus(`Загрузка ${data.modelName}: ${data.progress}%`);
  });

  elements.modelsOpenFolderButton?.addEventListener('click', async () => {
    try {
      await window.electronAPI.openModelsFolder();
    } catch (e) {
      await window.electronAPI.showDialog('Ошибка', e.message || String(e));
    }
  });

  elements.modelsCheckButton?.addEventListener('click', async () => {
    try {
      setStatus('Проверяю модели...');
      const result = await window.electronAPI.checkModelUpdates();
      if (!result.updatesAvailable) {
        setStatus(`Модели OK. Папка: ${result.modelsDir}`);
        await window.electronAPI.showDialog('Модели', 'Обновлений моделей нет.');
        return;
      }
      const lines = result.updates
        .map((u) => {
          if (u.reason === 'missing') return `• ${u.key}: отсутствует (нужна версия ${u.targetVersion || '?'})`;
          return `• ${u.key}: ${u.currentVersion || '?'} → ${u.targetVersion || '?'}`;
        })
        .join('\n');
      setStatus(`Найдены обновления: ${result.updates.length}`);
      await window.electronAPI.showDialog('Найдены обновления моделей', lines);
    } catch (e) {
      setStatus('Ошибка проверки моделей.');
      await window.electronAPI.showDialog('Ошибка', e.message || String(e));
    }
  });

  elements.modelsUpdateButton?.addEventListener('click', async () => {
    try {
      setStatus('Проверяю обновления...');
      const check = await window.electronAPI.checkModelUpdates();
      if (!check.updatesAvailable) {
        setStatus(`Модели OK. Папка: ${check.modelsDir}`);
        await window.electronAPI.showDialog('Модели', 'Обновлений моделей нет.');
        return;
      }
      const msg =
        `Будут скачаны/обновлены модели: ${check.updates.map((u) => u.key).join(', ')}\n\n` +
        `Примечание: ссылки на веса задаются в models-data/manifest.json (url + checksum).`;

      const ok = await window.electronAPI.showDialog('Обновление моделей', msg);
      if (!ok) {
        setStatus('Обновление отменено.');
        return;
      }

      setStatus('Загрузка моделей...');
      const res = await window.electronAPI.updateModels();
      setStatus(`Готово. Обновлено: ${res.updated.length}`);
      await window.electronAPI.showDialog('Модели', `Готово. Обновлено: ${res.updated.join(', ')}`);
    } catch (e) {
      setStatus('Ошибка обновления моделей.');
      await window.electronAPI.showDialog('Ошибка', e.message || String(e));
    }
  });

  // Clean up on unload.
  window.addEventListener('beforeunload', () => {
    try {
      unsubscribe();
    } catch (_e) {}
  });
}

initModelsPanel();
