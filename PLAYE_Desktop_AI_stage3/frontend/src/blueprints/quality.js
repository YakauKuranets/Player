export const createQualityBlueprint = () => ({
  name: "quality",
  init: ({ elements, actions, state }) => {
    const ensureTemporalCanvas = () => {
      if (!state.temporalCanvas) {
        state.temporalCanvas = document.createElement("canvas");
        state.temporalContext = state.temporalCanvas.getContext("2d", {
          willReadFrequently: true,
        });
      }
      state.temporalCanvas.width = elements.video.videoWidth;
      state.temporalCanvas.height = elements.video.videoHeight;
    };

    const initTemporalWorker = () => {
      if (state.temporalWorker) {
        return state.temporalWorker;
      }
      if (!window.Worker) {
        return null;
      }
      try {
        const worker = new Worker(new URL("../workers/temporal-denoise-worker.js", import.meta.url), {
          type: "module",
        });
        state.temporalWorker = worker;
        return worker;
      } catch (_error) {
        return null;
      }
    };

    const averageFramesSync = (frames) => {
      const pixels = frames[0].length;
      const output = new Uint8ClampedArray(pixels);
      for (let i = 0; i < pixels; i += 1) {
        let sum = 0;
        for (let f = 0; f < frames.length; f += 1) {
          sum += frames[f][i];
        }
        output[i] = Math.round(sum / frames.length);
      }
      return output;
    };

    const averageFramesAsync = (worker, width, height, frames) =>
      new Promise((resolve, reject) => {
        if (!worker) {
          resolve(averageFramesSync(frames));
          return;
        }

        const onMessage = (event) => {
          const { type, output, error } = event.data || {};
          if (type === "result") {
            worker.removeEventListener("message", onMessage);
            resolve(new Uint8ClampedArray(output));
          } else if (type === "error") {
            worker.removeEventListener("message", onMessage);
            reject(new Error(error || "temporal-worker-error"));
          }
        };

        worker.addEventListener("message", onMessage);
        worker.postMessage({
          type: "averageFrames",
          payload: { width, height, frames },
        });
      });

    const applyTemporalDenoise = async () => {
      if (!elements.temporalDenoiseToggle.checked) return;
      if (elements.video.readyState < 2) return;
      if (state.temporalBusy) return;

      ensureTemporalCanvas();
      const width = state.temporalCanvas.width;
      const height = state.temporalCanvas.height;
      if (!width || !height) return;

      state.temporalContext.drawImage(elements.video, 0, 0, width, height);
      const frame = state.temporalContext.getImageData(0, 0, width, height);
      const windowSize = Number.parseInt(elements.temporalWindowInput.value, 10);
      if (!Number.isFinite(windowSize) || windowSize <= 0) return;

      state.temporalFrames.unshift(new Uint8ClampedArray(frame.data));
      state.temporalFrames = state.temporalFrames.slice(0, windowSize);
      if (state.temporalFrames.length < 2) return;

      state.temporalBusy = true;
      try {
        const worker = initTemporalWorker();
        const averaged = await averageFramesAsync(worker, width, height, state.temporalFrames);
        const output = state.temporalContext.createImageData(width, height);
        output.data.set(averaged);
        state.temporalContext.putImageData(output, 0, 0);

        elements.video.style.opacity = "0";
        if (!elements.temporalPreview) return;
        elements.temporalPreview.style.opacity = "1";
        elements.temporalPreview.width = width;
        elements.temporalPreview.height = height;
        const previewContext = elements.temporalPreview.getContext("2d");
        if (previewContext) {
          previewContext.putImageData(output, 0, 0);
        }
      } catch (_error) {
        const fallback = state.temporalContext.createImageData(width, height);
        fallback.data.set(averageFramesSync(state.temporalFrames));
        const previewContext = elements.temporalPreview?.getContext("2d");
        if (previewContext) {
          previewContext.putImageData(fallback, 0, 0);
        }
      } finally {
        state.temporalBusy = false;
      }
    };
    const setupStabilizationCanvas = () => {
      if (!state.stabilizationCanvas) {
        state.stabilizationCanvas = document.createElement("canvas");
        state.stabilizationContext = state.stabilizationCanvas.getContext("2d", {
          willReadFrequently: true,
        });
      }
      state.stabilizationCanvas.width = elements.video.videoWidth;
      state.stabilizationCanvas.height = elements.video.videoHeight;
    };

    const stopAutoStabilization = () => {
      if (state.stabilizationLoopId) {
        cancelAnimationFrame(state.stabilizationLoopId);
      }
      state.stabilizationLoopId = null;
      state.stabilizationPrevFrame = null;
      state.stabilizationAutoOffset = { x: 0, y: 0 };
    };

    const updateAutoStabilization = () => {
      if (!elements.stabilizationAutoToggle.checked) {
        stopAutoStabilization();
        return;
      }
      if (elements.video.paused || elements.video.ended) {
        state.stabilizationLoopId = requestAnimationFrame(updateAutoStabilization);
        return;
      }

      if (elements.video.readyState < 2) {
        state.stabilizationLoopId = requestAnimationFrame(updateAutoStabilization);
        return;
      }

      setupStabilizationCanvas();
      const width = state.stabilizationCanvas.width;
      const height = state.stabilizationCanvas.height;
      state.stabilizationContext.drawImage(elements.video, 0, 0, width, height);
      const frame = state.stabilizationContext.getImageData(0, 0, width, height);

      if (state.stabilizationPrevFrame) {
        const step = 12;
        let count = 0;
        let sumX = 0;
        let sumY = 0;
        for (let y = 0; y < height; y += step) {
          for (let x = 0; x < width; x += step) {
            const idx = (y * width + x) * 4;
            const delta = Math.abs(frame.data[idx] - state.stabilizationPrevFrame.data[idx]);
            if (delta > 18) {
              count += 1;
              sumX += x;
              sumY += y;
            }
          }
        }

        if (count > 0) {
          const avgX = sumX / count - width / 2;
          const avgY = sumY / count - height / 2;
          const strength = Number.parseFloat(elements.stabilizationStrength.value);
          const strengthFactor = strength / 10;
          const smoothing = Number.parseFloat(elements.stabilizationSmoothing.value) / 100;
          const targetX = -avgX * strengthFactor;
          const targetY = -avgY * strengthFactor;
          const prev = state.stabilizationAutoOffset;
          const nextX = prev.x * smoothing + targetX * (1 - smoothing);
          const nextY = prev.y * smoothing + targetY * (1 - smoothing);
          const clampedX = Math.max(-50, Math.min(50, nextX));
          const clampedY = Math.max(-50, Math.min(50, nextY));
          state.stabilizationAutoOffset = { x: clampedX, y: clampedY };
          elements.stabilizationOffsetX.value = clampedX.toFixed(0);
          elements.stabilizationOffsetY.value = clampedY.toFixed(0);
          actions.updateZoom();
        }
      }

      state.stabilizationPrevFrame = frame;
      state.stabilizationLoopId = requestAnimationFrame(updateAutoStabilization);
    };

    const applyFilters = () => {
      if (elements.bypassFiltersToggle.checked) {
        elements.video.style.filter = "none";
        return;
      }
      const value = Number.parseFloat(elements.enhanceInput.value);
      const contrast = 100 + value * 0.4;
      const exposure = Number.parseFloat(elements.exposureInput.value);
      const brightness = 100 + value * 0.2 + exposure * 0.6;
      const temperature = Number.parseFloat(elements.temperatureInput.value);
      const hue = temperature * 0.6;
      const clarity = Number.parseFloat(elements.clarityInput.value);
      const clarityBoost = clarity * 0.5;
      const sharpness = Number.parseFloat(elements.sharpnessInput.value);
      const sharpnessBoost = sharpness * 0.6;
      const lowlightBoost = elements.lowlightBoostToggle.checked ? 10 : 0;
      const saturate =
        100 + value * 0.3 + clarity * 0.4 + sharpness * 0.2 + (lowlightBoost ? 8 : 0);
      const boostedBrightness = brightness + lowlightBoost + sharpness * 0.15;
      const denoiseValue = Number.parseFloat(elements.denoiseInput.value);
      const denoiseProfile = elements.denoiseProfile.value;
      const denoiseMultiplier = denoiseProfile === "heavy" ? 2 : denoiseProfile === "light" ? 1 : 1.5;
      const blur = Math.max(0, (denoiseValue * denoiseMultiplier) / 5);
      const grayscale = elements.grayscaleToggle.checked ? "grayscale(100%)" : "";
      const blurFilter = blur > 0 ? `blur(${blur}px)` : "";
      const hueFilter = hue !== 0 ? `hue-rotate(${hue}deg)` : "";
      elements.video.style.filter = [
        `contrast(${contrast + clarityBoost + sharpnessBoost}%)`,
        `brightness(${boostedBrightness}%)`,
        `saturate(${saturate}%)`,
        blurFilter,
        hueFilter,
        grayscale,
      ]
        .filter(Boolean)
        .join(" ");
    };

    const applyPreset = (preset) => {
      elements.enhanceInput.value = preset.enhance.toString();
      elements.exposureInput.value = preset.exposure.toString();
      elements.temperatureInput.value = preset.temperature.toString();
      elements.denoiseInput.value = preset.denoise.toString();
      elements.denoiseProfile.value = preset.denoiseProfile ?? "medium";
      elements.clarityInput.value = preset.clarity?.toString() ?? "0";
      elements.sharpnessInput.value = preset.sharpness?.toString() ?? "0";
      elements.lowlightBoostToggle.checked = preset.lowlightBoost ?? false;
      elements.grayscaleToggle.checked = preset.grayscale;
      elements.bypassFiltersToggle.checked = false;
      applyFilters();
      actions.recordLog("quality-preset", `Профиль: ${preset.label}`, {
        preset: preset.key,
      });
    };

    const applyStabilizationProfile = (profile) => {
      elements.stabilizationToggle.checked = true;
      elements.stabilizationAutoToggle.checked = profile.auto;
      elements.stabilizationStrength.value = profile.strength.toString();
      elements.stabilizationSmoothing.value = profile.smoothing.toString();
      elements.stabilizationOffsetX.value = "0";
      elements.stabilizationOffsetY.value = "0";
      if (profile.auto) {
        updateAutoStabilization();
      } else {
        stopAutoStabilization();
      }
      actions.updateZoom();
      actions.recordLog("stabilization-profile", `Профиль стабилизации: ${profile.label}`, {
        profile: profile.key,
      });
    };

    elements.enhanceInput.addEventListener("input", applyFilters);
    elements.exposureInput.addEventListener("input", applyFilters);
    elements.temperatureInput.addEventListener("input", applyFilters);
    elements.denoiseInput.addEventListener("input", applyFilters);
    elements.temporalDenoiseToggle.addEventListener("change", () => {
      if (!elements.temporalDenoiseToggle.checked) {
        elements.video.style.opacity = "1";
        if (elements.temporalPreview) {
          elements.temporalPreview.style.opacity = "0";
        }
        state.temporalFrames = [];
        state.temporalBusy = false;
      }
      actions.recordLog(
        "temporal-denoise-toggle",
        elements.temporalDenoiseToggle.checked
          ? "Temporal шумоподавление включено (demo)"
          : "Temporal шумоподавление выключено"
      );
    });
    elements.temporalWindowInput.addEventListener("change", () => {
      const value = Number.parseInt(elements.temporalWindowInput.value, 10);
      actions.recordLog("temporal-window", "Окно temporal (кадры)", { value });
    });
    elements.denoiseProfile.addEventListener("change", applyFilters);
    elements.clarityInput.addEventListener("input", applyFilters);
    elements.sharpnessInput.addEventListener("input", applyFilters);
    elements.lowlightBoostToggle.addEventListener("change", applyFilters);
    elements.grayscaleToggle.addEventListener("change", applyFilters);
    elements.bypassFiltersToggle.addEventListener("change", applyFilters);
    elements.upscaleFactor.addEventListener("change", () => {
      actions.updateZoom();
      actions.recordLog("upscale-factor", "Фактор апскейла изменен", {
        value: Number.parseFloat(elements.upscaleFactor.value),
      });
    });
    elements.stabilizationToggle.addEventListener("change", () => {
      if (!elements.stabilizationToggle.checked) {
        elements.stabilizationAutoToggle.checked = false;
        stopAutoStabilization();
      }
      actions.updateZoom();
      actions.recordLog(
        "stabilization-toggle",
        elements.stabilizationToggle.checked
          ? "Стабилизация включена (демо)"
          : "Стабилизация выключена"
      );
    });
    elements.stabilizationAutoToggle.addEventListener("change", () => {
      if (elements.stabilizationAutoToggle.checked) {
        elements.stabilizationToggle.checked = true;
        updateAutoStabilization();
      } else {
        stopAutoStabilization();
        elements.stabilizationOffsetX.value = "0";
        elements.stabilizationOffsetY.value = "0";
        actions.updateZoom();
      }
      actions.recordLog(
        "stabilization-auto-toggle",
        elements.stabilizationAutoToggle.checked
          ? "Авто-стабилизация включена (демо)"
          : "Авто-стабилизация выключена"
      );
    });
    elements.stabilizationStrength.addEventListener("input", () => {
      actions.updateZoom();
    });
    elements.stabilizationStrength.addEventListener("change", () => {
      const value = Number.parseFloat(elements.stabilizationStrength.value);
      actions.recordLog("stabilization-strength", "Сила стабилизации изменена", {
        value,
      });
    });
    elements.stabilizationSmoothing.addEventListener("change", () => {
      const value = Number.parseFloat(elements.stabilizationSmoothing.value);
      actions.recordLog("stabilization-smoothing", "Сглаживание стабилизации", {
        value,
      });
    });
    elements.stabilizationOffsetX.addEventListener("input", () => {
      actions.updateZoom();
    });
    elements.stabilizationOffsetY.addEventListener("input", () => {
      actions.updateZoom();
    });
    elements.stabilizationOffsetX.addEventListener("change", () => {
      const value = Number.parseFloat(elements.stabilizationOffsetX.value);
      actions.recordLog("stabilization-offset-x", "Смещение стабилизации X", {
        value,
      });
    });
    elements.stabilizationOffsetY.addEventListener("change", () => {
      const value = Number.parseFloat(elements.stabilizationOffsetY.value);
      actions.recordLog("stabilization-offset-y", "Смещение стабилизации Y", {
        value,
      });
    });
    elements.stabilizationProfileLight.addEventListener("click", () => {
      applyStabilizationProfile({
        key: "light",
        label: "Light",
        strength: 3,
        smoothing: 75,
        auto: true,
      });
    });
    elements.stabilizationProfileMedium.addEventListener("click", () => {
      applyStabilizationProfile({
        key: "medium",
        label: "Medium",
        strength: 5,
        smoothing: 60,
        auto: true,
      });
    });
    elements.stabilizationProfileStrong.addEventListener("click", () => {
      applyStabilizationProfile({
        key: "strong",
        label: "Strong",
        strength: 7,
        smoothing: 45,
        auto: true,
      });
    });
    elements.enhanceInput.addEventListener("change", () => {
      const value = Number.parseFloat(elements.enhanceInput.value);
      actions.recordLog("enhance", `Улучшение: ${value.toFixed(0)}%`, { value });
    });
    elements.exposureInput.addEventListener("change", () => {
      const value = Number.parseFloat(elements.exposureInput.value);
      actions.recordLog("exposure", `Экспозиция: ${value.toFixed(0)}`, { value });
    });
    elements.temperatureInput.addEventListener("change", () => {
      const value = Number.parseFloat(elements.temperatureInput.value);
      actions.recordLog("temperature", `Температура: ${value.toFixed(0)}`, {
        value,
      });
    });
    elements.denoiseInput.addEventListener("change", () => {
      const value = Number.parseFloat(elements.denoiseInput.value);
      actions.recordLog("denoise", `Шумоподавление: ${value.toFixed(1)}`, {
        value,
      });
    });
    elements.denoiseProfile.addEventListener("change", () => {
      actions.recordLog(
        "denoise-profile",
        `Профиль шумоподавления: ${elements.denoiseProfile.value}`
      );
    });
    elements.clarityInput.addEventListener("change", () => {
      const value = Number.parseFloat(elements.clarityInput.value);
      actions.recordLog("clarity", `Детализация: ${value.toFixed(0)}`, { value });
    });
    elements.sharpnessInput.addEventListener("change", () => {
      const value = Number.parseFloat(elements.sharpnessInput.value);
      actions.recordLog("sharpness", `Резкость: ${value.toFixed(0)}`, { value });
    });
    elements.lowlightBoostToggle.addEventListener("change", () => {
      actions.recordLog(
        "lowlight-boost",
        elements.lowlightBoostToggle.checked
          ? "Low-light усиление включено"
          : "Low-light усиление выключено"
      );
    });
    elements.grayscaleToggle.addEventListener("change", () => {
      actions.recordLog(
        "grayscale",
        elements.grayscaleToggle.checked ? "Ч/Б режим включен" : "Ч/Б режим выключен"
      );
    });
    elements.bypassFiltersToggle.addEventListener("change", () => {
      actions.recordLog(
        "bypass-filters",
        elements.bypassFiltersToggle.checked
          ? "Фильтры отключены (оригинал)"
          : "Фильтры включены"
      );
    });
    elements.resetFiltersButton.addEventListener("click", () => {
      elements.enhanceInput.value = "0";
      elements.exposureInput.value = "0";
      elements.temperatureInput.value = "0";
      elements.denoiseInput.value = "0";
      elements.temporalDenoiseToggle.checked = false;
      elements.temporalWindowInput.value = "3";
      elements.denoiseProfile.value = "medium";
      elements.clarityInput.value = "0";
      elements.sharpnessInput.value = "0";
      elements.lowlightBoostToggle.checked = false;
      elements.upscaleToggle.checked = false;
      elements.upscaleFactor.value = "2";
      elements.grayscaleToggle.checked = false;
      elements.bypassFiltersToggle.checked = false;
      elements.stabilizationToggle.checked = false;
      elements.stabilizationAutoToggle.checked = false;
      elements.stabilizationStrength.value = "0";
      elements.stabilizationSmoothing.value = "60";
      elements.stabilizationOffsetX.value = "0";
      elements.stabilizationOffsetY.value = "0";
      state.temporalFrames = [];
      elements.video.style.opacity = "1";
      if (elements.temporalPreview) {
        elements.temporalPreview.style.opacity = "0";
      }
      stopAutoStabilization();
      applyFilters();
      actions.updateZoom();
      actions.recordLog("filters-reset", "Сброс фильтров");
    });

    elements.presetLowlightButton.addEventListener("click", () => {
      applyPreset({
        key: "lowlight",
        label: "Low-light",
        enhance: 55,
        exposure: 15,
        temperature: 10,
        denoise: 2.5,
        temporalDenoise: true,
        temporalWindow: 4,
        denoiseProfile: "heavy",
        clarity: 10,
        sharpness: 10,
        lowlightBoost: true,
        grayscale: false,
      });
      elements.stabilizationToggle.checked = true;
      elements.stabilizationStrength.value = "4";
      elements.stabilizationOffsetX.value = "0";
      elements.stabilizationOffsetY.value = "0";
      elements.temporalDenoiseToggle.checked = true;
      elements.temporalWindowInput.value = "4";
      actions.updateZoom();
    });

    elements.presetNightButton.addEventListener("click", () => {
      applyPreset({
        key: "night",
        label: "Ночь",
        enhance: 70,
        exposure: 25,
        temperature: 18,
        denoise: 3,
        temporalDenoise: true,
        temporalWindow: 4,
        denoiseProfile: "heavy",
        clarity: 15,
        sharpness: 10,
        lowlightBoost: true,
        grayscale: false,
      });
      elements.stabilizationToggle.checked = true;
      elements.stabilizationStrength.value = "6";
      elements.stabilizationOffsetX.value = "0";
      elements.stabilizationOffsetY.value = "0";
      elements.temporalDenoiseToggle.checked = true;
      elements.temporalWindowInput.value = "4";
      actions.updateZoom();
    });

    elements.presetDetailButton.addEventListener("click", () => {
      applyPreset({
        key: "detail",
        label: "Детализация",
        enhance: 35,
        exposure: 0,
        temperature: 0,
        denoise: 1,
        temporalDenoise: false,
        temporalWindow: 3,
        denoiseProfile: "light",
        clarity: 25,
        sharpness: 20,
        lowlightBoost: false,
        grayscale: false,
      });
      elements.stabilizationToggle.checked = false;
      elements.stabilizationStrength.value = "0";
      elements.stabilizationOffsetX.value = "0";
      elements.stabilizationOffsetY.value = "0";
      elements.temporalDenoiseToggle.checked = false;
      elements.temporalWindowInput.value = "3";
      actions.updateZoom();
    });

    elements.presetUltraLowlightButton.addEventListener("click", () => {
      applyPreset({
        key: "ultra-lowlight",
        label: "Ultra Low-light",
        enhance: 80,
        exposure: 35,
        temperature: 20,
        denoise: 4,
        temporalDenoise: true,
        temporalWindow: 5,
        denoiseProfile: "heavy",
        clarity: 5,
        sharpness: 5,
        lowlightBoost: true,
        grayscale: false,
      });
      elements.stabilizationToggle.checked = true;
      elements.stabilizationStrength.value = "7";
      elements.stabilizationOffsetX.value = "0";
      elements.stabilizationOffsetY.value = "0";
      elements.temporalDenoiseToggle.checked = true;
      elements.temporalWindowInput.value = "5";
      actions.updateZoom();
    });

    elements.upscaleToggle.addEventListener("change", () => {
      actions.updateZoom();
      const factor = Number.parseFloat(elements.upscaleFactor.value) || 2;
      actions.recordLog(
        "upscale-toggle",
        elements.upscaleToggle.checked
          ? `Апскейл включен (${factor}x)`
          : "Апскейл выключен"
      );
    });

    elements.video.addEventListener("timeupdate", () => {
      applyTemporalDenoise();
    });
    applyFilters();
  },
});
