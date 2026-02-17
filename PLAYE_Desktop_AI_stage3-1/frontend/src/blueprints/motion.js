export const createMotionBlueprint = () => ({
  name: "motion",
  init: ({ elements, state, actions }) => {
    const syncRangeValue = (input, output, suffix = "") => {
      if (!output) return;
      output.textContent = `${input.value}${suffix}`;
    };

    const getSensitivity = () =>
      Number.parseFloat(elements.motionSensitivity.value || "800");
    const getCooldown = () =>
      Number.parseFloat(elements.motionCooldown.value || "3");

    const setupMotionCanvas = () => {
      if (!state.motionCanvas) {
        state.motionCanvas = document.createElement("canvas");
        state.motionContext = state.motionCanvas.getContext("2d", {
          willReadFrequently: true,
        });
      }
      state.motionCanvas.width = elements.video.videoWidth;
      state.motionCanvas.height = elements.video.videoHeight;
    };

    const detectMotion = () => {
      if (!state.motionDetectionActive || elements.video.paused || elements.video.ended) {
        state.motionLoopId = requestAnimationFrame(detectMotion);
        return;
      }

      setupMotionCanvas();
      state.motionContext.drawImage(
        elements.video,
        0,
        0,
        state.motionCanvas.width,
        state.motionCanvas.height
      );
      const frame = state.motionContext.getImageData(
        0,
        0,
        state.motionCanvas.width,
        state.motionCanvas.height
      );

      if (state.previousFrameData) {
        let diffCount = 0;
        for (let i = 0; i < frame.data.length; i += 16) {
          const delta = Math.abs(frame.data[i] - state.previousFrameData.data[i]);
          if (delta > 20) diffCount += 1;
        }
        const threshold = getSensitivity();
        const isActive = diffCount > threshold;
        elements.motionIndicator.classList.toggle("active", isActive);
        if (isActive && elements.motionMarkerToggle.checked) {
          const cooldown = getCooldown();
          const currentTime = elements.video.currentTime;
          const lastTime = state.motionLastMarkerTime;
          const canMark =
            lastTime === null || currentTime - lastTime >= cooldown;
          if (canMark) {
            const entry = {
              timestamp: new Date().toISOString(),
              time: currentTime,
              timecode: actions.formatTime(currentTime),
              type: "motion-auto",
              note: `Движение (diff=${diffCount})`,
            };
            state.markers.unshift(entry);
            actions.appendMarkerEntry(entry);
            actions.recordLog(
              "motion-marker",
              `Маркер движения на ${entry.timecode}`,
              {
                time: currentTime,
                diffCount,
              }
            );
            state.motionLastMarkerTime = currentTime;
            if (actions.refreshTimeline) {
              actions.refreshTimeline();
            }
          }
        }
      }

      state.previousFrameData = frame;
      state.motionLoopId = requestAnimationFrame(detectMotion);
    };

    elements.motionStart.addEventListener("click", () => {
      state.motionDetectionActive = true;
      elements.motionStart.disabled = true;
      elements.motionStop.disabled = false;
      elements.motionIndicator.classList.remove("active");
      state.previousFrameData = null;
      state.motionLastMarkerTime = null;
      detectMotion();
      actions.recordLog("motion-start", "Запуск детектора движения");
    });

    elements.motionStop.addEventListener("click", () => {
      state.motionDetectionActive = false;
      elements.motionStart.disabled = false;
      elements.motionStop.disabled = true;
      elements.motionIndicator.classList.remove("active");
      if (state.motionLoopId) {
        cancelAnimationFrame(state.motionLoopId);
      }
      actions.recordLog("motion-stop", "Остановка детектора движения");
    });

    syncRangeValue(elements.motionSensitivity, elements.motionSensitivityValue);
    syncRangeValue(elements.motionCooldown, elements.motionCooldownValue);
    elements.motionSensitivity.addEventListener("input", () => {
      syncRangeValue(elements.motionSensitivity, elements.motionSensitivityValue);
    });
    elements.motionCooldown.addEventListener("input", () => {
      syncRangeValue(elements.motionCooldown, elements.motionCooldownValue);
    });
    elements.motionSensitivity.addEventListener("change", () => {
      const value = getSensitivity();
      actions.recordLog("motion-sensitivity", "Чувствительность детектора", {
        value,
      });
    });
    elements.motionCooldown.addEventListener("change", () => {
      const value = getCooldown();
      actions.recordLog("motion-cooldown", "Интервал маркеров", { value });
    });
    elements.motionMarkerToggle.addEventListener("change", () => {
      actions.recordLog(
        "motion-marker-toggle",
        elements.motionMarkerToggle.checked
          ? "Авто-маркеры движения включены"
          : "Авто-маркеры движения выключены"
      );
    });
  },
});
