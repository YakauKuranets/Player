const initVideoTemporalPipeline = ({ elements, state, actions }) => {
  elements.temporalDenoiseButton?.addEventListener('click', async () => {
    if (!state.currentVideoFile) {
      actions.recordLog('temporal-denoise', 'Файл не выбран');
      return;
    }

    elements.temporalDenoiseButton.disabled = true;
    if (elements.temporalProgressBar) {
      elements.temporalProgressBar.style.width = '0%';
    }

    try {
      const formData = new FormData();
      formData.append('file', state.currentVideoFile);
      formData.append('operation', 'temporal_denoise');
      formData.append('fps', elements.fpsPicker?.value || '1.0');

      const resp = await fetch('http://127.0.0.1:8000/api/job/video/submit', {
        method: 'POST',
        headers: { Authorization: `Bearer ${state.apiToken || ''}` },
        body: formData,
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const taskId = data.result?.task_id;
      actions.recordLog('temporal-denoise', `Задача запущена: ${taskId}`);

      for (let i = 0; i < 120; i += 1) {
        await new Promise((r) => setTimeout(r, 1000));
        const st = await fetch(`http://127.0.0.1:8000/api/job/${taskId}/status`, {
          headers: { Authorization: `Bearer ${state.apiToken || ''}` },
        }).then((r) => r.json());

        const res = st.result || {};
        const pct = res.progress ?? 0;
        if (elements.temporalProgressBar) {
          elements.temporalProgressBar.style.width = `${pct}%`;
        }

        if (res.is_final) {
          if (res.error) throw new Error(res.error);
          actions.recordLog('temporal-denoise', `Готово: обработано ${res.result?.frames_processed} кадров`);
          if (elements.temporalResultPanel) {
            elements.temporalResultPanel.textContent = JSON.stringify(res.result, null, 2);
          }
          break;
        }
      }
    } catch (err) {
      actions.recordLog('temporal-denoise-error', err.message);
    } finally {
      elements.temporalDenoiseButton.disabled = false;
    }
  });

  elements.localDenoiseButton?.addEventListener('click', async () => {
    if (!state.canvasFrames?.length) {
      actions.recordLog('local-denoise', 'Нет кадров для обработки');
      return;
    }

    const worker = new Worker('/src/workers/temporal-denoise-worker.js');
    worker.postMessage({
      type: 'denoise',
      payload: {
        frames: state.canvasFrames,
        width: state.frameWidth,
        height: state.frameHeight,
        useOnnx: false,
      },
    });

    worker.onmessage = (e) => {
      if (e.data.type === 'progress') {
        actions.recordLog('local-denoise', `Кадр ${e.data.frame + 1}/${e.data.total}`);
      }
      if (e.data.type === 'result') {
        state.processedFrames = e.data.frames;
        actions.recordLog('local-denoise', `Готово: ${e.data.frames.length} кадров`);
        worker.terminate();
      }
      if (e.data.type === 'error') {
        actions.recordLog('local-denoise-error', e.data.error);
        worker.terminate();
      }
    };
  });
};

export const createMotionBlueprint = () => ({
  name: 'motion',
  init: ({ elements, state, actions }) => {
    const syncRangeValue = (input, output, suffix = '') => {
      if (!output) return;
      output.textContent = `${input.value}${suffix}`;
    };

    const getSensitivity = () => Number.parseFloat(elements.motionSensitivity.value || '800');
    const getCooldown = () => Number.parseFloat(elements.motionCooldown.value || '3');

    const setupMotionCanvas = () => {
      if (!state.motionCanvas) {
        state.motionCanvas = document.createElement('canvas');
        state.motionContext = state.motionCanvas.getContext('2d', { willReadFrequently: true });
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
      state.motionContext.drawImage(elements.video, 0, 0, state.motionCanvas.width, state.motionCanvas.height);
      const frame = state.motionContext.getImageData(0, 0, state.motionCanvas.width, state.motionCanvas.height);

      if (state.previousFrameData) {
        let diffCount = 0;
        for (let i = 0; i < frame.data.length; i += 16) {
          const delta = Math.abs(frame.data[i] - state.previousFrameData.data[i]);
          if (delta > 20) diffCount += 1;
        }
        const threshold = getSensitivity();
        const isActive = diffCount > threshold;
        elements.motionIndicator.classList.toggle('active', isActive);
      }

      state.previousFrameData = frame;
      state.motionLoopId = requestAnimationFrame(detectMotion);
    };

    elements.motionStart.addEventListener('click', () => {
      state.motionDetectionActive = true;
      elements.motionStart.disabled = true;
      elements.motionStop.disabled = false;
      elements.motionIndicator.classList.remove('active');
      state.previousFrameData = null;
      state.motionLastMarkerTime = null;
      detectMotion();
      actions.recordLog('motion-start', 'Запуск детектора движения');
    });

    elements.motionStop.addEventListener('click', () => {
      state.motionDetectionActive = false;
      elements.motionStart.disabled = false;
      elements.motionStop.disabled = true;
      elements.motionIndicator.classList.remove('active');
      if (state.motionLoopId) cancelAnimationFrame(state.motionLoopId);
      actions.recordLog('motion-stop', 'Остановка детектора движения');
    });

    syncRangeValue(elements.motionSensitivity, elements.motionSensitivityValue);
    syncRangeValue(elements.motionCooldown, elements.motionCooldownValue);
    elements.motionSensitivity.addEventListener('input', () => syncRangeValue(elements.motionSensitivity, elements.motionSensitivityValue));
    elements.motionCooldown.addEventListener('input', () => syncRangeValue(elements.motionCooldown, elements.motionCooldownValue));
    elements.motionSensitivity.addEventListener('change', () => actions.recordLog('motion-sensitivity', 'Чувствительность детектора', { value: getSensitivity() }));
    elements.motionCooldown.addEventListener('change', () => actions.recordLog('motion-cooldown', 'Интервал маркеров', { value: getCooldown() }));

    initVideoTemporalPipeline({ elements, state, actions });
  },
});
