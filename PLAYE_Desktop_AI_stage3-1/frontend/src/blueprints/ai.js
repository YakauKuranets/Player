export const createAiBlueprint = () => ({
  name: "ai",
  init: ({ elements, actions, state }) => {
    const overlay = elements.aiOverlay;
    const context = overlay.getContext("2d");
    let faceDetector = null;
    let trackingLoopId = null;
    let trackingActive = false;

    const trackingState = {
      nextTrackId: 1,
      tracks: [],
    };

    const sceneState = {
      previousLumaSample: null,
      detections: [],
    };

    const ensureAnalysisCanvas = () => {
      if (!state.aiAnalysisCanvas) {
        state.aiAnalysisCanvas = document.createElement("canvas");
        state.aiAnalysisContext = state.aiAnalysisCanvas.getContext("2d", {
          willReadFrequently: true,
        });
      }
      state.aiAnalysisCanvas.width = elements.video.videoWidth;
      state.aiAnalysisCanvas.height = elements.video.videoHeight;
    };

    const updateOverlaySize = () => {
      if (!elements.video.videoWidth || !elements.video.videoHeight) return;
      overlay.width = elements.video.videoWidth;
      overlay.height = elements.video.videoHeight;
    };

    const clearOverlay = () => {
      context.clearRect(0, 0, overlay.width, overlay.height);
    };

    const setStatus = (text) => {
      elements.aiStatus.textContent = text;
    };

    const applyAiProvider = (provider, { log = true } = {}) => {
      state.aiProvider = provider;
      state.aiRuntimeInfo = {
        provider,
        modelVersion: runtimeAdapters[provider]?.modelVersion || "unknown",
      };
      setStatus(`AI runtime provider: ${provider}`);
      if (log) {
        actions.recordLog("ai-provider-change", "Изменен AI runtime provider", {
          provider,
          modelVersion: state.aiRuntimeInfo.modelVersion,
        });
      }
    };
    const formatCapabilityStatus = (capabilities) => {
      const parts = [
        `WebGPU: ${capabilities.webgpu ? "yes" : "no"}`,
        `WebGL2: ${capabilities.webgl2 ? "yes" : "no"}`,
        `WebGL: ${capabilities.webgl ? "yes" : "no"}`,
        `FaceDetector: ${capabilities.faceDetector ? "yes" : "no"}`,
        `Fallback: ${capabilities.fallback}`,
      ];
      return `Capability: ${parts.join(" | ")}`;
    };

    const runCapabilityCheck = ({ log = true } = {}) => {
      const canvas = document.createElement("canvas");
      const hasWebGL2 = Boolean(canvas.getContext("webgl2"));
      const hasWebGL = hasWebGL2 || Boolean(canvas.getContext("webgl"));
      const hasWebGPU = Boolean(navigator.gpu);
      const hasFaceDetector = "FaceDetector" in window;
      const fallback = hasWebGPU ? "webgpu" : hasWebGL ? "webgl" : "cpu";

      state.aiCapabilities = {
        checkedAt: new Date().toISOString(),
        webgpu: hasWebGPU,
        webgl2: hasWebGL2,
        webgl: hasWebGL,
        faceDetector: hasFaceDetector,
        fallback,
      };

      if (elements.aiCapabilityStatus) {
        elements.aiCapabilityStatus.textContent = formatCapabilityStatus(state.aiCapabilities);
      }
      setStatus(`AI capability check завершен. Активный fallback: ${fallback.toUpperCase()}`);

      if (log) {
        actions.recordLog("ai-capability-check", "Проверены AI/GPU capability и fallback", {
          ...state.aiCapabilities,
        });
      }

      return state.aiCapabilities;
    };


    const applyAiSuperResolution = () => {
      const factor = Number.parseFloat(elements.aiSrFactor.value) || 2;
      state.aiSuperResolutionFactor = factor;
      state.aiSuperResolutionActive = true;
      actions.updateZoom();
      setStatus(`AI super-resolution (demo) активирован: ${factor}x`);
      actions.recordLog("ai-super-resolution-apply", "AI super-resolution (demo) включен", {
        factor,
      });
    };

    const resetAiSuperResolution = () => {
      const wasActive = state.aiSuperResolutionActive;
      state.aiSuperResolutionActive = false;
      state.aiSuperResolutionFactor = 1;
      actions.updateZoom();
      if (wasActive) {
        setStatus("AI super-resolution (demo) отключен.");
        actions.recordLog("ai-super-resolution-reset", "AI super-resolution (demo) выключен");
      }
    };

    const renderResultList = (listElement, rows, fallbackText) => {
      listElement.innerHTML = "";
      if (!rows.length) {
        const item = document.createElement("li");
        item.textContent = fallbackText;
        listElement.appendChild(item);
        return;
      }
      rows.forEach((row) => {
        const item = document.createElement("li");
        item.textContent = row;
        listElement.appendChild(item);
      });
    };

    const renderBoxes = (boxes, color) => {
      clearOverlay();
      context.strokeStyle = color;
      context.lineWidth = Math.max(2, overlay.width * 0.002);
      boxes.forEach(({ x, y, width, height }) => {
        context.strokeRect(x, y, width, height);
      });
    };

    const renderTracks = () => {
      clearOverlay();
      context.lineWidth = Math.max(2, overlay.width * 0.002);
      context.font = "12px Inter, system-ui, sans-serif";

      trackingState.tracks.forEach((track) => {
        const { box, id, history } = track;
        context.strokeStyle = "#34d399";
        context.strokeRect(box.x, box.y, box.width, box.height);

        context.fillStyle = "rgba(15, 23, 42, 0.8)";
        const label = `ID ${id}`;
        context.fillRect(box.x, Math.max(0, box.y - 16), 48, 14);
        context.fillStyle = "#a7f3d0";
        context.fillText(label, box.x + 6, Math.max(11, box.y - 5));

        if (history.length > 1) {
          context.strokeStyle = "rgba(52, 211, 153, 0.5)";
          context.beginPath();
          history.forEach((point, index) => {
            if (index === 0) {
              context.moveTo(point.x, point.y);
            } else {
              context.lineTo(point.x, point.y);
            }
          });
          context.stroke();
        }
      });
    };

    const ensureFaceDetector = () => {
      if (!("FaceDetector" in window)) {
        setStatus("FaceDetector не поддерживается в этом браузере.");
        actions.recordLog("ai-face-unsupported", "FaceDetector API недоступен");
        return null;
      }
      if (!faceDetector) {
        faceDetector = new FaceDetector({
          fastMode: true,
          maxDetectedFaces: 10,
        });
      }
      return faceDetector;
    };

    const detectObjectsFromFrame = () => {
      if (elements.video.readyState < 2) {
        return [];
      }

      updateOverlaySize();
      ensureAnalysisCanvas();
      const width = state.aiAnalysisCanvas.width;
      const height = state.aiAnalysisCanvas.height;
      if (!width || !height) return [];

      state.aiAnalysisContext.drawImage(elements.video, 0, 0, width, height);
      const frame = state.aiAnalysisContext.getImageData(0, 0, width, height);
      const data = frame.data;

      const cellSize = 24;
      const cols = Math.max(1, Math.floor(width / cellSize));
      const rows = Math.max(1, Math.floor(height / cellSize));
      const scores = new Array(cols * rows).fill(0);

      for (let row = 0; row < rows; row += 1) {
        for (let col = 0; col < cols; col += 1) {
          let score = 0;
          let samples = 0;
          const startX = col * cellSize;
          const startY = row * cellSize;
          const endX = Math.min(width - 2, startX + cellSize - 1);
          const endY = Math.min(height - 2, startY + cellSize - 1);

          for (let y = startY; y < endY; y += 3) {
            for (let x = startX; x < endX; x += 3) {
              const idx = (y * width + x) * 4;
              const rightIdx = (y * width + (x + 1)) * 4;
              const downIdx = ((y + 1) * width + x) * 4;

              const lum =
                data[idx] * 0.2126 +
                data[idx + 1] * 0.7152 +
                data[idx + 2] * 0.0722;
              const lumRight =
                data[rightIdx] * 0.2126 +
                data[rightIdx + 1] * 0.7152 +
                data[rightIdx + 2] * 0.0722;
              const lumDown =
                data[downIdx] * 0.2126 +
                data[downIdx + 1] * 0.7152 +
                data[downIdx + 2] * 0.0722;

              score += Math.abs(lum - lumRight) + Math.abs(lum - lumDown);
              samples += 1;
            }
          }

          const normalized = samples ? score / samples : 0;
          scores[row * cols + col] = normalized;
        }
      }

      const threshold = 28;
      const visited = new Set();
      const clusters = [];
      const neighbors = [
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
      ];

      const visitCluster = (startCol, startRow) => {
        const queue = [[startCol, startRow]];
        let queueIndex = 0;
        let minCol = startCol;
        let maxCol = startCol;
        let minRow = startRow;
        let maxRow = startRow;
        let points = 0;

        while (queueIndex < queue.length) {
          const [col, row] = queue[queueIndex];
          queueIndex += 1;
          const key = `${col}:${row}`;
          if (visited.has(key)) continue;
          visited.add(key);
          const idx = row * cols + col;
          if (scores[idx] < threshold) continue;

          points += 1;
          minCol = Math.min(minCol, col);
          maxCol = Math.max(maxCol, col);
          minRow = Math.min(minRow, row);
          maxRow = Math.max(maxRow, row);

          neighbors.forEach(([dx, dy]) => {
            const nextCol = col + dx;
            const nextRow = row + dy;
            if (
              nextCol >= 0 &&
              nextCol < cols &&
              nextRow >= 0 &&
              nextRow < rows &&
              !visited.has(`${nextCol}:${nextRow}`)
            ) {
              queue.push([nextCol, nextRow]);
            }
          });
        }

        if (points < 4) return null;
        return {
          x: minCol * cellSize,
          y: minRow * cellSize,
          width: Math.min(width - minCol * cellSize, (maxCol - minCol + 1) * cellSize),
          height: Math.min(height - minRow * cellSize, (maxRow - minRow + 1) * cellSize),
          points,
        };
      };

      for (let row = 0; row < rows; row += 1) {
        for (let col = 0; col < cols; col += 1) {
          const idx = row * cols + col;
          if (scores[idx] < threshold || visited.has(`${col}:${row}`)) continue;
          const cluster = visitCluster(col, row);
          if (cluster && cluster.width * cluster.height > 3200) {
            clusters.push(cluster);
          }
        }
      }

      return clusters
        .sort((a, b) => b.points - a.points)
        .slice(0, 8)
        .map(({ x, y, width: boxWidth, height: boxHeight }) => ({
          x,
          y,
          width: boxWidth,
          height: boxHeight,
        }));
    };

    const maybeCreateObjectMarker = (count, source) => {
      if (!elements.aiObjectMarkerToggle.checked) return;
      const time = elements.video.currentTime;
      const entry = {
        timestamp: new Date().toISOString(),
        time,
        timecode: actions.formatTime(time),
        type: "AI: объекты",
        note: `${source}: найдено объектов ${count}`,
      };
      state.markers.unshift(entry);
      actions.appendMarkerEntry(entry);
      actions.recordLog("ai-object-marker", `Добавлен маркер объектов: ${entry.timecode}`, {
        count,
        source,
      });
      if (actions.refreshTimeline) {
        actions.refreshTimeline();
      }
    };

    const refreshSceneList = () => {
      renderResultList(
        elements.aiSceneList,
        sceneState.detections.map((scene, index) =>
          `Сцена ${index + 1}: ${scene.timecode} | Δ ${scene.delta.toFixed(1)}`
        ),
        "Авто‑сцены не обнаружены."
      );
    };

    const runAutoSceneDelta = () => {
      if (elements.video.readyState < 2) {
        return { error: "Видео не готово для авто‑сцен." };
      }
      ensureAnalysisCanvas();
      const width = state.aiAnalysisCanvas.width;
      const height = state.aiAnalysisCanvas.height;
      if (!width || !height) {
        return { error: "Не удалось инициализировать кадр анализа." };
      }

      state.aiAnalysisContext.drawImage(elements.video, 0, 0, width, height);
      const frame = state.aiAnalysisContext.getImageData(0, 0, width, height);
      const data = frame.data;

      let sum = 0;
      let count = 0;
      for (let i = 0; i < data.length; i += 16) {
        const lum = data[i] * 0.2126 + data[i + 1] * 0.7152 + data[i + 2] * 0.0722;
        sum += lum;
        count += 1;
      }

      const averageLuma = count ? sum / count : 0;
      const threshold = Number.parseFloat(elements.aiSceneThreshold.value) || 28;
      return {
        averageLuma,
        threshold,
      };
    };

    const runFaceDetectorDemo = async () => {
      if (elements.video.readyState < 2) {
        return { error: "Видео не готово для анализа." };
      }

      const activeDetector = ensureFaceDetector();
      if (!activeDetector) {
        return { error: "FaceDetector API недоступен" };
      }

      const faces = await activeDetector.detect(elements.video);
      const boxes = faces.map((face) => face.boundingBox);
      return {
        faces,
        boxes,
      };
    };

    const createRuntimeAdapters = () => {
      const mockAdapter = {
        id: "mock",
        modelVersion: "demo-mock-1.0.0",
        detectFaces: async () => runFaceDetectorDemo(),
        detectObjects: async () => ({ objects: detectObjectsFromFrame() }),
        detectScenes: async () => runAutoSceneDelta(),
      };

      const createProxyAdapter = (id) => ({
        id,
        modelVersion: `${id}-proxy-1.0.0`,
        detectFaces: async () => {
          const result = await mockAdapter.detectFaces();
          return {
            ...result,
            provider: id,
            emulated: true,
          };
        },
        detectObjects: async () => {
          const result = await mockAdapter.detectObjects();
          return {
            ...result,
            provider: id,
            emulated: true,
          };
        },
        detectScenes: async () => {
          const result = await mockAdapter.detectScenes();
          return {
            ...result,
            provider: id,
            emulated: true,
          };
        },
      });

      return {
        mock: mockAdapter,
        onnx: createProxyAdapter("onnx"),
        torch: createProxyAdapter("torch"),
      };
    };

    const runtimeAdapters = createRuntimeAdapters();

    const runModelInference = async (task, payload = {}) => {
      const provider = state.aiProvider || "mock";
      const adapter = runtimeAdapters[provider] || runtimeAdapters.mock;
      const method = adapter[task];
      if (typeof method !== "function") {
        const fallback = runtimeAdapters.mock[task];
        const fallbackResult =
          typeof fallback === "function" ? await fallback(payload) : { error: "Метод адаптера недоступен" };
        actions.recordLog("ai-adapter-fallback", "Fallback к mock adapter для AI задачи", {
          provider,
          task,
        });
        return {
          ...fallbackResult,
          provider: "mock",
          fallback: true,
        };
      }
      const result = await method(payload);
      return {
        ...result,
        provider,
        modelVersion: adapter.modelVersion || "unknown",
      };
    };

    const runMockInferencePipeline = async (task, payload = {}) => {
      const startedAt = performance.now();
      const context = {
        task,
        payload,
        provider: state.aiProvider || "mock",
      };

      const preprocessed = {
        ...context,
        preparedAt: new Date().toISOString(),
      };

      const inferenceResult = await runModelInference(task, preprocessed.payload);

      const durationMs = Number((performance.now() - startedAt).toFixed(2));
      const result = {
        ...inferenceResult,
        pipeline: {
          preprocessed: true,
          postprocessed: true,
          durationMs,
          task,
          provider: inferenceResult.provider || context.provider,
        },
      };

      actions.recordLog("ai-mock-pipeline-run", "Выполнен mock‑pipeline инференса", {
        task,
        provider: result.pipeline.provider,
        modelVersion: result.modelVersion,
        durationMs,
        hasError: Boolean(result.error),
      });

      return result;
    };

    const detectAutoScenes = async () => {
      const { averageLuma, threshold, provider, emulated, pipeline, error } =
        await runMockInferencePipeline("detectScenes");
      if (error) {
        setStatus(error);
        return;
      }

      if (sceneState.previousLumaSample === null) {
        sceneState.previousLumaSample = averageLuma;
        setStatus("Авто‑сцены: базовый кадр зафиксирован.");
        actions.recordLog("ai-auto-scenes-baseline", "Авто‑сцены: базовый кадр зафиксирован", {
          luma: averageLuma,
        });
        return;
      }

      const delta = Math.abs(averageLuma - sceneState.previousLumaSample);
      sceneState.previousLumaSample = averageLuma;
      if (delta < threshold) {
        setStatus(`Авто‑сцены: переход не найден (Δ ${delta.toFixed(1)}).`);
        return;
      }

      const time = elements.video.currentTime;
      const sceneEntry = {
        timestamp: new Date().toISOString(),
        time,
        timecode: actions.formatTime(time),
        delta,
      };
      sceneState.detections.unshift(sceneEntry);
      sceneState.detections = sceneState.detections.slice(0, 25);
      refreshSceneList();

      const marker = {
        timestamp: sceneEntry.timestamp,
        time: sceneEntry.time,
        timecode: sceneEntry.timecode,
        type: "AI: сцена",
        note: `Авто‑сцена (Δ ${delta.toFixed(1)})`,
      };
      state.markers.unshift(marker);
      actions.appendMarkerEntry(marker);
      if (actions.refreshTimeline) {
        actions.refreshTimeline();
      }

      actions.recordLog("ai-auto-scenes-detect", "Обнаружен авто‑переход сцены (demo)", {
        delta,
        threshold,
        provider,
        emulated,
        pipeline: {
          durationMs: pipeline?.durationMs,
        },
        timecode: sceneEntry.timecode,
      });
      setStatus(`Авто‑сцена обнаружена: ${sceneEntry.timecode}`);
    };

    const clearAutoScenes = () => {
      sceneState.previousLumaSample = null;
      sceneState.detections = [];
      refreshSceneList();
      setStatus("Авто‑сцены очищены.");
      actions.recordLog("ai-auto-scenes-clear", "Список авто‑сцен очищен");
    };

    const detectFaces = async () => {
      setStatus("Идёт поиск лиц...");
      try {
        const { faces, boxes, provider, emulated, pipeline, error } = await runMockInferencePipeline("detectFaces");
        if (error) {
          setStatus(error);
          return;
        }
        updateOverlaySize();
        renderBoxes(boxes, "#38bdf8");

        renderResultList(
          elements.aiFaceList,
          faces.map((face, index) => {
            const box = face.boundingBox;
            return `Лицо ${index + 1}: ${Math.round(box.width)}x${Math.round(
              box.height
            )} px`;
          }),
          "Лица не найдены."
        );

        setStatus(`Найдено лиц: ${faces.length}`);
        actions.recordLog("ai-face-detect", "Демо детекция лиц", {
          count: faces.length,
          provider,
          emulated,
          pipeline: {
            durationMs: pipeline?.durationMs,
          },
        });

        if (elements.aiFaceMarkerToggle.checked) {
          const time = elements.video.currentTime;
          const entry = {
            timestamp: new Date().toISOString(),
            time,
            timecode: actions.formatTime(time),
            type: "AI: лица",
            note: `Найдено лиц: ${faces.length}`,
          };
          state.markers.unshift(entry);
          actions.appendMarkerEntry(entry);
          actions.recordLog("ai-face-marker", `Добавлен маркер лиц: ${entry.timecode}`, {
            count: faces.length,
          });
          if (actions.refreshTimeline) {
            actions.refreshTimeline();
          }
        }
      } catch (error) {
        setStatus("Ошибка анализа лиц.");
        actions.recordLog("ai-face-error", "Ошибка демо-анализа лиц", {
          message: error?.message,
        });
      }
    };

    const detectObjectsDemo = async () => {
      const { objects = [], provider, emulated, pipeline, error } = await runMockInferencePipeline("detectObjects");
      if (error) {
        setStatus(error);
        return;
      }
      if (!objects.length && elements.video.readyState < 2) {
        setStatus("Видео не готово для анализа.");
        return;
      }

      renderBoxes(objects, "#34d399");
      renderResultList(
        elements.aiObjectList,
        objects.map(
          (box, index) =>
            `Объект ${index + 1}: ${Math.round(box.width)}x${Math.round(box.height)} px`
        ),
        "Объекты не найдены."
      );

      setStatus(`Найдено объектов (demo): ${objects.length}`);
      actions.recordLog("ai-object-detect", "Демо детекция объектов", {
        count: objects.length,
        provider,
        emulated,
        pipeline: {
          durationMs: pipeline?.durationMs,
        },
      });
      maybeCreateObjectMarker(objects.length, "single-detect");
    };

    const boxCenter = (box) => ({
      x: box.x + box.width / 2,
      y: box.y + box.height / 2,
    });

    const distance = (a, b) => {
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      return Math.sqrt(dx * dx + dy * dy);
    };

    const updateTracking = (detections) => {
      const maxDistance = 90;
      const matchedTrackIds = new Set();
      const matchedDetectionIds = new Set();
      const newTrackIds = [];

      detections.forEach((detection, detectionIndex) => {
        const detectionCenter = boxCenter(detection);
        let bestTrack = null;
        let bestDistance = Number.POSITIVE_INFINITY;

        trackingState.tracks.forEach((track) => {
          if (matchedTrackIds.has(track.id)) return;
          const trackCenter = boxCenter(track.box);
          const dist = distance(detectionCenter, trackCenter);
          if (dist < bestDistance) {
            bestDistance = dist;
            bestTrack = track;
          }
        });

        if (bestTrack && bestDistance <= maxDistance) {
          bestTrack.box = detection;
          bestTrack.lastSeenTime = elements.video.currentTime;
          bestTrack.missed = 0;
          bestTrack.history.push(detectionCenter);
          bestTrack.history = bestTrack.history.slice(-12);
          matchedTrackIds.add(bestTrack.id);
          matchedDetectionIds.add(detectionIndex);
          return;
        }

        const newTrack = {
          id: trackingState.nextTrackId,
          box: detection,
          missed: 0,
          lastSeenTime: elements.video.currentTime,
          history: [detectionCenter],
        };
        trackingState.nextTrackId += 1;
        trackingState.tracks.push(newTrack);
        matchedTrackIds.add(newTrack.id);
        matchedDetectionIds.add(detectionIndex);
        newTrackIds.push(newTrack.id);
      });

      trackingState.tracks = trackingState.tracks
        .map((track) => {
          if (matchedTrackIds.has(track.id)) return track;
          return {
            ...track,
            missed: track.missed + 1,
          };
        })
        .filter((track) => track.missed <= 8);

      return newTrackIds;
    };

    const refreshTrackList = () => {
      renderResultList(
        elements.aiObjectList,
        trackingState.tracks.map((track) => {
          const { id, box, lastSeenTime } = track;
          return `Track #${id}: ${Math.round(box.width)}x${Math.round(
            box.height
          )} px @ ${actions.formatTime(lastSeenTime)}`;
        }),
        "Треки не обнаружены."
      );
    };

    const stopObjectTracking = (reason = "manual") => {
      trackingActive = false;
      if (trackingLoopId) {
        cancelAnimationFrame(trackingLoopId);
      }
      trackingLoopId = null;
      elements.aiTrackStartButton.disabled = false;
      elements.aiTrackStopButton.disabled = true;
      actions.recordLog("ai-object-track-stop", "Трекинг объектов остановлен (demo)", {
        reason,
      });
    };

    const trackFrame = () => {
      if (!trackingActive) return;
      if (elements.video.ended) {
        stopObjectTracking("video-ended");
        setStatus("Трекинг остановлен: видео завершено.");
        return;
      }

      if (!elements.video.paused && elements.video.readyState >= 2) {
        const detections = detectObjectsFromFrame();
        const newTrackIds = updateTracking(detections);
        refreshTrackList();
        renderTracks();
        setStatus(`Трекинг объектов (demo): активных треков ${trackingState.tracks.length}`);

        if (newTrackIds.length > 0) {
          actions.recordLog("ai-object-track-update", "Созданы новые треки объектов", {
            newTrackIds,
            activeTracks: trackingState.tracks.length,
          });
          maybeCreateObjectMarker(trackingState.tracks.length, "tracking-new-tracks");
        }
      }

      trackingLoopId = requestAnimationFrame(trackFrame);
    };

    const startObjectTracking = () => {
      if (trackingActive) return;
      if (elements.video.readyState < 2) {
        setStatus("Видео не готово для трекинга.");
        return;
      }

      trackingState.tracks = [];
      trackingState.nextTrackId = 1;
      elements.aiTrackStartButton.disabled = true;
      elements.aiTrackStopButton.disabled = false;
      setStatus("Трекинг объектов запущен (demo).");
      actions.recordLog("ai-object-track-start", "Трекинг объектов запущен (demo)");

      trackingActive = true;
      trackFrame();
    };

    elements.aiFaceDetectButton.addEventListener("click", () => {
      detectFaces();
    });

    elements.aiObjectDetectButton.addEventListener("click", () => {
      detectObjectsDemo();
    });

    elements.aiProviderSelect.addEventListener("change", () => {
      applyAiProvider(elements.aiProviderSelect.value);
    });

    elements.aiCapabilityCheckButton.addEventListener("click", () => {
      runCapabilityCheck();
    });

    elements.aiTrackStartButton.addEventListener("click", () => {
      startObjectTracking();
    });

    elements.aiTrackStopButton.addEventListener("click", () => {
      stopObjectTracking("manual");
      setStatus("Трекинг объектов остановлен.");
    });

    elements.aiSrApplyButton.addEventListener("click", () => {
      applyAiSuperResolution();
    });

    elements.aiSrResetButton.addEventListener("click", () => {
      resetAiSuperResolution();
    });

    elements.aiSrFactor.addEventListener("change", () => {
      const factor = Number.parseFloat(elements.aiSrFactor.value) || 2;
      actions.recordLog("ai-super-resolution-factor", "Изменен AI SR фактор (demo)", {
        factor,
      });
      if (state.aiSuperResolutionActive) {
        state.aiSuperResolutionFactor = factor;
        actions.updateZoom();
        setStatus(`AI super-resolution (demo) обновлён: ${factor}x`);
      }
    });

    elements.aiScenesDetectButton.addEventListener("click", () => {
      detectAutoScenes();
    });

    elements.aiScenesClearButton.addEventListener("click", () => {
      clearAutoScenes();
    });

    elements.aiSceneThreshold.addEventListener("change", () => {
      const threshold = Number.parseFloat(elements.aiSceneThreshold.value) || 28;
      actions.recordLog("ai-auto-scenes-threshold", "Изменен порог авто‑сцен (demo)", {
        threshold,
      });
    });

    elements.video.addEventListener("pause", () => {
      if (trackingActive) {
        setStatus("Трекинг на паузе (ожидание воспроизведения).");
      }
    });

    elements.video.addEventListener("loadedmetadata", () => {
      updateOverlaySize();
      clearOverlay();
      trackingState.tracks = [];
      sceneState.previousLumaSample = null;
      sceneState.detections = [];
      refreshTrackList();
      refreshSceneList();
      renderResultList(elements.aiFaceList, [], "Лица не найдены.");
    });

    elements.aiProviderSelect.value = state.aiProvider || "mock";
    applyAiProvider(elements.aiProviderSelect.value, { log: false });
    runCapabilityCheck({ log: false });

    renderResultList(elements.aiFaceList, [], "Лица не найдены.");
    renderResultList(elements.aiObjectList, [], "Объекты не найдены.");
    refreshSceneList();
    elements.aiSrFactor.value = "2";
    elements.aiSceneThreshold.value = "28";

    const observer = new ResizeObserver(() => {
      updateOverlaySize();
      if (trackingState.tracks.length) {
        renderTracks();
      }
    });
    observer.observe(elements.viewerSurface);
  },
});
