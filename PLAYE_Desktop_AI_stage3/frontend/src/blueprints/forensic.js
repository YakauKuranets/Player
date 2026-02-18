export const createForensicBlueprint = () => ({
  name: "forensic",
  init: ({ elements, state, actions }) => {
    const escapeHtml = (value) =>
      String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");

    const buildReportPayload = () => ({
      caseId: elements.caseId.value.trim(),
      owner: elements.caseOwner.value.trim(),
      status: elements.caseStatus.value,
      tags: elements.caseTags.value
        .split(",")
        .map((tag) => tag.trim())
        .filter(Boolean),
      summary: elements.caseSummary.value.trim(),
      createdAt: new Date().toISOString(),
      files: state.importedFiles,
      markers: state.markers,
      entries: state.logEntries,
      aiRuntime: {
        provider: state.aiRuntimeInfo?.provider || state.aiProvider || "mock",
        modelVersion: state.aiRuntimeInfo?.modelVersion || "unknown",
        capabilities: state.aiCapabilities,
      },
    });

    const buildReportHtml = (payload) => {
      const tags = payload.tags.length
        ? payload.tags.map(escapeHtml).join(", ")
        : "—";
      const filesRows = payload.files.length
        ? payload.files
            .map(
              (file) => `<tr>
          <td>${escapeHtml(file.name)}</td>
          <td>${escapeHtml(file.type || "—")}</td>
          <td>${escapeHtml(file.size || "—")}</td>
          <td>${escapeHtml(file.hash || "—")}</td>
        </tr>`
            )
            .join("")
        : `<tr><td colspan="4">Файлы не загружены</td></tr>`;
      const markersRows = payload.markers.length
        ? payload.markers
            .map(
              (marker) => `<tr>
          <td>${escapeHtml(marker.timecode)}</td>
          <td>${escapeHtml(marker.type)}</td>
          <td>${escapeHtml(marker.note)}</td>
          <td>${escapeHtml(marker.timestamp)}</td>
        </tr>`
            )
            .join("")
        : `<tr><td colspan="4">Маркеры отсутствуют</td></tr>`;
      const logsRows = payload.entries.length
        ? payload.entries
            .map(
              (entry) => `<tr>
          <td>${escapeHtml(entry.timestamp)}</td>
          <td>${escapeHtml(entry.action)}</td>
          <td>${escapeHtml(entry.message)}</td>
          <td>${escapeHtml(entry.owner || "—")}</td>
        </tr>`
            )
            .join("")
        : `<tr><td colspan="4">Журнал пуст</td></tr>`;
      return `<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8" />
    <title>Отчёт по делу ${escapeHtml(payload.caseId || "Без ID")}</title>
    <style>
      body { font-family: "Inter", Arial, sans-serif; color: #101828; margin: 32px; }
      h1 { font-size: 22px; margin-bottom: 8px; }
      h2 { font-size: 16px; margin-top: 28px; }
      .meta { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 24px; }
      .meta p { margin: 4px 0; }
      .muted { color: #667085; font-size: 12px; }
      table { width: 100%; border-collapse: collapse; margin-top: 12px; }
      th, td { border: 1px solid #d0d5dd; padding: 8px; text-align: left; font-size: 13px; }
      th { background: #f2f4f7; }
      .actions { margin-top: 24px; display: flex; gap: 12px; }
      button { padding: 8px 12px; border-radius: 8px; border: 1px solid #d0d5dd; cursor: pointer; }
    </style>
  </head>
  <body>
    <h1>Отчёт по делу</h1>
    <p class="muted">Сформировано: ${escapeHtml(payload.createdAt)}</p>

    <h2>Данные дела</h2>
    <div class="meta">
      <p><strong>ID дела:</strong> ${escapeHtml(payload.caseId || "—")}</p>
      <p><strong>Ответственный:</strong> ${escapeHtml(payload.owner || "—")}</p>
      <p><strong>Статус:</strong> ${escapeHtml(payload.status)}</p>
      <p><strong>Теги:</strong> ${tags}</p>
      <p><strong>Сводка:</strong> ${escapeHtml(payload.summary || "—")}</p>
    </div>

    <h2>AI runtime</h2>
    <div class="meta">
      <p><strong>Provider:</strong> ${escapeHtml(payload.aiRuntime.provider)}</p>
      <p><strong>Model version:</strong> ${escapeHtml(payload.aiRuntime.modelVersion)}</p>
      <p><strong>Fallback:</strong> ${escapeHtml(payload.aiRuntime.capabilities?.fallback || "unknown")}</p>
      <p><strong>WebGPU/WebGL2/WebGL:</strong> ${escapeHtml(
        `${payload.aiRuntime.capabilities?.webgpu ? "yes" : "no"} / ${
          payload.aiRuntime.capabilities?.webgl2 ? "yes" : "no"
        } / ${payload.aiRuntime.capabilities?.webgl ? "yes" : "no"}`
      )}</p>
    </div>

    <h2>Файлы</h2>
    <table>
      <thead>
        <tr>
          <th>Имя</th>
          <th>Тип</th>
          <th>Размер</th>
          <th>SHA-256</th>
        </tr>
      </thead>
      <tbody>
        ${filesRows}
      </tbody>
    </table>

    <h2>Маркеры</h2>
    <table>
      <thead>
        <tr>
          <th>Таймкод</th>
          <th>Тип</th>
          <th>Комментарий</th>
          <th>Добавлен</th>
        </tr>
      </thead>
      <tbody>
        ${markersRows}
      </tbody>
    </table>

    <h2>Журнал действий</h2>
    <table>
      <thead>
        <tr>
          <th>Время</th>
          <th>Действие</th>
          <th>Описание</th>
          <th>Оператор</th>
        </tr>
      </thead>
      <tbody>
        ${logsRows}
      </tbody>
    </table>

    <div class="actions">
      <button type="button" onclick="window.print()">Печать / PDF</button>
    </div>
  </body>
</html>`;
    };

    actions.loadCaseLibrary();
    actions.renderPipelineJobs();

    elements.logEntryButton.addEventListener("click", () => {
      actions.recordLog(
        "manual-entry",
        `Просмотр файла: ${elements.video.dataset.filename || "не выбран"}`
      );
    });

    elements.exportLogButton.addEventListener("click", () => {
      const payload = {
        caseId: elements.caseId.value.trim(),
        owner: elements.caseOwner.value.trim(),
        status: elements.caseStatus.value,
        tags: elements.caseTags.value
          .split(",")
          .map((tag) => tag.trim())
          .filter(Boolean),
        summary: elements.caseSummary.value.trim(),
        createdAt: new Date().toISOString(),
        entries: state.logEntries,
      };
      actions.downloadJson(payload, "forensic-log");
      actions.recordLog("export-log", "Экспорт журнала действий", {
        entries: state.logEntries.length,
      });
    });

    elements.exportReportButton.addEventListener("click", () => {
      const payload = buildReportPayload();
      actions.downloadJson(payload, "forensic-report");
      actions.recordLog("export-report", "Экспорт отчёта по делу", {
        files: state.importedFiles.length,
        markers: state.markers.length,
        entries: state.logEntries.length,
      });
    });

    elements.previewReportButton.addEventListener("click", () => {
      const payload = buildReportPayload();
      const reportWindow = window.open("", "_blank", "noopener,noreferrer");
      if (!reportWindow) {
        actions.recordLog(
          "report-preview-failed",
          "Браузер заблокировал открытие окна отчёта"
        );
        return;
      }
      reportWindow.document.open();
      reportWindow.document.write(buildReportHtml(payload));
      reportWindow.document.close();
      reportWindow.focus();
      actions.recordLog("report-preview", "Предпросмотр отчёта по делу", {
        files: state.importedFiles.length,
        markers: state.markers.length,
        entries: state.logEntries.length,
      });
    });

    elements.exportFfmpegJobButton.addEventListener("click", () => {
      const jobDraft = actions.buildFfmpegJobDraft("3.1.2");
      if (elements.ffmpegJobPreview) {
        elements.ffmpegJobPreview.value = JSON.stringify(jobDraft, null, 2);
      }
      actions.recordLog(
        "ffmpeg-job-draft",
        "Собран FFmpeg Job draft из текущих параметров обработки",
        {
          stage: "3.1.2",
          hasSource: Boolean(jobDraft.source),
          clipRange: {
            in: jobDraft.playback.clipIn,
            out: jobDraft.playback.clipOut,
          },
          readyFor: ["3.1.3"],
        }
      );
    });

    elements.downloadFfmpegJobButton.addEventListener("click", () => {
      const jobPayload = actions.buildFfmpegJobDraft("3.1.3");
      if (elements.ffmpegJobPreview) {
        elements.ffmpegJobPreview.value = JSON.stringify(jobPayload, null, 2);
      }
      const caseId = elements.caseId.value.trim() || "без-id";
      const safeCaseId = caseId.replace(/[^\p{L}\p{N}_-]+/gu, "-").toLowerCase();
      actions.downloadJson(jobPayload, `ffmpeg-job-${safeCaseId}`);
      actions.recordLog(
        "ffmpeg-job-export",
        "Экспортирован FFmpeg Job JSON для интеграционного pipeline",
        {
          stage: "3.1.3",
          schema: jobPayload.schema,
          hasSource: Boolean(jobPayload.source),
          clipRange: {
            in: jobPayload.playback.clipIn,
            out: jobPayload.playback.clipOut,
          },
        }
      );
    });

    elements.queueFfmpegJobButton.addEventListener("click", () => {
      const jobPayload = actions.buildFfmpegJobDraft("3.3.2");
      if (elements.ffmpegJobPreview) {
        elements.ffmpegJobPreview.value = JSON.stringify(jobPayload, null, 2);
      }
      const queued = actions.enqueuePipelineJob(jobPayload, "3.3.2");
      actions.recordLog("pipeline-queue-ui", "Job добавлен в очередь pipeline", {
        stage: "3.3.2",
        jobId: queued.id,
        hasSource: Boolean(jobPayload.source),
      });
    });

    elements.pipelinePauseButton?.addEventListener("click", () => {
      actions.pausePipelineQueue();
    });

    elements.pipelineResumeButton?.addEventListener("click", () => {
      actions.resumePipelineQueue();
    });

    elements.pipelineRetryFailedButton?.addEventListener("click", () => {
      actions.retryFailedPipelineJobs();
    });

    elements.pipelineClearTerminalButton?.addEventListener("click", () => {
      actions.clearTerminalPipelineJobs();
    });

    elements.addMarkerButton.addEventListener("click", () => {
      if (elements.video.readyState < 2) return;
      const timestamp = new Date().toISOString();
      const time = elements.video.currentTime;
      const type = elements.markerType.value;
      const noteInput = elements.markerNote.value.trim();
      const note = noteInput || `Маркер на ${actions.formatTime(time)}`;
      const entry = {
        timestamp,
        time,
        timecode: actions.formatTime(time),
        type,
        note,
      };
      state.markers.unshift(entry);
      actions.appendMarkerEntry(entry);
      actions.recordLog("marker-add", `Добавлен маркер на ${entry.timecode}`, {
        time,
        type,
        note,
      });
      elements.markerNote.value = "";
      if (actions.refreshTimeline) {
        actions.refreshTimeline();
      }
    });

    elements.exportMarkersButton.addEventListener("click", () => {
      const payload = {
        caseId: elements.caseId.value.trim(),
        createdAt: new Date().toISOString(),
        markers: state.markers,
      };
      actions.downloadJson(payload, "forensic-markers");
      actions.recordLog("export-markers", "Экспорт маркеров", {
        markers: state.markers.length,
      });
    });

    elements.caseSaveButton.addEventListener("click", () => {
      actions.saveCurrentCase();
    });

    elements.caseLoadButton.addEventListener("click", () => {
      actions.loadCaseFromLibrary();
    });

    elements.caseDeleteButton.addEventListener("click", () => {
      actions.deleteCaseFromLibrary();
    });

    elements.caseExportLibraryButton.addEventListener("click", () => {
      const payload = {
        exportedAt: new Date().toISOString(),
        cases: state.caseLibrary,
      };
      actions.downloadJson(payload, "forensic-case-library");
      actions.recordLog(
        "case-library-export",
        "Экспорт локальной библиотеки дел",
        { cases: state.caseLibrary.length }
      );
    });

    elements.caseImportLibraryButton.addEventListener("click", () => {
      elements.caseImportInput.click();
    });

    elements.caseImportInput.addEventListener("change", async (event) => {
      const [file] = event.target.files || [];
      if (!file) return;
      try {
        const text = await file.text();
        const payload = JSON.parse(text);
        const cases = Array.isArray(payload)
          ? payload
          : Array.isArray(payload?.cases)
            ? payload.cases
            : null;
        if (!cases) {
          actions.recordLog(
            "case-library-import-failed",
            "Некорректный формат импорта библиотеки"
          );
          return;
        }
        const existing = new Map(
          state.caseLibrary.map((caseItem) => [caseItem.id, caseItem])
        );
        cases.forEach((caseItem) => {
          if (!caseItem?.id) return;
          existing.set(caseItem.id, caseItem);
        });
        state.caseLibrary = Array.from(existing.values());
        actions.saveCaseLibrary();
        actions.loadCaseLibrary();
        actions.recordLog(
          "case-library-import",
          "Импорт библиотеки дел выполнен",
          { cases: cases.length }
        );
      } catch (error) {
        actions.recordLog(
          "case-library-import-failed",
          "Ошибка импорта библиотеки дел",
          { message: error?.message }
        );
      } finally {
        elements.caseImportInput.value = "";
      }
    });

    elements.caseSearch.addEventListener("input", (event) => {
      actions.refreshCaseLibraryOptions(event.target.value);
    });

    elements.caseClearSearchButton.addEventListener("click", () => {
      elements.caseSearch.value = "";
      actions.refreshCaseLibraryOptions("");
    });
  },
});
