const createMarkerElement = (marker, onSelect) => {
  const element = document.createElement("button");
  element.type = "button";
  element.className = "timeline-marker";
  element.title = marker.note;
  element.addEventListener("click", (event) => {
    event.stopPropagation();
    onSelect(marker);
  });
  return element;
};

export const createTimelineBlueprint = () => ({
  name: "timeline",
  init: ({ elements, state, actions }) => {
    const updateTimelineWindow = () => {
      const duration = elements.video.duration || 0;
      const zoom = state.timelineZoom;
      const windowSize = duration ? Math.max(1, duration / zoom) : 0;
      const currentTime = elements.video.currentTime || 0;
      let start = 0;
      if (duration && windowSize < duration) {
        start = Math.max(0, Math.min(duration - windowSize, currentTime - windowSize / 2));
      }
      const end = duration ? start + windowSize : 0;
      state.timelineWindow = { start, end, duration };
      elements.timeline.min = start;
      elements.timeline.max = end;
      elements.timeline.step = duration ? Math.max(0.01, windowSize / 500) : 0.01;
      elements.timeline.value = currentTime;
      elements.timelineCurrent.textContent = actions.formatTime(currentTime);
      elements.timelineDuration.textContent = actions.formatTime(duration);
      elements.timelineZoomValue.textContent = `${zoom.toFixed(1)}x`;
      renderMarkers();
    };

    actions.refreshTimeline = updateTimelineWindow;

    const renderMarkers = () => {
      elements.timelineMarkers.innerHTML = "";
      const { start, end } = state.timelineWindow;
      const windowSize = end - start;
      if (!windowSize) return;
      state.markers.forEach((marker) => {
        const element = createMarkerElement(marker, (selected) => {
          elements.video.currentTime = selected.time;
          updateTimelineWindow();
          actions.recordLog("timeline-marker", `Переход к маркеру ${selected.timecode}`, {
            time: selected.time,
          });
        });
        const position = ((marker.time - start) / windowSize) * 100;
        element.style.left = `${Math.max(0, Math.min(100, position))}%`;
        elements.timelineMarkers.appendChild(element);
      });
    };

    elements.timeline.addEventListener("input", () => {
      const nextTime = Number.parseFloat(elements.timeline.value);
      if (!Number.isNaN(nextTime)) {
        elements.video.currentTime = nextTime;
        updateTimelineWindow();
      }
    });

    elements.timeline.addEventListener("change", () => {
      actions.recordLog(
        "timeline-seek",
        `Переход на ${actions.formatTime(elements.video.currentTime)}`,
        { time: elements.video.currentTime }
      );
    });

    elements.timelineZoomIn.addEventListener("click", () => {
      state.timelineZoom = Math.min(10, state.timelineZoom + 0.5);
      updateTimelineWindow();
      actions.recordLog("timeline-zoom", `Увеличение масштаба таймлайна: ${state.timelineZoom.toFixed(1)}x`);
    });

    elements.timelineZoomOut.addEventListener("click", () => {
      state.timelineZoom = Math.max(1, state.timelineZoom - 0.5);
      updateTimelineWindow();
      actions.recordLog("timeline-zoom", `Уменьшение масштаба таймлайна: ${state.timelineZoom.toFixed(1)}x`);
    });

    elements.video.addEventListener("loadedmetadata", updateTimelineWindow);
    elements.video.addEventListener("timeupdate", updateTimelineWindow);

    updateTimelineWindow();
  },
});
