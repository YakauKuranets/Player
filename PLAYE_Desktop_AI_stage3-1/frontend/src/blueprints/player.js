export const createPlayerBlueprint = () => ({
  name: "player",
  init: ({ elements, state, actions }) => {
    elements.speedInput.addEventListener("input", actions.updateSpeed);
    elements.speedInput.addEventListener("change", () => {
      actions.recordLog(
        "speed-change",
        `Скорость воспроизведения: ${Number.parseFloat(elements.speedInput.value).toFixed(2)}x`
      );
    });

    elements.viewerSurface.addEventListener("wheel", (event) => {
      event.preventDefault();
      const delta = Math.sign(event.deltaY) * -0.1;
      state.zoomLevel = Math.min(5, Math.max(0.2, state.zoomLevel + delta));
      actions.updateZoom();
    });

    elements.viewerSurface.addEventListener("dblclick", () => {
      actions.resetZoom();
      actions.recordLog("zoom-reset", "Сброс масштабирования");
    });

    elements.frameBack.addEventListener("click", () => {
      if (elements.video.readyState >= 2) {
        elements.video.pause();
        elements.video.currentTime = Math.max(
          0,
          elements.video.currentTime - 1 / 30
        );
        actions.recordLog("frame-step", "Кадр назад", {
          time: elements.video.currentTime,
        });
      }
    });

    elements.frameForward.addEventListener("click", () => {
      if (elements.video.readyState >= 2) {
        elements.video.pause();
        elements.video.currentTime = Math.min(
          elements.video.duration,
          elements.video.currentTime + 1 / 30
        );
        actions.recordLog("frame-step", "Кадр вперед", {
          time: elements.video.currentTime,
        });
      }
    });

    actions.updateSpeed();
  },
});
