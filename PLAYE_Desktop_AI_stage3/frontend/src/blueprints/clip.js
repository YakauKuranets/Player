export const createClipBlueprint = () => ({
  name: "clip",
  init: ({ elements, state, actions }) => {
    elements.markInButton.addEventListener("click", () => {
      state.clipIn = elements.video.currentTime;
      elements.clipInValue.textContent = actions.formatTime(state.clipIn);
      actions.recordLog("clip-in", `Установлен IN: ${actions.formatTime(state.clipIn)}`, {
        time: state.clipIn,
      });
    });

    elements.markOutButton.addEventListener("click", () => {
      state.clipOut = elements.video.currentTime;
      elements.clipOutValue.textContent = actions.formatTime(state.clipOut);
      actions.recordLog("clip-out", `Установлен OUT: ${actions.formatTime(state.clipOut)}`, {
        time: state.clipOut,
      });
    });

    elements.exportClipButton.addEventListener("click", async () => {
      if (state.clipIn === null || state.clipOut === null || state.clipOut <= state.clipIn) {
        alert("Сначала задайте корректные IN и OUT.");
        return;
      }
      if (elements.video.readyState < 2) return;

      const stream = elements.video.captureStream();
      const recorder = new MediaRecorder(stream, {
        mimeType: "video/webm;codecs=vp9",
      });
      const chunks = [];

      recorder.addEventListener("dataavailable", (event) => {
        if (event.data.size) {
          chunks.push(event.data);
        }
      });

      recorder.addEventListener("stop", () => {
        const blob = new Blob(chunks, { type: "video/webm" });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = `clip-${Date.now()}.webm`;
        link.click();
        actions.recordLog(
          "clip-export",
          `Экспорт нарезки ${actions.formatTime(state.clipIn)} → ${actions.formatTime(
            state.clipOut
          )}`,
          {
            in: state.clipIn,
            out: state.clipOut,
          }
        );
      });

      elements.video.currentTime = state.clipIn;
      await elements.video.play();
      recorder.start();

      const stopAt = () => {
        if (elements.video.currentTime >= state.clipOut) {
          recorder.stop();
          elements.video.pause();
          elements.video.removeEventListener("timeupdate", stopAt);
        }
      };

      elements.video.addEventListener("timeupdate", stopAt);
    });
  },
});
