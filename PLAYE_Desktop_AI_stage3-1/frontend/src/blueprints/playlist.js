export const createPlaylistBlueprint = () => ({
  name: "playlist",
  init: ({ elements, state, actions }) => {
    const setVideoSource = (file) => {
      const url = URL.createObjectURL(file);
      elements.video.src = url;
      elements.video.dataset.filename = file.name;
      elements.video.load();
      actions.resetZoom();
      state.clipIn = null;
      state.clipOut = null;
      elements.clipInValue.textContent = actions.formatTime(state.clipIn);
      elements.clipOutValue.textContent = actions.formatTime(state.clipOut);
      actions.recordLog("video-select", `Выбран файл: ${file.name}`, {
        name: file.name,
        size: file.size,
        hash: file.hash || "—",
      });
    };

    const addToPlaylist = async (file) => {
      const item = document.createElement("li");
      item.textContent = file.name;
      const hash = await actions.hashFile(file);
      file.hash = hash;
      state.importedFiles.push({
        name: file.name,
        size: file.size,
        type: file.type,
        hash,
      });
      actions.recordLog("file-hash", `Хэш SHA-256 рассчитан для ${file.name}`, {
        name: file.name,
        hash,
      });
      item.addEventListener("click", () => {
        document.querySelectorAll(".playlist li").forEach((node) => {
          node.classList.remove("active");
        });
        item.classList.add("active");
        setVideoSource(file);
      });
      elements.playlist.appendChild(item);
      if (!elements.video.src) {
        item.click();
      }
    };

    elements.fileInput.addEventListener("change", (event) => {
      const files = Array.from(event.target.files || []);
      if (files.length) {
        actions.recordLog("video-import", `Импортировано файлов: ${files.length}`, {
          files: files.map((file) => ({
            name: file.name,
            size: file.size,
            type: file.type,
          })),
        });
      }
      files.forEach((file) => {
        addToPlaylist(file);
      });
      elements.fileInput.value = "";
    });
  },
});
