const drawComparison = async (canvas, leftFile, rightFile, split) => {
  const [leftBitmap, rightBitmap] = await Promise.all([
    createImageBitmap(leftFile),
    createImageBitmap(rightFile),
  ]);

  const width = Math.min(leftBitmap.width, rightBitmap.width);
  const height = Math.min(leftBitmap.height, rightBitmap.height);

  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d");
  const splitX = Math.floor(width * split);

  ctx.clearRect(0, 0, width, height);
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, splitX, height);
  ctx.clip();
  ctx.drawImage(leftBitmap, 0, 0, width, height);
  ctx.restore();

  ctx.save();
  ctx.beginPath();
  ctx.rect(splitX, 0, width - splitX, height);
  ctx.clip();
  ctx.drawImage(rightBitmap, 0, 0, width, height);
  ctx.restore();

  ctx.strokeStyle = "#22d3ee";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(splitX, 0);
  ctx.lineTo(splitX, height);
  ctx.stroke();
};

export const createCompareBlueprint = () => ({
  name: "compare",
  init: ({ elements, actions }) => {
    if (!elements.compareLeftInput || !elements.compareRightInput || !elements.compareCanvas) return;

    const redraw = async () => {
      const leftFile = elements.compareLeftInput.files?.[0];
      const rightFile = elements.compareRightInput.files?.[0];
      if (!leftFile || !rightFile) return;

      const split = Number.parseFloat(elements.compareSplitInput.value || "50") / 100;
      await drawComparison(elements.compareCanvas, leftFile, rightFile, split);
      if (elements.compareSplitValue) {
        elements.compareSplitValue.textContent = `${Math.round(split * 100)}%`;
      }
    };

    const render = () => {
      redraw().catch(() => {
        if (elements.compareStatus) {
          elements.compareStatus.textContent = "Не удалось отрисовать сравнение.";
        }
      });
    };

    elements.compareRenderButton.addEventListener("click", () => {
      render();
      actions.recordLog("compare-render", "Обновлено side-by-side сравнение кадров");
      if (elements.compareStatus) {
        elements.compareStatus.textContent = "Сравнение обновлено.";
      }
    });

    elements.compareSplitInput?.addEventListener("input", render);
  },
});
