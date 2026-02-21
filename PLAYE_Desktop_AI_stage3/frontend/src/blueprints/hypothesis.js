const HYPOTHESIS_TEMPLATES = [
  { id: "restore", label: "Face Restore", note: "Артефакты лица / low-light" },
  { id: "superres", label: "Super Resolution", note: "Мелкие детали и номерные зоны" },
  { id: "denoise", label: "Denoise", note: "Шум матрицы / компрессия" },
  { id: "deblur", label: "Deblur", note: "Смаз движения / фокуса" },
  { id: "detect", label: "Detect", note: "Лица / объекты / сцены" },
  { id: "reconstruct3d", label: "3D Reconstruct", note: "Перспектива и глубина сцены" },
];

export const createHypothesisBlueprint = () => ({
  name: "hypothesis",
  init: ({ elements, state, actions }) => {
    if (!elements.hypothesisList || !elements.hypothesisGenerateButton) return;

    const render = () => {
      elements.hypothesisList.innerHTML = "";
      const data = state.hypothesisClips || [];
      if (!data.length) {
        const item = document.createElement("li");
        item.textContent = "Гипотезы ещё не сформированы.";
        elements.hypothesisList.appendChild(item);
        return;
      }
      data.forEach((entry) => {
        const item = document.createElement("li");
        item.textContent = `${entry.timestamp} | ${entry.type} | ${entry.note}`;
        elements.hypothesisList.appendChild(item);
      });
    };

    const selectedTypes = () =>
      Array.from(document.querySelectorAll(".hypothesis-type:checked")).map((el) => el.value);

    elements.hypothesisGenerateButton.addEventListener("click", () => {
      const types = selectedTypes();
      if (!types.length) {
        elements.hypothesisStatus.textContent = "Выберите минимум одну гипотезу.";
        return;
      }

      state.hypothesisClips = types.map((type) => {
        const template = HYPOTHESIS_TEMPLATES.find((entry) => entry.id === type);
        return {
          id: crypto.randomUUID?.() || `hyp-${Date.now()}-${type}`,
          timestamp: new Date().toISOString(),
          type,
          label: template?.label || type,
          note: template?.note || "Пользовательская гипотеза",
        };
      });

      elements.hypothesisStatus.textContent = `Сформировано гипотез: ${state.hypothesisClips.length}`;
      actions.recordLog("hypothesis-generate", "Сформирован список гипотез для клипов", {
        types,
      });
      render();
    });

    elements.hypothesisExportButton?.addEventListener("click", () => {
      const payload = {
        generatedAt: new Date().toISOString(),
        clips: state.hypothesisClips || [],
      };
      actions.downloadJson(payload, "hypothesis-clips");
      actions.recordLog("hypothesis-export", "Экспортирован список гипотез");
    });

    render();
  },
});
