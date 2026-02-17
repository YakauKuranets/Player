# PLAYE PhotoLab Desktop (AI)

Desktop‑версия лаборатории (Electron + FastAPI backend).

## 1) Требования

**Windows (рекомендуется):**

- Node.js **18+**
- Python **3.10+** (лучше 3.10/3.11)

## 2) Установка

```powershell
cd <папка_проекта>
npm install
```

> `postinstall` проверяет наличие Python. Авто‑установка Python‑зависимостей выключена по умолчанию.

### 2.1 Авто‑подготовка backend (опционально)

Если хочешь, чтобы venv и зависимости поставились автоматически при `npm install`:

```powershell
$env:PLAYE_SETUP_BACKEND="1"
npm install
```

### 2.2 Ручная подготовка backend (Windows)

```powershell
cd backend
py -3.10 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
\.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
```

Важно:

- Команды вида `source venv/bin/activate` — это **Linux/macOS**. В PowerShell активируется через `Activate.ps1`.
- `torch/torchvision` очень большие. Если словишь `No space left on device` — освободи место на диске.

## 3) Запуск

```powershell
npm run dev
```

Electron сам поднимет Python backend (`backend/server.py`).

Проверка, что backend жив:

- В приложении есть проверка статуса.
- Или вручную открыть в браузере: `http://127.0.0.1:8000/health`

## 4) Модели (веса)

Веса моделей **не хранятся в репозитории**.

- Локальный манифест: `models-data/manifest.json`
- Папка для скачанных моделей (runtime): `userData/models` (кнопка **«Открыть папку»** в UI)

Чтобы включить скачивание:

1) В `models-data/manifest.json` пропиши реальные `url` (и желательно `checksum` sha256).
2) В UI нажми **«Проверить»** → **«Скачать/обновить»**.

## 5) Полезные команды

```bash
node scripts/check-updates.js
node scripts/update-models.js
node scripts/download-models.js --required
npm run ci:smoke
npm run test:api-client
npm run test:job-params
npm run test:task-status
npm run test:cancel-route
```
