// Main process for PLAYE PhotoLab desktop
// This file bootstraps the Electron application and manages the lifecycle of
// the embedded Python backend. It is adapted from the development
// instructions provided in Инструкция_Desktop_AI.md.

const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const { spawn, spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const axios = require('axios');

let mainWindow;
let pythonProcess;

// Determine paths based on dev/prod mode.
const isDev = process.argv.includes('--dev');
const backendPath = isDev
  ? path.join(__dirname, 'backend')
  : path.join(process.resourcesPath, 'backend');

function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

function detectPythonForRun() {
  // Prefer backend venv if exists.
  const isWin = process.platform === 'win32';
  const venvPython = isWin
    ? path.join(backendPath, '.venv', 'Scripts', 'python.exe')
    : path.join(backendPath, '.venv', 'bin', 'python');
  if (fs.existsSync(venvPython)) {
    return { cmd: venvPython, argsPrefix: [] };
  }

  if (isWin) {
    const py = spawnSync('py', ['-3.10', '--version'], { stdio: 'ignore' });
    if (py.status === 0) return { cmd: 'py', argsPrefix: ['-3.10'] };
    return { cmd: 'python', argsPrefix: [] };
  }

  return { cmd: 'python3', argsPrefix: [] };
}

/**
 * Spawn the Python backend using the appropriate interpreter. Resolves
 * when the backend prints the Uvicorn startup message. Rejects on
 * failure or after a timeout.
 */
function startPythonBackend() {
  return new Promise((resolve, reject) => {
    console.log('[Main] Starting Python backend...');

    const py = detectPythonForRun();
    const serverScript = path.join(backendPath, 'server.py');

    const modelsDir = path.join(app.getPath('userData'), 'models');
    ensureDir(modelsDir);

    pythonProcess = spawn(py.cmd, [...py.argsPrefix, serverScript], {
      cwd: backendPath,
      env: { ...process.env, PYTHONUNBUFFERED: '1', PLAYE_MODELS_DIR: modelsDir }
    });

    let timeoutHandle = null;

    pythonProcess.stdout.on('data', (data) => {
      const message = data.toString();
      console.log(`[Python] ${message}`);
      // When Uvicorn announces startup, resolve the promise.
      if (message.includes('Uvicorn running')) {
        console.log('[Main] Python backend started successfully');
        if (timeoutHandle) clearTimeout(timeoutHandle);
        resolve();
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`[Python Error] ${data}`);
    });

    pythonProcess.on('error', (error) => {
      console.error('[Main] Failed to start Python:', error);
      reject(error);
    });

    pythonProcess.on('close', (code) => {
      console.log(`[Main] Python process exited with code ${code}`);
    });

    // Guard against hanging startup by timing out after 15 seconds.
    timeoutHandle = setTimeout(() => {
      reject(new Error('Python backend timeout'));
    }, 15000);
  });
}

/**
 * Instantiate the main window. Loads the UI from the frontend directory.
 */
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 800,
    backgroundColor: '#1a1d24',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      sandbox: false
    },
    icon: path.join(__dirname, 'assets/icon.png')
  });

  // Load the frontend index.html file from the packaged UI.
  mainWindow.loadFile(path.join(__dirname, 'frontend/index.html'));

  if (isDev) {
    // Automatically open DevTools in development mode.
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// IPC Handlers for communication between renderer and main processes.
ipcMain.handle('check-python-backend', async () => {
  try {
    const res = await axios.get('http://127.0.0.1:8000/health', { timeout: 1500 });
    return res.status === 200;
  } catch (error) {
    return false;
  }
});

ipcMain.handle('get-models-path', () => {
  return path.join(app.getPath('userData'), 'models');
});

ipcMain.handle('check-model-updates', async () => {
  const { checkModelUpdates } = require('./scripts/check-updates.js');
  const modelsDir = path.join(app.getPath('userData'), 'models');
  ensureDir(modelsDir);
  return await checkModelUpdates({ modelsDir });
});

ipcMain.handle('download-model', async (event, modelName) => {
  const { downloadModel } = require('./scripts/download-models.js');
  const modelsDir = path.join(app.getPath('userData'), 'models');
  ensureDir(modelsDir);
  return await downloadModel(modelName, (progress) => {
    event.sender.send('download-progress', { modelName, progress });
  }, { modelsDir });
});

ipcMain.handle('update-models', async (event) => {
  const { updateModels } = require('./scripts/update-models.js');
  const modelsDir = path.join(app.getPath('userData'), 'models');
  ensureDir(modelsDir);
  return await updateModels({
    modelsDir,
    onModelProgress: ({ modelKey, progress }) => {
      event.sender.send('download-progress', { modelName: modelKey, progress });
    },
  });
});

ipcMain.handle('open-models-folder', async () => {
  const modelsDir = path.join(app.getPath('userData'), 'models');
  ensureDir(modelsDir);
  await shell.openPath(modelsDir);
  return true;
});

ipcMain.handle('show-dialog', async (event, options) => {
  return await dialog.showMessageBox(mainWindow, options);
});

// Application lifecycle events.
app.whenReady().then(async () => {
  console.log('[Main] App ready, starting...');
  try {
    await startPythonBackend();
    createWindow();
  } catch (error) {
    console.error('[Main] Startup error:', error);
    dialog.showErrorBox(
      'Ошибка запуска',
      'Не удалось запустить Python backend. Убедитесь, что Python установлен.'
    );
    app.quit();
  }
});

// Gracefully stop Python process when quitting.
app.on('will-quit', () => {
  if (pythonProcess) {
    console.log('[Main] Stopping Python backend...');
    pythonProcess.kill();
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});