// Preload script to expose limited APIs to the renderer process.
// This file acts as a bridge between Electron's main process and the
// browser (renderer) context. It exposes a safe set of functions
// through the `electronAPI` namespace using contextBridge.

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  /**
   * Check whether the Python backend is reachable. Returns a promise
   * resolving to a boolean.
   */
  checkPythonBackend: () => ipcRenderer.invoke('check-python-backend'),

  /**
   * Retrieve the path to the models directory used by the backend.
   */
  getModelsPath: () => ipcRenderer.invoke('get-models-path'),

  /**
   * Trigger a check for model updates. Returns a promise with update info.
   */
  checkModelUpdates: () => ipcRenderer.invoke('check-model-updates'),

  /**
   * Download all missing/new models based on checkModelUpdates().
   */
  updateModels: () => ipcRenderer.invoke('update-models'),

  /**
   * Subscribe to model download progress events.
   * Returns an unsubscribe function.
   */
  onDownloadProgress: (callback) => {
    const handler = (_event, data) => callback(data);
    ipcRenderer.on('download-progress', handler);
    return () => ipcRenderer.removeListener('download-progress', handler);
  },

  /**
   * Download a specific model by name. Accepts a callback to report
   * progress and returns a promise that resolves when the download
   * completes.
   *
   * @param {string} modelName - Name of the model to download
   * @param {function} onProgress - Callback with progress (0-100)
   */
  downloadModel: (modelName, onProgress) => {
    return new Promise((resolve, reject) => {
      const handler = (event, data) => {
        if (data.modelName === modelName) {
          onProgress(data.progress);
          if (data.progress >= 100) {
            ipcRenderer.removeListener('download-progress', handler);
          }
        }
      };
      ipcRenderer.on('download-progress', handler);

      ipcRenderer.invoke('download-model', modelName)
        .then((result) => {
          ipcRenderer.removeListener('download-progress', handler);
          resolve(result);
        })
        .catch((err) => {
          ipcRenderer.removeListener('download-progress', handler);
          reject(err);
        });
    });
  },

  /**
   * Open local models folder (userData/models) in the OS file explorer.
   */
  openModelsFolder: () => ipcRenderer.invoke('open-models-folder'),

  /**
   * Show a native dialog with given options. Useful for error/info boxes.
   */
  showDialog: (options) => ipcRenderer.invoke('show-dialog', options),

  /**
   * Wrapper for face enhancement AI call. Accepts a Blob or File and
   * returns JSON result from the backend.
   *
   * @param {Blob|File} imageData - Image data to process
   */
  enhanceFace: async (imageData) => {
    const formData = new FormData();
    formData.append('file', imageData);
    const response = await fetch('http://127.0.0.1:8000/ai/face-enhance', {
      method: 'POST',
      body: formData
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    // The backend returns a PNG stream. Convert it to a Blob so the
    // renderer can display or save it.
    return await response.blob();
  },

  /**
   * Wrapper for image upscaling. Accepts an image and scale factor.
   *
   * @param {Blob|File} imageData - Image data to process
   * @param {number} factor - Upscale factor
   */
  upscaleImage: async (imageData, factor) => {
    const formData = new FormData();
    formData.append('file', imageData);
    formData.append('factor', factor.toString());
    const response = await fetch('http://127.0.0.1:8000/ai/upscale', {
      method: 'POST',
      body: formData
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.blob();
  },

  /**
   * Wrapper for image denoising. Accepts an image and noise level.
   *
   * @param {Blob|File} imageData - Image data to process
   * @param {number} level - Denoise level
   */
  denoiseImage: async (imageData, level) => {
    const formData = new FormData();
    formData.append('file', imageData);
    formData.append('level', level);
    const response = await fetch('http://127.0.0.1:8000/ai/denoise', {
      method: 'POST',
      body: formData
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.blob();
  }
});