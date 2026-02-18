// Preload script to expose limited APIs to the renderer process.
const { contextBridge, ipcRenderer } = require('electron');

const API_BASE = 'http://127.0.0.1:8000/api';

function getAuthHeader() {
  const token = (globalThis.window && window._playeToken) ? window._playeToken : '';
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function blobToBase64(imageData) {
  const arrayBuf = await imageData.arrayBuffer();
  const bytes = new Uint8Array(arrayBuf);
  let binary = '';
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function decodeBase64ToBlob(base64Payload, mimeType = 'image/png') {
  const binary = atob(base64Payload);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: mimeType });
}


async function submitVideoAndPoll(videoFile, { operation = 'temporal_denoise', fps = 1.0, scene_threshold = 28.0, temporal_window = 3 } = {}) {
  const formData = new FormData();
  formData.append('file', videoFile);
  formData.append('operation', operation);
  formData.append('fps', String(fps));
  formData.append('scene_threshold', String(scene_threshold));
  formData.append('temporal_window', String(temporal_window));

  const submitResp = await fetch(`${API_BASE}/job/video/submit`, {
    method: 'POST',
    headers: {
      ...getAuthHeader(),
    },
    body: formData,
  });

  if (!submitResp.ok) throw new Error(`Video submit failed: ${submitResp.status}`);
  const submitData = await submitResp.json();
  const taskId = submitData?.result?.task_id;
  if (!taskId) throw new Error('No task_id returned for video job');

  for (let i = 0; i < 180; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    const statusResp = await fetch(`${API_BASE}/job/${taskId}/status`, { headers: { ...getAuthHeader() } });
    if (!statusResp.ok) throw new Error(`Video status check failed: ${statusResp.status}`);
    const statusData = await statusResp.json();
    const res = statusData?.result || {};
    if (res.is_final && res.error) throw new Error(String(res.error));
    if (res.is_final && !res.error) return res.result;
  }

  throw new Error('Timeout waiting for video job result');
}

async function submitAndPoll(operation, imageData, params = {}) {
  const image_base64 = await blobToBase64(imageData);

  const submitResp = await fetch(`${API_BASE}/job/submit`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeader(),
    },
    body: JSON.stringify({ operation, image_base64, params }),
  });

  if (!submitResp.ok) {
    throw new Error(`Submit failed: ${submitResp.status}`);
  }

  const submitData = await submitResp.json();
  const taskId = submitData?.result?.task_id;
  if (!taskId) {
    throw new Error('No task_id returned');
  }

  for (let i = 0; i < 60; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 800));
    const statusResp = await fetch(`${API_BASE}/job/${taskId}/status`, {
      headers: {
        ...getAuthHeader(),
      },
    });

    if (!statusResp.ok) {
      throw new Error(`Status check failed: ${statusResp.status}`);
    }

    const statusData = await statusResp.json();
    const res = statusData?.result || {};

    if (res.is_final && res.error) {
      throw new Error(String(res.error));
    }

    if (res.is_final && !res.error) {
      const payload = res.result;
      if (typeof payload === 'string') {
        return decodeBase64ToBlob(payload);
      }
      if (payload && typeof payload.image_base64 === 'string') {
        return decodeBase64ToBlob(payload.image_base64);
      }
      throw new Error('Unexpected final result payload');
    }
  }

  throw new Error('Timeout waiting for job result');
}

contextBridge.exposeInMainWorld('electronAPI', {
  checkPythonBackend: () => ipcRenderer.invoke('check-python-backend'),
  getModelsPath: () => ipcRenderer.invoke('get-models-path'),
  checkModelUpdates: () => ipcRenderer.invoke('check-model-updates'),
  updateModels: () => ipcRenderer.invoke('update-models'),

  onDownloadProgress: (callback) => {
    const handler = (_event, data) => callback(data);
    ipcRenderer.on('download-progress', handler);
    return () => ipcRenderer.removeListener('download-progress', handler);
  },

  downloadModel: (modelName, onProgress) => {
    return new Promise((resolve, reject) => {
      const handler = (_event, data) => {
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

  openModelsFolder: () => ipcRenderer.invoke('open-models-folder'),
  showDialog: (options) => ipcRenderer.invoke('show-dialog', options),

  enhanceFace: (imageData) => submitAndPoll('face_enhance', imageData),
  upscaleImage: (imageData, factor) => submitAndPoll('upscale', imageData, { factor }),
  denoiseImage: (imageData, level) => submitAndPoll('denoise', imageData, { level }),
  temporalDenoiseVideo: (videoFile, fps = 1.0) => submitVideoAndPoll(videoFile, { operation: 'temporal_denoise', fps }),
  detectVideoScenes: (videoFile, scene_threshold = 28.0, temporal_window = 3) => submitVideoAndPoll(videoFile, { operation: 'scene_detect', scene_threshold, temporal_window }),
});
