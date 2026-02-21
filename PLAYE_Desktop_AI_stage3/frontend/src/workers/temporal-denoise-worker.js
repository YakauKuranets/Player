importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

let _session = null;
let _modelUrl = null;

async function getSession(modelUrl) {
  if (_session && _modelUrl === modelUrl) return _session;
  self.postMessage({ type: 'status', message: 'Загрузка ONNX модели...' });
  _session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  _modelUrl = modelUrl;
  self.postMessage({ type: 'status', message: 'Модель загружена' });
  return _session;
}

function frameToTensor(imageData, width, height) {
  const data = imageData;
  const r = new Float32Array(width * height);
  const g = new Float32Array(width * height);
  const b = new Float32Array(width * height);
  for (let i = 0; i < width * height; i += 1) {
    r[i] = data[i * 4 + 0] / 255.0;
    g[i] = data[i * 4 + 1] / 255.0;
    b[i] = data[i * 4 + 2] / 255.0;
  }
  const tensor = new Float32Array(3 * height * width);
  tensor.set(r, 0);
  tensor.set(g, height * width);
  tensor.set(b, 2 * height * width);
  return new ort.Tensor('float32', tensor, [1, 3, height, width]);
}

function tensorToFrame(tensor, width, height) {
  const data = tensor.data;
  const output = new Uint8ClampedArray(width * height * 4);
  const pixels = width * height;
  for (let i = 0; i < pixels; i += 1) {
    output[i * 4 + 0] = Math.round(Math.min(1, Math.max(0, data[i])) * 255);
    output[i * 4 + 1] = Math.round(Math.min(1, Math.max(0, data[pixels + i])) * 255);
    output[i * 4 + 2] = Math.round(Math.min(1, Math.max(0, data[2 * pixels + i])) * 255);
    output[i * 4 + 3] = 255;
  }
  return output;
}

function motionCompensate(prev, curr, width, height, blockSize = 16, searchRange = 8) {
  const compensated = new Uint8ClampedArray(prev.length);
  compensated.set(curr);

  for (let by = 0; by < height; by += blockSize) {
    for (let bx = 0; bx < width; bx += blockSize) {
      let bestDx = 0;
      let bestDy = 0;
      let bestSAD = Infinity;

      for (let dy = -searchRange; dy <= searchRange; dy += 1) {
        for (let dx = -searchRange; dx <= searchRange; dx += 1) {
          let sad = 0;
          for (let py = 0; py < blockSize && by + py < height; py += 1) {
            for (let px = 0; px < blockSize && bx + px < width; px += 1) {
              const cy = by + py;
              const cx = bx + px;
              const ry = Math.max(0, Math.min(height - 1, cy + dy));
              const rx = Math.max(0, Math.min(width - 1, cx + dx));
              const ci = (cy * width + cx) * 4;
              const ri = (ry * width + rx) * 4;
              sad += Math.abs(curr[ci] - prev[ri])
                   + Math.abs(curr[ci + 1] - prev[ri + 1])
                   + Math.abs(curr[ci + 2] - prev[ri + 2]);
            }
          }
          if (sad < bestSAD) {
            bestSAD = sad;
            bestDx = dx;
            bestDy = dy;
          }
        }
      }

      for (let py = 0; py < blockSize && by + py < height; py += 1) {
        for (let px = 0; px < blockSize && bx + px < width; px += 1) {
          const cy = by + py;
          const cx = bx + px;
          const ry = Math.max(0, Math.min(height - 1, cy + bestDy));
          const rx = Math.max(0, Math.min(width - 1, cx + bestDx));
          const di = (cy * width + cx) * 4;
          const si = (ry * width + rx) * 4;
          compensated[di] = prev[si];
          compensated[di + 1] = prev[si + 1];
          compensated[di + 2] = prev[si + 2];
          compensated[di + 3] = 255;
        }
      }
    }
  }
  return compensated;
}

self.onmessage = async (event) => {
  const { type, payload } = event.data || {};

  if (type === 'denoise') {
    const { frames, width, height, modelUrl, useOnnx = true } = payload;

    if (!width || !height || !Array.isArray(frames) || frames.length < 1) {
      self.postMessage({ type: 'error', error: 'invalid-payload' });
      return;
    }

    try {
      const results = [];
      for (let i = 0; i < frames.length; i += 1) {
        self.postMessage({ type: 'progress', frame: i, total: frames.length });

        let frameData = frames[i];
        if (i > 0) {
          frameData = motionCompensate(frames[i - 1], frames[i], width, height);
        }

        let processed;
        if (useOnnx && modelUrl) {
          const session = await getSession(modelUrl);
          const inputTensor = frameToTensor(frameData, width, height);
          const feeds = { [session.inputNames[0]]: inputTensor };
          const outputMap = await session.run(feeds);
          const outputTensor = outputMap[session.outputNames[0]];
          processed = tensorToFrame(outputTensor, width, height);
        } else if (i === 0) {
          processed = new Uint8ClampedArray(frames[0]);
        } else {
          const pixels = width * height * 4;
          processed = new Uint8ClampedArray(pixels);
          const alpha = 0.7;
          for (let p = 0; p < pixels; p += 1) {
            processed[p] = Math.round(alpha * frames[i][p] + (1 - alpha) * frameData[p]);
          }
        }
        results.push(processed);
      }

      self.postMessage({ type: 'result', frames: results }, results.map((f) => f.buffer));
    } catch (err) {
      self.postMessage({ type: 'error', error: err.message });
    }
  }

  if (type === 'averageFrames') {
    const { width, height, frames } = payload || {};
    if (!width || !height || !Array.isArray(frames) || frames.length < 2) {
      self.postMessage({ type: 'error', error: 'invalid-payload' });
      return;
    }
    const pixels = width * height * 4;
    const output = new Uint8ClampedArray(pixels);
    for (let i = 0; i < pixels; i += 1) {
      let sum = 0;
      for (let f = 0; f < frames.length; f += 1) sum += frames[f][i] || 0;
      output[i] = Math.round(sum / frames.length);
    }
    self.postMessage({ type: 'result', output }, [output.buffer]);
  }
};
