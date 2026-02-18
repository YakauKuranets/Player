self.onmessage = (event) => {
  const { type, payload } = event.data || {};
  if (type !== "averageFrames") {
    return;
  }

  const { width, height, frames } = payload || {};
  if (!width || !height || !Array.isArray(frames) || frames.length < 2) {
    self.postMessage({ type: "error", error: "invalid-payload" });
    return;
  }

  const pixels = width * height * 4;
  const output = new Uint8ClampedArray(pixels);

  for (let i = 0; i < pixels; i += 1) {
    let sum = 0;
    for (let f = 0; f < frames.length; f += 1) {
      sum += frames[f][i] || 0;
    }
    output[i] = Math.round(sum / frames.length);
  }

  self.postMessage({ type: "result", output }, [output.buffer]);
};
