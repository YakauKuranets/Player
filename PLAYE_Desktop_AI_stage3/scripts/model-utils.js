/*
 * Shared helpers for model management.
 *
 * Goals:
 * - Read local manifest (models-data/manifest.json)
 * - Resolve the effective models directory (default: <project>/models-data, or PLAYE_MODELS_DIR)
 * - Maintain a lightweight local registry (models.json) with downloaded versions/checksums
 * - Download files with streaming, progress reporting and optional sha256 verification
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const http = require('http');
const https = require('https');

function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

function readJson(filePath, fallback = null) {
  try {
    const raw = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(raw);
  } catch (_e) {
    return fallback;
  }
}

function writeJson(filePath, value) {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2));
}

function getProjectRoot() {
  return path.join(__dirname, '..');
}

function getManifestPath() {
  return path.join(getProjectRoot(), 'models-data', 'manifest.json');
}

function readManifest() {
  const manifestPath = getManifestPath();
  const manifest = readJson(manifestPath);
  if (!manifest || typeof manifest !== 'object') {
    throw new Error(`Manifest not found or invalid: ${manifestPath}`);
  }
  if (!manifest.models || typeof manifest.models !== 'object') {
    throw new Error(`Manifest missing 'models' section: ${manifestPath}`);
  }
  return manifest;
}

function resolveModelsDir(overrideDir) {
  const envDir = process.env.PLAYE_MODELS_DIR;
  const modelsDir = overrideDir || envDir || path.join(getProjectRoot(), 'models-data');
  ensureDir(modelsDir);
  return modelsDir;
}

function getRegistryPath(modelsDir) {
  return path.join(modelsDir, 'models.json');
}

function readRegistry(modelsDir) {
  return readJson(getRegistryPath(modelsDir), { schema: 'playe.models.registry.v1', models: {} });
}

function writeRegistry(modelsDir, registry) {
  writeJson(getRegistryPath(modelsDir), registry);
}

function normalizeSha256(checksum) {
  if (!checksum) return null;
  const text = String(checksum).trim();
  if (!text) return null;
  return text.startsWith('sha256:') ? text.slice('sha256:'.length) : text;
}

function isPlaceholderUrl(url) {
  if (!url) return true;
  const u = String(url);
  return u.includes('example.com') || u.includes('YOUR_SERVER') || u.includes('your-server');
}

function requestStream(url) {
  return new Promise((resolve, reject) => {
    const client = url.startsWith('https:') ? https : http;
    const req = client.get(url, (res) => {
      if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        resolve(requestStream(res.headers.location));
        return;
      }
      if (res.statusCode !== 200) {
        reject(new Error(`HTTP ${res.statusCode} for ${url}`));
        return;
      }
      resolve({ res, headers: res.headers });
    });
    req.on('error', reject);
  });
}

async function downloadToFile({ url, destPath, expectedSha256, onProgress }) {
  ensureDir(path.dirname(destPath));
  const tmpPath = `${destPath}.download`;
  if (fs.existsSync(tmpPath)) {
    fs.rmSync(tmpPath, { force: true });
  }

  const { res, headers } = await requestStream(url);
  const total = Number.parseInt(headers['content-length'] || '0', 10);
  const hasTotal = Number.isFinite(total) && total > 0;

  const hash = crypto.createHash('sha256');
  let downloaded = 0;
  let lastPercent = -1;

  const file = fs.createWriteStream(tmpPath);

  return await new Promise((resolve, reject) => {
    const cleanup = (err) => {
      try { file.close(); } catch (_e) {}
      try { if (fs.existsSync(tmpPath)) fs.rmSync(tmpPath, { force: true }); } catch (_e) {}
      reject(err);
    };

    res.on('data', (chunk) => {
      downloaded += chunk.length;
      hash.update(chunk);
      if (onProgress) {
        if (hasTotal) {
          const percent = Math.floor((downloaded / total) * 100);
          if (percent !== lastPercent) {
            lastPercent = percent;
            onProgress(percent);
          }
        } else {
          const approx = Math.min(99, Math.floor(downloaded / (5 * 1024 * 1024)));
          if (approx !== lastPercent) {
            lastPercent = approx;
            onProgress(approx);
          }
        }
      }
    });

    res.on('error', cleanup);
    file.on('error', cleanup);

    res.pipe(file);

    file.on('finish', () => {
      file.close(() => {
        const actualSha256 = hash.digest('hex');
        if (expectedSha256 && actualSha256 !== expectedSha256) {
          cleanup(new Error(`SHA256 mismatch. Expected ${expectedSha256}, got ${actualSha256}`));
          return;
        }
        fs.renameSync(tmpPath, destPath);
        if (onProgress) onProgress(100);
        resolve({ destPath, sha256: actualSha256, bytes: downloaded });
      });
    });
  });
}

module.exports = {
  ensureDir,
  readManifest,
  resolveModelsDir,
  readRegistry,
  writeRegistry,
  normalizeSha256,
  isPlaceholderUrl,
  downloadToFile,
};
