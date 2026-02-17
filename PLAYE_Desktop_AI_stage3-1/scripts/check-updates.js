/*
 * Model update checker.
 *
 * It can work in two modes:
 * - Offline: check missing required models against the local manifest.
 * - Online: if PLAYE_REMOTE_MANIFEST_URL is set, it will fetch the remote manifest
 *   and compare versions against the local registry.
 */

const https = require('https');
const http = require('http');

const {
  readManifest,
  resolveModelsDir,
  readRegistry,
} = require('./model-utils');

function fetchJson(url) {
  return new Promise((resolve, reject) => {
    const client = url.startsWith('https:') ? https : http;
    client
      .get(url, (res) => {
        if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          resolve(fetchJson(res.headers.location));
          return;
        }
        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode} for ${url}`));
          return;
        }
        let raw = '';
        res.setEncoding('utf8');
        res.on('data', (chunk) => (raw += chunk));
        res.on('end', () => {
          try {
            resolve(JSON.parse(raw));
          } catch (e) {
            reject(new Error(`Invalid JSON from ${url}`));
          }
        });
      })
      .on('error', reject);
  });
}

function compareVersions(a, b) {
  // Lightweight semver-ish compare: split by dots, compare numeric parts.
  const pa = String(a || '0').split('.').map((x) => Number.parseInt(x, 10) || 0);
  const pb = String(b || '0').split('.').map((x) => Number.parseInt(x, 10) || 0);
  const n = Math.max(pa.length, pb.length);
  for (let i = 0; i < n; i += 1) {
    const da = pa[i] || 0;
    const db = pb[i] || 0;
    if (da > db) return 1;
    if (da < db) return -1;
  }
  return 0;
}

async function checkModelUpdates(options = {}) {
  const localManifest = readManifest();
  const modelsDir = resolveModelsDir(options.modelsDir);
  const registry = readRegistry(modelsDir);

  const remoteUrl = options.remoteManifestUrl || process.env.PLAYE_REMOTE_MANIFEST_URL || null;
  const remoteManifest = remoteUrl ? await fetchJson(remoteUrl) : null;
  const manifestToUse = remoteManifest && remoteManifest.models ? remoteManifest : localManifest;

  const updates = [];

  for (const [key, entry] of Object.entries(manifestToUse.models)) {
    const local = registry.models[key] || null;

    if (entry.required && !local) {
      updates.push({
        key,
        name: entry.name || key,
        reason: 'missing',
        currentVersion: null,
        targetVersion: entry.version || null,
      });
      continue;
    }

    if (remoteManifest && local && entry.version && local.version) {
      if (compareVersions(entry.version, local.version) > 0) {
        updates.push({
          key,
          name: entry.name || key,
          reason: 'new-version',
          currentVersion: local.version,
          targetVersion: entry.version,
        });
      }
    }
  }

  return {
    updatesAvailable: updates.length > 0,
    modelsDir,
    remoteManifestUrl: remoteUrl,
    updates,
  };
}

// CLI
async function main() {
  const result = await checkModelUpdates();
  console.log(JSON.stringify(result, null, 2));
}

if (require.main === module) {
  main().catch((e) => {
    console.error('[check-updates] Error:', e.message || e);
    process.exit(1);
  });
}

module.exports = { checkModelUpdates };
