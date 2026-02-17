/*
 * Download AI model weights defined in models-data/manifest.json.
 *
 * Can be used in two ways:
 * 1) Required as a module by Electron main process.
 * 2) Run as a CLI:
 *    - node scripts/download-models.js --required
 *    - node scripts/download-models.js --all
 *    - node scripts/download-models.js restoreformer
 */

const path = require('path');

const {
  readManifest,
  resolveModelsDir,
  readRegistry,
  writeRegistry,
  normalizeSha256,
  isPlaceholderUrl,
  downloadToFile,
} = require('./model-utils');

async function downloadModel(modelKey, onProgress, options = {}) {
  const manifest = readManifest();
  const entry = manifest.models[modelKey];
  if (!entry) {
    throw new Error(`Unknown model '${modelKey}'. Check models-data/manifest.json`);
  }

  const modelsDir = resolveModelsDir(options.modelsDir);
  const registry = readRegistry(modelsDir);

  const url = entry.url;
  if (isPlaceholderUrl(url)) {
    throw new Error(
      `URL для модели '${modelKey}' не настроен (manifest содержит placeholder). ` +
        `Укажи реальный url в models-data/manifest.json`
    );
  }

  const filename = entry.filename || `${modelKey}.pth`;
  const destPath = path.join(modelsDir, filename);
  const expectedSha256 = normalizeSha256(entry.checksum);

  const result = await downloadToFile({
    url,
    destPath,
    expectedSha256,
    onProgress,
  });

  registry.models[modelKey] = {
    key: modelKey,
    name: entry.name || modelKey,
    filename,
    version: entry.version || null,
    sha256: result.sha256,
    bytes: result.bytes,
    downloadedAt: new Date().toISOString(),
  };
  writeRegistry(modelsDir, registry);

  return { success: true, modelKey, path: result.destPath, sha256: result.sha256 };
}

async function downloadMany({ keys, modelsDir, onModelProgress }) {
  const updated = [];
  for (const key of keys) {
    await downloadModel(
      key,
      (progress) => onModelProgress && onModelProgress({ modelKey: key, progress }),
      { modelsDir }
    );
    updated.push(key);
  }
  return updated;
}

// CLI
async function main() {
  const args = process.argv.slice(2);
  const manifest = readManifest();
  const allKeys = Object.keys(manifest.models);

  const wantAll = args.includes('--all');
  const wantRequired = args.includes('--required');
  const keysFromArgs = args.filter((a) => !a.startsWith('--'));

  let keys = [];
  if (wantAll) {
    keys = allKeys;
  } else if (wantRequired) {
    keys = allKeys.filter((k) => !!manifest.models[k].required);
  } else if (keysFromArgs.length > 0) {
    keys = keysFromArgs;
  } else {
    // By default: do nothing (safer for postinstall).
    console.log('[download-models] No targets specified. Use --required / --all / <modelKey>.');
    return;
  }

  console.log(`[download-models] Targets: ${keys.join(', ')}`);
  await downloadMany({
    keys,
    onModelProgress: ({ modelKey, progress }) => {
      process.stdout.write(`\r[download-models] ${modelKey}: ${String(progress).padStart(3, ' ')}%`);
      if (progress >= 100) process.stdout.write('\n');
    },
  });
  console.log('[download-models] Done.');
}

if (require.main === module) {
  main().catch((e) => {
    console.error('[download-models] Error:', e.message || e);
    process.exit(1);
  });
}

module.exports = { downloadModel, downloadMany };
