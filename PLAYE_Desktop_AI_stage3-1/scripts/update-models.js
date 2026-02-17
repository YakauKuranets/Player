/*
 * Update models by:
 *  - running checkModelUpdates
 *  - downloading missing/new models
 */

const { checkModelUpdates } = require('./check-updates');
const { downloadMany } = require('./download-models');

async function updateModels(options = {}) {
  const check = await checkModelUpdates(options);
  if (!check.updatesAvailable) {
    return { updated: [], ...check };
  }

  const keys = check.updates.map((u) => u.key);
  const updated = await downloadMany({
    keys,
    modelsDir: check.modelsDir,
    onModelProgress: options.onModelProgress,
  });

  return { updated, ...check };
}

// CLI
async function main() {
  const result = await updateModels({
    onModelProgress: ({ modelKey, progress }) => {
      process.stdout.write(`\r[update-models] ${modelKey}: ${String(progress).padStart(3, ' ')}%`);
      if (progress >= 100) process.stdout.write('\n');
    },
  });
  console.log('\n' + JSON.stringify(result, null, 2));
}

if (require.main === module) {
  main().catch((e) => {
    console.error('[update-models] Error:', e.message || e);
    process.exit(1);
  });
}

module.exports = { updateModels };
