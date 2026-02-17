/*
 * Create backend venv and install requirements.
 *
 * Called by check-python.js when PLAYE_SETUP_BACKEND=1.
 *
 * Usage (internal):
 *   node scripts/setup-backend.js <pythonCmd> [...pythonArgsPrefix]
 */

const { spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');

function run(cmd, args, cwd) {
  const r = spawnSync(cmd, args, { cwd, stdio: 'inherit' });
  if (r.status !== 0) {
    throw new Error(`Command failed: ${cmd} ${args.join(' ')}`);
  }
}

function exists(p) {
  try {
    return fs.existsSync(p);
  } catch (_e) {
    return false;
  }
}

function getVenvPython(backendDir) {
  const isWin = process.platform === 'win32';
  const p = isWin
    ? path.join(backendDir, '.venv', 'Scripts', 'python.exe')
    : path.join(backendDir, '.venv', 'bin', 'python');
  return p;
}

function main() {
  const args = process.argv.slice(2);
  const pythonCmd = args[0];
  const prefix = args.slice(1);
  if (!pythonCmd) {
    console.error('Usage: node scripts/setup-backend.js <pythonCmd> [...pythonArgsPrefix]');
    process.exit(2);
  }

  const projectRoot = path.join(__dirname, '..');
  const backendDir = path.join(projectRoot, 'backend');
  const requirements = path.join(backendDir, 'requirements.txt');

  if (!exists(requirements)) {
    console.error(`[setup-backend] requirements.txt not found: ${requirements}`);
    process.exit(2);
  }

  const venvPython = getVenvPython(backendDir);

  if (!exists(venvPython)) {
    console.log('[setup-backend] Creating venv in backend/.venv ...');
    run(pythonCmd, [...prefix, '-m', 'venv', '.venv'], backendDir);
  } else {
    console.log('[setup-backend] venv already exists.');
  }

  if (!exists(venvPython)) {
    console.error('[setup-backend] venv python not found after creation.');
    process.exit(2);
  }

  console.log('[setup-backend] Upgrading pip/setuptools/wheel ...');
  run(venvPython, ['-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'], backendDir);

  // NOTE: torch/torchvision are huge. Use --no-cache-dir to reduce disk pressure.
  const pipArgs = ['-m', 'pip', 'install', '--no-cache-dir', '-r', 'requirements.txt'];
  console.log('[setup-backend] Installing backend requirements (this can take a while) ...');
  run(venvPython, pipArgs, backendDir);

  console.log('[setup-backend] Backend is ready.');
}

try {
  main();
} catch (e) {
  console.error('[setup-backend] Error:', e.message || e);
  process.exit(1);
}
