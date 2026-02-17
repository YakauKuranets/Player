/*
 * Postinstall helper.
 *
 * What it does:
 * - Detect Python (Windows: prefer `py -3.10`, otherwise `python`; Linux/macOS: `python3`)
 * - Print clear instructions if Python is missing
 * - Optionally run backend setup (create venv + pip install) if PLAYE_SETUP_BACKEND=1
 */

const { spawnSync } = require('child_process');
const path = require('path');

function run(cmd, args) {
  return spawnSync(cmd, args, { stdio: 'pipe', encoding: 'utf8' });
}

function detectPython() {
  const isWin = process.platform === 'win32';

  if (isWin) {
    // Prefer `py -3.10` if available.
    const py = run('py', ['-3.10', '--version']);
    if (py.status === 0) return { cmd: 'py', argsPrefix: ['-3.10'] };

    const python = run('python', ['--version']);
    if (python.status === 0) return { cmd: 'python', argsPrefix: [] };
  } else {
    const python3 = run('python3', ['--version']);
    if (python3.status === 0) return { cmd: 'python3', argsPrefix: [] };
    const python = run('python', ['--version']);
    if (python.status === 0) return { cmd: 'python', argsPrefix: [] };
  }
  return null;
}

function printHelp() {
  console.error('\n[PLAYE] Python не найден. Установи Python 3.10+ и повтори установку.');
  console.error('Windows: https://www.python.org/downloads/windows/');
  console.error('После установки проверь:  python --version  (или)  py -3.10 --version\n');
}

async function maybeSetupBackend(python) {
  if (process.env.PLAYE_SETUP_BACKEND !== '1') {
    console.log('[PLAYE] Backend setup skipped (set PLAYE_SETUP_BACKEND=1 to enable).');
    return;
  }

  console.log('[PLAYE] Setting up backend (venv + pip install)...');
  const setupPath = path.join(__dirname, 'setup-backend.js');
  const node = process.execPath;
  const r = spawnSync(node, [setupPath, python.cmd, ...python.argsPrefix], { stdio: 'inherit' });
  if (r.status !== 0) {
    throw new Error('Backend setup failed.');
  }
}

async function main() {
  const python = detectPython();
  if (!python) {
    printHelp();
    process.exit(1);
  }

  const ver = run(python.cmd, [...python.argsPrefix, '--version']);
  const versionStr = (ver.stdout || ver.stderr || '').trim();
  console.log(`[PLAYE] Python найден: ${versionStr}`);
  await maybeSetupBackend(python);
}

main().catch((e) => {
  console.error('[PLAYE] Ошибка:', e.message || e);
  process.exit(1);
});
