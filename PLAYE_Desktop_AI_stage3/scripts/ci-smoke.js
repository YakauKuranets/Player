#!/usr/bin/env node

const { execSync } = require('child_process');

const checks = [
  'node --check frontend/src/main.js',
  'node --check frontend/src/api-client.js',
  'node scripts/test-api-client.js',
  'python3 scripts/test-job-params.py',
  'python3 scripts/test-task-status.py',
  'python3 scripts/test-cancel-route.py',
  'python3 scripts/test-enterprise-report-filters.py',
  'python3 scripts/test-enterprise-report-schema.py',
  'python3 scripts/test-rbac-route-matching.py',
  'node --check frontend/src/blueprints/forensic.js',
  'node --check frontend/src/blueprints/ai.js',
  'node --check frontend/src/blueprints/quality.js',
  'node --check frontend/src/workers/temporal-denoise-worker.js',
  'python3 -c "import ast; ast.parse(open(\'backend/server.py\', encoding=\'utf-8\').read())"',
  'python3 -c "import ast; ast.parse(open(\'backend/app/api/routes.py\', encoding=\'utf-8\').read())"',
  'python3 -c "import ast; ast.parse(open(\'backend/app/main.py\', encoding=\'utf-8\').read())"',
  'node -e "require(\'./scripts/model-utils\')"',
];

for (const cmd of checks) {
  process.stdout.write(`[ci-smoke] ${cmd}\n`);
  execSync(cmd, { stdio: 'inherit' });
}

process.stdout.write('[ci-smoke] All checks passed.\n');
