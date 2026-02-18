#!/usr/bin/env node

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

function loadApiClientFactory() {
  const filePath = path.resolve(__dirname, '../frontend/src/api-client.js');
  const source = fs.readFileSync(filePath, 'utf-8');
  const transformed = source.replace('export const createApiClient =', 'const createApiClient =');
  const factory = new Function(`${transformed}\nreturn { createApiClient };`);
  return factory().createApiClient;
}

const makeJsonResponse = (status, payload) => ({
  ok: status >= 200 && status < 300,
  status,
  async json() {
    return payload;
  },
});

async function run() {
  const createApiClient = loadApiClientFactory();
  const calls = [];

  global.fetch = async (url, options = {}) => {
    calls.push({ url, options });

    if (String(url).endsWith('/hello')) {
      return makeJsonResponse(200, { status: 'done', result: { message: 'ok' } });
    }

    if (String(url).endsWith('/job/submit')) {
      return makeJsonResponse(202, { status: 'queued', result: { task_id: 'task-1' } });
    }


    if (String(url).endsWith('/job/task-1/cancel')) {
      return makeJsonResponse(200, { status: 'done', result: { task_id: 'task-1', status: 'canceled', is_final: true } });
    }

    if (String(url).endsWith('/job/task-1/status')) {
      const pollAttempt = calls.filter((c) => String(c.url).endsWith('/job/task-1/status')).length;
      if (pollAttempt < 3) {
        return makeJsonResponse(200, { status: 'done', result: { status: 'running', progress: pollAttempt * 25, is_final: false, poll_after_ms: 1 } });
      }
      return makeJsonResponse(200, { status: 'done', result: { status: 'done', progress: 100, is_final: true, poll_after_ms: 0 } });
    }

    return makeJsonResponse(404, { error: 'not-found' });
  };

  global.window = { setTimeout };

  const client = createApiClient({ baseUrl: 'http://127.0.0.1:8000/api', token: 'jwt-token' });
  const ping = await client.ping();
  assert.equal(ping.result.message, 'ok');

  const submit = await client.submitJob({ operation: 'detect_objects', image_base64: 'aGVsbG8=', params: {} });
  assert.equal(submit.result.task_id, 'task-1');

  const cancel = await client.cancelJob('task-1');
  assert.equal(cancel.result.status, 'canceled');

  const progressEvents = [];
  const final = await client.pollJobUntilFinal('task-1', {
    maxAttempts: 5,
    intervalMs: 1,
    onProgress: ({ status, progress }) => progressEvents.push({ status, progress }),
  });

  assert.equal(final.final, 'success');
  assert.ok(progressEvents.length >= 2);

  const submitCall = calls.find((c) => String(c.url).endsWith('/job/submit'));
  assert.equal(submitCall.options.headers.Authorization, 'Bearer jwt-token');
  const cancelCall = calls.find((c) => String(c.url).endsWith('/job/task-1/cancel'));
  assert.equal(cancelCall.options.headers.Authorization, 'Bearer jwt-token');

  console.log('[test-api-client] passed');
}

run().catch((error) => {
  console.error('[test-api-client] failed:', error.message || error);
  process.exit(1);
});
