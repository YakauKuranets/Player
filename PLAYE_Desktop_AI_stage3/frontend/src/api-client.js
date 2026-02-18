const DEFAULT_BASE_URL = "http://127.0.0.1:8000/api";

const FINAL_SUCCESS_STATES = new Set(["success", "done"]);
const FINAL_FAILURE_STATES = new Set(["failure", "failed", "error"]);

const buildHeaders = (token) => {
  const headers = { "Content-Type": "application/json" };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
};

const delay = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

export const createApiClient = ({ baseUrl = DEFAULT_BASE_URL, token = "" } = {}) => {
  const normalizedBaseUrl = baseUrl.replace(/\/$/, "");

  const parseResponse = async (response, fallbackMessage) => {
    const json = await response.json();
    if (!response.ok) {
      throw new Error(json?.error || json?.detail || fallbackMessage);
    }
    return json;
  };

  const client = {
    async ping() {
      const response = await fetch(`${normalizedBaseUrl}/hello`, { method: "GET" });
      return parseResponse(response, "Ping failed");
    },

    async submitJob(payload) {
      const response = await fetch(`${normalizedBaseUrl}/job/submit`, {
        method: "POST",
        headers: buildHeaders(token),
        body: JSON.stringify(payload),
      });
      return parseResponse(response, "Job submit failed");
    },

    async getJobStatus(taskId) {
      const response = await fetch(`${normalizedBaseUrl}/job/${taskId}/status`, {
        method: "GET",
        headers: buildHeaders(token),
      });
      return parseResponse(response, "Job status failed");
    },

    async cancelJob(taskId) {
      const response = await fetch(`${normalizedBaseUrl}/job/${taskId}/cancel`, {
        method: "POST",
        headers: buildHeaders(token),
      });
      return parseResponse(response, "Job cancel failed");
    },

    async pollJobUntilFinal(taskId, { maxAttempts = 25, intervalMs = 600, onProgress } = {}) {
      let attempts = 0;
      let nextIntervalMs = intervalMs;
      while (attempts < maxAttempts) {
        await delay(nextIntervalMs);
        const statusPayload = await client.getJobStatus(taskId);
        const backend = statusPayload?.result || {};
        const status = String(backend.status || "pending").toLowerCase();
        const progress = Number.isFinite(backend.progress) ? backend.progress : 0;
        const isFinal = Boolean(backend.is_final);

        if (onProgress) {
          onProgress({ status, progress, payload: backend });
        }

        if (isFinal && !backend.error) {
          return { final: "success", payload: backend };
        }

        if (isFinal && backend.error) {
          return { final: "failure", payload: backend };
        }

        if (FINAL_SUCCESS_STATES.has(status)) {
          return { final: "success", payload: backend };
        }

        if (FINAL_FAILURE_STATES.has(status)) {
          return { final: "failure", payload: backend };
        }

        const pollAfter = Number.parseInt(backend.poll_after_ms, 10);
        if (Number.isFinite(pollAfter) && pollAfter >= 0) {
          nextIntervalMs = pollAfter;
        }

        attempts += 1;
      }

      return { final: "timeout", payload: null };
    },
  };

  return client;
};
