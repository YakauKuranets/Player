/**
 * Enterprise Blueprint — управление пользователями, командами и audit log.
 * Подключается к /api/enterprise/* эндпоинтам.
 */
export const createEnterpriseBlueprint = () => ({
  name: "enterprise",
  init: ({ elements, state, actions }) => {

    const apiGet = async (path) => {
      const resp = await fetch(`http://127.0.0.1:8000${path}`, {
        headers: { Authorization: `Bearer ${state.apiToken || ""}` },
      });
      if (!resp.ok) throw new Error(`${path} → HTTP ${resp.status}`);
      return resp.json();
    };

    const apiPost = async (path, body) => {
      const resp = await fetch(`http://127.0.0.1:8000${path}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${state.apiToken || ""}`,
        },
        body: JSON.stringify(body),
      });
      if (!resp.ok) throw new Error(`${path} → HTTP ${resp.status}`);
      return resp.json();
    };

    const apiPatch = async (path, body = {}) => {
      const resp = await fetch(`http://127.0.0.1:8000${path}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${state.apiToken || ""}`,
        },
        body: JSON.stringify(body),
      });
      if (!resp.ok) throw new Error(`${path} → HTTP ${resp.status}`);
      return resp.json();
    };

    // ─── Auth ──────────────────────────────────────────────────────────────

    elements.loginButton?.addEventListener("click", async () => {
      const email = elements.loginEmail?.value?.trim();
      const password = elements.loginPassword?.value;
      if (!email || !password) return;
      try {
        const data = await apiPost("/auth/login", { email, password });
        const token = data?.result?.token;
        if (!token) throw new Error("No token in response");
        state.apiToken = token;
        // Сохрани токен глобально для preload.js
        if (typeof window !== "undefined") window._playeToken = token;
        actions.recordLog("enterprise-login", `Вход выполнен: ${email}`);
        if (elements.loginStatus) {
          elements.loginStatus.textContent = `✅ Авторизован: ${email}`;
        }
        await loadDashboard();
      } catch (err) {
        actions.recordLog("enterprise-login-error", err.message);
        if (elements.loginStatus) {
          elements.loginStatus.textContent = `❌ Ошибка: ${err.message}`;
        }
      }
    });

    elements.logoutButton?.addEventListener("click", async () => {
      try {
        await apiPost("/auth/logout", {});
      } catch (_) { /* игнорируем */ }
      state.apiToken = "";
      if (typeof window !== "undefined") window._playeToken = "";
      if (elements.loginStatus) elements.loginStatus.textContent = "Не авторизован";
      actions.recordLog("enterprise-logout", "Выход выполнен");
    });

    // ─── Dashboard ─────────────────────────────────────────────────────────

    const loadDashboard = async () => {
      try {
        const data = await apiGet("/api/enterprise/reports/dashboard");
        const summary = data?.result?.summary || {};
        if (elements.dashboardPanel) {
          elements.dashboardPanel.textContent = JSON.stringify(summary, null, 2);
        }
        actions.recordLog("enterprise-dashboard", "Дашборд загружен");
      } catch (err) {
        actions.recordLog("enterprise-dashboard-error", err.message);
      }
    };

    elements.dashboardRefreshButton?.addEventListener("click", loadDashboard);

    // ─── GPU Status ────────────────────────────────────────────────────────

    elements.gpuStatusButton?.addEventListener("click", async () => {
      try {
        const data = await apiGet("/api/system/gpu");
        const gpus = data?.result?.gpus || [];
        if (elements.gpuStatusPanel) {
          elements.gpuStatusPanel.innerHTML = gpus.map((gpu) => `
            <div class="gpu-slot">
              <strong>${gpu.device_name}</strong> (${gpu.queue})<br>
              Активных задач: ${gpu.active_tasks} |
              Память: ${gpu.free_memory_mb}/${gpu.total_memory_mb} МБ |
              ${gpu.healthy ? "✅ OK" : "❌ Недоступен"}
            </div>
          `).join("") || "<p>GPU не обнаружены (CPU режим)</p>";
        }
        actions.recordLog("gpu-status", `GPU: ${gpus.length} устройств`);
      } catch (err) {
        actions.recordLog("gpu-status-error", err.message);
      }
    });

    // ─── Users ─────────────────────────────────────────────────────────────

    const loadUsers = async () => {
      try {
        const data = await apiGet("/api/enterprise/users");
        const users = data?.result?.users || [];
        if (elements.usersTableBody) {
          elements.usersTableBody.innerHTML = users.map((u) => `
            <tr data-user-id="${u.id}">
              <td>${u.id}</td>
              <td>${u.email}</td>
              <td>${u.username}</td>
              <td>
                <select class="role-select" data-user-id="${u.id}">
                  <option value="viewer"  ${u.role === "viewer"  ? "selected" : ""}>viewer</option>
                  <option value="analyst" ${u.role === "analyst" ? "selected" : ""}>analyst</option>
                  <option value="admin"   ${u.role === "admin"   ? "selected" : ""}>admin</option>
                </select>
              </td>
              <td>${u.is_active ? "✅" : "❌"}</td>
              <td>${u.last_login_at ? new Date(u.last_login_at).toLocaleString("ru") : "—"}</td>
              <td>
                <button class="deactivate-btn" data-user-id="${u.id}" ${!u.is_active ? "disabled" : ""}>
                  Деактив.
                </button>
              </td>
            </tr>
          `).join("");

          // Role change handlers
          elements.usersTableBody.querySelectorAll(".role-select").forEach((sel) => {
            sel.addEventListener("change", async (e) => {
              const userId = e.target.dataset.userId;
              const newRole = e.target.value;
              try {
                await apiPatch(`/api/enterprise/users/${userId}/role?role=${newRole}`);
                actions.recordLog("enterprise-role-change", `User ${userId} → ${newRole}`);
              } catch (err) {
                actions.recordLog("enterprise-role-error", err.message);
              }
            });
          });

          // Deactivate handlers
          elements.usersTableBody.querySelectorAll(".deactivate-btn").forEach((btn) => {
            btn.addEventListener("click", async (e) => {
              const userId = e.target.dataset.userId;
              if (!confirm(`Деактивировать пользователя #${userId}?`)) return;
              try {
                await apiPatch(`/api/enterprise/users/${userId}/deactivate`);
                actions.recordLog("enterprise-deactivate", `User ${userId} деактивирован`);
                await loadUsers();
              } catch (err) {
                actions.recordLog("enterprise-deactivate-error", err.message);
              }
            });
          });
        }
      } catch (err) {
        actions.recordLog("enterprise-users-error", err.message);
      }
    };

    elements.usersRefreshButton?.addEventListener("click", loadUsers);

    // ─── Teams ─────────────────────────────────────────────────────────────

    elements.teamCreateButton?.addEventListener("click", async () => {
      const name = elements.teamNameInput?.value?.trim();
      const desc = elements.teamDescInput?.value?.trim() || null;
      if (!name) return;
      try {
        await apiPost("/api/enterprise/teams", { name, description: desc });
        actions.recordLog("enterprise-team-create", `Команда создана: ${name}`);
        if (elements.teamNameInput) elements.teamNameInput.value = "";
        await loadTeams();
      } catch (err) {
        actions.recordLog("enterprise-team-error", err.message);
      }
    });

    elements.teamAddUserButton?.addEventListener("click", async () => {
      const teamId = elements.teamIdInput?.value?.trim();
      const userId = elements.teamUserIdInput?.value?.trim();
      if (!teamId || !userId) return;
      try {
        await apiPost(`/api/enterprise/teams/${teamId}/add-user`, { user_id: parseInt(userId, 10) });
        actions.recordLog("enterprise-team-add-user", `User ${userId} → Team ${teamId}`);
      } catch (err) {
        actions.recordLog("enterprise-team-add-user-error", err.message);
      }
    });

    const loadTeams = async () => {
      try {
        const data = await apiGet("/api/enterprise/teams");
        const teams = data?.result?.teams || [];
        if (elements.teamsPanel) {
          elements.teamsPanel.innerHTML = teams.map((t) =>
            `<div><strong>#${t.id}</strong> ${t.name} — ${t.members} участников</div>`
          ).join("") || "<p>Команды не созданы</p>";
        }
      } catch (err) {
        actions.recordLog("enterprise-teams-error", err.message);
      }
    };

    elements.teamsRefreshButton?.addEventListener("click", loadTeams);

    // ─── Audit Log ─────────────────────────────────────────────────────────

    const loadAuditLog = async () => {
      const limit = parseInt(elements.auditLimitInput?.value || "50", 10);
      const action = elements.auditActionFilter?.value?.trim() || undefined;
      try {
        const params = new URLSearchParams({ limit: String(limit) });
        if (action) params.set("action", action);
        const data = await apiGet(`/api/enterprise/audit?${params}`);
        const entries = data?.result?.entries || [];
        if (elements.auditTableBody) {
          elements.auditTableBody.innerHTML = entries.map((e) => `
            <tr>
              <td>${new Date(e.timestamp).toLocaleString("ru")}</td>
              <td>${e.action}</td>
              <td>${e.user_id ?? "—"}</td>
              <td>${e.resource_type ?? "—"}</td>
              <td>${e.resource_id ?? "—"}</td>
              <td><span class="${e.status === "success" ? "status-ok" : "status-err"}">${e.status}</span></td>
              <td>${e.ip_address ?? "—"}</td>
              <td>${e.request_id ? e.request_id.slice(0, 8) + "…" : "—"}</td>
            </tr>
          `).join("") || "<tr><td colspan='8'>Записей нет</td></tr>";
        }
        actions.recordLog("enterprise-audit", `Загружено ${entries.length} записей`);
      } catch (err) {
        actions.recordLog("enterprise-audit-error", err.message);
      }
    };

    elements.auditRefreshButton?.addEventListener("click", loadAuditLog);

    elements.auditExportCsvButton?.addEventListener("click", async () => {
      try {
        const resp = await fetch("/api/enterprise/reports/audit.csv", {
          headers: { Authorization: `Bearer ${state.apiToken || ""}` },
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `audit-${new Date().toISOString().slice(0, 10)}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        actions.recordLog("enterprise-audit-export", "CSV экспорт audit log");
      } catch (err) {
        actions.recordLog("enterprise-audit-export-error", err.message);
      }
    });

    // ─── Timeseries ────────────────────────────────────────────────────────

    elements.timeseriesRefreshButton?.addEventListener("click", async () => {
      try {
        const data = await apiGet("/api/enterprise/reports/timeseries");
        const rows = data?.result?.rows || [];
        if (elements.timeseriesPanel) {
          elements.timeseriesPanel.innerHTML = rows.map((r) =>
            `<div>${r.bucket_start?.slice(0, 16)} | всего: ${r.total} | ok: ${r.success} | err: ${r.error ?? 0}</div>`
          ).join("") || "<p>Данных нет</p>";
        }
      } catch (err) {
        actions.recordLog("enterprise-timeseries-error", err.message);
      }
    });

    // ─── Startup ───────────────────────────────────────────────────────────

    // Если токен уже есть (например сохранён в state), загружаем дашборд сразу
    if (state.apiToken) {
      loadDashboard();
    }
  },
});
