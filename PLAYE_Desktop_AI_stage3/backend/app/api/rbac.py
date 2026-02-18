"""Role-based access control helpers."""

from __future__ import annotations

import re
from functools import lru_cache

from fastapi import HTTPException, Request

from app.db.models import UserRole

ROLE_HIERARCHY = {
    UserRole.viewer.value: 0,
    UserRole.analyst.value: 1,
    UserRole.admin.value: 2,
}

ENDPOINT_ROLES = {
    # Jobs
    "POST /api/job/submit": UserRole.analyst.value,
    "POST /api/job/batch/submit": UserRole.analyst.value,
    "POST /api/job/video/submit": UserRole.analyst.value,
    "GET /api/system/gpu": UserRole.analyst.value,
    # Auth & enterprise
    "POST /auth/register": UserRole.admin.value,
    "GET /api/enterprise/users": UserRole.admin.value,
    "GET /api/enterprise/users/{user_id}": UserRole.admin.value,
    "GET /api/enterprise/users/{user_id}/activity": UserRole.admin.value,
    "PATCH /api/enterprise/users/{user_id}/role": UserRole.admin.value,
    "PATCH /api/enterprise/users/{user_id}/deactivate": UserRole.admin.value,
    "PATCH /api/enterprise/users/{user_id}/activate": UserRole.admin.value,
    "GET /api/enterprise/users/{user_id}/sessions": UserRole.admin.value,
    "PATCH /api/enterprise/sessions/{session_id}/revoke": UserRole.admin.value,
    "GET /api/enterprise/audit": UserRole.admin.value,
    "GET /api/enterprise/audit/actions": UserRole.admin.value,
    "POST /api/enterprise/teams": UserRole.admin.value,
    "POST /api/enterprise/teams/{team_id}/add-user": UserRole.admin.value,
    "GET /api/enterprise/teams": UserRole.analyst.value,
    "GET /api/enterprise/teams/{team_id}/workspaces": UserRole.analyst.value,
    "GET /api/enterprise/teams/{team_id}/members": UserRole.analyst.value,
    "GET /api/enterprise/workspaces": UserRole.analyst.value,
    "GET /api/enterprise/dashboard/summary": UserRole.analyst.value,
    "POST /api/enterprise/workspaces": UserRole.analyst.value,
    # Reports
    "GET /api/enterprise/reports/audit": UserRole.admin.value,
    "GET /api/enterprise/reports/audit.csv": UserRole.admin.value,
    "GET /api/enterprise/reports/actions": UserRole.analyst.value,
    "GET /api/enterprise/reports/actions.csv": UserRole.analyst.value,
    "GET /api/enterprise/reports/dashboard": UserRole.analyst.value,
    "GET /api/enterprise/reports/timeseries": UserRole.analyst.value,
    "GET /api/enterprise/reports/timeseries.csv": UserRole.analyst.value,
    "GET /api/enterprise/reports/dashboard.csv": UserRole.analyst.value,
    "GET /api/enterprise/reports/users/{user_id}/activity": UserRole.admin.value,
    "GET /api/enterprise/reports/users/{user_id}/activity.csv": UserRole.admin.value,
    "GET /api/enterprise/reports/manifest": UserRole.analyst.value,
    "GET /api/enterprise/reports/manifest.csv": UserRole.analyst.value,
}


def _assert_role(request: Request, min_role: str) -> None:
    jwt_payload = getattr(request.state, "jwt_payload", {})
    user_role = jwt_payload.get("role", UserRole.viewer.value)
    user_level = ROLE_HIERARCHY.get(user_role, 0)
    required_level = ROLE_HIERARCHY.get(min_role, 999)

    if user_level < required_level:
        raise HTTPException(
            status_code=403,
            detail=f"Insufficient permissions. Required: {min_role}, got: {user_role}",
        )


def require_role(min_role: str):
    async def _check(request: Request):
        _assert_role(request, min_role)

    return _check


@lru_cache(maxsize=1)
def _compiled_endpoint_patterns() -> list[tuple[str, re.Pattern[str], str]]:
    compiled: list[tuple[str, re.Pattern[str], str]] = []
    for key, role in ENDPOINT_ROLES.items():
        method, path = key.split(" ", 1)
        # convert /api/users/{id}/x => ^/api/users/[^/]+/x$
        regex_path = "^" + re.sub(r"\{[^/{}]+\}", r"[^/]+", path.rstrip("/")) + "$"
        compiled.append((method.upper(), re.compile(regex_path), role))
    return compiled


def endpoint_required_role(method: str, path: str) -> str | None:
    """Return min role for endpoint using exact and templated matching."""
    normalized_path = path.rstrip("/") or "/"
    key = f"{method.upper()} {normalized_path}"
    role = ENDPOINT_ROLES.get(key)
    if role is not None:
        return role

    for endpoint_method, pattern, endpoint_role in _compiled_endpoint_patterns():
        if endpoint_method == method.upper() and pattern.match(normalized_path):
            return endpoint_role
    return None


require_viewer = require_role(UserRole.viewer.value)
require_analyst = require_role(UserRole.analyst.value)
require_admin = require_role(UserRole.admin.value)
