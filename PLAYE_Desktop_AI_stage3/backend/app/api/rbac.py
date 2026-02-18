"""Role-based access control helpers."""

from __future__ import annotations

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
    "PATCH /api/enterprise/users/{user_id}/role": UserRole.admin.value,
    "PATCH /api/enterprise/users/{user_id}/deactivate": UserRole.admin.value,
    "PATCH /api/enterprise/users/{user_id}/activate": UserRole.admin.value,
    "GET /api/enterprise/users/{user_id}/sessions": UserRole.admin.value,
    "PATCH /api/enterprise/sessions/{session_id}/revoke": UserRole.admin.value,
    "GET /api/enterprise/audit": UserRole.admin.value,
    "POST /api/enterprise/teams": UserRole.admin.value,
    "POST /api/enterprise/teams/{team_id}/add-user": UserRole.admin.value,
    "GET /api/enterprise/teams": UserRole.analyst.value,
    "GET /api/enterprise/teams/{team_id}/workspaces": UserRole.analyst.value,
    "GET /api/enterprise/workspaces": UserRole.analyst.value,
    "POST /api/enterprise/workspaces": UserRole.analyst.value,
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


def endpoint_required_role(method: str, path: str) -> str | None:
    """Return min role for known endpoint key (best-effort lookup)."""
    key = f"{method.upper()} {path}"
    return ENDPOINT_ROLES.get(key)


require_viewer = require_role(UserRole.viewer.value)
require_analyst = require_role(UserRole.analyst.value)
require_admin = require_role(UserRole.admin.value)
