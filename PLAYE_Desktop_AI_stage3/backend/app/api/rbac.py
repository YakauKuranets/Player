"""Role-based access control helpers."""

from fastapi import HTTPException, Request

from app.db.models import UserRole

ROLE_HIERARCHY = {
    UserRole.viewer.value: 0,
    UserRole.analyst.value: 1,
    UserRole.admin.value: 2,
}


ENDPOINT_ROLES = {
    "POST /api/job/submit": UserRole.analyst.value,
    "POST /api/job/batch/submit": UserRole.analyst.value,
    "POST /api/job/video/submit": UserRole.analyst.value,
    "GET /api/system/gpu": UserRole.analyst.value,
    "POST /auth/register": UserRole.admin.value,
    "GET /api/enterprise/users": UserRole.admin.value,
    "GET /api/enterprise/audit": UserRole.admin.value,
}


def require_role(min_role: str):
    async def _check(request: Request):
        jwt_payload = getattr(request.state, "jwt_payload", {})
        user_role = jwt_payload.get("role", "viewer")
        user_level = ROLE_HIERARCHY.get(user_role, 0)
        required_level = ROLE_HIERARCHY.get(min_role, 999)
        if user_level < required_level:
            raise HTTPException(status_code=403, detail=f"Insufficient permissions. Required: {min_role}, got: {user_role}")

    return _check


require_viewer = require_role("viewer")
require_analyst = require_role("analyst")
require_admin = require_role("admin")
