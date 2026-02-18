"""Enterprise admin endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.auth_routes import get_db
from app.api.rbac import require_admin, require_analyst
from app.api.response import success_response
from app.api.routes import auth_required
from app.audit.enterprise import get_audit_log, log_action
from app.db.models import Team, User, UserRole, Workspace

router = APIRouter(prefix="/enterprise", tags=["enterprise"])


@router.get("/users")
async def list_users(
    request: Request,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_admin),
):
    users = db.query(User).all()
    return success_response(
        request,
        status="done",
        result={
            "users": [
                {
                    "id": u.id,
                    "email": u.email,
                    "username": u.username,
                    "role": u.role.value,
                    "team_id": u.team_id,
                    "is_active": u.is_active,
                    "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
                }
                for u in users
            ]
        },
    )


@router.patch("/users/{user_id}/role")
async def update_user_role(
    request: Request,
    user_id: int,
    role: str,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    try:
        user.role = UserRole[role]
    except KeyError as err:
        raise HTTPException(400, "Invalid role") from err
    db.commit()
    jwt_payload = getattr(request.state, "jwt_payload", {})
    log_action(
        db,
        "user_role_change",
        user_id=int(jwt_payload.get("sub") or 0) or None,
        resource_type="user",
        resource_id=str(user_id),
        details={"new_role": role},
        ip_address=request.client.host if request.client else "",
        request_id=getattr(request.state, "request_id", None),
    )
    return success_response(request, status="done", result={"user_id": user_id, "role": role})


@router.patch("/users/{user_id}/deactivate")
async def deactivate_user(
    request: Request,
    user_id: int,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    user.is_active = False
    db.commit()
    return success_response(request, status="done", result={"user_id": user_id, "active": False})


class TeamCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None


@router.post("/teams")
async def create_team(
    request: Request,
    payload: TeamCreateRequest,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_admin),
):
    team = Team(name=payload.name, description=payload.description)
    db.add(team)
    db.commit()
    db.refresh(team)
    return success_response(request, status="done", result={"id": team.id, "name": team.name})


@router.get("/teams")
async def list_teams(
    request: Request,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_analyst),
):
    teams = db.query(Team).all()
    return success_response(request, status="done", result={"teams": [{"id": t.id, "name": t.name, "members": len(t.members)} for t in teams]})


@router.post("/teams/{team_id}/add-user")
async def add_user_to_team(
    request: Request,
    team_id: int,
    user_id: int,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    user.team_id = team_id
    db.commit()
    return success_response(request, status="done", result={"user_id": user_id, "team_id": team_id})


@router.get("/audit")
async def enterprise_audit(
    request: Request,
    team_id: Optional[int] = None,
    user_id: Optional[int] = None,
    action: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_admin),
):
    entries = get_audit_log(db, team_id=team_id, user_id=user_id, action=action, limit=limit, offset=offset)
    return success_response(request, status="done", result={"entries": entries, "limit": limit, "offset": offset})


@router.post("/workspaces")
async def create_workspace(
    request: Request,
    name: str,
    team_id: int,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_analyst),
):
    ws = Workspace(name=name, team_id=team_id)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return success_response(request, status="done", result={"id": ws.id, "name": ws.name, "team_id": ws.team_id})
