"""Enterprise admin endpoints: users, teams, workspaces, sessions, audit."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.auth_routes import get_db
from app.api.rbac import require_admin, require_analyst
from app.api.response import success_response
from app.api.routes import auth_required
from app.audit.enterprise import get_audit_log, log_action
from app.db.models import Team, User, UserRole, UserSession, Workspace

router = APIRouter(prefix="/enterprise", tags=["enterprise"])


def _actor_info(request: Request) -> tuple[Optional[int], Optional[int]]:
    payload = getattr(request.state, "jwt_payload", {}) or {}
    user_id = payload.get("sub")
    team_id = payload.get("team_id")
    return (
        int(user_id) if str(user_id).isdigit() else None,
        int(team_id) if str(team_id).isdigit() else None,
    )


def _audit(
    request: Request,
    db: Session,
    action: str,
    *,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    details: Optional[dict] = None,
    status: str = "success",
) -> None:
    actor_user_id, actor_team_id = _actor_info(request)
    log_action(
        db=db,
        action=action,
        user_id=actor_user_id,
        team_id=actor_team_id,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        ip_address=request.client.host if request.client else None,
        request_id=getattr(request.state, "request_id", None),
        status=status,
    )


class TeamCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)


class WorkspaceCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    team_id: int


class UserTeamAssignRequest(BaseModel):
    user_id: int


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
                    "created_at": u.created_at.isoformat() if u.created_at else None,
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
    _audit(request, db, "user_role_change", resource_type="user", resource_id=str(user_id), details={"new_role": role})
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
    _audit(request, db, "user_deactivate", resource_type="user", resource_id=str(user_id))
    return success_response(request, status="done", result={"user_id": user_id, "active": False})


@router.patch("/users/{user_id}/activate")
async def activate_user(
    request: Request,
    user_id: int,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")

    user.is_active = True
    db.commit()
    _audit(request, db, "user_activate", resource_type="user", resource_id=str(user_id))
    return success_response(request, status="done", result={"user_id": user_id, "active": True})


@router.get("/users/{user_id}/sessions")
async def list_user_sessions(
    request: Request,
    user_id: int,
    include_revoked: bool = Query(False),
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_admin),
):
    query = db.query(UserSession).filter(UserSession.user_id == user_id)
    if not include_revoked:
        query = query.filter(UserSession.revoked.is_(False))
    sessions = query.order_by(UserSession.created_at.desc()).all()

    return success_response(
        request,
        status="done",
        result={
            "user_id": user_id,
            "sessions": [
                {
                    "id": s.id,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "expires_at": s.expires_at.isoformat() if s.expires_at else None,
                    "revoked": s.revoked,
                    "ip_address": s.ip_address,
                    "user_agent": s.user_agent,
                }
                for s in sessions
            ],
        },
    )


@router.patch("/sessions/{session_id}/revoke")
async def revoke_user_session(
    request: Request,
    session_id: int,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_admin),
):
    session = db.query(UserSession).filter(UserSession.id == session_id).first()
    if not session:
        raise HTTPException(404, "Session not found")

    session.revoked = True
    db.commit()
    _audit(request, db, "session_revoke", resource_type="session", resource_id=str(session_id))
    return success_response(request, status="done", result={"session_id": session_id, "revoked": True})


@router.post("/teams")
async def create_team(
    request: Request,
    payload: TeamCreateRequest,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_admin),
):
    existing = db.query(Team).filter(Team.name == payload.name).first()
    if existing:
        raise HTTPException(409, "Team already exists")

    team = Team(name=payload.name, description=payload.description)
    db.add(team)
    db.commit()
    db.refresh(team)

    _audit(request, db, "team_create", resource_type="team", resource_id=str(team.id), details={"name": team.name})
    return success_response(request, status="done", result={"id": team.id, "name": team.name})


@router.get("/teams")
async def list_teams(
    request: Request,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_analyst),
):
    teams = db.query(Team).all()
    return success_response(
        request,
        status="done",
        result={
            "teams": [
                {
                    "id": t.id,
                    "name": t.name,
                    "description": t.description,
                    "members": len(t.members),
                    "workspaces": len(t.workspaces),
                }
                for t in teams
            ]
        },
    )


@router.post("/teams/{team_id}/add-user")
async def add_user_to_team(
    request: Request,
    team_id: int,
    payload: UserTeamAssignRequest,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_admin),
):
    team = db.query(Team).filter(Team.id == team_id).first()
    if not team:
        raise HTTPException(404, "Team not found")

    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(404, "User not found")

    user.team_id = team_id
    db.commit()
    _audit(
        request,
        db,
        "team_add_user",
        resource_type="team",
        resource_id=str(team_id),
        details={"user_id": payload.user_id},
    )

    return success_response(request, status="done", result={"user_id": payload.user_id, "team_id": team_id})


@router.get("/teams/{team_id}/workspaces")
async def list_team_workspaces(
    request: Request,
    team_id: int,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_analyst),
):
    workspaces = db.query(Workspace).filter(Workspace.team_id == team_id).all()
    return success_response(
        request,
        status="done",
        result={
            "team_id": team_id,
            "workspaces": [
                {
                    "id": w.id,
                    "name": w.name,
                    "created_at": w.created_at.isoformat() if w.created_at else None,
                    "cases_count": len(w.cases),
                }
                for w in workspaces
            ],
        },
    )


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
    payload: WorkspaceCreateRequest,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_analyst),
):
    team = db.query(Team).filter(Team.id == payload.team_id).first()
    if not team:
        raise HTTPException(404, "Team not found")

    ws = Workspace(name=payload.name, team_id=payload.team_id)
    db.add(ws)
    db.commit()
    db.refresh(ws)

    _audit(
        request,
        db,
        "workspace_create",
        resource_type="workspace",
        resource_id=str(ws.id),
        details={"team_id": payload.team_id, "name": payload.name},
    )

    return success_response(request, status="done", result={"id": ws.id, "name": ws.name, "team_id": ws.team_id})


@router.get("/workspaces")
async def list_workspaces(
    request: Request,
    team_id: Optional[int] = None,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_analyst),
):
    query = db.query(Workspace)
    if team_id is not None:
        query = query.filter(Workspace.team_id == team_id)
    workspaces = query.order_by(Workspace.created_at.desc()).all()

    return success_response(
        request,
        status="done",
        result={
            "workspaces": [
                {
                    "id": w.id,
                    "name": w.name,
                    "team_id": w.team_id,
                    "created_at": w.created_at.isoformat() if w.created_at else None,
                    "cases_count": len(w.cases),
                }
                for w in workspaces
            ]
        },
    )
