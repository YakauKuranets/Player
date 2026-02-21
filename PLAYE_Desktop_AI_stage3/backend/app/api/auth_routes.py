"""Auth endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.response import success_response
from app.api.routes import auth_required
from app.audit.enterprise import log_action
from app.auth.service import create_user, login, revoke_session
from app.config import settings
from app.db.database import SessionLocal
from app.db.models import User, UserRole

router = APIRouter(prefix="/auth", tags=["auth"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()




def _audit_auth(request: Request, db: Session, action: str, details: dict | None = None, status: str = "success") -> None:
    payload = getattr(request.state, "jwt_payload", {}) or {}
    user_id = payload.get("sub")
    team_id = payload.get("team_id")
    log_action(
        db=db,
        action=action,
        user_id=int(user_id) if str(user_id).isdigit() else None,
        team_id=int(team_id) if str(team_id).isdigit() else None,
        details=details or {},
        ip_address=request.client.host if request.client else None,
        request_id=getattr(request.state, "request_id", None),
        status=status,
    )

class RegisterRequest(BaseModel):
    email: str
    username: str
    password: str
    role: str = "analyst"


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/register")
async def register(request: Request, payload: RegisterRequest, db: Session = Depends(get_db)):
    users_count = db.query(User).count()
    if users_count > 0:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(403, "Only admin can register new users")
        from app.api.routes import _validate_session_payload, verify_jwt

        jwt_payload = verify_jwt(auth_header.split(" ", 1)[1].strip())
        _validate_session_payload(jwt_payload)
        if jwt_payload.get("role") != UserRole.admin.value:
            raise HTTPException(403, "Only admin can register new users")

    try:
        role = UserRole[payload.role]
    except KeyError as err:
        raise HTTPException(400, "Invalid role. Use: admin, analyst, viewer") from err

    try:
        user = create_user(db=db, email=payload.email, username=payload.username, password=payload.password, role=role)
    except Exception as exc:
        raise HTTPException(409, f"User already exists: {exc}") from exc

    _audit_auth(request, db, "register", {"email": user.email, "role": user.role.value})
    return success_response(request, status="done", result={"id": user.id, "email": user.email, "username": user.username, "role": user.role.value})


@router.post("/login")
async def auth_login(request: Request, payload: LoginRequest, db: Session = Depends(get_db)):
    ip = request.client.host if request.client else ""
    ua = request.headers.get("User-Agent", "")
    token = login(db=db, email=payload.email, password=payload.password, ip=ip, user_agent=ua, jwt_secret=settings.JWT_SECRET)
    if not token:
        _audit_auth(request, db, "login", {"email": payload.email}, status="failure")
        raise HTTPException(401, "Invalid credentials")
    _audit_auth(request, db, "login", {"email": payload.email}, status="success")
    return success_response(request, status="done", result={"token": token})


@router.post("/logout")
async def auth_logout(request: Request, auth: None = Depends(auth_required)):
    jwt_payload = getattr(request.state, "jwt_payload", {})
    session_id = jwt_payload.get("session_id")
    if session_id:
        db = SessionLocal()
        try:
            revoke_session(db, int(session_id))
            _audit_auth(request, db, "logout", {"session_id": session_id})
        finally:
            db.close()
    return success_response(request, status="done", result={"message": "Logged out"})


@router.get("/me")
async def me(request: Request, auth: None = Depends(auth_required)):
    payload = getattr(request.state, "jwt_payload", {})
    if not payload:
        raise HTTPException(401, "Not authenticated")
    return success_response(request, status="done", result=payload)
