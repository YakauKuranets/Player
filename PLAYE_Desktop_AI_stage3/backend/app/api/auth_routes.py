"""Auth endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.response import success_response
from app.api.routes import auth_required
from app.auth.service import create_user, login, revoke_session
from app.config import settings
from app.db.database import SessionLocal
from app.db.models import UserRole

router = APIRouter(prefix="/auth", tags=["auth"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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
    try:
        role = UserRole[payload.role]
    except KeyError as err:
        raise HTTPException(400, "Invalid role. Use: admin, analyst, viewer") from err

    try:
        user = create_user(db=db, email=payload.email, username=payload.username, password=payload.password, role=role)
    except Exception as exc:
        raise HTTPException(409, f"User already exists: {exc}") from exc

    return success_response(request, status="done", result={"id": user.id, "email": user.email, "username": user.username, "role": user.role.value})


@router.post("/login")
async def auth_login(request: Request, payload: LoginRequest, db: Session = Depends(get_db)):
    ip = request.client.host if request.client else ""
    ua = request.headers.get("User-Agent", "")
    token = login(db=db, email=payload.email, password=payload.password, ip=ip, user_agent=ua, jwt_secret=settings.JWT_SECRET)
    if not token:
        raise HTTPException(401, "Invalid credentials")
    return success_response(request, status="done", result={"token": token})


@router.post("/logout")
async def auth_logout(request: Request, auth: None = Depends(auth_required)):
    jwt_payload = getattr(request.state, "jwt_payload", {})
    session_id = jwt_payload.get("session_id")
    if session_id:
        db = SessionLocal()
        try:
            revoke_session(db, int(session_id))
        finally:
            db.close()
    return success_response(request, status="done", result={"message": "Logged out"})


@router.get("/me")
async def me(request: Request, auth: None = Depends(auth_required)):
    payload = getattr(request.state, "jwt_payload", {})
    if not payload:
        raise HTTPException(401, "Not authenticated")
    return success_response(request, status="done", result=payload)
