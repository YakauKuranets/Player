"""Enterprise auth service."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.db.models import User, UserRole, UserSession

SESSION_TTL_HOURS = 8
BCRYPT_ROUNDS = 12


def _hash_password(password: str) -> str:
    try:
        import bcrypt

        return bcrypt.hashpw(password.encode(), bcrypt.gensalt(BCRYPT_ROUNDS)).decode()
    except ImportError:
        salt = secrets.token_hex(16)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260000)
        return f"pbkdf2${salt}${dk.hex()}"


def _verify_password(password: str, password_hash: str) -> bool:
    if password_hash.startswith("pbkdf2$"):
        _, salt, dk_hex = password_hash.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260000)
        return hmac.compare_digest(dk.hex(), dk_hex)
    try:
        import bcrypt

        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ImportError:
        return False


def _make_jwt(payload: dict, secret: str) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).rstrip(b"=").decode()
    sig_input = f"{header}.{body}".encode()
    sig = hmac.new(secret.encode(), sig_input, hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
    return f"{header}.{body}.{sig_b64}"


def create_user(
    db: Session,
    email: str,
    username: str,
    password: str,
    role: UserRole = UserRole.analyst,
    team_id: Optional[int] = None,
) -> User:
    user = User(
        email=email.lower().strip(),
        username=username.strip(),
        password_hash=_hash_password(password),
        role=role,
        team_id=team_id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def login(db: Session, email: str, password: str, ip: str = "", user_agent: str = "", jwt_secret: str = "") -> Optional[str]:
    user = db.query(User).filter(User.email == email.lower()).first()
    if not user or not user.is_active or not _verify_password(password, user.password_hash):
        return None

    user.last_login_at = datetime.now(timezone.utc)
    db.commit()

    token_raw = secrets.token_hex(32)
    token_hash = hashlib.sha256(token_raw.encode()).hexdigest()
    expires = datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS)
    session = UserSession(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires,
        ip_address=ip,
        user_agent=user_agent,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    payload = {
        "sub": str(user.id),
        "email": user.email,
        "username": user.username,
        "role": user.role.value,
        "team_id": user.team_id,
        "session_id": session.id,
        "exp": int(expires.timestamp()),
        "iat": int(time.time()),
    }
    return _make_jwt(payload, jwt_secret)


def revoke_session(db: Session, session_id: int):
    session = db.query(UserSession).filter(UserSession.id == session_id).first()
    if session:
        session.revoked = True
        db.commit()
