"""Enterprise audit trail in DB."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def log_action(
    db,
    action: str,
    user_id: Optional[int] = None,
    team_id: Optional[int] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    details: Optional[dict] = None,
    ip_address: Optional[str] = None,
    request_id: Optional[str] = None,
    status: str = "success",
):
    from app.db.models import EnterpriseAuditLog

    try:
        entry = EnterpriseAuditLog(
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            team_id=team_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=json.dumps(details, ensure_ascii=False) if details else None,
            ip_address=ip_address,
            request_id=request_id,
            status=status,
        )
        db.add(entry)
        db.commit()
    except Exception as exc:
        logger.error("Failed to write enterprise audit log: %s", exc)


def get_audit_log(
    db,
    team_id: Optional[int] = None,
    user_id: Optional[int] = None,
    action: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    from app.db.models import EnterpriseAuditLog

    query = db.query(EnterpriseAuditLog)
    if team_id is not None:
        query = query.filter(EnterpriseAuditLog.team_id == team_id)
    if user_id is not None:
        query = query.filter(EnterpriseAuditLog.user_id == user_id)
    if action is not None:
        query = query.filter(EnterpriseAuditLog.action == action)

    entries = query.order_by(EnterpriseAuditLog.timestamp.desc()).limit(limit).offset(offset).all()
    return [
        {
            "id": e.id,
            "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            "action": e.action,
            "user_id": e.user_id,
            "team_id": e.team_id,
            "resource_type": e.resource_type,
            "resource_id": e.resource_id,
            "details": json.loads(e.details) if e.details else None,
            "ip_address": e.ip_address,
            "request_id": e.request_id,
            "status": e.status,
        }
        for e in entries
    ]
