"""Enterprise audit trail persistence and analytics helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func

logger = logging.getLogger(__name__)


def _to_json(value: Optional[dict]) -> Optional[str]:
    return json.dumps(value, ensure_ascii=False) if value else None


def _to_dt(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


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
    """Write a single enterprise audit log row."""
    from app.db.models import EnterpriseAuditLog

    try:
        entry = EnterpriseAuditLog(
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            team_id=team_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=_to_json(details),
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
    status: Optional[str] = None,
    resource_type: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Fetch audit rows with filtering and pagination."""
    from app.db.models import EnterpriseAuditLog

    query = db.query(EnterpriseAuditLog)

    if team_id is not None:
        query = query.filter(EnterpriseAuditLog.team_id == team_id)
    if user_id is not None:
        query = query.filter(EnterpriseAuditLog.user_id == user_id)
    if action is not None:
        query = query.filter(EnterpriseAuditLog.action == action)
    if status is not None:
        query = query.filter(EnterpriseAuditLog.status == status)
    if resource_type is not None:
        query = query.filter(EnterpriseAuditLog.resource_type == resource_type)

    dt_since = _to_dt(since)
    dt_until = _to_dt(until)
    if dt_since is not None:
        query = query.filter(EnterpriseAuditLog.timestamp >= dt_since)
    if dt_until is not None:
        query = query.filter(EnterpriseAuditLog.timestamp <= dt_until)

    entries = query.order_by(EnterpriseAuditLog.timestamp.desc()).limit(max(1, min(1000, limit))).offset(max(0, offset)).all()

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


def get_action_breakdown(db, *, team_id: Optional[int] = None, days: int = 7) -> list[dict[str, Any]]:
    """Aggregate counts by action for a recent period."""
    from app.db.models import EnterpriseAuditLog

    since = datetime.now(timezone.utc) - timedelta(days=max(1, min(days, 365)))

    query = db.query(
        EnterpriseAuditLog.action.label("action"),
        func.count(EnterpriseAuditLog.id).label("count"),
    ).filter(EnterpriseAuditLog.timestamp >= since)

    if team_id is not None:
        query = query.filter(EnterpriseAuditLog.team_id == team_id)

    rows = query.group_by(EnterpriseAuditLog.action).order_by(func.count(EnterpriseAuditLog.id).desc()).all()
    return [{"action": row.action, "count": int(row.count)} for row in rows]


def get_activity_timeseries(
    db,
    *,
    team_id: Optional[int] = None,
    days: int = 14,
    bucket_hours: int = 6,
) -> list[dict[str, Any]]:
    """Build a time-bucketed activity series for dashboard/reporting charts."""
    from app.db.models import EnterpriseAuditLog

    safe_days = max(1, min(days, 365))
    safe_bucket = max(1, min(bucket_hours, 24))
    since = datetime.now(timezone.utc) - timedelta(days=safe_days)

    query = db.query(EnterpriseAuditLog).filter(EnterpriseAuditLog.timestamp >= since)
    if team_id is not None:
        query = query.filter(EnterpriseAuditLog.team_id == team_id)

    rows = query.order_by(EnterpriseAuditLog.timestamp.asc()).all()

    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        ts = row.timestamp
        if ts is None:
            continue
        dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        floored_hour = (dt.hour // safe_bucket) * safe_bucket
        bucket_start = dt.replace(hour=floored_hour, minute=0, second=0, microsecond=0)
        key = bucket_start.isoformat()

        if key not in buckets:
            buckets[key] = {
                "bucket_start": key,
                "bucket_hours": safe_bucket,
                "total": 0,
                "success": 0,
                "failure": 0,
            }

        buckets[key]["total"] += 1
        if row.status == "failure":
            buckets[key]["failure"] += 1
        else:
            buckets[key]["success"] += 1

    return [buckets[k] for k in sorted(buckets.keys())]


def get_user_activity_summary(db, *, user_id: int, days: int = 30) -> dict[str, Any]:
    """Get summary stats and recent events for a user."""
    from app.db.models import EnterpriseAuditLog

    since = datetime.now(timezone.utc) - timedelta(days=max(1, min(days, 365)))
    base = db.query(EnterpriseAuditLog).filter(
        EnterpriseAuditLog.user_id == user_id,
        EnterpriseAuditLog.timestamp >= since,
    )

    total = int(base.count())
    success = int(base.filter(EnterpriseAuditLog.status == "success").count())
    failure = int(base.filter(EnterpriseAuditLog.status == "failure").count())

    latest = (
        base.order_by(EnterpriseAuditLog.timestamp.desc())
        .limit(20)
        .all()
    )

    return {
        "user_id": user_id,
        "window_days": days,
        "events_total": total,
        "events_success": success,
        "events_failure": failure,
        "recent_events": [
            {
                "id": e.id,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                "action": e.action,
                "status": e.status,
                "resource_type": e.resource_type,
                "resource_id": e.resource_id,
                "request_id": e.request_id,
            }
            for e in latest
        ],
    }


def get_dashboard_summary(db, *, team_id: Optional[int] = None, days: int = 7) -> dict[str, Any]:
    """Return compact dashboard summary for enterprise panel."""
    from app.db.models import EnterpriseAuditLog, Team, User, Workspace

    since = datetime.now(timezone.utc) - timedelta(days=max(1, min(days, 365)))

    users_q = db.query(User)
    workspaces_q = db.query(Workspace)
    teams_q = db.query(Team)

    if team_id is not None:
        users_q = users_q.filter(User.team_id == team_id)
        workspaces_q = workspaces_q.filter(Workspace.team_id == team_id)
        teams_q = teams_q.filter(Team.id == team_id)

    events_q = db.query(EnterpriseAuditLog).filter(EnterpriseAuditLog.timestamp >= since)
    if team_id is not None:
        events_q = events_q.filter(EnterpriseAuditLog.team_id == team_id)

    return {
        "window_days": days,
        "team_id": team_id,
        "teams_total": int(teams_q.count()),
        "users_total": int(users_q.count()),
        "users_active": int(users_q.filter(User.is_active.is_(True)).count()),
        "workspaces_total": int(workspaces_q.count()),
        "audit_events_total": int(events_q.count()),
        "audit_events_failure": int(events_q.filter(EnterpriseAuditLog.status == "failure").count()),
        "top_actions": get_action_breakdown(db, team_id=team_id, days=days)[:10],
    }
