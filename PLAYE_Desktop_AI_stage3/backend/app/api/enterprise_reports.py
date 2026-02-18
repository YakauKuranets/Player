"""Enterprise reporting endpoints (CSV/JSON exports)."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.api.auth_routes import get_db
from app.api.rbac import require_admin, require_analyst
from app.api.response import success_response
from app.api.routes import auth_required
from app.audit.enterprise import (
    get_action_breakdown,
    get_activity_timeseries,
    get_audit_log,
    get_dashboard_summary,
    get_user_activity_summary,
    log_action,
)

router = APIRouter(prefix="/enterprise/reports", tags=["enterprise-reports"])

AUDIT_CSV_FIELDS = [
    "id",
    "timestamp",
    "action",
    "user_id",
    "team_id",
    "resource_type",
    "resource_id",
    "details",
    "ip_address",
    "request_id",
    "status",
]

USER_ACTIVITY_CSV_FIELDS = [
    "id",
    "timestamp",
    "action",
    "status",
    "resource_type",
    "resource_id",
    "request_id",
]

ACTION_BREAKDOWN_CSV_FIELDS = ["action", "count"]

TIMESERIES_CSV_FIELDS = [
    "bucket_start",
    "bucket_hours",
    "total",
    "success",
    "failure",
]

DASHBOARD_CSV_FIELDS = [
    "window_days",
    "team_id",
    "teams_total",
    "users_total",
    "users_active",
    "workspaces_total",
    "audit_events_total",
    "audit_events_failure",
]

MANIFEST_CSV_FIELDS = ["path", "format", "role", "description"]

REPORTS_MANIFEST = [
    {
        "path": "/api/enterprise/reports/audit",
        "format": "json",
        "role": "admin",
        "description": "Raw enterprise audit entries with filters and pagination",
    },
    {
        "path": "/api/enterprise/reports/audit.csv",
        "format": "csv",
        "role": "admin",
        "description": "CSV export of filtered enterprise audit entries",
    },
    {
        "path": "/api/enterprise/reports/actions",
        "format": "json",
        "role": "analyst",
        "description": "Action frequency ranking for selected team/time window",
    },
    {
        "path": "/api/enterprise/reports/actions.csv",
        "format": "csv",
        "role": "analyst",
        "description": "CSV export of action frequency ranking",
    },
    {
        "path": "/api/enterprise/reports/timeseries",
        "format": "json",
        "role": "analyst",
        "description": "Time-bucketed audit activity trend (success/failure)",
    },
    {
        "path": "/api/enterprise/reports/timeseries.csv",
        "format": "csv",
        "role": "analyst",
        "description": "CSV export of time-bucketed audit trend",
    },
    {
        "path": "/api/enterprise/reports/dashboard",
        "format": "json",
        "role": "analyst",
        "description": "Aggregated users/workspaces/audit event metrics",
    },
    {
        "path": "/api/enterprise/reports/dashboard.csv",
        "format": "csv",
        "role": "analyst",
        "description": "CSV snapshot of aggregated dashboard metrics",
    },
    {
        "path": "/api/enterprise/reports/users/{user_id}/activity",
        "format": "json",
        "role": "admin",
        "description": "Per-user audit activity summary + recent events",
    },
    {
        "path": "/api/enterprise/reports/users/{user_id}/activity.csv",
        "format": "csv",
        "role": "admin",
        "description": "CSV export of a user's recent activity events",
    },
    {
        "path": "/api/enterprise/reports/manifest",
        "format": "json",
        "role": "analyst",
        "description": "Catalog of all available enterprise report endpoints",
    },
    {
        "path": "/api/enterprise/reports/manifest.csv",
        "format": "csv",
        "role": "analyst",
        "description": "CSV export of enterprise report catalog",
    },
]


def _parse_iso_utc(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid {field_name} datetime: {value}") from exc


def _build_csv(rows: list[dict], fieldnames: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        normalized = {}
        for key in fieldnames:
            value = row.get(key)
            if isinstance(value, (dict, list)):
                normalized[key] = json.dumps(value, ensure_ascii=False)
            else:
                normalized[key] = value
        writer.writerow(normalized)
    return buf.getvalue()


def _export_response(csv_payload: str, *, filename_prefix: str) -> Response:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{filename_prefix}_{stamp}.csv"
    return Response(
        content=csv_payload,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _jwt_payload(request: Request) -> dict[str, Any]:
    payload = getattr(request.state, "jwt_payload", {}) or {}
    return payload if isinstance(payload, dict) else {}


def _resolve_team_scope(request: Request, requested_team_id: int | None) -> int | None:
    payload = _jwt_payload(request)
    role = str(payload.get("role", "viewer"))
    token_team_id = payload.get("team_id")

    if role == "analyst":
        return int(token_team_id) if str(token_team_id).isdigit() else None

    return requested_team_id


def _audit_report_event(request: Request, db: Session, action: str, details: dict | None = None) -> None:
    payload = _jwt_payload(request)
    user_id = payload.get("sub")
    team_id = payload.get("team_id")
    log_action(
        db=db,
        action=action,
        user_id=int(user_id) if str(user_id).isdigit() else None,
        team_id=int(team_id) if str(team_id).isdigit() else None,
        resource_type="report",
        details=details or {},
        ip_address=request.client.host if request.client else None,
        request_id=getattr(request.state, "request_id", None),
        status="success",
    )


def _normalize_audit_filters(
    request: Request,
    *,
    team_id: int | None,
    user_id: int | None,
    action: str | None,
    status: str | None,
    resource_type: str | None,
    since: str | None,
    until: str | None,
) -> dict[str, Any]:
    normalized_since = _parse_iso_utc(since, "since")
    normalized_until = _parse_iso_utc(until, "until")

    if normalized_since and normalized_until and normalized_since > normalized_until:
        raise HTTPException(status_code=422, detail="since must be earlier than until")

    return {
        "team_id": _resolve_team_scope(request, team_id),
        "user_id": user_id,
        "action": action,
        "status": status,
        "resource_type": resource_type,
        "since": normalized_since,
        "until": normalized_until,
    }


@router.get("/audit")
async def report_audit_json(
    request: Request,
    team_id: Optional[int] = None,
    user_id: Optional[int] = None,
    action: Optional[str] = None,
    status: Optional[str] = None,
    resource_type: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_admin),
):
    filters = _normalize_audit_filters(
        request,
        team_id=team_id,
        user_id=user_id,
        action=action,
        status=status,
        resource_type=resource_type,
        since=since,
        until=until,
    )

    entries = get_audit_log(
        db,
        team_id=filters["team_id"],
        user_id=filters["user_id"],
        action=filters["action"],
        status=filters["status"],
        resource_type=filters["resource_type"],
        since=filters["since"],
        until=filters["until"],
        limit=limit,
        offset=offset,
    )

    _audit_report_event(request, db, "report_audit_json", {**filters, "count": len(entries)})
    return success_response(
        request,
        status="done",
        result={
            "filters": filters,
            "count": len(entries),
            "entries": entries,
        },
    )


@router.get("/audit.csv")
async def report_audit_csv(
    request: Request,
    team_id: Optional[int] = None,
    user_id: Optional[int] = None,
    action: Optional[str] = None,
    status: Optional[str] = None,
    resource_type: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = Query(1000, ge=1, le=10000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_admin),
):
    filters = _normalize_audit_filters(
        request,
        team_id=team_id,
        user_id=user_id,
        action=action,
        status=status,
        resource_type=resource_type,
        since=since,
        until=until,
    )

    entries = get_audit_log(
        db,
        team_id=filters["team_id"],
        user_id=filters["user_id"],
        action=filters["action"],
        status=filters["status"],
        resource_type=filters["resource_type"],
        since=filters["since"],
        until=filters["until"],
        limit=limit,
        offset=offset,
    )

    _audit_report_event(request, db, "report_audit_csv", {**filters, "count": len(entries)})
    csv_payload = _build_csv(entries, AUDIT_CSV_FIELDS)
    return _export_response(csv_payload, filename_prefix="enterprise_audit")


@router.get("/actions")
async def report_action_breakdown(
    request: Request,
    team_id: Optional[int] = None,
    days: int = Query(14, ge=1, le=365),
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_analyst),
):
    scoped_team_id = _resolve_team_scope(request, team_id)
    actions = get_action_breakdown(db, team_id=scoped_team_id, days=days)
    _audit_report_event(request, db, "report_actions_json", {"team_id": scoped_team_id, "days": days, "count": len(actions)})
    return success_response(
        request,
        status="done",
        result={
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "team_id": scoped_team_id,
            "days": days,
            "actions": actions,
        },
    )


@router.get("/actions.csv")
async def report_action_breakdown_csv(
    request: Request,
    team_id: Optional[int] = None,
    days: int = Query(14, ge=1, le=365),
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_analyst),
):
    scoped_team_id = _resolve_team_scope(request, team_id)
    actions = get_action_breakdown(db, team_id=scoped_team_id, days=days)
    _audit_report_event(request, db, "report_actions_csv", {"team_id": scoped_team_id, "days": days, "count": len(actions)})
    csv_payload = _build_csv(actions, ACTION_BREAKDOWN_CSV_FIELDS)
    return _export_response(csv_payload, filename_prefix="enterprise_actions")




@router.get("/timeseries")
async def report_activity_timeseries(
    request: Request,
    team_id: Optional[int] = None,
    days: int = Query(14, ge=1, le=365),
    bucket_hours: int = Query(6, ge=1, le=24),
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_analyst),
):
    scoped_team_id = _resolve_team_scope(request, team_id)
    series = get_activity_timeseries(
        db,
        team_id=scoped_team_id,
        days=days,
        bucket_hours=bucket_hours,
    )
    _audit_report_event(
        request,
        db,
        "report_timeseries_json",
        {
            "team_id": scoped_team_id,
            "days": days,
            "bucket_hours": bucket_hours,
            "count": len(series),
        },
    )
    return success_response(
        request,
        status="done",
        result={
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "team_id": scoped_team_id,
            "days": days,
            "bucket_hours": bucket_hours,
            "series": series,
        },
    )


@router.get("/timeseries.csv")
async def report_activity_timeseries_csv(
    request: Request,
    team_id: Optional[int] = None,
    days: int = Query(14, ge=1, le=365),
    bucket_hours: int = Query(6, ge=1, le=24),
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_analyst),
):
    scoped_team_id = _resolve_team_scope(request, team_id)
    series = get_activity_timeseries(
        db,
        team_id=scoped_team_id,
        days=days,
        bucket_hours=bucket_hours,
    )
    _audit_report_event(
        request,
        db,
        "report_timeseries_csv",
        {
            "team_id": scoped_team_id,
            "days": days,
            "bucket_hours": bucket_hours,
            "count": len(series),
        },
    )
    csv_payload = _build_csv(series, TIMESERIES_CSV_FIELDS)
    return _export_response(csv_payload, filename_prefix="enterprise_timeseries")


@router.get("/dashboard")
async def report_dashboard(
    request: Request,
    team_id: Optional[int] = None,
    days: int = Query(14, ge=1, le=365),
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_analyst),
):
    scoped_team_id = _resolve_team_scope(request, team_id)
    summary = get_dashboard_summary(db, team_id=scoped_team_id, days=days)
    action_breakdown = get_action_breakdown(db, team_id=scoped_team_id, days=days)

    _audit_report_event(request, db, "report_dashboard", {"team_id": scoped_team_id, "days": days})
    return success_response(
        request,
        status="done",
        result={
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "actions": action_breakdown,
        },
    )


@router.get("/dashboard.csv")
async def report_dashboard_csv(
    request: Request,
    team_id: Optional[int] = None,
    days: int = Query(14, ge=1, le=365),
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_analyst),
):
    scoped_team_id = _resolve_team_scope(request, team_id)
    summary = get_dashboard_summary(db, team_id=scoped_team_id, days=days)
    _audit_report_event(request, db, "report_dashboard_csv", {"team_id": scoped_team_id, "days": days})
    csv_payload = _build_csv([summary], DASHBOARD_CSV_FIELDS)
    return _export_response(csv_payload, filename_prefix="enterprise_dashboard")


@router.get("/users/{user_id}/activity")
async def report_user_activity(
    request: Request,
    user_id: int,
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_admin),
):
    summary = get_user_activity_summary(db, user_id=user_id, days=days)
    _audit_report_event(request, db, "report_user_activity", {"user_id": user_id, "days": days})
    return success_response(request, status="done", result=summary)


@router.get("/users/{user_id}/activity.csv")
async def report_user_activity_csv(
    request: Request,
    user_id: int,
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_admin),
):
    summary = get_user_activity_summary(db, user_id=user_id, days=days)
    rows = summary.get("recent_events", [])
    _audit_report_event(request, db, "report_user_activity_csv", {"user_id": user_id, "days": days, "rows": len(rows)})

    csv_payload = _build_csv(rows, USER_ACTIVITY_CSV_FIELDS)
    return _export_response(csv_payload, filename_prefix=f"user_{user_id}_activity")


@router.get("/manifest")
async def report_manifest(
    request: Request,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_analyst),
):
    reports = list(REPORTS_MANIFEST)
    _audit_report_event(request, db, "report_manifest", {"reports": len(reports)})
    return success_response(
        request,
        status="done",
        result={
            "reports": reports,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


@router.get("/manifest.csv")
async def report_manifest_csv(
    request: Request,
    db: Session = Depends(get_db),
    auth: None = Depends(auth_required),
    rbac: None = Depends(require_analyst),
):
    reports = list(REPORTS_MANIFEST)
    _audit_report_event(request, db, "report_manifest_csv", {"reports": len(reports)})
    csv_payload = _build_csv(reports, MANIFEST_CSV_FIELDS)
    return _export_response(csv_payload, filename_prefix="enterprise_reports_manifest")
