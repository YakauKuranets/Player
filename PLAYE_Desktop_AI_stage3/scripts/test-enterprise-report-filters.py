#!/usr/bin/env python3

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def load_helpers():
    reports_path = Path(__file__).resolve().parents[1] / "backend" / "app" / "api" / "enterprise_reports.py"
    source = reports_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    wanted = {"_jwt_payload", "_resolve_team_scope", "_parse_iso_utc", "_normalize_audit_filters"}
    selected = []
    for candidate in tree.body:
        if isinstance(candidate, ast.FunctionDef) and candidate.name in wanted:
            selected.append(candidate)

    found = {node.name for node in selected}
    missing = wanted - found
    if missing:
        raise RuntimeError(f"Missing enterprise report helper(s): {sorted(missing)}")

    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {
        "Any": object,
        "Request": object,
        "HTTPException": HTTPException,
        "datetime": datetime,
        "timezone": timezone,
    }
    exec(compile(module, str(reports_path), "exec"), namespace)

    return (
        namespace["_parse_iso_utc"],
        namespace["_resolve_team_scope"],
        namespace["_normalize_audit_filters"],
    )


def assert_equal(actual, expected, msg):
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected!r}, got {actual!r}")


def assert_raises_422(fn, *, contains: str):
    try:
        fn()
    except HTTPException as exc:
        assert_equal(exc.status_code, 422, "HTTP status")
        if contains not in exc.detail:
            raise AssertionError(f"error detail mismatch: expected substring {contains!r}, got {exc.detail!r}")
        return

    raise AssertionError("Expected HTTPException(422)")


def make_request(payload: dict):
    return SimpleNamespace(state=SimpleNamespace(jwt_payload=payload))


parse_iso_utc, resolve_team_scope, normalize_audit_filters = load_helpers()

# ISO parser behavior
parsed = parse_iso_utc("2025-02-18T10:20:30+03:00", "since")
assert_equal(parsed, "2025-02-18T07:20:30+00:00", "timezone conversion")

parsed_naive = parse_iso_utc("2025-02-18T10:20:30", "until")
assert_equal(parsed_naive, "2025-02-18T10:20:30+00:00", "naive datetime normalized to UTC")

assert_equal(parse_iso_utc(None, "since"), None, "None passthrough")
assert_raises_422(lambda: parse_iso_utc("not-a-date", "since"), contains="Invalid since datetime")

# Team scope behavior
admin_request = make_request({"role": "admin", "team_id": 22})
assert_equal(resolve_team_scope(admin_request, 7), 7, "admin can request arbitrary scope")

analyst_request = make_request({"role": "analyst", "team_id": "5"})
assert_equal(resolve_team_scope(analyst_request, 9), 5, "analyst forced to token team")

analyst_bad_team_request = make_request({"role": "analyst", "team_id": "abc"})
assert_equal(resolve_team_scope(analyst_bad_team_request, 9), None, "invalid analyst team falls back to None")

# Filter normalization and date ordering check
normalized = normalize_audit_filters(
    admin_request,
    team_id=2,
    user_id=3,
    action="report",
    status="success",
    resource_type="report",
    since="2025-02-01T00:00:00Z",
    until="2025-02-02T00:00:00Z",
)
assert_equal(normalized["team_id"], 2, "team scope from request")
assert_equal(normalized["user_id"], 3, "user_id passthrough")
assert_equal(normalized["since"], "2025-02-01T00:00:00+00:00", "since normalized")
assert_equal(normalized["until"], "2025-02-02T00:00:00+00:00", "until normalized")

assert_raises_422(
    lambda: normalize_audit_filters(
        admin_request,
        team_id=None,
        user_id=None,
        action=None,
        status=None,
        resource_type=None,
        since="2025-02-03T00:00:00Z",
        until="2025-02-02T00:00:00Z",
    ),
    contains="since must be earlier than until",
)

print("[test-enterprise-report-filters] passed")
