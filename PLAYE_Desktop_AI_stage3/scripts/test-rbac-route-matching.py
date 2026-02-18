#!/usr/bin/env python3

from __future__ import annotations

import ast
import re
from enum import Enum
from functools import lru_cache
from pathlib import Path


class UserRole(Enum):
    viewer = "viewer"
    analyst = "analyst"
    admin = "admin"


def load_endpoint_matcher():
    rbac_path = Path(__file__).resolve().parents[1] / "backend" / "app" / "api" / "rbac.py"
    source = rbac_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    wanted_assigns = {"ENDPOINT_ROLES"}
    wanted_funcs = {"_compiled_endpoint_patterns", "endpoint_required_role"}

    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target_ids = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if target_ids & wanted_assigns:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_funcs:
            selected.append(node)

    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "UserRole": UserRole,
        "lru_cache": lru_cache,
        "re": re,
    }
    exec(compile(module, str(rbac_path), "exec"), namespace)
    return namespace["endpoint_required_role"]


def assert_equal(actual, expected, msg):
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected!r}, got {actual!r}")


endpoint_required_role = load_endpoint_matcher()

# exact match
assert_equal(endpoint_required_role("GET", "/api/enterprise/reports/manifest"), "analyst", "manifest role")

# trailing slash normalization
assert_equal(endpoint_required_role("GET", "/api/enterprise/reports/manifest/"), "analyst", "manifest trailing slash")

assert_equal(endpoint_required_role("GET", "/api/enterprise/reports/manifest.csv"), "analyst", "manifest csv role")

# templated route matching
assert_equal(
    endpoint_required_role("GET", "/api/enterprise/reports/users/42/activity"),
    "admin",
    "templated user activity role",
)
assert_equal(
    endpoint_required_role("PATCH", "/api/enterprise/users/77/role"),
    "admin",
    "templated enterprise user role patch",
)

# known analyst route
assert_equal(endpoint_required_role("GET", "/api/system/gpu"), "analyst", "gpu endpoint role")

# unknown route should stay unrestricted in middleware map
assert_equal(endpoint_required_role("DELETE", "/api/enterprise/reports/manifest"), None, "unknown method")
assert_equal(endpoint_required_role("GET", "/api/not-registered"), None, "unknown path")

print("[test-rbac-route-matching] passed")
