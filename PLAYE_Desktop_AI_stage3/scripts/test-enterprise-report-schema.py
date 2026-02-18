#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


REPORTS_PATH = Path(__file__).resolve().parents[1] / "backend" / "app" / "api" / "enterprise_reports.py"


class EndpointCheckError(AssertionError):
    pass


def _router_get_path(node: ast.AsyncFunctionDef) -> str | None:
    for deco in node.decorator_list:
        if not isinstance(deco, ast.Call):
            continue
        if not isinstance(deco.func, ast.Attribute):
            continue
        if not isinstance(deco.func.value, ast.Name):
            continue
        if deco.func.value.id != "router" or deco.func.attr != "get":
            continue
        if deco.args and isinstance(deco.args[0], ast.Constant) and isinstance(deco.args[0].value, str):
            return deco.args[0].value
    return None


def _returns_named_call(node: ast.AsyncFunctionDef, func_name: str) -> bool:
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Return):
            continue
        if not isinstance(candidate.value, ast.Call):
            continue
        if isinstance(candidate.value.func, ast.Name) and candidate.value.func.id == func_name:
            return True
    return False


def _assert(condition: bool, msg: str):
    if not condition:
        raise EndpointCheckError(msg)


def main() -> None:
    tree = ast.parse(REPORTS_PATH.read_text(encoding="utf-8"))

    json_routes = []
    csv_routes = []

    for node in tree.body:
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        route_path = _router_get_path(node)
        if route_path is None:
            continue
        if route_path.endswith(".csv"):
            csv_routes.append((route_path, node))
        else:
            json_routes.append((route_path, node))

    _assert(len(json_routes) >= 1, "No JSON report routes found")
    _assert(len(csv_routes) >= 1, "No CSV report routes found")

    for route_path, node in json_routes:
        _assert(
            _returns_named_call(node, "success_response"),
            f"JSON report route {route_path} must return success_response(...)",
        )

    for route_path, node in csv_routes:
        _assert(
            _returns_named_call(node, "_export_response"),
            f"CSV report route {route_path} must return _export_response(...)",
        )

    json_paths = {path for path, _ in json_routes}
    csv_paths = {path for path, _ in csv_routes}
    _assert("/manifest" in json_paths, "Manifest JSON route is required")
    _assert("/manifest.csv" in csv_paths, "Manifest CSV route is required")

    print("[test-enterprise-report-schema] passed")


if __name__ == "__main__":
    main()
