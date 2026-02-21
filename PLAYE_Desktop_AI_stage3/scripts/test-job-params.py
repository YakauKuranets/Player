#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def load_normalize_fn():
    routes_path = Path(__file__).resolve().parents[1] / "backend" / "app" / "api" / "routes.py"
    source = routes_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    selected_nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "PRESET_DEFAULTS" in targets:
                selected_nodes.append(node)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "PRESET_DEFAULTS":
            selected_nodes.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "normalize_job_params":
            selected_nodes.append(node)
            break

    if not selected_nodes:
        raise RuntimeError("normalize_job_params not found")

    module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {"HTTPException": HTTPException, "Dict": dict, "Any": object, "Tuple": tuple, "List": list}
    exec(compile(module, str(routes_path), "exec"), namespace)
    return namespace["normalize_job_params"]


def assert_equal(actual, expected, msg):
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected!r}, got {actual!r}")


def assert_raises(fn, contains):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        detail = getattr(exc, "detail", str(exc))
        if contains not in str(detail):
            raise AssertionError(f"error detail mismatch: {detail!r}") from exc
        return
    raise AssertionError("expected exception was not raised")


normalize_job_params = load_normalize_fn()

op, args, meta = normalize_job_params("upscale", {"factor": "4"})
assert_equal(op, "upscale", "operation normalize")
assert_equal(args, [4], "upscale args")
assert_equal(meta["factor"], 4, "upscale meta")

op, args, meta = normalize_job_params("denoise", {"level": "heavy"})
assert_equal(op, "denoise", "denoise op")
assert_equal(args, ["heavy"], "denoise args")
assert_equal(meta["level"], "heavy", "denoise meta")

op, args, meta = normalize_job_params("detect_objects", {})
assert_equal(args, [None, None], "detect_objects args")
assert_equal(meta, {}, "detect_objects meta empty")

op, args, meta = normalize_job_params("detect_objects", {"scene_threshold": 32.5, "temporal_window": 4})
assert_equal(args, [32.5, 4], "detect_objects args with meta")
assert_equal(meta["scene_threshold"], 32.5, "scene_threshold meta")
assert_equal(meta["temporal_window"], 4, "temporal_window meta")

op, args, meta = normalize_job_params("denoise", {"preset": "forensic_safe"})
assert_equal(args, ["light"], "denoise preset default")
assert_equal(meta["preset"], "forensic_safe", "preset value")

op, args, meta = normalize_job_params("upscale", {"preset": "presentation"})
assert_equal(args, [8], "upscale presentation default")

assert_raises(lambda: normalize_job_params("upscale", {"factor": 3}), "one of 2, 4, 8")
assert_raises(lambda: normalize_job_params("denoise", {"level": "x"}), "one of light, medium, heavy")
assert_raises(lambda: normalize_job_params("detect_objects", {"scene_threshold": "abc"}), "scene_threshold must be numeric")
assert_raises(lambda: normalize_job_params("detect_objects", {"scene_threshold": 120}), "between 0 and 100")
assert_raises(lambda: normalize_job_params("detect_objects", {"temporal_window": 0}), "between 1 and 12")
assert_raises(lambda: normalize_job_params("denoise", {"preset": "random"}), "preset must be one of")
assert_raises(lambda: normalize_job_params("abc", {}), "Unsupported operation")

print("[test-job-params] passed")
