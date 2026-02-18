#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


def load_mapper():
    routes_path = Path(__file__).resolve().parents[1] / "backend" / "app" / "api" / "routes.py"
    source = routes_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    node = None
    for candidate in tree.body:
        if isinstance(candidate, ast.FunctionDef) and candidate.name == "_to_task_status":
            node = candidate
            break

    if node is None:
        raise RuntimeError("_to_task_status not found")

    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Dict": dict, "Any": object, "AsyncResult": object}
    exec(compile(module, str(routes_path), "exec"), namespace)
    return namespace["_to_task_status"]


def assert_equal(actual, expected, msg):
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected!r}, got {actual!r}")


class DummyAsyncResult:
    def __init__(self, state, result=None, info=None):
        self.state = state
        self.result = result
        self.info = info


mapper = load_mapper()

pending = mapper(DummyAsyncResult("PENDING"), "task-1")
assert_equal(pending["status"], "pending", "pending status")
assert_equal(pending["progress"], 0, "pending progress")
assert_equal(pending["is_final"], False, "pending non-final")
assert_equal(pending["poll_after_ms"], 700, "pending poll")

received = mapper(DummyAsyncResult("RECEIVED"), "task-2")
assert_equal(received["status"], "queued", "received mapped")
assert_equal(received["poll_after_ms"], 700, "received poll")

progressed = mapper(DummyAsyncResult("PROGRESS", info={"progress": 55, "stage": "detect", "message": "running"}), "task-3")
assert_equal(progressed["status"], "running", "progress mapped")
assert_equal(progressed["progress"], 55, "progress carried")
assert_equal(progressed["meta"]["stage"], "detect", "meta stage")
assert_equal(progressed["is_final"], False, "progress non-final")

finished = mapper(DummyAsyncResult("SUCCESS", result={"ok": True}), "task-4")
assert_equal(finished["status"], "done", "success mapped")
assert_equal(finished["progress"], 100, "success progress")
assert_equal(finished["result"], {"ok": True}, "success payload")
assert_equal(finished["is_final"], True, "success final")
assert_equal(finished["poll_after_ms"], 0, "success no poll")

failed = mapper(DummyAsyncResult("FAILURE", result=RuntimeError("boom")), "task-5")
assert_equal(failed["status"], "failed", "failure mapped")
assert "boom" in failed["error"], "failure error string"
assert_equal(failed["is_final"], True, "failure final")

canceled = mapper(DummyAsyncResult("REVOKED", result="killed"), "task-6")
assert_equal(canceled["status"], "canceled", "revoked mapped")
assert_equal(canceled["is_final"], True, "revoked final")

retry = mapper(DummyAsyncResult("RETRY"), "task-7")
assert_equal(retry["status"], "retry", "retry mapped")
assert_equal(retry["poll_after_ms"], 900, "retry poll")

print("[test-task-status] passed")
