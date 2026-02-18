#!/usr/bin/env python3

from __future__ import annotations

import ast
import asyncio
from pathlib import Path


def load_cancel_fn():
    routes_path = Path(__file__).resolve().parents[1] / "backend" / "app" / "api" / "routes.py"
    source = routes_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    cancel_node = None
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "cancel_job":
            cancel_node = node
            break

    if cancel_node is None:
        raise RuntimeError("cancel_job not found")

    cancel_node.decorator_list = []
    cancel_node.args.defaults = [ast.Constant(value=None) for _ in cancel_node.args.defaults]
    module = ast.Module(body=[cancel_node], type_ignores=[])
    ast.fix_missing_locations(module)

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    namespace = {
        "AsyncResult": None,
        "success_response": lambda _req, status, result: {"status": status, "result": result},
        "HTTPException": HTTPException,
        "Depends": lambda x: x,
        "Request": object,
        "_log_enterprise_action": lambda *_args, **_kwargs: None,
    }
    exec(compile(module, str(routes_path), "exec"), namespace)
    return namespace["cancel_job"], namespace


class RevokableResult:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.revoked = False

    def revoke(self, terminate=False):
        self.revoked = True


class PlainResult:
    def __init__(self, task_id: str):
        self.task_id = task_id


async def main():
    cancel_job, ns = load_cancel_fn()

    ns["AsyncResult"] = RevokableResult
    payload = await cancel_job(request=None, task_id="t-1", auth=None)
    assert payload["result"]["status"] == "canceled"
    assert payload["result"]["is_final"] is True

    ns["AsyncResult"] = PlainResult
    payload = await cancel_job(request=None, task_id="t-2", auth=None)
    assert payload["result"]["status"] == "cancel-unsupported"
    assert payload["result"]["is_final"] is False
    assert payload["result"]["poll_after_ms"] == 1000

    print("[test-cancel-route] passed")


asyncio.run(main())
