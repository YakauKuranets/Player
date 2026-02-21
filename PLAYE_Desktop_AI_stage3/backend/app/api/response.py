"""Utilities for unified API responses.

The roadmap asks for a shared response contract across endpoints. This module
provides small helpers to keep response payloads consistent.
"""

from typing import Any, Dict, Optional

from fastapi import Request
from fastapi.responses import JSONResponse


def get_request_id(request: Optional[Request]) -> Optional[str]:
    """Return request identifier stored by middleware, if present."""
    if request is None:
        return None
    return getattr(request.state, "request_id", None)


def success_response(
    request: Optional[Request],
    *,
    status: str,
    result: Optional[Dict[str, Any]] = None,
    status_code: int = 200,
) -> JSONResponse:
    """Build successful API response with the unified schema."""
    request_id = get_request_id(request)
    payload: Dict[str, Any] = {
        "request_id": request_id,
        "status": status,
        "error": None,
        "result": result or {},
    }
    response = JSONResponse(status_code=status_code, content=payload)
    if request_id:
        response.headers["X-Request-ID"] = request_id
    return response


def error_response(
    request: Optional[Request],
    *,
    error: str,
    status_code: int,
    status: str = "error",
) -> JSONResponse:
    """Build error response with the unified schema."""
    request_id = get_request_id(request)
    payload: Dict[str, Any] = {
        "request_id": request_id,
        "status": status,
        "error": error,
        "result": None,
    }
    response = JSONResponse(status_code=status_code, content=payload)
    if request_id:
        response.headers["X-Request-ID"] = request_id
    return response
