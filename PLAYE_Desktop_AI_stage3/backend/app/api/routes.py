"""API routes for PLAYE PhotoLab backend."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.api.response import success_response
from app.config import settings
from app.db.database import SessionLocal
from app.db.models import UserSession
from app.audit.enterprise import log_action
from app.queue.gpu_router import gpu_router
from app.queue.tasks import (
    batch_denoise_task,
    batch_face_enhance_task,
    batch_upscale_task,
    denoise_task,
    detect_faces_task,
    detect_objects_task,
    face_enhance_task,
    upscale_task,
    video_scene_detect_task,
    video_temporal_denoise_task,
)

try:
    from celery.result import AsyncResult  # type: ignore
except ImportError:  # pragma: no cover
    class AsyncResult:  # type: ignore
        def __init__(self, task_id: str):
            self.id = task_id
            self.state = "SUCCESS"
            self.result = None


RATE_LIMIT = 1.0
_request_times: Dict[str, float] = {}
logger = logging.getLogger(__name__)
router = APIRouter()


class JobSubmitRequest(BaseModel):
    operation: str = Field(...)
    image_base64: str = Field(...)
    params: Dict[str, Any] = Field(default_factory=dict)


PRESET_DEFAULTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "forensic_safe": {
        "upscale": {"factor": 2},
        "denoise": {"level": "light"},
        "detect_objects": {"scene_threshold": 18.0, "temporal_window": 5},
    },
    "balanced": {
        "upscale": {"factor": 4},
        "denoise": {"level": "medium"},
        "detect_objects": {"scene_threshold": 28.0, "temporal_window": 3},
    },
    "presentation": {
        "upscale": {"factor": 8},
        "denoise": {"level": "heavy"},
        "detect_objects": {"scene_threshold": 45.0, "temporal_window": 2},
    },
}


def normalize_job_params(operation: str, raw_params: Dict[str, Any]) -> Tuple[str, List[Any], Dict[str, Any]]:
    params = dict(raw_params or {})
    normalized: Dict[str, Any] = {}

    preset = params.get("preset")
    if preset is not None:
        preset = str(preset).strip().lower()
        if preset not in PRESET_DEFAULTS:
            raise HTTPException(status_code=400, detail="preset must be one of forensic_safe, balanced, presentation")
        normalized["preset"] = preset

    merged_params = dict(PRESET_DEFAULTS.get(preset, {}).get(operation, {}))
    merged_params.update({k: v for k, v in params.items() if k != "preset"})

    if operation == "face_enhance":
        return operation, [], normalized

    if operation == "upscale":
        try:
            factor = int(merged_params.get("factor", 2))
        except Exception as err:
            raise HTTPException(status_code=400, detail="upscale.factor must be an integer") from err
        if factor not in {2, 4, 8}:
            raise HTTPException(status_code=400, detail="upscale.factor must be one of 2, 4, 8")
        normalized["factor"] = factor
        return operation, [factor], normalized

    if operation == "denoise":
        level = str(merged_params.get("level", "light")).lower().strip()
        if level not in {"light", "medium", "heavy"}:
            raise HTTPException(status_code=400, detail="denoise.level must be one of light, medium, heavy")
        normalized["level"] = level
        return operation, [level], normalized

    if operation == "detect_faces":
        return operation, [], normalized

    if operation == "detect_objects":
        scene_threshold = merged_params.get("scene_threshold", None)
        temporal_window = merged_params.get("temporal_window", None)

        if scene_threshold is not None:
            try:
                scene_threshold = float(scene_threshold)
            except Exception as err:
                raise HTTPException(status_code=400, detail="detect_objects.scene_threshold must be numeric") from err
            if scene_threshold < 0 or scene_threshold > 100:
                raise HTTPException(status_code=400, detail="detect_objects.scene_threshold must be between 0 and 100")
            normalized["scene_threshold"] = scene_threshold

        if temporal_window is not None:
            try:
                temporal_window = int(temporal_window)
            except Exception as err:
                raise HTTPException(status_code=400, detail="detect_objects.temporal_window must be integer") from err
            if temporal_window < 1 or temporal_window > 12:
                raise HTTPException(status_code=400, detail="detect_objects.temporal_window must be between 1 and 12")
            normalized["temporal_window"] = temporal_window

        return operation, [normalized.get("scene_threshold"), normalized.get("temporal_window")], normalized

    raise HTTPException(status_code=400, detail=f"Unsupported operation: {operation}")


def _base64url_decode(input_str: str) -> bytes:
    padding = "=" * (-len(input_str) % 4)
    return base64.urlsafe_b64decode(input_str + padding)


def verify_jwt(token: str) -> dict:
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
    except ValueError as err:
        raise ValueError("Invalid token format") from err

    payload_bytes = _base64url_decode(payload_b64)
    signature = _base64url_decode(signature_b64)
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    secret = settings.JWT_SECRET.encode("utf-8")
    expected_sig = hmac.new(secret, signing_input, hashlib.sha256).digest()

    if not hmac.compare_digest(expected_sig, signature):
        raise ValueError("Invalid token signature")

    return json.loads(payload_bytes.decode("utf-8"))


def _validate_session_payload(payload: dict) -> None:
    exp = payload.get("exp")
    if exp is not None:
        try:
            exp_ts = int(exp)
        except Exception as err:
            raise HTTPException(status_code=401, detail="Invalid token exp") from err
        if exp_ts < int(time.time()):
            raise HTTPException(status_code=401, detail="Token expired")

    session_id = payload.get("session_id")
    if session_id is None:
        return

    db = SessionLocal()
    try:
        session = db.query(UserSession).filter(UserSession.id == int(session_id)).first()
        if session is None:
            raise HTTPException(status_code=401, detail="Session not found")
        if session.revoked:
            raise HTTPException(status_code=401, detail="Session revoked")
        if session.expires_at and session.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
            raise HTTPException(status_code=401, detail="Session expired")
    finally:
        db.close()


async def auth_required(request: Request) -> None:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = auth_header.split(" ", 1)[1].strip()
    try:
        payload = verify_jwt(token)
        _validate_session_payload(payload)
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=401, detail="Invalid token") from err

    request.state.jwt_payload = payload




def _log_enterprise_action(request: Request, action: str, details: dict | None = None, status: str = "success") -> None:
    payload = getattr(request.state, "jwt_payload", {}) or {}
    db = SessionLocal()
    try:
        user_id = payload.get("sub")
        team_id = payload.get("team_id")
        log_action(
            db=db,
            action=action,
            user_id=int(user_id) if str(user_id).isdigit() else None,
            team_id=int(team_id) if str(team_id).isdigit() else None,
            details=details or {},
            ip_address=request.client.host if request.client else None,
            request_id=getattr(request.state, "request_id", None),
            status=status,
        )
    except Exception:
        pass
    finally:
        db.close()

async def rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "anonymous"
    now = time.time()
    last_time = _request_times.get(ip)
    if last_time is not None and (now - last_time) < RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too Many Requests")
    _request_times[ip] = now


def _to_task_status(async_result: AsyncResult, task_id: str) -> Dict[str, Any]:
    state = str(async_result.state or "PENDING").upper()
    status_map = {
        "PENDING": "pending",
        "RECEIVED": "queued",
        "STARTED": "running",
        "RETRY": "retry",
        "PROGRESS": "running",
        "SUCCESS": "done",
        "FAILURE": "failed",
        "REVOKED": "canceled",
    }
    default_progress = {
        "PENDING": 0,
        "RECEIVED": 5,
        "STARTED": 15,
        "RETRY": 10,
        "PROGRESS": 25,
        "SUCCESS": 100,
        "FAILURE": 100,
        "REVOKED": 100,
    }

    response: Dict[str, Any] = {
        "task_id": task_id,
        "status": status_map.get(state, state.lower()),
        "raw_state": state,
        "progress": default_progress.get(state, 0),
        "result": None,
        "error": None,
        "meta": None,
        "is_final": False,
        "poll_after_ms": 600,
    }

    info = getattr(async_result, "info", None)
    if isinstance(info, dict):
        if "progress" in info:
            try:
                response["progress"] = max(0, min(100, int(info.get("progress", response["progress"]))))
            except Exception:
                pass
        response["meta"] = {
            "stage": info.get("stage"),
            "message": info.get("message"),
            "attempt": info.get("attempt"),
        }

    if state == "SUCCESS":
        response["result"] = async_result.result

    if state in {"SUCCESS", "FAILURE", "REVOKED"}:
        response["is_final"] = True
        response["poll_after_ms"] = 0

    if state == "RETRY":
        response["poll_after_ms"] = 900
    elif state in {"PENDING", "RECEIVED"}:
        response["poll_after_ms"] = 700

    if state in {"FAILURE", "REVOKED"}:
        response["error"] = str(async_result.result or info)

    return response


@router.get("/hello")
async def hello_world(request: Request):
    return success_response(request, status="done", result={"message": "Hello from PLAYE PhotoLab backend!"})


@router.post("/job/submit")
async def submit_job(
    request: Request,
    payload: JobSubmitRequest,
    auth: None = Depends(auth_required),
    limiter: None = Depends(rate_limit),
):
    operation, extra_args, normalized_params = normalize_job_params(payload.operation, payload.params)
    task_map = {
        "face_enhance": face_enhance_task,
        "upscale": upscale_task,
        "denoise": denoise_task,
        "detect_faces": detect_faces_task,
        "detect_objects": detect_objects_task,
    }

    try:
        image_bytes = base64.b64decode(payload.image_base64)
    except Exception as err:
        raise HTTPException(status_code=400, detail="Invalid image_base64 payload") from err

    task_fn = task_map[operation]
    if hasattr(task_fn, "apply_async"):
        queue = gpu_router.get_best_queue()
        task_result = task_fn.apply_async(
            args=[image_bytes, *extra_args],
            queue=queue,
        )
        _log_enterprise_action(
            request,
            "job_submit",
            {"operation": operation, "mode": "async", "params": normalized_params, "queue": queue},
        )
        return success_response(
            request,
            status="queued",
            status_code=202,
            result={
                "task_id": task_result.id,
                "operation": operation,
                "status": "pending",
                "params": normalized_params,
                "queue": queue,
            },
        )

    output = task_fn(image_bytes, *extra_args)
    _log_enterprise_action(request, "job_submit", {"operation": operation, "mode": "sync", "params": normalized_params})
    return success_response(
        request,
        status="done",
        result={
            "task_id": f"sync-{int(time.time() * 1000)}",
            "operation": operation,
            "status": "done",
            "params": normalized_params,
            "result": output,
        },
    )


@router.post("/job/video/submit")
async def submit_video_job(
    request: Request,
    file: UploadFile = File(...),
    operation: str = Form("temporal_denoise"),
    fps: float = Form(1.0),
    scene_threshold: float = Form(28.0),
    temporal_window: int = Form(3),
    auth: None = Depends(auth_required),
    limiter: None = Depends(rate_limit),
):
    video_bytes = await file.read()

    if operation == "temporal_denoise":
        task_result = video_temporal_denoise_task.delay(video_bytes, fps)
    elif operation == "scene_detect":
        task_result = video_scene_detect_task.delay(video_bytes, scene_threshold, temporal_window)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown video operation: {operation}")

    _log_enterprise_action(request, "video_job_submit", {"operation": operation, "fps": fps, "filename": file.filename})
    return success_response(
        request,
        status="queued",
        status_code=202,
        result={"task_id": task_result.id, "operation": operation, "filename": file.filename},
    )


@router.get("/job/{task_id}/status")
async def get_job_status_v2(
    request: Request,
    task_id: str,
    auth: None = Depends(auth_required),
):
    result = AsyncResult(task_id)
    return success_response(request, status="done", result=_to_task_status(result, task_id))


@router.post("/job/{task_id}/cancel")
async def cancel_job(
    request: Request,
    task_id: str,
    auth: None = Depends(auth_required),
):
    result = AsyncResult(task_id)
    if not hasattr(result, "revoke"):
        _log_enterprise_action(request, "job_cancel", {"task_id": task_id, "status": "unsupported"})
        return success_response(
            request,
            status="done",
            result={"task_id": task_id, "status": "cancel-unsupported", "is_final": False, "poll_after_ms": 1000},
        )

    try:
        result.revoke(terminate=False)
    except Exception as err:
        raise HTTPException(status_code=500, detail="Unable to cancel task") from err

    _log_enterprise_action(request, "job_cancel", {"task_id": task_id, "status": "canceled"})
    return success_response(
        request,
        status="done",
        result={"task_id": task_id, "status": "canceled", "is_final": True, "poll_after_ms": 0},
    )


@router.get("/system/gpu")
async def gpu_status(request: Request, auth: None = Depends(auth_required)):
    return success_response(
        request,
        status="done",
        result={"gpus": gpu_router.status(), "strategy": "least_loaded"},
    )


@router.post("/job/batch/submit")
async def submit_batch_job(
    request: Request,
    operation: str = Form(...),
    files: list[UploadFile] = File(...),
    factor: int = Form(2),
    level: str = Form("medium"),
    auth: None = Depends(auth_required),
    limiter: None = Depends(rate_limit),
):
    images_b64 = [base64.b64encode(await f.read()).decode() for f in files]
    queue = gpu_router.get_best_queue()

    batch_task_map = {
        "upscale": (batch_upscale_task, [images_b64, factor]),
        "denoise": (batch_denoise_task, [images_b64, level]),
        "face_enhance": (batch_face_enhance_task, [images_b64]),
    }
    task_entry = batch_task_map.get(operation)
    if task_entry is None:
        raise HTTPException(400, f"Unsupported batch operation: {operation}")

    task_fn, task_args = task_entry
    if hasattr(task_fn, "apply_async"):
        task_result = task_fn.apply_async(args=task_args, queue=queue)
        task_id = task_result.id
    else:
        _ = task_fn(*task_args)
        task_id = f"sync-{int(time.time() * 1000)}"

    _log_enterprise_action(request, "batch_job_submit", {"operation": operation, "batch_size": len(files), "queue": queue})
    return success_response(
        request,
        status="queued",
        status_code=202,
        result={
            "task_id": task_id,
            "operation": operation,
            "batch_size": len(files),
            "queue": queue,
        },
    )


# Backward-compatible direct upload endpoints
@router.post("/ai/face-enhance")
async def api_face_enhance(
    request: Request,
    file: UploadFile = File(...),
    auth: None = Depends(auth_required),
    limiter: None = Depends(rate_limit),
):
    image_bytes = await file.read()
    if hasattr(face_enhance_task, "delay"):
        task_result = face_enhance_task.delay(image_bytes)
        return success_response(
            request,
            status="queued",
            result={"task_id": task_result.id, "filename": file.filename},
            status_code=202,
        )

    result_data = face_enhance_task(image_bytes)
    return success_response(request, status="done", result={"result": result_data, "filename": file.filename})


@router.post("/ai/upscale")
async def api_upscale(
    request: Request,
    file: UploadFile = File(...),
    factor: int = Form(2),
    auth: None = Depends(auth_required),
    limiter: None = Depends(rate_limit),
):
    image_bytes = await file.read()
    if hasattr(upscale_task, "delay"):
        task_result = upscale_task.delay(image_bytes, factor)
        return success_response(
            request,
            status="queued",
            result={"task_id": task_result.id, "filename": file.filename, "factor": factor},
            status_code=202,
        )

    result_data = upscale_task(image_bytes, factor)
    return success_response(request, status="done", result={"result": result_data, "filename": file.filename, "factor": factor})


@router.post("/ai/denoise")
async def api_denoise(
    request: Request,
    file: UploadFile = File(...),
    level: str = Form("light"),
    auth: None = Depends(auth_required),
    limiter: None = Depends(rate_limit),
):
    image_bytes = await file.read()
    if hasattr(denoise_task, "delay"):
        task_result = denoise_task.delay(image_bytes, level)
        return success_response(
            request,
            status="queued",
            result={"task_id": task_result.id, "filename": file.filename, "level": level},
            status_code=202,
        )

    result_data = denoise_task(image_bytes, level)
    return success_response(request, status="done", result={"result": result_data, "filename": file.filename, "level": level})


@router.post("/ai/detect-faces")
async def api_detect_faces(
    request: Request,
    file: UploadFile = File(...),
    auth: None = Depends(auth_required),
    limiter: None = Depends(rate_limit),
):
    image_bytes = await file.read()
    if hasattr(detect_faces_task, "delay"):
        task_result = detect_faces_task.delay(image_bytes)
        return success_response(
            request,
            status="queued",
            result={"task_id": task_result.id, "filename": file.filename},
            status_code=202,
        )

    faces = detect_faces_task(image_bytes)
    return success_response(request, status="done", result={"faces": faces, "filename": file.filename})


@router.post("/ai/detect-objects")
async def api_detect_objects(
    request: Request,
    file: UploadFile = File(...),
    auth: None = Depends(auth_required),
    limiter: None = Depends(rate_limit),
):
    image_bytes = await file.read()
    if hasattr(detect_objects_task, "delay"):
        task_result = detect_objects_task.delay(image_bytes)
        return success_response(
            request,
            status="queued",
            result={"task_id": task_result.id, "filename": file.filename},
            status_code=202,
        )

    objects = detect_objects_task(image_bytes)
    return success_response(request, status="done", result={"objects": objects, "filename": file.filename})


@router.get("/jobs/{task_id}")
async def get_job_status_alias(
    request: Request,
    task_id: str,
    auth: None = Depends(auth_required),
):
    result = AsyncResult(task_id)
    return success_response(request, status="done", result=_to_task_status(result, task_id))
