"""API routes for PLAYE PhotoLab backend."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.api.response import success_response
from app.config import settings
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
        factor = int(merged_params.get("factor", 2))
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
            scene_threshold = float(scene_threshold)
            if scene_threshold < 0 or scene_threshold > 100:
                raise HTTPException(status_code=400, detail="detect_objects.scene_threshold must be between 0 and 100")
            normalized["scene_threshold"] = scene_threshold
        if temporal_window is not None:
            temporal_window = int(temporal_window)
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


async def auth_required(request: Request) -> None:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ", 1)[1].strip()
    request.state.jwt_payload = verify_jwt(token)


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
    response: Dict[str, Any] = {
        "task_id": task_id,
        "status": status_map.get(state, state.lower()),
        "raw_state": state,
        "progress": 0,
        "result": None,
        "error": None,
        "meta": None,
        "is_final": False,
        "poll_after_ms": 600,
    }
    if state == "SUCCESS":
        response["result"] = async_result.result
    if state in {"SUCCESS", "FAILURE", "REVOKED"}:
        response["is_final"] = True
        response["poll_after_ms"] = 0
    if state in {"FAILURE", "REVOKED"}:
        response["error"] = str(async_result.result)
    return response


@router.get("/hello")
async def hello_world(request: Request):
    return success_response(request, status="done", result={"message": "Hello from PLAYE PhotoLab backend!"})


@router.post("/job/submit")
async def submit_job(request: Request, payload: JobSubmitRequest, auth: None = Depends(auth_required), limiter: None = Depends(rate_limit)):
    operation, extra_args, normalized_params = normalize_job_params(payload.operation, payload.params)
    task_map = {
        "face_enhance": face_enhance_task,
        "upscale": upscale_task,
        "denoise": denoise_task,
        "detect_faces": detect_faces_task,
        "detect_objects": detect_objects_task,
    }
    image_bytes = base64.b64decode(payload.image_base64)
    task_fn = task_map[operation]
    if hasattr(task_fn, "delay"):
        task_result = task_fn.delay(image_bytes, *extra_args)
        return success_response(request, status="queued", status_code=202, result={"task_id": task_result.id, "operation": operation, "status": "pending", "params": normalized_params})

    output = task_fn(image_bytes, *extra_args)
    return success_response(request, status="done", result={"task_id": f"sync-{int(time.time() * 1000)}", "operation": operation, "status": "done", "params": normalized_params, "result": output})


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
    return success_response(request, status="queued", status_code=202, result={"task_id": task_result.id, "operation": operation, "filename": file.filename})


@router.get("/job/{task_id}/status")
async def get_job_status_v2(request: Request, task_id: str, auth: None = Depends(auth_required)):
    result = AsyncResult(task_id)
    return success_response(request, status="done", result=_to_task_status(result, task_id))


@router.post("/job/{task_id}/cancel")
async def cancel_job(request: Request, task_id: str, auth: None = Depends(auth_required)):
    result = AsyncResult(task_id)
    if hasattr(result, "revoke"):
        result.revoke(terminate=False)
    return success_response(request, status="done", result={"task_id": task_id, "status": "canceled", "is_final": True, "poll_after_ms": 0})


@router.get("/system/gpu")
async def gpu_status(request: Request, auth: None = Depends(auth_required)):
    return success_response(request, status="done", result={"gpus": gpu_router.status(), "strategy": "least_loaded"})


@router.post("/job/batch/submit")
async def submit_batch_job(
    request: Request,
    operation: str = Form(...),
    files: list[UploadFile] = File(...),
    auth: None = Depends(auth_required),
    limiter: None = Depends(rate_limit),
):
    images_b64 = [base64.b64encode(await f.read()).decode() for f in files]
    queue = gpu_router.get_best_queue()
    batch_task_map = {
        "upscale": batch_upscale_task,
        "denoise": batch_denoise_task,
        "face_enhance": batch_face_enhance_task,
    }
    task_fn = batch_task_map.get(operation)
    if task_fn is None:
        raise HTTPException(400, f"Unsupported batch operation: {operation}")

    if hasattr(task_fn, "apply_async"):
        task_result = task_fn.apply_async(args=[images_b64], queue=queue)
    else:
        task_result = type("Result", (), {"id": f"sync-{int(time.time() * 1000)}"})
    return success_response(request, status="queued", status_code=202, result={"task_id": task_result.id, "operation": operation, "batch_size": len(files), "queue": queue})
