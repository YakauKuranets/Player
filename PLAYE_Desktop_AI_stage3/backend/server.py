"""
FastAPI server for the PLAYE PhotoLab desktop application.

This module exposes three AI endpoints for enhancing faces, upscaling images
and denoising pictures. Models are loaded at startup using simple wrapper
classes defined in the ``backend/models`` package. Each endpoint accepts an
uploaded file and optional parameters, passes the image through the
corresponding model and returns the processed result as a PNG stream.

The device (CPU or CUDA) is selected automatically depending on the
availability of a GPU. If a model fails to load or run, the endpoint
responds with a 500/503 status and logs the error.
"""

from pathlib import Path
from typing import Any, Callable, Optional
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
import time
import uuid
import re

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError
import numpy as np
import torch
import uvicorn

MAX_IMAGE_BYTES = 20 * 1024 * 1024
ALLOWED_UPSCALE_FACTORS = {2, 4, 8}
AUDIT_DIR_NAME = "audit"
AUDIT_LOG_FILE = "events.jsonl"
AUDIT_REPORTS_DIR = "reports"

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

load_restoreformer: Optional[Callable] = None
load_realesrgan: Optional[Callable] = None
load_nafnet: Optional[Callable] = None

try:
    from models.restoreformer import load_restoreformer
except Exception as exc:
    logger.error("Error importing RestoreFormer loader: %s", exc)

try:
    from models.realesrgan import load_realesrgan
except Exception as exc:
    logger.error("Error importing Real-ESRGAN loader: %s", exc)

try:
    from models.nafnet import load_nafnet
except Exception as exc:
    logger.error("Error importing NAFNet loader: %s", exc)

try:
    from models.model_paths import get_models_dir
except Exception as exc:
    logger.error("Error importing get_models_dir: %s", exc)

    def get_models_dir() -> Path:
        return Path(__file__).resolve().parents[1] / "models-data"


app = FastAPI(title="PLAYE PhotoLab Desktop Backend")
models = {}
manifest_models: dict = {}
manifest_meta_cache: dict = {}
audit_enabled = os.getenv("PLAYE_AUDIT_LOG", "1") == "1"
audit_log_path: Optional[Path] = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid.uuid4()))


def _get_operator(request: Request) -> str:
    operator = request.headers.get("X-Operator")
    if operator:
        return operator.strip()[:128] or "unknown"
    return "unknown"


def _canonical_json_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)



REPORT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _normalize_report_id(raw_request_id: str) -> Optional[str]:
    if not raw_request_id:
        return None
    candidate = str(raw_request_id).strip()
    if REPORT_ID_RE.fullmatch(candidate):
        return candidate
    return None
def _get_manifest_path() -> Path:
    models_dir_manifest = Path(get_models_dir()) / "manifest.json"
    if models_dir_manifest.exists():
        return models_dir_manifest
    return Path(__file__).resolve().parents[1] / "models-data" / "manifest.json"


def _load_manifest_models() -> dict:
    path = _get_manifest_path()
    if not path.exists():
        logger.warning("Manifest not found: %s", path)
        return {}

    try:
        with path.open("r", encoding="utf-8") as inp:
            data = json.load(inp)
        return data.get("models", {}) if isinstance(data, dict) else {}
    except Exception as exc:
        logger.error("Failed to read manifest (%s): %s", path, exc)
        return {}


def _build_manifest_meta_cache(models_map: dict) -> dict:
    cache = {}
    for key, entry in models_map.items():
        if isinstance(entry, dict):
            cache[key] = {
                "model_name": entry.get("name", key),
                "model_version": entry.get("version"),
                "model_filename": entry.get("filename"),
                "model_checksum": entry.get("checksum"),
            }
    return cache


def _get_model_manifest_meta(model_key: str) -> dict:
    return manifest_meta_cache.get(model_key, {"model_name": model_key})


def _get_audit_log_path() -> Path:
    global audit_log_path
    if audit_log_path is not None:
        return audit_log_path

    audit_dir = Path(get_models_dir()) / AUDIT_DIR_NAME
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_log_path = audit_dir / AUDIT_LOG_FILE
    return audit_log_path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _get_audit_reports_dir() -> Path:
    reports_dir = _get_audit_log_path().parent / AUDIT_REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def _write_forensic_report(event: dict) -> Optional[Path]:
    if not audit_enabled:
        return None

    request_id = _normalize_report_id(event.get("request_id"))
    if not request_id:
        logger.warning("Skip forensic report write due to invalid request_id: %r", event.get("request_id"))
        return None

    report_payload: dict[str, Any] = {
        "schema_version": "1.1",
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "operation": event.get("operation"),
        "status": event.get("status"),
        "operator": event.get("operator", "unknown"),
        "input": {
            "sha256": event.get("input_sha256"),
            "filename": event.get("file_name"),
        },
        "output": {
            "sha256": event.get("output_sha256"),
            "bytes": event.get("output_bytes"),
        },
        "processing": {
            "duration_ms": event.get("duration_ms"),
            "parameters": {
                "scale": event.get("scale"),
                "level": event.get("level"),
            },
        },
        "model": {
            "key": event.get("model"),
            "name": event.get("model_name"),
            "version": event.get("model_version"),
            "filename": event.get("model_filename"),
            "checksum": event.get("model_checksum"),
        },
        "chain_of_custody": {
            "audit_log": str(_get_audit_log_path()),
            "audit_timestamp": event.get("timestamp"),
        },
        "disclaimer": (
            "Результат AI-обработки носит вспомогательный характер и требует верификации "
            "экспертом."
        ),
    }

    integrity_scope = {
        "request_id": report_payload["request_id"],
        "operation": report_payload["operation"],
        "status": report_payload["status"],
        "operator": report_payload["operator"],
        "input": report_payload["input"],
        "output": report_payload["output"],
        "processing": report_payload["processing"],
        "model": report_payload["model"],
        "chain_of_custody": report_payload["chain_of_custody"],
    }
    report_payload["integrity"] = {
        "algorithm": "sha256",
        "scope": "core_fields",
        "digest": _canonical_json_sha256(integrity_scope),
    }

    report_path = _get_audit_reports_dir() / f"{request_id}.json"
    try:
        with report_path.open("w", encoding="utf-8") as out:
            json.dump(report_payload, out, ensure_ascii=False, indent=2)
        return report_path
    except Exception as exc:
        logger.error("Failed to write forensic report for %s: %s", request_id, exc)
        return None


def _append_audit_event(event: dict) -> None:
    if not audit_enabled:
        return

    event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    try:
        with _get_audit_log_path().open("a", encoding="utf-8") as out:
            out.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.error("Failed to write audit event: %s", exc)


def _error(status_code: int, message: str, request_id: Optional[str] = None) -> JSONResponse:
    payload = {"error": message}
    if request_id:
        payload["request_id"] = request_id

    response = JSONResponse(status_code=status_code, content=payload)
    if request_id:
        response.headers["X-Request-ID"] = request_id
    return response


def _encode_png_bytes(result: np.ndarray) -> bytes:
    output = Image.fromarray(result)
    buf = io.BytesIO()
    output.save(buf, format='PNG')
    return buf.getvalue()


def _to_png_response(png_bytes: bytes, request_id: str) -> StreamingResponse:
    response = StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")
    response.headers["X-Request-ID"] = request_id
    return response


def _read_upload_as_rgb(contents: bytes) -> np.ndarray:
    if not contents:
        raise ValueError("Empty file")
    if len(contents) > MAX_IMAGE_BYTES:
        raise ValueError(f"File is too large (max {MAX_IMAGE_BYTES} bytes)")

    with Image.open(io.BytesIO(contents)) as image:
        return np.asarray(image.convert('RGB'))


def _load_model(model_key: str, model_name: str, loader: Optional[Callable]) -> None:
    if loader is None:
        logger.error("%s loader is unavailable; check import errors above.", model_name)
        return

    try:
        models[model_key] = loader(device)
        logger.info("Loaded model: %s", model_name)
    except Exception as exc:
        logger.error("Failed to load %s: %s", model_name, exc)


async def _process_image_operation(
    request: Request,
    file: UploadFile,
    operation: str,
    model_key: str,
    model_label: str,
    run_inference: Callable,
    extra_audit: Optional[dict] = None,
):
    request_id = _get_request_id(request)
    operator = _get_operator(request)
    started = time.perf_counter()

    try:
        model = models.get(model_key)
        if model is None:
            _append_audit_event({
                "request_id": request_id,
                "operation": operation,
                "status": "model_unavailable",
                "operator": operator,
                "error": f"{model_label} model not loaded",
                **_get_model_manifest_meta(model_key),
            })
            return _error(503, f"{model_label} model not loaded", request_id)

        contents = await file.read()
        image = _read_upload_as_rgb(contents)
        result = run_inference(model, image)
        png_bytes = _encode_png_bytes(result)

        if audit_enabled:
            duration_ms = int((time.perf_counter() - started) * 1000)
            event = {
                "request_id": request_id,
                "operation": operation,
                "model": model_key,
                "file_name": file.filename,
                "input_sha256": _sha256_bytes(contents),
                "output_sha256": _sha256_bytes(png_bytes),
                "output_bytes": len(png_bytes),
                "duration_ms": duration_ms,
                "status": "success",
                "operator": operator,
                **_get_model_manifest_meta(model_key),
            }
            if extra_audit:
                event.update(extra_audit)
            _append_audit_event(event)
            _write_forensic_report(event)

        return _to_png_response(png_bytes, request_id)
    except (ValueError, UnidentifiedImageError) as exc:
        _append_audit_event({
            "request_id": request_id,
            "operation": operation,
            "status": "client_error",
            "operator": operator,
            "error": str(exc),
            **_get_model_manifest_meta(model_key),
        })
        return _error(400, str(exc), request_id)
    except Exception as exc:
        logger.error("Error in %s [request_id=%s]: %s", operation, request_id, exc)
        _append_audit_event({
            "request_id": request_id,
            "operation": operation,
            "status": "server_error",
            "operator": operator,
            "error": str(exc),
            **_get_model_manifest_meta(model_key),
        })
        return _error(500, "Internal server error", request_id)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.on_event("startup")
async def startup_event():
    """Load AI models into memory when the server starts."""
    global manifest_models, manifest_meta_cache, audit_enabled, audit_log_path
    logger.info("Starting backend on device: %s", device)
    logger.info("Loading AI models...")

    audit_enabled = os.getenv("PLAYE_AUDIT_LOG", "1") == "1"
    audit_log_path = None

    manifest_models = _load_manifest_models()
    manifest_meta_cache = _build_manifest_meta_cache(manifest_models)

    if audit_enabled:
        _get_audit_log_path()

    _load_model('restoreformer', 'RestoreFormer', load_restoreformer)
    _load_model('realesrgan', 'Real-ESRGAN', load_realesrgan)
    _load_model('nafnet', 'NAFNet', load_nafnet)

    loaded = [name for name, model in models.items() if model is not None]
    logger.info("Models loaded: %s", loaded)


@app.get("/health")
async def health_check():
    """Return basic information about the backend's status."""
    return {
        "status": "ok",
        "device": device,
        "models_dir": str(get_models_dir()),
        "models": {k: (v is not None) for k, v in models.items()},
        "gpu_available": torch.cuda.is_available(),
        "audit_enabled": audit_enabled,
        "audit_log": str(_get_audit_log_path()) if audit_enabled else None,
        "manifest_path": str(_get_manifest_path()),
    }


@app.post("/ai/face-enhance")
async def enhance_face(request: Request, file: UploadFile = File(...)):
    """Enhance the quality of a face in the uploaded image using RestoreFormer."""
    return await _process_image_operation(
        request=request,
        file=file,
        operation="face-enhance",
        model_key="restoreformer",
        model_label="RestoreFormer",
        run_inference=lambda model, image: model.enhance(image),
    )


@app.post("/ai/upscale")
async def upscale_image(request: Request, file: UploadFile = File(...), factor: int = Form(2)):
    """Upscale the uploaded image by a given factor using Real-ESRGAN."""
    request_id = _get_request_id(request)
    if factor not in ALLOWED_UPSCALE_FACTORS:
        allowed = ", ".join(str(v) for v in sorted(ALLOWED_UPSCALE_FACTORS))
        _append_audit_event({
            "request_id": request_id,
            "operation": "upscale",
            "status": "client_error",
            "operator": _get_operator(request),
            "error": f"factor must be one of: {allowed}",
            "provided_factor": factor,
            **_get_model_manifest_meta("realesrgan"),
        })
        return _error(422, f"factor must be one of: {allowed}", request_id)

    return await _process_image_operation(
        request=request,
        file=file,
        operation="upscale",
        model_key="realesrgan",
        model_label="Real-ESRGAN",
        run_inference=lambda model, image: model.upscale(image, scale=factor),
        extra_audit={"scale": factor},
    )


@app.post("/ai/denoise")
async def denoise_image(request: Request, file: UploadFile = File(...), level: str = Form('medium')):
    """Denoise the uploaded image using NAFNet."""
    return await _process_image_operation(
        request=request,
        file=file,
        operation="denoise",
        model_key="nafnet",
        model_label="NAFNet",
        run_inference=lambda model, image: model.denoise(image, level=level),
        extra_audit={"level": level},
    )


@app.get("/forensic/report/{request_id}")
async def forensic_report(request_id: str):
    """Fetch a generated forensic report by request_id."""
    if not audit_enabled:
        return _error(404, "Audit reporting is disabled")

    normalized_request_id = _normalize_report_id(request_id)
    if normalized_request_id is None:
        return _error(422, "Invalid request_id format")

    report_path = _get_audit_reports_dir() / f"{normalized_request_id}.json"
    if not report_path.exists():
        return _error(404, f"Report not found for request_id={normalized_request_id}")

    try:
        with report_path.open("r", encoding="utf-8") as inp:
            payload = json.load(inp)
    except Exception as exc:
        logger.error("Failed to read forensic report %s: %s", report_path, exc)
        return _error(500, "Failed to load report")

    expected_digest = payload.get("integrity", {}).get("digest")
    chain = payload.get("chain_of_custody", {})
    integrity_scope = {
        "request_id": payload.get("request_id"),
        "operation": payload.get("operation"),
        "status": payload.get("status"),
        "operator": payload.get("operator"),
        "input": payload.get("input"),
        "output": payload.get("output"),
        "processing": payload.get("processing"),
        "model": payload.get("model"),
        "chain_of_custody": {
            "audit_log": chain.get("audit_log"),
            "audit_timestamp": chain.get("audit_timestamp"),
        },
    }
    digest_ok = expected_digest == _canonical_json_sha256(integrity_scope) if expected_digest else False

    return {
        "status": "ok",
        "request_id": normalized_request_id,
        "report": payload,
        "integrity": {
            "verified": digest_ok,
            "algorithm": payload.get("integrity", {}).get("algorithm", "sha256"),
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
