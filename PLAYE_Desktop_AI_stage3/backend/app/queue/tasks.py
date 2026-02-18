"""Celery tasks for PLAYE PhotoLab backend."""

from __future__ import annotations

import asyncio
from typing import Any

try:
    from celery import Celery  # type: ignore
except ImportError:  # pragma: no cover
    class Celery:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def task(self, name: str = None, **opts):  # type: ignore
            def decorator(fn):
                return fn

            return decorator

        def send_task(self, *args, **kwargs):  # type: ignore
            raise RuntimeError("Celery is not available")


from app.models.denoise import denoise_image
from app.models.detect_faces import detect_faces
from app.models.detect_objects import detect_objects
from app.models.face_enhance import enhance_face
from app.models.upscale import upscale_image

celery_app = Celery("playe_photo_lab")

TASK_RETRY_OPTS = {
    "autoretry_for": (Exception,),
    "retry_backoff": True,
    "retry_jitter": False,
    "max_retries": 3,
}


@celery_app.task(name="tasks.face_enhance", **TASK_RETRY_OPTS)
def face_enhance_task(image: bytes) -> Any:
    return asyncio.run(enhance_face(image))


@celery_app.task(name="tasks.upscale", **TASK_RETRY_OPTS)
def upscale_task(image: bytes, factor: int = 2) -> Any:
    return asyncio.run(upscale_image(image, factor))


@celery_app.task(name="tasks.denoise", **TASK_RETRY_OPTS)
def denoise_task(image: bytes, level: str = "light") -> Any:
    return asyncio.run(denoise_image(image, level))


@celery_app.task(name="tasks.detect_faces", **TASK_RETRY_OPTS)
def detect_faces_task(image: bytes) -> Any:
    return asyncio.run(detect_faces(image))


@celery_app.task(name="tasks.detect_objects", **TASK_RETRY_OPTS)
def detect_objects_task(
    image: bytes,
    scene_threshold: float | None = None,
    temporal_window: int | None = None,
) -> Any:
    return asyncio.run(
        detect_objects(
            image,
            scene_threshold=scene_threshold,
            temporal_window=temporal_window,
        )
    )
