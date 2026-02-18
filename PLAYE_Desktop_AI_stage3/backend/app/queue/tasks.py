"""Celery tasks for PLAYE PhotoLab backend."""

from __future__ import annotations

import asyncio
import base64
from typing import Any

from app.models.denoise import denoise_image
from app.models.detect_faces import detect_faces
from app.models.detect_objects import detect_objects
from app.models.face_enhance import enhance_face
from app.models.upscale import upscale_image
from app.models.video_pipeline import detect_scene_changes, extract_frames, process_video_frames
from app.queue.worker import celery_app

TASK_RETRY_OPTS = {
    "autoretry_for": (Exception,),
    "retry_backoff": True,
    "retry_jitter": False,
    "max_retries": 3,
}


def _encode_image_result(result: Any) -> Any:
    if isinstance(result, (bytes, bytearray)):
        return {"image_base64": base64.b64encode(bytes(result)).decode("ascii"), "mime_type": "image/png"}
    return result


def _decode_images_b64(images_b64: list[str]) -> list[bytes]:
    return [base64.b64decode(v) for v in images_b64]


def _encode_images_b64(images: list[bytes]) -> list[str]:
    return [base64.b64encode(v).decode("ascii") for v in images]


@celery_app.task(name="tasks.face_enhance", **TASK_RETRY_OPTS)
def face_enhance_task(image: bytes) -> Any:
    return _encode_image_result(asyncio.run(enhance_face(image)))


@celery_app.task(name="tasks.upscale", **TASK_RETRY_OPTS)
def upscale_task(image: bytes, factor: int = 2) -> Any:
    return _encode_image_result(asyncio.run(upscale_image(image, factor)))


@celery_app.task(name="tasks.denoise", **TASK_RETRY_OPTS)
def denoise_task(image: bytes, level: str = "light") -> Any:
    return _encode_image_result(asyncio.run(denoise_image(image, level)))


@celery_app.task(name="tasks.detect_faces", **TASK_RETRY_OPTS)
def detect_faces_task(image: bytes) -> Any:
    return asyncio.run(detect_faces(image))


@celery_app.task(name="tasks.detect_objects", **TASK_RETRY_OPTS)
def detect_objects_task(image: bytes, scene_threshold: float | None = None, temporal_window: int | None = None) -> Any:
    return asyncio.run(
        detect_objects(image, scene_threshold=scene_threshold, temporal_window=temporal_window)
    )


@celery_app.task(name="tasks.video_temporal_denoise", **TASK_RETRY_OPTS)
def video_temporal_denoise_task(video: bytes, fps: float = 1.0) -> Any:
    async def _run():
        return await process_video_frames(
            video_bytes=video,
            operation="temporal_denoise",
            params={},
            frame_processor=lambda frame, _: denoise_image(frame, "medium"),
            fps=fps,
        )

    return asyncio.run(_run())


@celery_app.task(name="tasks.video_scene_detect", **TASK_RETRY_OPTS)
def video_scene_detect_task(video: bytes, scene_threshold: float = 28.0, temporal_window: int = 3) -> Any:
    async def _run():
        frames = await extract_frames(video, fps=2.0)
        scenes = await detect_scene_changes(frames, threshold=scene_threshold)
        scene_objects = []
        for idx in scenes:
            if idx < len(frames):
                objs = await detect_objects(
                    frames[idx],
                    scene_threshold=scene_threshold,
                    temporal_window=temporal_window,
                )
                scene_objects.append({"frame": idx, "objects": objs})
        return {"total_frames": len(frames), "scene_cuts": scenes, "scene_objects": scene_objects}

    return asyncio.run(_run())


@celery_app.task(name="tasks.batch_upscale", **TASK_RETRY_OPTS)
def batch_upscale_task(images_b64: list[str], factor: int = 2) -> Any:
    async def _run():
        imgs = _decode_images_b64(images_b64)
        outs = [await upscale_image(img, factor) for img in imgs]
        return {"images_base64": _encode_images_b64(outs), "count": len(outs)}

    return asyncio.run(_run())


@celery_app.task(name="tasks.batch_denoise", **TASK_RETRY_OPTS)
def batch_denoise_task(images_b64: list[str], level: str = "medium") -> Any:
    async def _run():
        imgs = _decode_images_b64(images_b64)
        outs = [await denoise_image(img, level) for img in imgs]
        return {"images_base64": _encode_images_b64(outs), "count": len(outs)}

    return asyncio.run(_run())


@celery_app.task(name="tasks.batch_face_enhance", **TASK_RETRY_OPTS)
def batch_face_enhance_task(images_b64: list[str]) -> Any:
    async def _run():
        imgs = _decode_images_b64(images_b64)
        outs = [await enhance_face(img) for img in imgs]
        return {"images_base64": _encode_images_b64(outs), "count": len(outs)}

    return asyncio.run(_run())
