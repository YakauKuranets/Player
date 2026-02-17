"""
Celery tasks for PLAYE PhotoLab backend.

This module defines asynchronous tasks that wrap the heavy AI models. The
tasks are executed by the Celery worker defined in ``worker.py``. For now
the tasks simply forward the image data to the corresponding model
functions and return the result. When integrating with real PyTorch models
you would perform the heavy computation here and write outputs to disk or
a blob store.
"""

from __future__ import annotations

import asyncio
from typing import Any

try:
    # Celery is an optional dependency. In development environments where
    # Celery is not installed, we define a minimal stub so that this module
    # can still be imported without raising ImportError. The stub mimics
    # enough of the Celery API used here to decorate functions but does
    # not provide any actual task queue functionality.
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

from app.models.face_enhance import enhance_face
from app.models.upscale import upscale_image
from app.models.denoise import denoise_image
from app.models.detect_faces import detect_faces
from app.models.detect_objects import detect_objects

# Celery application will be configured in worker.py. Note: tasks are
# registered on the shared ``celery_app`` defined in worker.py.
celery_app = Celery('playe_photo_lab')


@celery_app.task(name='tasks.face_enhance')
def face_enhance_task(image: bytes) -> Any:
    """Celery task to enhance faces.

    This synchronous wrapper calls the asynchronous ``enhance_face``
    function using ``asyncio.run``. The input image is expected to be
    raw bytes. The result is returned as-is; when integrating real models
    you might return processed bytes or store the output to a file and
    return a path.
    """
    return asyncio.run(enhance_face(image))


@celery_app.task(name='tasks.upscale')
def upscale_task(image: bytes, factor: int = 2) -> Any:
    """Celery task to upscale images.

    :param image: The raw image bytes to upscale.
    :param factor: The upscale factor.
    :return: The upscaled image (currently unchanged).
    """
    return asyncio.run(upscale_image(image, factor))


@celery_app.task(name='tasks.denoise')
def denoise_task(image: bytes, level: str = 'light') -> Any:
    """Celery task to denoise images.

    :param image: The raw image bytes to denoise.
    :param level: The denoising level ('light', 'medium', 'heavy').
    :return: The denoised image (currently unchanged).
    """
    return asyncio.run(denoise_image(image, level))


@celery_app.task(name='tasks.detect_faces')
def detect_faces_task(image: bytes) -> Any:
    """Celery task to detect faces in an image.

    :param image: The raw image bytes.
    :return: A list of detected faces (currently empty).
    """
    return asyncio.run(detect_faces(image))


@celery_app.task(name='tasks.detect_objects')
def detect_objects_task(image: bytes) -> Any:
    """Celery task to detect objects in an image.

    :param image: The raw image bytes.
    :return: A list of detected objects (currently empty).
    """
    return asyncio.run(detect_objects(image))