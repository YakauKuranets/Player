"""Celery worker with multi-GPU queues."""

from __future__ import annotations

import os

from celery import Celery
from celery.signals import task_postrun, task_prerun

from app.queue.gpu_router import gpu_router


def create_celery_app() -> Celery:
    celery_app = Celery(
        "playe_photo_lab",
        broker=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        backend=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    )

    queue = gpu_router.get_best_queue()
    celery_app.conf.update(
        task_routes={
            "tasks.upscale": {"queue": queue},
            "tasks.face_enhance": {"queue": queue},
            "tasks.denoise": {"queue": queue},
            "tasks.detect_faces": {"queue": queue},
            "tasks.detect_objects": {"queue": queue},
            "tasks.video_temporal_denoise": {"queue": queue},
            "tasks.video_scene_detect": {"queue": queue},
            "tasks.batch_upscale": {"queue": queue},
            "tasks.batch_denoise": {"queue": queue},
            "tasks.batch_face_enhance": {"queue": queue},
        },
        task_default_queue="cpu",
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        worker_max_tasks_per_child=50,
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        task_soft_time_limit=300,
        task_time_limit=600,
    )
    celery_app.autodiscover_tasks(["app.queue"])
    return celery_app


celery_app = create_celery_app()


@task_prerun.connect
def on_task_start(task_id, task, **kwargs):
    queue = kwargs.get("routing_key", "cpu")
    gpu_router.increment(queue)


@task_postrun.connect
def on_task_end(task_id, task, **kwargs):
    queue = kwargs.get("routing_key", "cpu")
    gpu_router.decrement(queue)


if __name__ == "__main__":
    celery_app.worker_main()
