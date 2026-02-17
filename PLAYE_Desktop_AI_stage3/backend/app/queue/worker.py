"""
Celery worker for PLAYE PhotoLab backend.

This module configures the Celery application and starts the worker process
when run as a script. It loads configuration from environment variables.
"""

from celery import Celery
import os


def create_celery_app() -> Celery:
    """Create and configure Celery application."""
    celery_app = Celery(
        'playe_photo_lab',
        broker=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
        backend=os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    )
    # Additional configuration can be set here
    # Autodiscover tasks in the app.queue package so that Celery registers
    # functions decorated with @celery_app.task.
    celery_app.autodiscover_tasks(['app.queue'])
    return celery_app


celery_app = create_celery_app()


if __name__ == '__main__':
    # When running this module directly, start the Celery worker
    celery_app.worker_main()