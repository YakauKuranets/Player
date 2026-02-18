"""Configuration for backend."""

from __future__ import annotations

import os

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/playelab"

    # Redis (для Celery)
    REDIS_URL: str = "redis://localhost:6379/0"

    # Models
    MODELS_DIR: str = "/models"

    # GPU
    CUDA_VISIBLE_DEVICES: Optional[str] = "0"

    # Authentication
    JWT_SECRET: str = "changeme"

    # Processing
    MAX_IMAGE_SIZE: int = 4096
    BATCH_SIZE: int = 4

    def model_post_init(self, __context) -> None:
        if self.JWT_SECRET == "changeme":
            if os.environ.get("PLAYE_ALLOW_INSECURE_JWT") != "1":
                raise RuntimeError(
                    "JWT_SECRET is set to default 'changeme'. "
                    "Set JWT_SECRET env variable to a secure random value. "
                    "Generate one: python -c \"import secrets; print(secrets.token_hex(32))\""
                )

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
