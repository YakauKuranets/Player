"""
Configuration for backend
"""

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
    # Secret key used to sign and verify JWT tokens. In a production
    # environment this value should be set via an environment variable
    # (e.g. JWT_SECRET) and kept secret. Clients must include a valid
    # Bearer token signed with this secret when accessing protected
    # endpoints.
    JWT_SECRET: str = "changeme"
    
    # Processing
    MAX_IMAGE_SIZE: int = 4096  # Max dimension
    BATCH_SIZE: int = 4
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()