import os
from pathlib import Path


def get_models_dir() -> Path:
    """Resolve the directory where model weights live.

    In Desktop mode, Electron sets PLAYE_MODELS_DIR to something like:
      %APPDATA%/PLAYE.../models

    Fallback for dev: <project>/models-data
    """

    env = os.getenv("PLAYE_MODELS_DIR")
    if env:
        return Path(env)

    # project_root/backend/models/model_paths.py -> project_root
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "models-data"


def model_path(filename: str) -> Path:
    return get_models_dir() / filename
