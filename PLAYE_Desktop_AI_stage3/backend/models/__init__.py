"""
Model loading subpackage for the PLAYE PhotoLab desktop backend.

This package contains lightweight wrapper classes for the AI models used by
the FastAPI server. Each wrapper encapsulates the logic for loading a
PyTorch model from disk and exposes a simple inference method. The loaders
defined here are intentionally minimal: they do not implement the actual
model architectures, leaving that to the user to fill in according to
their training code. If a model file is missing, the loader will raise
a FileNotFoundError.
"""

from .restoreformer import load_restoreformer  # noqa: F401
from .realesrgan import load_realesrgan  # noqa: F401
from .nafnet import load_nafnet  # noqa: F401