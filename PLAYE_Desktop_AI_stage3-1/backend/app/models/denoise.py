"""
NAFNet denoising module.

This module defines a function to denoise images using a pretrained NAFNet
model. It provides a lazy loader for the heavy NAFNet weights. As with the
other modules, the current implementation does not perform actual
denoising—it simply ensures the weights exist and then returns the
original image. Extend this module with real model loading and inference
when the heavy NAFNet model becomes available.
"""

from __future__ import annotations

from typing import Any, Optional
from pathlib import Path

from .download_models import download_nafnet_heavy

# Lazy initialisation of the NAFNet heavy weights path. The actual model
# will not be loaded in this placeholder implementation.
_nafnet_path: Optional[Path] = None


def _ensure_nafnet_weights() -> Path:
    """Ensure that the heavy NAFNet weights are present on disk.

    Returns the path to the weights file. If the file does not exist, it
    will be created as a placeholder via the ``download_nafnet_heavy``
    function.
    """
    global _nafnet_path
    if _nafnet_path is None:
        _nafnet_path = download_nafnet_heavy()
    return _nafnet_path


async def denoise_image(image: Any, level: str = "light") -> Any:
    """Remove noise from the given image.

    Ensures that the heavy NAFNet weights exist and then returns the input
    image unchanged. Use the ``level`` parameter to choose between light,
    medium and heavy denoising when a real implementation is added.

    :param image: Image data (e.g., bytes or numpy array or PIL image)
    :param level: Denoising level ('light', 'medium', or 'heavy')
    :return: Denoised image (currently the same as input)
    """
    _ensure_nafnet_weights()
    # TODO: Load and run the NAFNet model on the image.
    return image