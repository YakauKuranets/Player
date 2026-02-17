"""
Wrapper for the Real-ESRGAN model used for image upscaling.

This module provides a ``RealESRGAN`` class that encapsulates loading
the PyTorch model and running inference. The actual model
architecture and weight loading must be filled in by the user. At the
moment the implementation simply passes the input through unchanged.
"""

from pathlib import Path

from .model_paths import model_path
import numpy as np
import torch


class RealESRGAN:
    """A minimal wrapper around a super-resolution model."""

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model = self._load_model(model_path)
        if hasattr(self.model, 'eval'):
            self.model.eval()

    def _load_model(self, model_path: str):
        # TODO: implement model loading for Real-ESRGAN.
        class _NoOpModel:
            def __call__(self, x):
                return x
            def to(self, device):
                return self
            def eval(self):
                return self
        return _NoOpModel()

    def upscale(self, image: np.ndarray, scale: int = 2) -> np.ndarray:
        tensor = self._preprocess(image)
        with torch.no_grad():
            output = self.model(tensor)
        result = self._postprocess(output)
        return result

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = tensor.squeeze(0).permute(1, 2, 0)
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return array


def load_realesrgan(device: str = 'cuda') -> RealESRGAN:
    weights = model_path('realesrgan.pth')
    if not weights.exists():
        raise FileNotFoundError(f"Model not found: {weights}")
    return RealESRGAN(str(weights), device)