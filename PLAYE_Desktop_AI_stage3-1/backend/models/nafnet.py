"""
Wrapper for the NAFNet model used for image denoising.

This module defines a ``NAFNet`` class that encapsulates model loading
and inference. The internal model implementation is currently a stub
that returns the input unchanged; users should integrate their own
pretrained weights and architecture when available.
"""

from pathlib import Path

from .model_paths import model_path
import numpy as np
import torch


class NAFNet:
    """A minimal wrapper for denoising using NAFNet."""

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model = self._load_model(model_path)
        if hasattr(self.model, 'eval'):
            self.model.eval()

    def _load_model(self, model_path: str):
        # TODO: implement model loading for NAFNet.
        class _NoOpModel:
            def __call__(self, x):
                return x
            def to(self, device):
                return self
            def eval(self):
                return self
        return _NoOpModel()

    def denoise(self, image: np.ndarray, level: str = 'medium') -> np.ndarray:
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


def load_nafnet(device: str = 'cuda') -> NAFNet:
    weights = model_path('nafnet.pth')
    if not weights.exists():
        raise FileNotFoundError(f"Model not found: {weights}")
    return NAFNet(str(weights), device)