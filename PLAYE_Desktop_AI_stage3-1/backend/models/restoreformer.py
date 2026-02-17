"""
Wrapper for the RestoreFormer model used for facial enhancement.

This module defines a ``RestoreFormer`` class with methods for loading
the model weights, preprocessing numpy arrays into PyTorch tensors,
running inference and converting the result back into a numpy array.

The ``load_restoreformer`` function locates the model weights in the
``models-data`` directory relative to the project root and instantiates
the wrapper. If the weights are missing the function raises a
FileNotFoundError so the caller can report an error to the user.

The actual model architecture and weight-loading code should be added
where marked. For demonstration purposes the current implementation
simply returns the input image unchanged.
"""

from pathlib import Path

from .model_paths import model_path
import numpy as np
import torch


class RestoreFormer:
    """A minimal wrapper around a facial enhancement model."""

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model = self._load_model(model_path)
        # Switch the model to eval mode if it defines eval()
        if hasattr(self.model, 'eval'):
            self.model.eval()

    def _load_model(self, model_path: str):
        """
        Load the underlying PyTorch model.

        Replace this stub with your model class and weight-loading logic.
        """
        # TODO: import and instantiate your trained model class here.
        # Example:
        # model = MyRestoreFormerModel()
        # state = torch.load(model_path, map_location=self.device)
        # model.load_state_dict(state)
        # return model.to(self.device)
        # For now return a no-op lambda that just returns the input.
        class _NoOpModel:
            def __call__(self, x):
                return x

            def to(self, device):
                return self

            def eval(self):
                return self

        return _NoOpModel()

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance the quality of a face image.

        Args:
            image: numpy array of shape (H, W, 3) in RGB format.

        Returns:
            numpy array of the same shape containing the enhanced image.
        """
        tensor = self._preprocess(image)
        with torch.no_grad():
            output = self.model(tensor)
        result = self._postprocess(output)
        return result

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert a numpy array (H, W, 3) into a PyTorch tensor."""
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a tensor back into a numpy array (H, W, 3)."""
        tensor = tensor.squeeze(0).permute(1, 2, 0)
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return array


def load_restoreformer(device: str = 'cuda') -> RestoreFormer:
    """
    Load the RestoreFormer model from the models-data directory.

    The weights file must be named ``restoreformer.pth`` and reside in
    ``models-data`` relative to the project root.
    """
    # Compute the path relative to this file. ``Path(__file__).parents``
    # returns the directory hierarchy; the weight file lives two levels
    # up in models-data/restoreformer.pth.
    weights = model_path('restoreformer.pth')
    if not weights.exists():
        raise FileNotFoundError(f"Model not found: {weights}")
    return RestoreFormer(str(weights), device)