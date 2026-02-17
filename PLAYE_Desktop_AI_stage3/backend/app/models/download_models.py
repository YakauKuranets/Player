"""
Model downloader for heavy PyTorch models used in the PLAYE PhotoLab backend.

This module defines helper functions that simulate downloading large PyTorch
weights required for the cloud backend. Since the environment in which this
project runs does not have access to the internet, the "download" functions
create placeholder files of a fixed size to represent the presence of the
model weights. In a real deployment, these functions could download the
actual `.pth` or `.pt` files from a model zoo or cloud storage bucket.

The placeholder weights are created in the ``weights`` subdirectory of this
package. Each file is only created if it does not already exist. The
allocated size is intentionally small (~10 MiB) to avoid exhausting disk
space in this example implementation. You can adjust the ``SIZE_MB``
constants to match the approximate size of the real models if desired.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

__all__ = [
    "download_restoreformer_pp",
    "download_realesrgan_x4",
    "download_nafnet_heavy",
    "download_all",
]


# Directory where simulated weights will be stored.  It resides next to this file.
WEIGHTS_DIR: Path = Path(__file__).resolve().parent / "weights"

def _create_placeholder(weight_path: Path, size_mb: int, model_name: str) -> None:
    """Create a placeholder file of the given size.

    If the file already exists, this function does nothing. Otherwise it
    creates the parent directory if necessary and writes ``size_mb``
    megabytes of zero bytes to the file. A message is printed to the console
    whenever a placeholder is created.

    :param weight_path: The path where the file should be created.
    :param size_mb: The approximate size of the file in megabytes.
    :param model_name: The human-friendly name of the model being created.
    """
    if weight_path.exists():
        return
    # Ensure the directory exists
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    # Write the file in 1 MiB chunks to avoid keeping all bytes in memory
    chunk = b"\0" * (1024 * 1024)
    with weight_path.open("wb") as f:
        for _ in range(size_mb):
            f.write(chunk)
    print(f"Created placeholder weight for {model_name} at {weight_path} ({size_mb} MiB)")


def download_restoreformer_pp(size_mb: int = 10) -> Path:
    """Ensure the RestoreFormer++ weights are available on disk.

    This function simulates downloading the RestoreFormer++ model by
    creating a placeholder file if it does not already exist. The returned
    path points to ``restoreformer_pp.pth`` inside the weights directory.

    :param size_mb: The size of the placeholder file in megabytes.
    :returns: Path to the placeholder weight file.
    """
    weight_path = WEIGHTS_DIR / "restoreformer_pp.pth"
    _create_placeholder(weight_path, size_mb, "RestoreFormer++")
    return weight_path


def download_realesrgan_x4(size_mb: int = 10) -> Path:
    """Ensure the Real-ESRGAN-x4 weights are available on disk.

    This function simulates downloading the Real-ESRGAN x4 model by
    creating a placeholder file if it does not already exist. The returned
    path points to ``realesrgan_x4.pth`` inside the weights directory.

    :param size_mb: The size of the placeholder file in megabytes.
    :returns: Path to the placeholder weight file.
    """
    weight_path = WEIGHTS_DIR / "realesrgan_x4.pth"
    _create_placeholder(weight_path, size_mb, "Real-ESRGAN-x4")
    return weight_path


def download_nafnet_heavy(size_mb: int = 10) -> Path:
    """Ensure the NAFNet heavy weights are available on disk.

    This function simulates downloading the heavy NAFNet model by creating
    a placeholder file if it does not already exist. The returned path
    points to ``nafnet_heavy.pth`` inside the weights directory.

    :param size_mb: The size of the placeholder file in megabytes.
    :returns: Path to the placeholder weight file.
    """
    weight_path = WEIGHTS_DIR / "nafnet_heavy.pth"
    _create_placeholder(weight_path, size_mb, "NAFNet-heavy")
    return weight_path


def download_all(size_mb: int = 10) -> None:
    """Download all heavy models by creating their placeholder files.

    Invokes each individual download function in sequence. The ``size_mb``
    parameter applies to all models.

    :param size_mb: The size in megabytes for each placeholder weight file.
    """
    download_restoreformer_pp(size_mb)
    download_realesrgan_x4(size_mb)
    download_nafnet_heavy(size_mb)