"""Batch image processing helpers."""

from __future__ import annotations

import io
import logging
from typing import Callable, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def images_to_batch_tensor(images: list[bytes], target_size: tuple[int, int] = (512, 512)):
    import torch

    tensors = []
    for img_bytes in images:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize(target_size)
        arr = np.array(img, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(tensors)


def batch_tensor_to_images(tensor) -> list[bytes]:
    results: list[bytes] = []
    for i in range(tensor.shape[0]):
        arr = tensor[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        arr_uint8 = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr_uint8)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        results.append(buf.getvalue())
    return results


async def process_batch(
    images: list[bytes],
    model_fn: Callable,
    batch_size: int = 4,
    device: str = "cpu",
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list[bytes]:
    import torch

    total = len(images)
    results: list[bytes | None] = [None] * total

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_images = images[start:end]
        try:
            batch_tensor = images_to_batch_tensor(batch_images).to(device)
            with torch.no_grad():
                output_tensor = model_fn(batch_tensor)
            batch_results = batch_tensor_to_images(output_tensor.cpu())
            for i, result in enumerate(batch_results):
                results[start + i] = result
        except Exception as exc:
            logger.error("Batch [%d:%d] failed: %s", start, end, exc)
            for i in range(len(batch_images)):
                results[start + i] = batch_images[i]

        if on_progress:
            on_progress(end, total)

    return [r if r is not None else b"" for r in results]
