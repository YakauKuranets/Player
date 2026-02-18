"""Real-ESRGAN upscaling module with graceful fallback."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .download_models import download_realesrgan_x4

logger = logging.getLogger(__name__)

_realesrgan_path: Path | None = None
_upscaler_cache: dict[int, object] = {}


def _ensure_realesrgan_weights() -> Path:
    global _realesrgan_path
    if _realesrgan_path is None:
        _realesrgan_path = download_realesrgan_x4()
    return _realesrgan_path


def _get_upscaler(factor: int):
    if factor in _upscaler_cache:
        return _upscaler_cache[factor]

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )
    weights_path = str(_ensure_realesrgan_weights())

    upscaler = RealESRGANer(
        scale=4,
        model_path=weights_path,
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    _upscaler_cache[factor] = upscaler
    return upscaler


async def upscale_image(image: bytes, factor: int = 2) -> bytes:
    weights = _ensure_realesrgan_weights()
    if weights.stat().st_size < 1_000_000:
        return image

    try:
        img_pil = Image.open(io.BytesIO(image)).convert("RGB")
        img_np = np.array(img_pil)
        upscaler = _get_upscaler(factor)
        output_np, _ = upscaler.enhance(img_np, outscale=factor)
        output_pil = Image.fromarray(output_np)
        buf = io.BytesIO()
        output_pil.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:  # pragma: no cover - depends on optional heavy deps
        logger.warning("Real-ESRGAN unavailable, fallback to original image: %s", exc)
        return image
