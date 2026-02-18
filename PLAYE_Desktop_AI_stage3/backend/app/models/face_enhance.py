"""GFPGAN/RestoreFormer face enhancement module with graceful fallback."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from .download_models import download_restoreformer_pp

logger = logging.getLogger(__name__)

_restoreformer_path: Path | None = None
_gfpgan_model = None


def _ensure_restoreformer_weights() -> Path:
    global _restoreformer_path
    if _restoreformer_path is None:
        _restoreformer_path = download_restoreformer_pp()
    return _restoreformer_path


def _load_gfpgan():
    global _gfpgan_model
    if _gfpgan_model is not None:
        return _gfpgan_model

    weights_path = _ensure_restoreformer_weights()
    if weights_path.stat().st_size < 1_000_000:
        return None

    from gfpgan import GFPGANer

    model = GFPGANer(
        model_path=str(weights_path),
        upscale=1,
        arch="RestoreFormer",
        channel_multiplier=2,
        bg_upsampler=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    _gfpgan_model = model
    return model


async def enhance_face(image: bytes) -> bytes:
    try:
        model = _load_gfpgan()
        if model is None:
            return image

        try:
            import cv2
        except Exception:
            return image

        img_np = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        if img_np is None:
            return image

        _, _, restored_img = model.enhance(
            img_np,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )

        if restored_img is None:
            return image

        ok, buf = cv2.imencode(".png", restored_img)
        if not ok:
            return image
        return buf.tobytes()
    except Exception as exc:  # pragma: no cover - depends on optional heavy deps
        logger.warning("GFPGAN unavailable, fallback to original image: %s", exc)
        return image
