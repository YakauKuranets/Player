"""NAFNet denoising module with graceful fallback."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .download_models import download_nafnet_heavy

logger = logging.getLogger(__name__)

_nafnet_path: Path | None = None
_nafnet_model = None
_nafnet_device = "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_nafnet_weights() -> Path:
    global _nafnet_path
    if _nafnet_path is None:
        _nafnet_path = download_nafnet_heavy()
    return _nafnet_path


def _load_nafnet():
    global _nafnet_model
    if _nafnet_model is not None:
        return _nafnet_model

    weights_path = _ensure_nafnet_weights()
    if weights_path.stat().st_size < 1_000_000:
        return None

    checkpoint = torch.load(str(weights_path), map_location="cpu")
    from basicsr.models.archs.NAFNet_arch import NAFNet

    model = NAFNet(
        img_channel=3,
        width=32,
        middle_blk_num=1,
        enc_blks=[1, 1, 1, 28],
        dec_blks=[1, 1, 1, 1],
    )
    model.load_state_dict(checkpoint.get("params", checkpoint), strict=False)
    model.eval()
    model.to(_nafnet_device)
    _nafnet_model = model
    return model


async def denoise_image(image: bytes, level: str = "light") -> bytes:
    strength_map = {"light": 0.3, "medium": 0.6, "heavy": 1.0}
    strength = strength_map.get(level, 0.3)

    try:
        model = _load_nafnet()
        if model is None:
            return image

        img_pil = Image.open(io.BytesIO(image)).convert("RGB")
        img_np = np.array(img_pil).astype(np.float32) / 255.0

        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(_nafnet_device)

        with torch.no_grad():
            output = model(tensor)

        blend = float(max(0.0, min(1.0, strength)))
        output = output * blend + tensor * (1.0 - blend)

        out_np = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
        out_uint8 = (out_np * 255).astype(np.uint8)

        result_pil = Image.fromarray(out_uint8)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:  # pragma: no cover - depends on optional heavy deps
        logger.warning("NAFNet unavailable, fallback to original image: %s", exc)
        return image
