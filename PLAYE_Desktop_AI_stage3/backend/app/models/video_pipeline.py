"""Video temporal processing pipeline."""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

try:
    import ffmpeg  # type: ignore

    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    logger.warning("ffmpeg-python not installed — video pipeline disabled")


async def extract_frames(video_bytes: bytes, fps: float = 1.0) -> list[bytes]:
    if not FFMPEG_AVAILABLE:
        return []

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    frames: list[bytes] = []
    try:
        out, _ = (
            ffmpeg.input(tmp_path)
            .filter("fps", fps=fps)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        probe = ffmpeg.probe(tmp_path)
        video_stream = next((s for s in probe["streams"] if s.get("codec_type") == "video"), None)
        if video_stream:
            width = int(video_stream["width"])
            height = int(video_stream["height"])
            frame_size = width * height * 3
            for i in range(0, len(out), frame_size):
                raw = out[i : i + frame_size]
                if len(raw) == frame_size:
                    from PIL import Image

                    img = Image.frombytes("RGB", (width, height), raw)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    frames.append(buf.getvalue())
    except Exception as exc:
        logger.error("Frame extraction failed: %s", exc)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return frames


async def process_video_frames(
    video_bytes: bytes,
    operation: str,
    params: dict,
    frame_processor: Callable[[bytes, dict], Any],
    fps: float = 1.0,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> dict[str, Any]:
    frames = await extract_frames(video_bytes, fps=fps)
    total = len(frames)
    if total == 0:
        return {"frames_processed": 0, "frames_total": 0, "results": [], "operation": operation}

    results = []
    for i, frame in enumerate(frames):
        try:
            result = await frame_processor(frame, params)
            results.append({"frame": i, "status": "ok", "data": result})
        except Exception as exc:
            logger.error("Frame %d processing failed: %s", i, exc)
            results.append({"frame": i, "status": "error", "error": str(exc)})
        if on_progress:
            on_progress(i + 1, total)

    return {
        "frames_processed": len([r for r in results if r["status"] == "ok"]),
        "frames_total": total,
        "results": results,
        "operation": operation,
    }


async def detect_scene_changes(frames: list[bytes], threshold: float = 28.0) -> list[int]:
    import numpy as np
    from PIL import Image

    scene_cuts: list[int] = []
    prev_array = None
    for i, frame_bytes in enumerate(frames):
        img = Image.open(io.BytesIO(frame_bytes)).convert("L").resize((64, 64))
        curr_array = np.array(img, dtype=np.float32)
        if prev_array is not None:
            mad = float(np.mean(np.abs(curr_array - prev_array)))
            if mad > threshold:
                scene_cuts.append(i)
        prev_array = curr_array
    return scene_cuts
