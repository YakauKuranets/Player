"""YOLOv8 object detection module with graceful fallback."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

_yolo_model = None


def _load_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO

        _yolo_model = YOLO("yolov8n.pt")
        return _yolo_model
    except Exception:
        return None


async def detect_objects(
    image: bytes,
    scene_threshold: Optional[float] = None,
    temporal_window: Optional[int] = None,
) -> List[Dict[str, Any]]:
    _ = temporal_window
    model = _load_yolo()
    if model is None:
        return []

    try:
        import cv2
    except Exception:
        return []

    img_np = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if img_np is None:
        return []

    conf_threshold = 0.25
    if scene_threshold is not None:
        conf_threshold = max(0.0, min(1.0, float(scene_threshold) / 100.0))

    try:
        results = model(img_np, conf=conf_threshold, verbose=False)
    except Exception:
        return []

    output: List[Dict[str, Any]] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            output.append(
                {
                    "class_id": cls_id,
                    "class_name": model.names.get(cls_id, "unknown"),
                    "confidence": float(box.conf[0]),
                    "bbox": {
                        "x": int(box.xyxy[0][0]),
                        "y": int(box.xyxy[0][1]),
                        "w": int(box.xyxy[0][2] - box.xyxy[0][0]),
                        "h": int(box.xyxy[0][3] - box.xyxy[0][1]),
                    },
                }
            )
    return output
