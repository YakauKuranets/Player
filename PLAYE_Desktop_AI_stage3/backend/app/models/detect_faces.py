"""RetinaFace face detection module with OpenCV fallback."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


async def detect_faces(image: bytes) -> List[Dict[str, Any]]:
    try:
        from retinaface import RetinaFace
    except ImportError:
        return _detect_faces_opencv(image)

    try:
        import cv2
    except Exception:
        return []

    img_np = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if img_np is None:
        return []

    try:
        faces = RetinaFace.detect_faces(img_np)
    except Exception:
        return []

    if not isinstance(faces, dict):
        return []

    result: List[Dict[str, Any]] = []
    for face_id, face_data in faces.items():
        area = face_data.get("facial_area", [])
        if not isinstance(area, (list, tuple)) or len(area) < 4:
            continue
        landmarks = face_data.get("landmarks", {})
        score = face_data.get("score", 0.0)
        result.append(
            {
                "id": str(face_id),
                "bbox": {
                    "x": int(area[0]),
                    "y": int(area[1]),
                    "w": int(area[2] - area[0]),
                    "h": int(area[3] - area[1]),
                },
                "confidence": float(score),
                "landmarks": landmarks if isinstance(landmarks, dict) else {},
            }
        )
    return result


def _detect_faces_opencv(image: bytes) -> List[Dict[str, Any]]:
    try:
        import cv2
    except Exception:
        return []

    img_np = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if img_np is None:
        return []

    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return [
        {
            "id": f"face_{i}",
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "confidence": 0.8,
            "landmarks": {},
        }
        for i, (x, y, w, h) in enumerate(faces)
    ]
