"""Cached, optional suggestion providers. Suggestions never mutate labels."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..inference import FactoryPredictor
from .projects import sha256_file


def provider_fingerprint(provider: str, checkpoint: str | None = None) -> str:
    payload: dict[str, Any] = {"provider": provider}
    if provider == "legacy":
        implementation = (
            Path(__file__).resolve().parents[2] / "providers" / "side_3ddfa.py"
        )
        payload["implementation_sha256"] = sha256_file(implementation)
    elif provider == "custom":
        if not checkpoint:
            raise ValueError("custom suggestions require a checkpoint")
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        payload["checkpoint"] = str(checkpoint_path)
        payload["checkpoint_sha256"] = sha256_file(checkpoint_path)
    else:
        raise ValueError("provider must be legacy or custom")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def cache_key(image_hash: str, bbox: list[float], fingerprint: str) -> str:
    bbox_hash = hashlib.sha256(
        json.dumps([round(value, 6) for value in bbox]).encode("utf-8")
    ).hexdigest()
    return hashlib.sha256(
        f"{image_hash}:{bbox_hash}:{fingerprint}".encode("utf-8")
    ).hexdigest()


@lru_cache(maxsize=3)
def _custom_predictor(checkpoint: str) -> FactoryPredictor:
    return FactoryPredictor(checkpoint)


def legacy_suggestions(image: Image.Image) -> dict[str, dict[str, Any]]:
    from third_party.providers.side_3ddfa import _detect_side_from_image

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    result = _detect_side_from_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if result.get("error"):
        raise RuntimeError(str(result["error"]))
    width, height = image.size
    return {
        name: {
            "x": float(item["x"]) * width,
            "y": float(item["y"]) * height,
            "confidence": item.get("confidence"),
        }
        for name, item in result.get("landmarks", {}).items()
    }


def custom_suggestions(
    image: Image.Image,
    checkpoint: str,
    *,
    selected_landmark_index: int,
) -> dict[int, dict[str, Any]]:
    result = _custom_predictor(
        str(Path(checkpoint).expanduser().resolve())
    ).predict_upload(
        image,
        selected_heatmap_index=selected_landmark_index,
    )
    if result.get("error"):
        raise RuntimeError(str(result["error"]))
    width, height = image.size
    predictions = {
        int(item["model_index"]): {
            "name": item["name"],
            "x": float(item["x"]) * width,
            "y": float(item["y"]) * height,
            "confidence": item.get("confidence"),
        }
        for item in result["predictions"]
    }
    selected = predictions.get(selected_landmark_index)
    if selected is not None:
        selected["heatmap_png"] = result.get("selected_heatmap_png")
    return predictions


@lru_cache(maxsize=1)
def _faceboxes():
    from third_party.providers.side_3ddfa import _load_faceboxes_onnx_class

    return _load_faceboxes_onnx_class()()


def detect_bbox(image: Image.Image) -> list[float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    boxes = _faceboxes()(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not boxes:
        raise RuntimeError("FaceBoxes could not find a face")
    best = max(
        boxes,
        key=lambda box: float(box[2] - box[0]) * float(box[3] - box[1]),
    )
    return [float(value) for value in best[:4]]
