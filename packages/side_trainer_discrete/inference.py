"""Checkpoint-safe inference for one discrete landmark specialist."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from .config import resolve_device
from .model import DiscreteLandmarkModel, soft_argmax_2d


class DiscreteLandmarkPredictor:
    def __init__(self, checkpoint_path: str | Path, *, device: str = "auto") -> None:
        self.device = resolve_device(device)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if checkpoint.get("format") != "mini-faceiq-discrete-v1":
            raise ValueError("Not a Mini-FaceIQ discrete landmark checkpoint")
        self.landmark_id = str(checkpoint["landmark_id"])
        self.image_size = int(checkpoint["config"]["image_size"])
        self.model = DiscreteLandmarkModel(pretrained=False).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict[str, Any]:
        rgb = image.convert("RGB")
        original_width, original_height = rgb.size
        resized = rgb.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32).copy()
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).contiguous() / 255.0
        logits = self.model(tensor.to(self.device))
        coordinates, confidence = soft_argmax_2d(logits)
        normalized_x, normalized_y = coordinates[0, 0].cpu().tolist()
        return {
            "landmark": self.landmark_id,
            "x": normalized_x * (original_width - 1),
            "y": normalized_y * (original_height - 1),
            "normalized_x": normalized_x,
            "normalized_y": normalized_y,
            "confidence": float(confidence[0, 0].cpu()),
        }


def checkpoint_summary(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    return {
        "format": checkpoint.get("format"),
        "landmark": checkpoint.get("landmark_id"),
        "epoch": checkpoint.get("epoch"),
        "labels": checkpoint.get("label_count"),
        "best_validation_nme": checkpoint.get("best_validation_nme"),
    }
