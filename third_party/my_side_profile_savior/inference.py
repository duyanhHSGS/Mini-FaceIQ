"""Checkpoint inference for the private side-profile model factory."""

from __future__ import annotations

import base64
from functools import lru_cache
import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch

from .dataset import _pil_to_float_tensor, _square_crop_box
from .checkpoint_contract import validate_checkpoint_layout
from .factory_config import resolve_device
from .model import LANDMARK_COUNT, ProfileLandmarkModel, soft_argmax_2d


@lru_cache(maxsize=1)
def _faceboxes_model():
    from third_party.providers.side_3ddfa import _load_faceboxes_onnx_class

    return _load_faceboxes_onnx_class()()


def _output_layout_from_checkpoint(
    checkpoint: dict[str, Any],
) -> list[dict[str, Any]]:
    return validate_checkpoint_layout(checkpoint)


def _load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Factory checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state" not in checkpoint:
        raise ValueError(f"Not a factory checkpoint: {checkpoint_path}")
    return checkpoint


class FactoryPredictor:
    """Cached model wrapper that safely loads CUDA checkpoints on CPU."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "auto") -> None:
        self.device = resolve_device(device)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.checkpoint = _load_checkpoint(self.checkpoint_path, self.device)
        self.config = dict(self.checkpoint.get("config", {}))
        self.image_size = int(self.config.get("image_size", 256))
        self.bbox_scale = float(self.config.get("bbox_scale", 1.25))
        self.output_layout = _output_layout_from_checkpoint(self.checkpoint)
        self.names_by_index = {
            item["model_index"]: item["name"] for item in self.output_layout
        }

        self.model = ProfileLandmarkModel(
            landmark_count=LANDMARK_COUNT,
            pretrained=False,
        )
        self.model.load_state_dict(self.checkpoint["model_state"], strict=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_crop(
        self,
        image: Image.Image,
        *,
        mirror: bool = False,
        selected_heatmap_index: int | None = None,
    ) -> dict[str, Any]:
        original = image.convert("RGB")
        model_image = ImageOps.mirror(original) if mirror else original
        resized = model_image.resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.BILINEAR,
        )
        tensor = _pil_to_float_tensor(resized).unsqueeze(0).to(
            self.device,
            non_blocking=self.device.type == "cuda",
        )
        amp_enabled = self.device.type == "cuda" and bool(
            self.config.get("amp", True)
        )
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = self.model(tensor)
            coordinates, confidence = soft_argmax_2d(logits)
        selected_heatmap_png = None
        if selected_heatmap_index is not None:
            if not 0 <= selected_heatmap_index < LANDMARK_COUNT:
                raise ValueError("selected_heatmap_index must be 0-30")
            heatmap = logits[0, selected_heatmap_index].float()
            heatmap = heatmap - heatmap.min()
            heatmap = heatmap / heatmap.max().clamp_min(1e-12)
            heatmap_pixels = (
                heatmap.mul(255).byte().cpu().numpy()
            )
            buffer = io.BytesIO()
            Image.fromarray(heatmap_pixels, mode="L").save(buffer, format="PNG")
            selected_heatmap_png = (
                "data:image/png;base64,"
                + base64.b64encode(buffer.getvalue()).decode("ascii")
            )
        coordinates_np = coordinates[0].float().cpu().numpy()
        confidence_np = confidence[0].float().cpu().numpy()
        if mirror:
            coordinates_np[:, 0] = 1.0 - coordinates_np[:, 0]

        predictions = []
        for item in self.output_layout:
            model_index = int(item["model_index"])
            dataset_index = item.get("dataset_index")
            name = str(item["name"])
            x, y = coordinates_np[model_index]
            predictions.append(
                {
                    "name": name,
                    "model_index": model_index,
                    "dataset_index": dataset_index,
                    "x": float(x),
                    "y": float(y),
                    "confidence": float(confidence_np[model_index]),
                }
            )
        return {
            "predictions": predictions,
            "all_coordinates": coordinates_np,
            "all_confidence": confidence_np,
            "device": str(self.device),
            "mirror": mirror,
            "checkpoint": str(self.checkpoint_path),
            "selected_heatmap_index": selected_heatmap_index,
            "selected_heatmap_png": selected_heatmap_png,
        }

    def predict_upload(
        self,
        image: Image.Image,
        *,
        mirror: bool = False,
        selected_heatmap_index: int | None = None,
    ) -> dict[str, Any]:
        original = image.convert("RGB")
        rgb = np.asarray(original, dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        boxes = _faceboxes_model()(bgr)
        if not boxes:
            return {
                "error": "FaceBoxes could not find a face",
                "device": str(self.device),
            }
        best_box = max(
            boxes,
            key=lambda box: float(box[2] - box[0]) * float(box[3] - box[1]),
        )
        bbox = np.asarray(best_box[:4], dtype=np.float32)
        crop_box = _square_crop_box(bbox, scale=self.bbox_scale)
        crop = original.crop(crop_box)
        result = self.predict_crop(
            crop,
            mirror=mirror,
            selected_heatmap_index=selected_heatmap_index,
        )

        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        crop_width = float(crop_x2 - crop_x1)
        crop_height = float(crop_y2 - crop_y1)
        image_width, image_height = original.size
        for prediction in result["predictions"]:
            prediction["crop_x"] = prediction["x"]
            prediction["crop_y"] = prediction["y"]
            prediction["x"] = (
                crop_x1 + prediction["crop_x"] * crop_width
            ) / image_width
            prediction["y"] = (
                crop_y1 + prediction["crop_y"] * crop_height
            ) / image_height
        result["bbox_xyxy"] = bbox.tolist()
        result["crop_xyxy"] = list(crop_box)
        result["image_size"] = [image_width, image_height]
        return result


def checkpoint_summary(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(
        Path(path).expanduser().resolve(),
        map_location="cpu",
    )
    layout = _output_layout_from_checkpoint(checkpoint)
    return {
        "epoch": int(checkpoint.get("epoch", -1)),
        "best_validation_nme": checkpoint.get("best_validation_nme"),
        "active_count": len(layout),
        "output_count": LANDMARK_COUNT,
        "active_mapping": {
            item["model_index"]: item["name"] for item in layout
        },
        "config": checkpoint.get("config", {}),
    }
