"""Checkpoint inference for the private side-profile model factory."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch

from .dataset import _pil_to_float_tensor, _square_crop_box
from .factory_config import resolve_device
from .model import LANDMARK_COUNT, ProfileLandmarkModel, soft_argmax_2d


@lru_cache(maxsize=1)
def _faceboxes_model():
    from third_party.providers.side_3ddfa import _load_faceboxes_onnx_class

    return _load_faceboxes_onnx_class()()


def _output_layout_from_checkpoint(
    checkpoint: dict[str, Any],
) -> list[dict[str, Any]]:
    mapping_snapshot = checkpoint.get("mapping", {})
    try:
        mapping_count = int(mapping_snapshot.get("landmark_count", -1))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Checkpoint mapping snapshot is invalid") from exc
    if mapping_count != LANDMARK_COUNT:
        raise ValueError(
            f"Checkpoint must describe exactly {LANDMARK_COUNT} model outputs"
        )
    mapping_entries = mapping_snapshot.get("entries")
    if (
        not isinstance(mapping_entries, list)
        or len(mapping_entries) != LANDMARK_COUNT
    ):
        raise ValueError(
            f"Checkpoint mapping must contain {LANDMARK_COUNT} entries"
        )
    entries = checkpoint.get("output_layout")
    if not isinstance(entries, list) or len(entries) != LANDMARK_COUNT:
        raise ValueError(
            f"Checkpoint output_layout must contain {LANDMARK_COUNT} entries"
        )

    layout: list[dict[str, Any]] = []
    for expected_index, (entry, mapping_entry) in enumerate(
        zip(entries, mapping_entries)
    ):
        if not isinstance(entry, dict) or not isinstance(mapping_entry, dict):
            raise ValueError("Checkpoint layout entries must be objects")
        try:
            model_index = int(entry["model_index"])
            name = str(entry["name"])
            mapping_model_index = int(mapping_entry["model_index"])
            mapping_name = str(mapping_entry["name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Checkpoint layout entries require model_index and name"
            ) from exc
        if (
            model_index != expected_index
            or mapping_model_index != expected_index
            or mapping_name != name
            or mapping_entry.get("dataset_index") != entry.get("dataset_index")
        ):
            raise ValueError(
                "Checkpoint mapping and output_layout must agree in 0-30 order"
            )
        dataset_index = entry.get("dataset_index")
        if dataset_index is not None:
            layout.append(
                {
                    "model_index": model_index,
                    "dataset_index": int(dataset_index),
                    "name": name,
                }
            )
    if not layout:
        raise ValueError("Checkpoint has no confirmed mapping entries")
    return layout


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
            item["dataset_index"]: item["name"] for item in self.output_layout
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
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = self.model(tensor)
            coordinates, confidence = soft_argmax_2d(logits)
        coordinates_np = coordinates[0].float().cpu().numpy()
        confidence_np = confidence[0].float().cpu().numpy()
        if mirror:
            coordinates_np[:, 0] = 1.0 - coordinates_np[:, 0]

        predictions = []
        for item in self.output_layout:
            model_index = int(item["model_index"])
            dataset_index = int(item["dataset_index"])
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
        }

    def predict_upload(
        self,
        image: Image.Image,
        *,
        mirror: bool = False,
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
        result = self.predict_crop(crop, mirror=mirror)

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
        "confirmed_count": len(layout),
        "output_count": LANDMARK_COUNT,
        "confirmed_mapping": {
            item["dataset_index"]: item["name"] for item in layout
        },
        "config": checkpoint.get("config", {}),
    }
