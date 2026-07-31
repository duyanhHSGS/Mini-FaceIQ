"""Immutable human-export dataset with per-sample 31-slot visibility."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .dataset import (
    AugmentationSettings,
    SubjectSplit,
    _augment_crop,
    _normalize_points,
    _pil_to_float_tensor,
)
from .model import LANDMARK_COUNT


@dataclass(frozen=True)
class HumanAnnotation:
    image_path: Path
    relative_image_path: str
    sha256: str
    bbox_xyxy: np.ndarray
    crop_xyxy: np.ndarray
    landmarks_xy: np.ndarray
    visibility: np.ndarray
    subject_id: str
    camera_code: str
    split_name: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_export(export_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = export_dir / "manifest.json"
    labels_path = export_dir / "labels.jsonl"
    if not manifest_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError("Human export requires manifest.json and labels.jsonl")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("export_id") != export_dir.name:
        raise ValueError("Human export directory and manifest export_id disagree")
    mapping = manifest.get("landmark_snapshot", {})
    entries = mapping.get("entries") if isinstance(mapping, dict) else None
    if (
        mapping.get("landmark_count") != LANDMARK_COUNT
        or not isinstance(entries, list)
        or len(entries) != LANDMARK_COUNT
    ):
        raise ValueError("Human export must freeze exactly 31 landmark entries")
    rows = []
    with labels_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"labels.jsonl:{line_number}: invalid JSON") from exc
            if len(row.get("landmarks", [])) != LANDMARK_COUNT:
                raise ValueError(
                    f"labels.jsonl:{line_number}: expected 31 landmark objects"
                )
            rows.append(row)
    if not rows:
        raise ValueError("Human export contains no images")
    return manifest, rows


def _subject_split(manifest: dict[str, Any]) -> SubjectSplit:
    split = manifest.get("split", {})
    return SubjectSplit(
        train=tuple(split["train"]),
        validation=tuple(split["validation"]),
        test=tuple(split["test"]),
        seed_text=str(split["seed_text"]),
        seed=int(split["seed"]),
    )


class HumanExportLandmarkDataset(Dataset):
    """Load only explicit exported human truth; never consult Multi-PIE labels."""

    def __init__(
        self,
        export_path: str | Path,
        *,
        image_size: int = 256,
        verify_images: bool = True,
        augmentation: AugmentationSettings | None = None,
    ) -> None:
        self.export_dir = Path(export_path).expanduser().resolve()
        self.manifest, rows = _load_export(self.export_dir)
        self.export_id = str(self.manifest["export_id"])
        self.export_hash = _sha256_file(self.export_dir / "labels.jsonl")
        self.image_size = int(image_size)
        self.augmentation = augmentation or AugmentationSettings()
        self.split = _subject_split(self.manifest)
        source_kind = self.manifest.get("source", {}).get("kind")
        project_dir = self.export_dir.parents[1]
        repository_root = Path(__file__).resolve().parents[2]
        self.records: list[HumanAnnotation] = []
        for row in rows:
            relative = str(row["image_path"])
            image_path = (
                project_dir / relative
                if source_kind == "arbitrary"
                else repository_root / relative
            ).resolve()
            if verify_images:
                if not image_path.is_file():
                    raise FileNotFoundError(f"Export image is missing: {image_path}")
                if _sha256_file(image_path) != row["sha256"]:
                    raise ValueError(f"Export image hash mismatch: {image_path}")
            points = np.zeros((LANDMARK_COUNT, 2), dtype=np.float32)
            visibility = np.zeros(LANDMARK_COUNT, dtype=np.bool_)
            for expected_index, landmark in enumerate(row["landmarks"]):
                if int(landmark["model_index"]) != expected_index:
                    raise ValueError("Human export model indices must be ordered 0-30")
                if bool(landmark.get("training_visible")):
                    if landmark.get("truth_state") != "placed":
                        raise ValueError("training_visible requires placed truth")
                    points[expected_index] = (
                        float(landmark["truth_x"]),
                        float(landmark["truth_y"]),
                    )
                    visibility[expected_index] = True
            self.records.append(
                HumanAnnotation(
                    image_path=image_path,
                    relative_image_path=relative,
                    sha256=str(row["sha256"]),
                    bbox_xyxy=np.asarray(row["bbox"], dtype=np.float32),
                    crop_xyxy=np.asarray(row["crop"], dtype=np.float32),
                    landmarks_xy=points,
                    visibility=visibility,
                    subject_id=str(row["subject_id"]),
                    camera_code=str(row.get("camera") or ""),
                    split_name=str(row["split"]),
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        crop_box = tuple(float(value) for value in record.crop_xyxy)
        with Image.open(record.image_path) as source:
            crop = source.convert("RGB").crop(crop_box).resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BILINEAR,
            )
        landmarks = _normalize_points(record.landmarks_xy, crop_box)
        auxiliary = np.zeros((5, 2), dtype=np.float32)
        crop, landmarks_np, _ = _augment_crop(
            crop,
            landmarks.numpy(),
            auxiliary,
            self.augmentation,
        )
        visibility = torch.from_numpy(record.visibility.copy())
        finite_inside = (
            torch.isfinite(torch.from_numpy(landmarks_np)).all(dim=1)
            & (torch.from_numpy(landmarks_np)[:, 0] >= 0.0)
            & (torch.from_numpy(landmarks_np)[:, 0] <= 1.0)
            & (torch.from_numpy(landmarks_np)[:, 1] >= 0.0)
            & (torch.from_numpy(landmarks_np)[:, 1] <= 1.0)
        )
        visibility &= finite_inside
        landmarks_tensor = torch.from_numpy(landmarks_np).masked_fill(
            ~visibility.view(-1, 1), 0.0
        )
        return {
            "image": _pil_to_float_tensor(crop),
            "landmarks": landmarks_tensor,
            "landmarks_flat": landmarks_tensor.reshape(-1),
            "auxiliary_points": torch.zeros((5, 2), dtype=torch.float32),
            "visibility": visibility,
            "bbox_xyxy": torch.from_numpy(record.bbox_xyxy.copy()),
            "crop_xyxy": torch.from_numpy(record.crop_xyxy.copy()),
            "image_path": str(record.image_path),
            "relative_image_path": record.relative_image_path,
            "subject_id": record.subject_id,
            "camera_code": record.camera_code,
        }
