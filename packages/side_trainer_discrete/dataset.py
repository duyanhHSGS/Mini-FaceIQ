"""Dataset adapter from one human-data JSONL shard to one model target."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from .config import encoded_seed


@dataclass(frozen=True)
class LandmarkRecord:
    image_name: str
    image_path: Path
    point_xy: tuple[float, float]
    subject_id: str


@dataclass(frozen=True)
class SubjectSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    seed_text: str
    seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "seed_text": self.seed_text,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class AugmentationSettings:
    enabled: bool = False
    rotation_degrees: float = 7.0
    translation_fraction: float = 0.035
    scale_jitter: float = 0.07
    brightness_jitter: float = 0.12
    contrast_jitter: float = 0.12
    blur_probability: float = 0.08


def load_records(shard_path: str | Path, images_root: str | Path) -> list[LandmarkRecord]:
    shard_path = Path(shard_path).expanduser().resolve()
    images_root = Path(images_root).expanduser().resolve()
    landmark_id = shard_path.stem
    records: list[LandmarkRecord] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(shard_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{shard_path.name}:{line_number} is invalid JSON") from exc
        if value.get("landmark") != landmark_id:
            raise ValueError(f"{shard_path.name}:{line_number} contains the wrong landmark")
        image_name = str(value.get("image", ""))
        if image_name in seen:
            raise ValueError(f"Duplicate label for {image_name}")
        seen.add(image_name)
        state = value.get("state")
        if state == "unavailable":
            continue
        if state != "placed":
            raise ValueError(f"{shard_path.name}:{line_number} has invalid state {state!r}")
        try:
            x, y = float(value["x"]), float(value["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{shard_path.name}:{line_number} has invalid coordinates") from exc
        image_path = (images_root / image_name).resolve()
        try:
            image_path.relative_to(images_root)
        except ValueError as exc:
            raise ValueError(f"Unsafe image path: {image_name}") from exc
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing labeled image: {image_path}")
        subject_id = Path(image_name).stem.split("_")[0]
        records.append(LandmarkRecord(image_name, image_path, (x, y), subject_id))
    if not records:
        raise ValueError(f"No placed labels found in {shard_path}")
    return records


def make_subject_split(records: list[LandmarkRecord], seed_text: str) -> SubjectSplit:
    subjects = sorted({record.subject_id for record in records})
    if len(subjects) < 3:
        raise ValueError("At least three labeled subjects are required")
    rng = random.Random(encoded_seed(seed_text))
    rng.shuffle(subjects)
    test_count = max(1, round(len(subjects) * 0.15))
    validation_count = max(1, round(len(subjects) * 0.15))
    train_count = len(subjects) - validation_count - test_count
    if train_count < 1:
        raise ValueError("Not enough subjects for train/validation/test")
    return SubjectSplit(
        train=tuple(sorted(subjects[:train_count])),
        validation=tuple(sorted(subjects[train_count:train_count + validation_count])),
        test=tuple(sorted(subjects[train_count + validation_count:])),
        seed_text=seed_text,
        seed=encoded_seed(seed_text),
    )


def indices_for_subjects(records: list[LandmarkRecord], subjects: tuple[str, ...]) -> list[int]:
    allowed = set(subjects)
    return [index for index, record in enumerate(records) if record.subject_id in allowed]


def _augment(image: Image.Image, point: np.ndarray, settings: AugmentationSettings) -> tuple[Image.Image, np.ndarray]:
    if not settings.enabled:
        return image, point
    width, height = image.size
    angle = random.uniform(-settings.rotation_degrees, settings.rotation_degrees)
    scale = random.uniform(1.0 - settings.scale_jitter, 1.0 + settings.scale_jitter)
    translation = settings.translation_fraction
    tx = random.uniform(-translation, translation) * width
    ty = random.uniform(-translation, translation) * height
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    pixels = cv2.warpAffine(
        np.asarray(image), matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )
    pixel_point = np.array([point[0] * (width - 1), point[1] * (height - 1), 1.0])
    transformed = matrix @ pixel_point
    normalized = np.array(
        [transformed[0] / (width - 1), transformed[1] / (height - 1)],
        dtype=np.float32,
    )
    if not np.logical_and(normalized >= 0.0, normalized <= 1.0).all():
        return image, point
    result = Image.fromarray(pixels.astype(np.uint8), mode="RGB")
    if settings.brightness_jitter:
        result = ImageEnhance.Brightness(result).enhance(
            random.uniform(1 - settings.brightness_jitter, 1 + settings.brightness_jitter)
        )
    if settings.contrast_jitter:
        result = ImageEnhance.Contrast(result).enhance(
            random.uniform(1 - settings.contrast_jitter, 1 + settings.contrast_jitter)
        )
    if random.random() < settings.blur_probability:
        result = result.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 1.0)))
    return result, normalized


class DiscreteLandmarkDataset(Dataset):
    """One output channel and one coordinate target per placed human label."""

    def __init__(
        self,
        shard_path: str | Path,
        images_root: str | Path,
        *,
        image_size: int = 256,
        augmentation: AugmentationSettings | None = None,
    ) -> None:
        self.records = load_records(shard_path, images_root)
        self.image_size = image_size
        self.augmentation = augmentation or AugmentationSettings()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with Image.open(record.image_path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            x, y = record.point_xy
            if not (0 <= x <= width - 1 and 0 <= y <= height - 1):
                raise ValueError(f"Label outside image: {record.image_name} -> {(x, y)}")
            point = np.array([x / (width - 1), y / (height - 1)], dtype=np.float32)
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        image, point = _augment(image, point, self.augmentation)
        pixels = np.asarray(image, dtype=np.float32).copy()
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous() / 255.0
        return {
            "image": tensor,
            "target": torch.from_numpy(point).view(1, 2),
            "image_name": record.image_name,
            "original_size": torch.tensor([width, height], dtype=torch.float32),
        }
