"""Configuration and atomic artifact helpers for discrete landmark models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

try:
    from ..human_data_factory.schema import LANDMARKS
except ImportError:  # Direct package execution fallback.
    from human_data_factory.schema import LANDMARKS


PACKAGE_DIR = Path(__file__).resolve().parent
HUMAN_FACTORY_DIR = PACKAGE_DIR.parent / "human_data_factory"
DEFAULT_IMAGES_ROOT = HUMAN_FACTORY_DIR / "source_images" / "multipie"
DEFAULT_CONTRIBUTIONS_ROOT = (
    HUMAN_FACTORY_DIR / "contributions" / "multipie-profile-v1"
)
DEFAULT_RUNS_ROOT = PACKAGE_DIR / "discrete_factory_runs"
LANDMARK_IDS = tuple(str(item["id"]) for item in LANDMARKS)


def encoded_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def seed_everything(seed_text: str) -> int:
    seed = encoded_seed(seed_text)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    return seed


def resolve_device(requested: str) -> torch.device:
    value = requested.strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot see a CUDA GPU")
    if value not in {"cuda", "cpu"}:
        raise ValueError("device must be auto, cuda, or cpu")
    return torch.device(value)


@dataclass
class TrainerConfig:
    landmark_id: str = "porion"
    images_root: str = str(DEFAULT_IMAGES_ROOT)
    contributions_root: str = str(DEFAULT_CONTRIBUTIONS_ROOT)
    runs_root: str = str(DEFAULT_RUNS_ROOT)
    device: str = "auto"
    seed_text: str = "Mini-FaceIQ-discrete"
    image_size: int = 256
    heatmap_size: int = 64
    gaussian_sigma: float = 1.5
    map_loss_weight: float = 1.0
    coordinate_loss_weight: float = 1.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    workers: int = 4
    epochs: int = 150
    early_stopping_patience: int = 20
    pretrained: bool = True
    amp: bool = True
    augmentation: bool = True
    rotation_degrees: float = 7.0
    translation_fraction: float = 0.035
    scale_jitter: float = 0.07
    brightness_jitter: float = 0.12
    contrast_jitter: float = 0.12
    blur_probability: float = 0.08
    resume_checkpoint: str = ""

    @property
    def shard_path(self) -> Path:
        return Path(self.contributions_root) / f"{self.landmark_id}.jsonl"

    @property
    def landmark_runs_root(self) -> Path:
        return Path(self.runs_root) / self.landmark_id

    def validate(self) -> None:
        if self.landmark_id not in LANDMARK_IDS:
            raise ValueError(f"Unknown landmark: {self.landmark_id}")
        if self.image_size <= 0 or self.image_size % 32:
            raise ValueError("image_size must be positive and divisible by 32")
        if self.heatmap_size != self.image_size // 4:
            raise ValueError("heatmap_size must equal image_size / 4")
        if self.batch_size <= 0 or self.epochs <= 0 or self.workers < 0:
            raise ValueError("batch_size/epochs must be positive; workers cannot be negative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if self.gaussian_sigma <= 0 or self.early_stopping_patience <= 0:
            raise ValueError("sigma and patience must be positive")
        if not self.shard_path.is_file():
            raise FileNotFoundError(f"Missing landmark shard: {self.shard_path}")
        if not Path(self.images_root).is_dir():
            raise FileNotFoundError(f"Missing image directory: {self.images_root}")
        resolve_device(self.device)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainerConfig":
        return cls(**values)


def create_run_directory(root: Path) -> Path:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = root / stamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stamp}-{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def atomic_json_save(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, destination)


def atomic_torch_save(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, destination)
