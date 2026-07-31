"""Configuration, device, seeding, and atomic artifact helpers."""

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


DEFAULT_SEED_TEXT = "Mini-FaceIQ"


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
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was explicitly requested, but torch.cuda.is_available() "
                "is false. Install a compatible CUDA-enabled PyTorch build or "
                "choose --device cpu/auto."
            )
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, cuda, cpu")


@dataclass
class FactoryConfig:
    annotation_path: str
    mapping_path: str
    runs_root: str
    device: str = "auto"
    seed_text: str = DEFAULT_SEED_TEXT
    image_size: int = 256
    heatmap_size: int = 64
    bbox_scale: float = 1.25
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
    rotation_degrees: float = 8.0
    translation_fraction: float = 0.04
    scale_jitter: float = 0.08
    brightness_jitter: float = 0.12
    contrast_jitter: float = 0.12
    blur_probability: float = 0.10
    resume_checkpoint: str = ""
    initial_checkpoint: str = ""
    run_benchmark: bool = True
    truth_source: str = "multipie"
    human_export_path: str = ""

    def validate(self) -> None:
        if self.image_size <= 0 or self.heatmap_size <= 0:
            raise ValueError("image_size and heatmap_size must be positive")
        if self.image_size % 32 != 0:
            raise ValueError("image_size must be divisible by 32")
        if self.heatmap_size != self.image_size // 4:
            raise ValueError(
                "MobileNetV3 decoder requires heatmap_size == image_size / 4"
            )
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.workers < 0:
            raise ValueError("workers cannot be negative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if self.gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be positive")
        if self.resume_checkpoint and self.initial_checkpoint:
            raise ValueError(
                "Choose either resume_checkpoint or initial_checkpoint, not both"
            )
        if self.truth_source not in {"multipie", "human"}:
            raise ValueError("truth_source must be multipie or human")
        if self.truth_source == "human" and not self.human_export_path:
            raise ValueError("human truth_source requires human_export_path")
        if self.truth_source == "multipie" and self.human_export_path:
            raise ValueError(
                "human_export_path is only valid when truth_source is human"
            )
        resolve_device(self.device)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FactoryConfig":
        return cls(**values)


def create_run_directory(runs_root: str | Path) -> Path:
    root = Path(runs_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}-{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def atomic_torch_save(value: object, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def atomic_json_save(value: object, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, destination)
