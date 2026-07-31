"""Pure annotator rules: coordinates, crops, states, consensus, and splits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
import re
from typing import Any, Iterable

from ..factory_config import DEFAULT_SEED_TEXT, encoded_seed


LANDMARK_STATES = {"unreviewed", "placed", "occluded", "out_of_frame"}
CONFIDENCE_STATES = {None, "easy", "uncertain", "hard"}
ORIGINS = {None, "manual", "accepted_suggestion", "adjusted_suggestion"}
FACINGS = {"left", "right"}
CROP_SCALE = 1.5
PROJECT_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")

LANDMARK_GROUPS = (
    ("Head", ("top_of_head", "occiput", "hairline_profile", "forehead", "glabella")),
    ("Eye", ("orbitale", "corneal_apex", "eyelid_end", "lower_eyelid", "cheekbone")),
    (
        "Nose",
        (
            "nasal_bridge_root", "rhinion", "supratip", "nose_tip", "infratip",
            "columella", "subnasale", "subalare",
        ),
    ),
    (
        "Mouth / chin",
        (
            "upper_lip", "mouth_corner", "lower_lip", "labiomental_fold",
            "chin_point", "chin_bottom",
        ),
    ),
    (
        "Neck / jaw",
        ("cervical_point", "neck_point", "lower_jaw_angle", "upper_jaw_angle"),
    ),
    ("Ear", ("tragus", "intertragic_notch", "porion")),
)
ANATOMICAL_SWEEP = tuple(name for _, names in LANDMARK_GROUPS for name in names)


def validate_project_id(value: str) -> str:
    normalized = value.strip().lower()
    if not PROJECT_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "project_id must be 1-64 lowercase letters, numbers, hyphens, or underscores"
        )
    return normalized


def validate_annotation(value: dict[str, Any]) -> dict[str, Any]:
    state = value.get("state")
    if state not in LANDMARK_STATES:
        raise ValueError(f"state must be one of: {sorted(LANDMARK_STATES)}")
    confidence = value.get("confidence")
    if confidence not in CONFIDENCE_STATES:
        raise ValueError("confidence must be easy, uncertain, hard, or null")
    origin = value.get("origin")
    if origin not in ORIGINS:
        raise ValueError(
            "origin must be manual, accepted_suggestion, adjusted_suggestion, or null"
        )
    x = value.get("x")
    y = value.get("y")
    if state == "placed":
        if isinstance(x, bool) or isinstance(y, bool):
            raise ValueError("placed coordinates must be finite numbers")
        try:
            x = float(x)
            y = float(y)
        except (TypeError, ValueError) as exc:
            raise ValueError("placed coordinates must be finite numbers") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("placed coordinates must be finite numbers")
        if origin is None:
            origin = "manual"
    elif x is not None or y is not None:
        raise ValueError(f"{state} annotations cannot contain coordinates")
    return {
        "state": state,
        "x": x if state == "placed" else None,
        "y": y if state == "placed" else None,
        "confidence": confidence,
        "origin": origin,
        "provider_fingerprint": value.get("provider_fingerprint"),
        "suggestion_exposed": bool(value.get("suggestion_exposed", False)),
    }


def square_crop(
    bbox: Iterable[float],
    image_width: int,
    image_height: int,
    *,
    scale: float = CROP_SCALE,
) -> list[float]:
    x1, y1, x2, y2 = (float(item) for item in bbox)
    if not (0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height):
        raise ValueError("bbox must be a non-empty rectangle inside the source image")
    side = max(x2 - x1, y2 - y1) * scale
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return [cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0]


def point_in_crop(x: float, y: float, crop: Iterable[float]) -> bool:
    x1, y1, x2, y2 = crop
    return x1 <= x <= x2 and y1 <= y <= y2


def source_to_view(
    x: float,
    y: float,
    *,
    width: float,
    mirrored: bool,
) -> tuple[float, float]:
    return ((width - x) if mirrored else x, y)


def view_to_source(
    x: float,
    y: float,
    *,
    width: float,
    mirrored: bool,
) -> tuple[float, float]:
    return source_to_view(x, y, width=width, mirrored=mirrored)


def source_to_crop(x: float, y: float, crop: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = crop
    return ((x - x1) / (x2 - x1), (y - y1) / (y2 - y1))


def crop_to_source(x: float, y: float, crop: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = crop
    return (x1 + x * (x2 - x1), y1 + y * (y2 - y1))


def deterministic_split(subject_ids: Iterable[str]) -> dict[str, Any]:
    subjects = sorted(set(subject_ids))
    if len(subjects) < 3:
        raise ValueError("At least three unique subjects are required")
    seed = encoded_seed(DEFAULT_SEED_TEXT)
    random.Random(seed).shuffle(subjects)
    train_count = max(1, int(round(len(subjects) * 0.70)))
    validation_count = max(1, int(round(len(subjects) * 0.15)))
    if train_count + validation_count >= len(subjects):
        validation_count = 1
        train_count = len(subjects) - 2
    return {
        "seed_text": DEFAULT_SEED_TEXT,
        "seed": seed,
        "train": sorted(subjects[:train_count]),
        "validation": sorted(subjects[train_count : train_count + validation_count]),
        "test": sorted(subjects[train_count + validation_count :]),
    }


@dataclass(frozen=True)
class Consensus:
    state: str
    x: float | None
    y: float | None
    requires_adjudication: bool
    reason: str


def consensus(first: dict[str, Any], second: dict[str, Any], bbox: Iterable[float]) -> Consensus:
    if first["state"] != second["state"]:
        return Consensus("unreviewed", None, None, True, "state_disagreement")
    state = first["state"]
    if state != "placed":
        return Consensus(state, None, None, False, "matching_state")
    x1, y1, x2, y2 = bbox
    diagonal = math.hypot(x2 - x1, y2 - y1)
    distance = math.hypot(first["x"] - second["x"], first["y"] - second["y"])
    if distance <= diagonal * 0.01:
        return Consensus(
            "placed",
            (first["x"] + second["x"]) / 2.0,
            (first["y"] + second["y"]) / 2.0,
            False,
            "within_one_percent",
        )
    return Consensus("unreviewed", None, None, True, "distance_disagreement")


def canonical_json_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
