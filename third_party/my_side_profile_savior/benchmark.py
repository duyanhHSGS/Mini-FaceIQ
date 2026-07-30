"""Frozen custom-versus-legacy landmark benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .dataset import ProfileLandmarkDataset, SubjectSplit, subject_disjoint_indices
from .factory_config import atomic_json_save
from .inference import FactoryPredictor
from .mapping import LandmarkMapping


PCK_THRESHOLDS = (0.02, 0.05, 0.10)
FAILURE_PENALTY = 1.0


def normalized_landmark_error(
    predicted_xy: np.ndarray | None,
    target_xy: np.ndarray,
    *,
    crop_xyxy: np.ndarray,
    bbox_xyxy: np.ndarray,
) -> float:
    if predicted_xy is None:
        return FAILURE_PENALTY
    predicted = np.asarray(predicted_xy, dtype=np.float64)
    target = np.asarray(target_xy, dtype=np.float64)
    if (
        predicted.shape != (2,)
        or not np.isfinite(predicted).all()
        or np.any(predicted < 0.0)
        or np.any(predicted > 1.0)
    ):
        return FAILURE_PENALTY

    crop_width = float(crop_xyxy[2] - crop_xyxy[0])
    crop_height = float(crop_xyxy[3] - crop_xyxy[1])
    bbox_width = float(bbox_xyxy[2] - bbox_xyxy[0])
    bbox_height = float(bbox_xyxy[3] - bbox_xyxy[1])
    denominator = float(np.hypot(bbox_width, bbox_height))
    if denominator <= 0:
        return FAILURE_PENALTY
    difference_pixels = (predicted - target) * np.asarray(
        [crop_width, crop_height],
        dtype=np.float64,
    )
    error = float(np.linalg.norm(difference_pixels) / denominator)
    return error if np.isfinite(error) else FAILURE_PENALTY


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _legacy_fingerprint(
    annotation_path: Path,
    mapping: LandmarkMapping,
    dataset: ProfileLandmarkDataset,
) -> str:
    provider_path = (
        Path(__file__).resolve().parents[1] / "providers" / "side_3ddfa.py"
    )
    payload = {
        "annotation_sha256": _file_digest(annotation_path),
        "mapping": mapping.snapshot(),
        "provider_sha256": _file_digest(provider_path),
        "image_size": dataset.image_size,
        "bbox_scale": dataset.bbox_scale,
        "failure_penalty": FAILURE_PENALTY,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _legacy_on_crop(crop: Image.Image) -> dict[str, Any]:
    from third_party.providers.side_3ddfa import _detect_side_from_image

    rgb = np.asarray(crop.convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return _detect_side_from_image(bgr)


def _empty_accumulator(mapping: LandmarkMapping) -> dict[str, list[float]]:
    return {entry.name: [] for entry in mapping.confirmed_entries}


def _summarize(errors: dict[str, list[float]]) -> dict[str, Any]:
    per_landmark: dict[str, Any] = {}
    all_errors: list[float] = []
    for name, values in errors.items():
        array = np.asarray(values, dtype=np.float64)
        all_errors.extend(array.tolist())
        per_landmark[name] = {
            "count": int(len(array)),
            "mean_nme": float(array.mean()) if len(array) else FAILURE_PENALTY,
            "failures": int(np.sum(array >= FAILURE_PENALTY)),
            "pck": {
                f"{threshold:.2f}": float(np.mean(array <= threshold))
                if len(array)
                else 0.0
                for threshold in PCK_THRESHOLDS
            },
        }
    aggregate = np.asarray(all_errors, dtype=np.float64)
    return {
        "per_landmark": per_landmark,
        "aggregate": {
            "count": int(len(aggregate)),
            "mean_nme": float(aggregate.mean())
            if len(aggregate)
            else FAILURE_PENALTY,
            "failures": int(np.sum(aggregate >= FAILURE_PENALTY)),
            "pck": {
                f"{threshold:.2f}": float(np.mean(aggregate <= threshold))
                if len(aggregate)
                else 0.0
                for threshold in PCK_THRESHOLDS
            },
        },
    }


def per_landmark_wins(
    custom_summary: dict[str, Any],
    legacy_summary: dict[str, Any],
    landmark_names: list[str],
) -> dict[str, bool]:
    return {
        name: (
            float(custom_summary["per_landmark"][name]["mean_nme"])
            < float(legacy_summary["per_landmark"][name]["mean_nme"])
        )
        for name in landmark_names
    }


def run_benchmark(
    *,
    checkpoint_path: str | Path,
    dataset: ProfileLandmarkDataset,
    split: SubjectSplit,
    mapping: LandmarkMapping,
    output_path: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    """Compare confirmed custom points with legacy 3DDFA on annotation crops."""

    output_path = Path(output_path).expanduser().resolve()
    cache_path = output_path.with_name("legacy_baseline_cache.json")
    fingerprint = _legacy_fingerprint(dataset.annotation_path, mapping, dataset)
    legacy_cache: dict[str, Any] = {}
    if cache_path.is_file():
        with cache_path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("fingerprint") == fingerprint:
            legacy_cache = dict(cached.get("samples", {}))

    predictor = FactoryPredictor(checkpoint_path, device=device)
    custom_errors = _empty_accumulator(mapping)
    legacy_errors = _empty_accumulator(mapping)
    test_indices = subject_disjoint_indices(dataset.records, set(split.test))

    for dataset_index in test_indices:
        sample = dataset[dataset_index]
        record = dataset.records[dataset_index]
        crop_box = tuple(int(value) for value in sample["crop_xyxy"].tolist())
        with Image.open(record.image_path) as source:
            crop = source.convert("RGB").crop(crop_box)

        custom = predictor.predict_crop(crop)
        custom_by_name = {
            item["name"]: np.asarray([item["x"], item["y"]], dtype=np.float64)
            for item in custom["predictions"]
        }

        cache_key = record.relative_image_path
        legacy_result = legacy_cache.get(cache_key)
        if legacy_result is None:
            raw_legacy_result = _legacy_on_crop(crop)
            legacy_result = {
                "error": raw_legacy_result.get("error"),
                "landmarks": raw_legacy_result.get("landmarks", {}),
            }
            legacy_cache[cache_key] = legacy_result

        legacy_landmarks = legacy_result.get("landmarks", {})
        targets = sample["landmarks"].numpy()
        crop_xyxy = sample["crop_xyxy"].numpy()
        bbox_xyxy = sample["bbox_xyxy"].numpy()
        for entry in mapping.confirmed_entries:
            target = targets[int(entry.dataset_index)]
            custom_errors[entry.name].append(
                normalized_landmark_error(
                    custom_by_name.get(entry.name),
                    target,
                    crop_xyxy=crop_xyxy,
                    bbox_xyxy=bbox_xyxy,
                )
            )
            legacy_item = legacy_landmarks.get(entry.name)
            legacy_xy = None
            if legacy_item is not None:
                legacy_xy = np.asarray(
                    [legacy_item.get("x"), legacy_item.get("y")],
                    dtype=np.float64,
                )
            legacy_errors[entry.name].append(
                normalized_landmark_error(
                    legacy_xy,
                    target,
                    crop_xyxy=crop_xyxy,
                    bbox_xyxy=bbox_xyxy,
                )
            )

    atomic_json_save(
        {"fingerprint": fingerprint, "samples": legacy_cache},
        cache_path,
    )
    custom_summary = _summarize(custom_errors)
    legacy_summary = _summarize(legacy_errors)
    wins = per_landmark_wins(
        custom_summary,
        legacy_summary,
        [entry.name for entry in mapping.confirmed_entries],
    )
    result = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "test_subjects": list(split.test),
        "test_images": len(test_indices),
        "confirmed_count": mapping.confirmed_count,
        "failure_penalty_nme": FAILURE_PENALTY,
        "pck_thresholds": list(PCK_THRESHOLDS),
        "custom": custom_summary,
        "legacy": legacy_summary,
        "per_landmark_wins": wins,
        "graduated": bool(wins) and all(wins.values()),
    }
    atomic_json_save(result, output_path)
    return result
