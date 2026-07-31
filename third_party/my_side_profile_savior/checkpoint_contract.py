"""One strict checkpoint-layout contract for the 31-slot profile factory."""

from __future__ import annotations

from typing import Any

from .model import LANDMARK_COUNT


def validate_checkpoint_layout(
    checkpoint: dict[str, Any],
    *,
    expected_layout: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Validate and return every active output in model-index order.

    Dataset indices are optional provenance. They never decide whether an
    output exists; the explicit boolean ``active`` field does.
    """

    mapping = checkpoint.get("mapping")
    if not isinstance(mapping, dict):
        raise ValueError("Checkpoint mapping snapshot must be an object")
    if mapping.get("landmark_count") != LANDMARK_COUNT:
        raise ValueError(
            f"Checkpoint mapping.landmark_count must equal {LANDMARK_COUNT}"
        )
    mapping_entries = mapping.get("entries")
    layout = checkpoint.get("output_layout")
    if not isinstance(mapping_entries, list) or len(mapping_entries) != LANDMARK_COUNT:
        raise ValueError(
            f"Checkpoint mapping must contain exactly {LANDMARK_COUNT} entries"
        )
    if not isinstance(layout, list) or len(layout) != LANDMARK_COUNT:
        raise ValueError(
            f"Checkpoint output_layout must contain exactly {LANDMARK_COUNT} entries"
        )
    if expected_layout is not None and len(expected_layout) != LANDMARK_COUNT:
        raise ValueError(
            f"Expected output layout must contain exactly {LANDMARK_COUNT} entries"
        )

    active: list[dict[str, Any]] = []
    for index in range(LANDMARK_COUNT):
        mapped = mapping_entries[index]
        slot = layout[index]
        if not isinstance(mapped, dict) or not isinstance(slot, dict):
            raise ValueError("Checkpoint mapping and layout entries must be objects")
        try:
            mapped_index = int(mapped["model_index"])
            slot_index = int(slot["model_index"])
            mapped_name = str(mapped["name"])
            slot_name = str(slot["name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Checkpoint mapping and layout entries require model_index and name"
            ) from exc
        if mapped_index != index or slot_index != index:
            raise ValueError("Checkpoint model indices must be ordered 0-30")
        if mapped_name != slot_name:
            raise ValueError("Checkpoint mapping and output_layout names must agree")
        if mapped.get("dataset_index") != slot.get("dataset_index"):
            raise ValueError(
                "Checkpoint mapping and output_layout dataset_index values must agree"
            )
        if type(slot.get("active")) is not bool:
            raise ValueError("Every checkpoint output_layout slot needs boolean active")
        if expected_layout is not None:
            expected = expected_layout[index]
            comparable_keys = ("model_index", "name", "dataset_index", "active")
            if any(slot.get(key) != expected.get(key) for key in comparable_keys):
                raise ValueError(
                    "Checkpoint output_layout does not match the requested 31-slot schema"
                )
        if slot["active"]:
            active.append(
                {
                    "model_index": index,
                    "name": slot_name,
                    "dataset_index": slot.get("dataset_index"),
                }
            )
    if not active:
        raise ValueError("Checkpoint has no active output slots")
    return active
