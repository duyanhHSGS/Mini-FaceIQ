"""Parse the human-owned side-profile landmark worksheet."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch


DATASET_LANDMARK_COUNT = 39
INACTIVE_VALUES = {"", "NONE", "?"}


@dataclass(frozen=True)
class LandmarkMappingEntry:
    model_index: int
    name: str
    legacy_source: str
    legacy_reference: str
    dataset_index: int | None
    user_value: str

    @property
    def confirmed(self) -> bool:
        return self.dataset_index is not None


@dataclass(frozen=True)
class LandmarkMapping:
    source_path: Path
    entries: tuple[LandmarkMappingEntry, ...]

    @property
    def confirmed_entries(self) -> tuple[LandmarkMappingEntry, ...]:
        return tuple(entry for entry in self.entries if entry.confirmed)

    @property
    def confirmed_count(self) -> int:
        return len(self.confirmed_entries)

    @property
    def names_by_dataset_index(self) -> dict[int, str]:
        return {
            int(entry.dataset_index): entry.name
            for entry in self.confirmed_entries
            if entry.dataset_index is not None
        }

    @property
    def names_by_model_index(self) -> dict[int, str]:
        return {entry.model_index: entry.name for entry in self.entries}

    def active_mask(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        mask = torch.zeros(len(self.entries), dtype=dtype)
        for entry in self.confirmed_entries:
            mask[entry.model_index] = 1
        return mask

    def output_layout(self) -> list[dict[str, object]]:
        return [
            {
                "model_index": entry.model_index,
                "name": entry.name,
                "dataset_index": entry.dataset_index,
                "active": entry.confirmed,
            }
            for entry in self.entries
        ]

    def snapshot(self) -> dict[str, object]:
        return {
            "source_path": str(self.source_path),
            "landmark_count": len(self.entries),
            "dataset_landmark_count": DATASET_LANDMARK_COUNT,
            "confirmed_count": self.confirmed_count,
            "entries": [asdict(entry) for entry in self.entries],
        }


def load_landmark_mapping(path: str | Path) -> LandmarkMapping:
    """Load integer mappings while preserving NONE, blank, and question marks."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Landmark mapping file not found: {source_path}")

    entries: list[LandmarkMappingEntry] = []
    used_indices: dict[int, str] = {}
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            fields = [field.strip() for field in raw_line.rstrip("\r\n").split("|")]
            if len(fields) != 4:
                raise ValueError(
                    f"{source_path}:{line_number}: expected four pipe-separated "
                    f"fields, got {len(fields)}"
                )

            name, legacy_source, legacy_reference, user_value = fields
            if not name:
                raise ValueError(f"{source_path}:{line_number}: landmark name is empty")

            normalized_value = user_value.upper()
            dataset_index: int | None
            if normalized_value in INACTIVE_VALUES or normalized_value.endswith("?"):
                dataset_index = None
            else:
                try:
                    dataset_index = int(user_value)
                except ValueError as exc:
                    raise ValueError(
                        f"{source_path}:{line_number}: dataset index must be "
                        f"0-{DATASET_LANDMARK_COUNT - 1}, NONE, ?, an uncertain "
                        f"value "
                        f"ending in ?, or blank"
                    ) from exc
                if not 0 <= dataset_index < DATASET_LANDMARK_COUNT:
                    raise ValueError(
                        f"{source_path}:{line_number}: dataset index "
                        f"{dataset_index} is outside "
                        f"0-{DATASET_LANDMARK_COUNT - 1}"
                    )
                duplicate_name = used_indices.get(dataset_index)
                if duplicate_name is not None:
                    raise ValueError(
                        f"{source_path}:{line_number}: dataset index {dataset_index} "
                        f"is already assigned to {duplicate_name}"
                    )
                used_indices[dataset_index] = name

            entries.append(
                LandmarkMappingEntry(
                    model_index=len(entries),
                    name=name,
                    legacy_source=legacy_source,
                    legacy_reference=legacy_reference,
                    dataset_index=dataset_index,
                    user_value=user_value,
                )
            )

    if not entries:
        raise ValueError(f"No landmark mapping rows found in: {source_path}")
    if not used_indices:
        raise ValueError(f"No confirmed integer mappings found in: {source_path}")
    return LandmarkMapping(source_path=source_path, entries=tuple(entries))
