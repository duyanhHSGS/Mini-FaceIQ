"""PyTorch dataset for the Multi-PIE 39-point profile annotations.

Each annotation row has this exact layout:

    image_path
    bbox_x1 bbox_y1 bbox_x2 bbox_y2
    five auxiliary (x, y) point pairs
    thirty-nine profile-landmark (x, y) pairs

The auxiliary points and the 39 training targets are deliberately kept
separate.  Images are cropped around an expanded square version of the supplied
face bounding box, resized, and returned as float tensors in the [0, 1] range.
Landmark coordinates are normalized relative to that crop; they are not
clamped, so broken annotations remain visible instead of being silently hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


PROFILE_LANDMARK_COUNT = 39
AUXILIARY_POINT_COUNT = 5
BOUNDING_BOX_VALUE_COUNT = 4
EXPECTED_FIELD_COUNT = (
    1
    + BOUNDING_BOX_VALUE_COUNT
    + AUXILIARY_POINT_COUNT * 2
    + PROFILE_LANDMARK_COUNT * 2
)


@dataclass(frozen=True)
class ProfileAnnotation:
    """One parsed row from ``MultiPIE_profile_train.txt``."""

    image_path: Path
    relative_image_path: str
    bbox_xyxy: np.ndarray
    auxiliary_points_xy: np.ndarray
    landmarks_xy: np.ndarray
    subject_id: str
    camera_code: str


def parse_annotation_line(
    line: str,
    *,
    annotation_root: Path,
    line_number: int,
) -> ProfileAnnotation:
    """Parse and validate one profile-annotation row."""

    fields = line.split()
    if len(fields) != EXPECTED_FIELD_COUNT:
        raise ValueError(
            f"Line {line_number}: expected {EXPECTED_FIELD_COUNT} fields "
            f"(path + bbox + {AUXILIARY_POINT_COUNT} auxiliary points + "
            f"{PROFILE_LANDMARK_COUNT} landmarks), got {len(fields)}"
        )

    relative_image_path = fields[0].replace("\\", "/")
    image_path = Path(relative_image_path)
    if not image_path.is_absolute():
        image_path = annotation_root / image_path

    try:
        values = np.asarray(fields[1:], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(
            f"Line {line_number}: annotation coordinates must be numbers"
        ) from exc

    bbox_end = BOUNDING_BOX_VALUE_COUNT
    auxiliary_end = bbox_end + AUXILIARY_POINT_COUNT * 2

    bbox_xyxy = values[:bbox_end].copy()
    auxiliary_points_xy = values[bbox_end:auxiliary_end].reshape(
        AUXILIARY_POINT_COUNT, 2
    )
    landmarks_xy = values[auxiliary_end:].reshape(PROFILE_LANDMARK_COUNT, 2)

    x1, y1, x2, y2 = bbox_xyxy
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Line {line_number}: invalid bbox {bbox_xyxy.tolist()}"
        )

    filename_parts = Path(relative_image_path).stem.split("_")
    subject_id = filename_parts[0] if filename_parts else ""
    camera_code = filename_parts[3] if len(filename_parts) > 3 else ""

    return ProfileAnnotation(
        image_path=image_path.resolve(),
        relative_image_path=relative_image_path,
        bbox_xyxy=bbox_xyxy,
        auxiliary_points_xy=auxiliary_points_xy.copy(),
        landmarks_xy=landmarks_xy.copy(),
        subject_id=subject_id,
        camera_code=camera_code,
    )


def load_profile_annotations(annotation_path: str | Path) -> list[ProfileAnnotation]:
    """Load all non-empty rows and fail loudly on malformed data."""

    annotation_path = Path(annotation_path).expanduser().resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    records: list[ProfileAnnotation] = []
    with annotation_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            records.append(
                parse_annotation_line(
                    line,
                    annotation_root=annotation_path.parent,
                    line_number=line_number,
                )
            )

    if not records:
        raise ValueError(f"No annotation rows found in: {annotation_path}")
    return records


def _square_crop_box(
    bbox_xyxy: np.ndarray,
    *,
    scale: float,
) -> tuple[int, int, int, int]:
    if scale <= 0:
        raise ValueError(f"bbox_scale must be positive, got {scale}")

    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * scale
    half_side = side / 2.0

    crop_x1 = int(np.floor(center_x - half_side))
    crop_y1 = int(np.floor(center_y - half_side))
    crop_x2 = int(np.ceil(center_x + half_side))
    crop_y2 = int(np.ceil(center_y + half_side))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise ValueError(f"Calculated an empty crop from bbox {bbox_xyxy.tolist()}")
    return crop_x1, crop_y1, crop_x2, crop_y2


def _normalize_points(
    points_xy: np.ndarray,
    crop_box: tuple[int, int, int, int],
) -> torch.Tensor:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    crop_width = float(crop_x2 - crop_x1)
    crop_height = float(crop_y2 - crop_y1)

    normalized = points_xy.astype(np.float32, copy=True)
    normalized[:, 0] = (normalized[:, 0] - crop_x1) / crop_width
    normalized[:, 1] = (normalized[:, 1] - crop_y1) / crop_height
    return torch.from_numpy(normalized)


def _pil_to_float_tensor(image: Image.Image) -> torch.Tensor:
    pixels = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous() / 255.0


class ProfileLandmarkDataset(Dataset):
    """Return model-ready crops and normalized 39-point landmark targets."""

    def __init__(
        self,
        annotation_path: str | Path,
        *,
        image_size: int = 256,
        bbox_scale: float = 1.25,
        verify_images: bool = False,
    ) -> None:
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")

        self.annotation_path = Path(annotation_path).expanduser().resolve()
        self.image_size = int(image_size)
        self.bbox_scale = float(bbox_scale)
        self.records = load_profile_annotations(self.annotation_path)

        if verify_images:
            missing = [
                str(record.image_path)
                for record in self.records
                if not record.image_path.is_file()
            ]
            if missing:
                preview = "\n".join(missing[:10])
                suffix = "" if len(missing) <= 10 else f"\n...and {len(missing) - 10} more"
                raise FileNotFoundError(
                    f"{len(missing)} annotated images are missing:\n{preview}{suffix}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        if not record.image_path.is_file():
            raise FileNotFoundError(f"Annotated image not found: {record.image_path}")

        crop_box = _square_crop_box(
            record.bbox_xyxy,
            scale=self.bbox_scale,
        )

        with Image.open(record.image_path) as source:
            rgb_image = source.convert("RGB")
            crop = rgb_image.crop(crop_box)
            crop = crop.resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BILINEAR,
            )
            image_tensor = _pil_to_float_tensor(crop)

        landmarks = _normalize_points(record.landmarks_xy, crop_box)
        auxiliary_points = _normalize_points(
            record.auxiliary_points_xy,
            crop_box,
        )

        return {
            "image": image_tensor,
            "landmarks": landmarks,
            "landmarks_flat": landmarks.reshape(-1),
            "auxiliary_points": auxiliary_points,
            "bbox_xyxy": torch.from_numpy(record.bbox_xyxy.copy()),
            "crop_xyxy": torch.tensor(crop_box, dtype=torch.float32),
            "image_path": str(record.image_path),
            "relative_image_path": record.relative_image_path,
            "subject_id": record.subject_id,
            "camera_code": record.camera_code,
        }


def subject_disjoint_indices(
    records: Iterable[ProfileAnnotation],
    subject_ids: set[str],
) -> list[int]:
    """Return indices for a subject-level split without image leakage."""

    return [
        index
        for index, record in enumerate(records)
        if record.subject_id in subject_ids
    ]
