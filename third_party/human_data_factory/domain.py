"""Pure validation and traversal rules for human annotation."""

from __future__ import annotations

import math
import re
from typing import Any

from .schema import LANDMARK_COUNT


LABEL_STATES = {"unreviewed", "placed", "unavailable"}
PROJECT_ID_PATTERN = re.compile(
    r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$"
)


def validate_project_id(value: str) -> str:
    normalized = value.strip().lower()
    if not PROJECT_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "project_id must use 1-64 lowercase letters, numbers, "
            "hyphens, or underscores"
        )
    return normalized


def validate_annotator(value: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 80:
        raise ValueError("annotator must contain 1-80 characters")
    return normalized


def validate_landmark_index(value: int) -> int:
    index = int(value)
    if not 0 <= index < LANDMARK_COUNT:
        raise ValueError(
            f"landmark_index must be between 0 and {LANDMARK_COUNT - 1}"
        )
    return index


def validate_label(
    value: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    state = str(value.get("state", ""))
    if state not in LABEL_STATES:
        raise ValueError(f"state must be one of: {sorted(LABEL_STATES)}")
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
        if not (0 <= x <= image_width - 1 and 0 <= y <= image_height - 1):
            raise ValueError("placed coordinates must stay inside the image")
    elif x is not None or y is not None:
        raise ValueError(f"{state} labels cannot contain coordinates")
    return {
        "state": state,
        "x": x if state == "placed" else None,
        "y": y if state == "placed" else None,
    }


def advance_position(
    *,
    mode: str,
    image_index: int,
    landmark_index: int,
    image_count: int,
) -> tuple[int, int, bool]:
    """Return the next image/landmark and whether the queue is finished."""

    if mode not in {"all", "focused"}:
        raise ValueError("mode must be all or focused")
    if image_count < 1:
        return 0, landmark_index, True
    if mode == "focused":
        if image_index + 1 >= image_count:
            return image_index, landmark_index, True
        return image_index + 1, landmark_index, False
    if landmark_index + 1 < LANDMARK_COUNT:
        return image_index, landmark_index + 1, False
    if image_index + 1 >= image_count:
        return image_index, landmark_index, True
    return image_index + 1, 0, False


def retreat_position(
    *,
    mode: str,
    image_index: int,
    landmark_index: int,
) -> tuple[int, int]:
    if mode == "focused":
        return max(0, image_index - 1), landmark_index
    if landmark_index > 0:
        return image_index, landmark_index - 1
    if image_index > 0:
        return image_index - 1, LANDMARK_COUNT - 1
    return 0, 0


def source_to_view(
    x: float,
    y: float,
    *,
    image_width: float,
    mirrored: bool,
) -> tuple[float, float]:
    return ((image_width - 1 - x) if mirrored else x, y)


def view_to_source(
    x: float,
    y: float,
    *,
    image_width: float,
    mirrored: bool,
) -> tuple[float, float]:
    return source_to_view(
        x,
        y,
        image_width=image_width,
        mirrored=mirrored,
    )
