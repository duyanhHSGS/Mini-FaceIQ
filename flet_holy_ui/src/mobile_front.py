"""Android/iOS front-face analysis using the ORIGINAL MediaPipe Task + mapper."""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path


def _front_asset(name: str) -> bytes:
    if Path(name).name != name:
        raise ValueError("Invalid front model asset")
    relative = "assets/models/front/" + name
    disk = Path(__file__).resolve().parent / relative
    if disk.is_file():
        return disk.read_bytes()
    path = str(__file__)
    if ".zip/" in path:
        with zipfile.ZipFile(path.split(".zip/", 1)[0] + ".zip") as data:
            return data.read(relative)
    raise FileNotFoundError(relative)


def get_front_task_model() -> bytes:
    return _front_asset("face_landmarker.task")


def map_mediapipe_front(
    normalized: list[list[float]], image_data: bytes
) -> list[dict]:
    """Map 478 MediaPipe landmarks to the exact original project point IDs.

    Desktop MediaPipe rounds down to integer pixels BEFORE applying the map,
    so we reproduce that truncation for coordinate parity.
    """
    from local_engine import image_dimensions

    from front_landmarks import FRONT_LANDMARK_DEFS

    mapping = json.loads(_front_asset("mapper.json"))
    width, height = image_dimensions(image_data)
    if width <= 0 or height <= 0:
        raise ValueError("Unable to read image dimensions for normalized coordinates")
    if not normalized:
        return []
    points = []
    for coords in normalized:
        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            raise ValueError("MediaPipe returned malformed coordinates")
        x, y = float(coords[0]), float(coords[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("MediaPipe returned non-finite coordinates")
        points.append((int(x * width), int(y * height)))
    values = []
    direct = mapping["DIRECT_MP_MAP"]
    averages = mapping["AVERAGE_MP_MAP"]
    for definition in FRONT_LANDMARK_DEFS:
        point_id = definition["id"]
        indexes = averages.get(point_id)
        if indexes is not None:
            available = [points[i] for i in indexes if i < len(points)]
            if not available:
                continue
            x = sum(item[0] for item in available) / len(available)
            y = sum(item[1] for item in available) / len(available)
        else:
            index = direct.get(point_id)
            if index is None or index >= len(points):
                continue
            x, y = points[index]
        values.append({
            "id": point_id,
            "label": definition["label"],
            "x": round(max(0, min(1, x / width)), 6),
            "y": round(max(0, min(1, y / height)), 6),
        })
    return values
