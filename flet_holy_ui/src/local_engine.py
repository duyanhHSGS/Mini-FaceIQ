"""Local, in-process Mini-FaceIQ inference: no Flask, HTTP, or API server.

Reuse the repository's detectors/calculators, not its Flask route handlers:
those handlers depend on request.files and Flask request context.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

MAX_IMAGE_BYTES = 20 * 1024 * 1024
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Desktop layout:
#   Mini-FaceIQ/
#     front_calculator.py, side_calculator.py, packages/, ...
#     flet_holy_ui/src/{main.py,local_engine.py}
#
# This does not start Flask or contact localhost. It does use the model files
# installed in the containing Mini-FaceIQ checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class AnalysisError(Exception):
    """An expected local analysis or photo-validation error."""


class _MemoryUpload:
    """Minimum in-memory file API expected by the existing model adapters."""

    def __init__(self, data: bytes, filename: str) -> None:
        self.filename = filename
        self._stream = io.BytesIO(data)

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def save(self, destination) -> None:
        # adapter_features uses file_storage.save(open_temp_file)
        if hasattr(destination, "write"):
            destination.write(self._stream.getvalue())
        else:
            with open(destination, "wb") as output:
                output.write(self._stream.getvalue())


def image_aspect_ratio(contents: bytes) -> float:
    """Read PNG/JPEG/WebP pixel aspect without adding a Pillow dependency."""
    width = height = 0

    if contents.startswith(bytes.fromhex("89504e470d0a1a0a")) and len(contents) >= 24:
        width = int.from_bytes(contents[16:20], "big")
        height = int.from_bytes(contents[20:24], "big")

    elif contents.startswith(bytes.fromhex("ffd8")):
        cursor = 2
        sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
        while cursor + 8 <= len(contents):
            if contents[cursor] != 0xFF:
                cursor += 1
                continue
            while cursor < len(contents) and contents[cursor] == 0xFF:
                cursor += 1
            if cursor >= len(contents):
                break
            marker = contents[cursor]
            cursor += 1
            if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
                continue
            if marker == 0xDA or cursor + 2 > len(contents):
                break
            seg_length = int.from_bytes(contents[cursor : cursor + 2], "big")
            if seg_length < 2 or cursor + seg_length > len(contents):
                break
            if marker in sof and seg_length >= 7:
                height = int.from_bytes(contents[cursor + 3 : cursor + 5], "big")
                width = int.from_bytes(contents[cursor + 5 : cursor + 7], "big")
                break
            cursor += seg_length

    elif contents.startswith(b"RIFF") and contents[8:12] == b"WEBP":
        chunk = contents[12:16]
        if chunk == b"VP8X" and len(contents) >= 30:
            width = 1 + int.from_bytes(contents[24:27], "little")
            height = 1 + int.from_bytes(contents[27:30], "little")
        elif chunk == b"VP8L" and len(contents) >= 25 and contents[20] == 0x2F:
            bits = int.from_bytes(contents[21:25], "little")
            width = (bits & 0x3FFF) + 1
            height = ((bits >> 14) & 0x3FFF) + 1
        elif chunk == b"VP8 " and len(contents) >= 30 and contents[23:26] == bytes.fromhex("9d012a"):
            width = int.from_bytes(contents[26:28], "little") & 0x3FFF
            height = int.from_bytes(contents[28:30], "little") & 0x3FFF

    return width / height if width > 0 and height > 0 else 1.0


def front_autolandmarks(data: bytes, filename: str = "photo.jpg") -> list[dict]:
    """Direct equivalent of the Flask front-autolandmarks endpoint."""
    from packages import detect_front_landmarks_from_upload

    return detect_front_landmarks_from_upload(_MemoryUpload(data, filename))


def side_autolandmarks(data: bytes, filename: str = "photo.jpg") -> list[dict]:
    """Direct equivalent of the Flask side-autolandmarks endpoint."""
    from packages import detect_side_landmarks_from_upload

    return detect_side_landmarks_from_upload(_MemoryUpload(data, filename))


def features_rating(data: bytes, filename: str) -> dict:
    """Direct feature-rating inference, without HTTP or Flask."""
    from packages import analyze_features_from_upload

    suffix = Path(filename).suffix.lower()
    return analyze_features_from_upload(_MemoryUpload(data, filename), suffix)


def front_metrics(landmarks: list[dict], gender: str, ethnicity: str, aspect: float) -> dict:
    from front_calculator import calculate_front_analysis

    return calculate_front_analysis(landmarks, gender=gender, ethnicity=ethnicity, front_aspect=aspect)


def side_metrics(landmarks: list[dict], gender: str, ethnicity: str, aspect: float) -> dict:
    from side_calculator import calculate_side_analysis

    return calculate_side_analysis(landmarks, gender=gender, ethnicity=ethnicity, side_aspect=aspect)


def _missing_landmarks(landmarks: list[dict], mode: str) -> list[str]:
    # Mirrors the validation currently used in the Flask API.
    if mode == "front":
        from front_landmarks import FRONT_LANDMARK_DEFS as definitions
    else:
        from side_landmarks import SIDE_LANDMARK_DEFS as definitions

    required = {item["id"] for item in definitions}
    placed: set[str] = set()
    for landmark in landmarks:
        if not isinstance(landmark, dict) or landmark.get("id") not in required:
            continue
        try:
            float(landmark["x"])
            float(landmark["y"])
        except (KeyError, ValueError, TypeError):
            continue
        placed.add(landmark["id"])
    return sorted(required - placed)


def analyze_photo(
    mode: str,
    filename: str,
    image_data: bytes,
    gender: str = "male",
    ethnicity: str = "asian",
) -> dict:
    """Compute an analysis inside this Python process, without networking."""
    if mode not in {"front", "side", "features"}:
        raise AnalysisError("Choose Front, Side, or Feature rating.")
    if Path(filename).suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise AnalysisError("Select a JPG, JPEG, PNG, or WEBP photo.")
    if not image_data or len(image_data) > MAX_IMAGE_BYTES:
        raise AnalysisError("Photo must be non-empty and no larger than 20 MB.")

    if mode == "features":
        return {"mode": mode, "data": features_rating(image_data, filename), "landmark_count": None}

    detector = front_autolandmarks if mode == "front" else side_autolandmarks
    landmarks = detector(image_data, filename)
    if not isinstance(landmarks, list) or not landmarks:
        raise AnalysisError("No usable landmarks detected. Try a clearer photo.")

    missing = _missing_landmarks(landmarks, mode)
    if missing:
        return {
            "mode": mode,
            "data": None,
            "landmark_count": len(landmarks),
            "missing_ids": missing,
            "warning": f"Place all {mode} landmarks first. Missing {len(missing)}.",
        }

    analyzer = front_metrics if mode == "front" else side_metrics
    result = analyzer(landmarks, gender, ethnicity, image_aspect_ratio(image_data))
    return {"mode": mode, "data": result, "landmark_count": len(landmarks)}
