"""
Side Face Landmark Detector API.

Public function:
    detect_side_landmarks_from_upload(file_storage) -> list[dict]
        Reads a side‑view photo from a Flask FileStorage,
        runs 3DDFA_V2 face alignment, maps the result to the app's side landmark
        definitions, and returns a list of landmarks: [{id, x, y, label}, ...].

Also exposed (for direct use):
    detect_side(image_path: str) -> dict
        Direct file‑path interface to 3DDFA_V2 detection (returns raw result).
"""
from side_landmarks import SIDE_LANDMARK_DEFS

from .providers import side_3ddfa


def detect_side_landmarks_from_upload(file_storage):
    result = side_3ddfa.detect_side_from_upload(file_storage)
    if result.get("error"):
        raise ValueError(result["error"])
    return _side_landmark_list(result)


def detect_side(image_path):
    return side_3ddfa.detect_side(image_path)


def _side_landmark_list(result):
    detected = result.get("landmarks", {})
    items = []
    for item in SIDE_LANDMARK_DEFS:
        landmark = detected.get(item["id"])
        if not landmark:
            continue
        items.append(
            {
                "id": item["id"],
                "label": landmark.get("label", item["label"]),
                "x": landmark["x"],
                "y": landmark["y"],
            }
        )
    return items