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
