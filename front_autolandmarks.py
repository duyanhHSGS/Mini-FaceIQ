from front_landmarks import FRONT_LANDMARK_DEFS


DIRECT_MP_MAP = {
    "hairline": 10,
    "left_pupil": 468,
    "right_pupil": 473,
    "left_nose_side": 48,
    "right_nose_side": 278,
    "lower_lip_center": 17,
    "chin_bottom": 152,
    "left_temple": 54,
    "right_temple": 284,
    "left_medial_canthus": 243,
    "left_lateral_canthus": 130,
    "left_upper_eyelid": 159,
    "left_lower_eyelid": 145,
    "left_eyelid_hood_end": 247,
    "left_brow_head": 107,
    "left_brow_inner_corner": 55,
    "left_brow_arch": 53,
    "left_brow_tail": 70,
    "left_upper_eyelid_crease": 470,
    "right_medial_canthus": 463,
    "right_lateral_canthus": 359,
    "right_upper_eyelid": 386,
    "right_lower_eyelid": 374,
    "right_eyelid_hood_end": 467,
    "right_brow_head": 336,
    "right_brow_inner_corner": 285,
    "right_brow_arch": 283,
    "right_brow_tail": 300,
    "right_upper_eyelid_crease": 475,
    "nose_bottom": 2,
    "left_nose_bridge": 196,
    "right_nose_bridge": 419,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
    "inner_cupids_bow": 0,
    "left_upper_jaw_angle": 138,
    "right_upper_jaw_angle": 367,
    "left_lower_jaw_angle": 136,
    "right_lower_jaw_angle": 365,
    "left_chin": 149,
    "right_chin": 378,
}

AVERAGE_MP_MAP = {
    "left_brow_peak": [105, 63],
    "right_brow_peak": [334, 293],
    "nasal_base": [97, 326],
    "cupids_bow": [37, 267],
    "mouth_middle": [13, 14],
    "left_cheekbone": [34, 227],
    "right_cheekbone": [264, 447],
}


def detect_front_landmarks_from_upload(file_storage):
    import cv2
    import numpy as np

    from face_analyzer import get_landmarks_mp

    data = np.frombuffer(file_storage.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not read image file")

    detected = get_landmarks_mp(img_bgr)
    if detected is None:
        raise ValueError("No face detected")

    h, w = img_bgr.shape[:2]
    labels = {item["id"]: item["label"] for item in FRONT_LANDMARK_DEFS}
    points = []

    for item in FRONT_LANDMARK_DEFS:
        lid = item["id"]
        if lid in AVERAGE_MP_MAP:
            point = _average_point(detected, AVERAGE_MP_MAP[lid])
        else:
            point = _direct_point(detected, DIRECT_MP_MAP.get(lid))
        if point is None:
            continue

        x, y = point
        points.append(
            {
                "id": lid,
                "x": round(_clamp(x / w), 6),
                "y": round(_clamp(y / h), 6),
                "label": labels.get(lid, lid),
            }
        )

    return points


def _direct_point(points, index):
    if index is None or index >= len(points):
        return None
    return points[index]


def _average_point(points, indices):
    selected = [points[index] for index in indices if index < len(points)]
    if not selected:
        return None
    x = sum(point[0] for point in selected) / len(selected)
    y = sum(point[1] for point in selected) / len(selected)
    return x, y


def _clamp(value):
    return max(0.0, min(1.0, value))
