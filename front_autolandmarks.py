from front_landmarks import FRONT_LANDMARK_DEFS


DIRECT_MP_MAP = {
    "hairline": 10,
    "left_nose_side": 98,
    "right_nose_side": 327,
    "lower_lip_center": 17,
    "chin_bottom": 152,
    "left_temple": 127,
    "right_temple": 356,
    "left_medial_canthus": 133,
    "left_lateral_canthus": 33,
    "left_upper_eyelid": 159,
    "left_lower_eyelid": 145,
    "left_eyelid_hood_end": 173,
    "left_upper_eyelid_crease": 159,
    "right_medial_canthus": 362,
    "right_lateral_canthus": 263,
    "right_upper_eyelid": 386,
    "right_lower_eyelid": 374,
    "right_eyelid_hood_end": 398,
    "right_upper_eyelid_crease": 386,
    "nasal_base": 94,
    "nose_bottom": 2,
    "left_nose_bridge": 193,
    "right_nose_bridge": 417,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
    "cupids_bow": 0,
    "inner_cupids_bow": 13,
    "mouth_middle": 14,
    "left_upper_jaw_angle": 172,
    "right_upper_jaw_angle": 397,
    "left_lower_jaw_angle": 136,
    "right_lower_jaw_angle": 365,
    "left_chin": 150,
    "right_chin": 379,
    "left_cheekbone": 234,
    "right_cheekbone": 454,
}

AVERAGE_MP_MAP = {
    "left_pupil": [33, 133, 159, 145],
    "right_pupil": [362, 263, 386, 374],
    "left_brow_head": [105, 107],
    "left_brow_inner_corner": [66, 107],
    "left_brow_arch": [52, 65],
    "left_brow_peak": [63, 105],
    "left_brow_tail": [46, 53],
    "right_brow_head": [334, 336],
    "right_brow_inner_corner": [296, 336],
    "right_brow_arch": [282, 295],
    "right_brow_peak": [293, 334],
    "right_brow_tail": [276, 283],
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
