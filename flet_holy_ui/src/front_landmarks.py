FRONT_LANDMARK_DEFS = [
    {"id": "hairline", "label": "Hairline", "group": "head", "color": "#3b82f6"},
    {"id": "left_pupil", "label": "Left Pupil", "group": "eyes", "color": "#ef4444"},
    {"id": "right_pupil", "label": "Right Pupil", "group": "eyes", "color": "#ef4444"},
    {"id": "left_nose_side", "label": "Left Nose Side", "group": "nose", "color": "#10b981"},
    {"id": "right_nose_side", "label": "Right Nose Side", "group": "nose", "color": "#10b981"},
    {"id": "lower_lip_center", "label": "Lower Lip Center", "group": "mouth", "color": "#8b5cf6"},
    {"id": "chin_bottom", "label": "Chin Bottom", "group": "chin", "color": "#f59e0b"},
    {"id": "left_temple", "label": "Left Temple", "group": "head", "color": "#3b82f6"},
    {"id": "right_temple", "label": "Right Temple", "group": "head", "color": "#3b82f6"},
    {"id": "left_medial_canthus", "label": "Left Medial Canthus", "group": "eyes", "color": "#ef4444"},
    {"id": "left_lateral_canthus", "label": "Left Lateral Canthus", "group": "eyes", "color": "#ef4444"},
    {"id": "left_upper_eyelid", "label": "Left Upper Eyelid", "group": "eyes", "color": "#ef4444"},
    {"id": "left_lower_eyelid", "label": "Left Lower Eyelid", "group": "eyes", "color": "#ef4444"},
    {"id": "left_eyelid_hood_end", "label": "Left Eyelid Hood End", "group": "eyes", "color": "#ef4444"},
    {"id": "left_brow_head", "label": "Left Brow Head", "group": "brows", "color": "#f97316"},
    {"id": "left_brow_inner_corner", "label": "Left Brow Inner Corner", "group": "brows", "color": "#f97316"},
    {"id": "left_brow_arch", "label": "Left Brow Arch", "group": "brows", "color": "#f97316"},
    {"id": "left_brow_peak", "label": "Left Brow Peak", "group": "brows", "color": "#f97316"},
    {"id": "left_brow_tail", "label": "Left Brow Tail", "group": "brows", "color": "#f97316"},
    {"id": "left_upper_eyelid_crease", "label": "Left Upper Eyelid Crease", "group": "eyes", "color": "#ef4444"},
    {"id": "right_medial_canthus", "label": "Right Medial Canthus", "group": "eyes", "color": "#ef4444"},
    {"id": "right_lateral_canthus", "label": "Right Lateral Canthus", "group": "eyes", "color": "#ef4444"},
    {"id": "right_upper_eyelid", "label": "Right Upper Eyelid", "group": "eyes", "color": "#ef4444"},
    {"id": "right_lower_eyelid", "label": "Right Lower Eyelid", "group": "eyes", "color": "#ef4444"},
    {"id": "right_eyelid_hood_end", "label": "Right Eyelid Hood End", "group": "eyes", "color": "#ef4444"},
    {"id": "right_brow_head", "label": "Right Brow Head", "group": "brows", "color": "#f97316"},
    {"id": "right_brow_inner_corner", "label": "Right Brow Inner Corner", "group": "brows", "color": "#f97316"},
    {"id": "right_brow_arch", "label": "Right Brow Arch", "group": "brows", "color": "#f97316"},
    {"id": "right_brow_peak", "label": "Right Brow Peak", "group": "brows", "color": "#f97316"},
    {"id": "right_brow_tail", "label": "Right Brow Tail", "group": "brows", "color": "#f97316"},
    {"id": "right_upper_eyelid_crease", "label": "Right Upper Eyelid Crease", "group": "eyes", "color": "#ef4444"},
    {"id": "nasal_base", "label": "Nasal Base", "group": "nose", "color": "#10b981"},
    {"id": "nose_bottom", "label": "Nose Bottom", "group": "nose", "color": "#10b981"},
    {"id": "left_nose_bridge", "label": "Left Nose Bridge", "group": "nose", "color": "#10b981"},
    {"id": "right_nose_bridge", "label": "Right Nose Bridge", "group": "nose", "color": "#10b981"},
    {"id": "left_mouth_corner", "label": "Left Mouth Corner", "group": "mouth", "color": "#8b5cf6"},
    {"id": "right_mouth_corner", "label": "Right Mouth Corner", "group": "mouth", "color": "#8b5cf6"},
    {"id": "cupids_bow", "label": "Cupid's Bow", "group": "mouth", "color": "#8b5cf6"},
    {"id": "inner_cupids_bow", "label": "Inner Cupid's Bow", "group": "mouth", "color": "#8b5cf6"},
    {"id": "mouth_middle", "label": "Mouth Middle", "group": "mouth", "color": "#8b5cf6"},
    {"id": "left_upper_jaw_angle", "label": "Left Upper Jaw Angle", "group": "jaw", "color": "#f59e0b"},
    {"id": "right_upper_jaw_angle", "label": "Right Upper Jaw Angle", "group": "jaw", "color": "#f59e0b"},
    {"id": "left_lower_jaw_angle", "label": "Left Lower Jaw Angle", "group": "jaw", "color": "#f59e0b"},
    {"id": "right_lower_jaw_angle", "label": "Right Lower Jaw Angle", "group": "jaw", "color": "#f59e0b"},
    {"id": "left_chin", "label": "Left Chin", "group": "chin", "color": "#f59e0b"},
    {"id": "right_chin", "label": "Right Chin", "group": "chin", "color": "#f59e0b"},
    {"id": "left_cheekbone", "label": "Left Cheekbone", "group": "cheeks", "color": "#ec4899"},
    {"id": "right_cheekbone", "label": "Right Cheekbone", "group": "cheeks", "color": "#ec4899"},
]


def blank_front_landmarks():
    return [{"id": item["id"], "x": None, "y": None, "label": item["label"]} for item in FRONT_LANDMARK_DEFS]


def normalize_front_landmarks(items):
    landmarks = {}
    for item in items or []:
        lid = item.get("id")
        if not lid:
            continue
        try:
            x = float(item.get("x"))
            y = float(item.get("y"))
        except (TypeError, ValueError):
            continue
        landmarks[lid] = {
            "id": lid,
            "x": x,
            "y": y,
            "label": item.get("label", lid),
        }
    return landmarks
