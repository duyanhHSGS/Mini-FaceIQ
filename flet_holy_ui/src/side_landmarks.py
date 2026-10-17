SIDE_LANDMARK_DEFS = [
    {"id": "top_of_head", "label": "Top of Head", "group": "head", "color": "#3b82f6"},
    {"id": "occiput", "label": "Occiput", "group": "head", "color": "#3b82f6"},
    {"id": "nose_tip", "label": "Nose Tip", "group": "nose", "color": "#10b981"},
    {"id": "neck_point", "label": "Neck Point", "group": "neck", "color": "#6b7280"},
    {"id": "porion", "label": "Porion", "group": "ears", "color": "#ec4899"},
    {"id": "orbitale", "label": "Orbitale", "group": "eyes", "color": "#ef4444"},
    {"id": "tragus", "label": "Tragus", "group": "ears", "color": "#ec4899"},
    {"id": "intertragic_notch", "label": "Intertragic Notch", "group": "ears", "color": "#ec4899"},
    {"id": "corneal_apex", "label": "Corneal Apex", "group": "eyes", "color": "#ef4444"},
    {"id": "cheekbone", "label": "Cheekbone", "group": "cheeks", "color": "#ec4899"},
    {"id": "eyelid_end", "label": "Eyelid End", "group": "eyes", "color": "#ef4444"},
    {"id": "lower_eyelid", "label": "Lower Eyelid", "group": "eyes", "color": "#ef4444"},
    {"id": "hairline_profile", "label": "Hairline (Profile)", "group": "head", "color": "#3b82f6"},
    {"id": "glabella", "label": "Glabella", "group": "head", "color": "#3b82f6"},
    {"id": "forehead", "label": "Forehead", "group": "head", "color": "#3b82f6"},
    {"id": "nasal_bridge_root", "label": "Nasal Bridge Root", "group": "nose", "color": "#10b981"},
    {"id": "rhinion", "label": "Rhinion", "group": "nose", "color": "#10b981"},
    {"id": "supratip", "label": "Supratip", "group": "nose", "color": "#10b981"},
    {"id": "infratip", "label": "Infratip", "group": "nose", "color": "#10b981"},
    {"id": "columella", "label": "Columella", "group": "nose", "color": "#10b981"},
    {"id": "subnasale", "label": "Subnasale", "group": "nose", "color": "#10b981"},
    {"id": "subalare", "label": "Subalare", "group": "nose", "color": "#10b981"},
    {"id": "upper_lip", "label": "Upper Lip", "group": "mouth", "color": "#8b5cf6"},
    {"id": "mouth_corner", "label": "Mouth Corner", "group": "mouth", "color": "#8b5cf6"},
    {"id": "lower_lip", "label": "Lower Lip", "group": "mouth", "color": "#8b5cf6"},
    {"id": "labiomental_fold", "label": "Labiomental Fold", "group": "chin", "color": "#f59e0b"},
    {"id": "chin_point", "label": "Chin Point", "group": "chin", "color": "#f59e0b"},
    {"id": "chin_bottom", "label": "Chin Bottom", "group": "chin", "color": "#f59e0b"},
    {"id": "cervical_point", "label": "Cervical Point", "group": "neck", "color": "#6b7280"},
    {"id": "upper_jaw_angle", "label": "Upper Jaw Angle", "group": "jaw", "color": "#f59e0b"},
    {"id": "lower_jaw_angle", "label": "Lower Jaw Angle", "group": "jaw", "color": "#f59e0b"},
]


def blank_side_landmarks():
    return [{"id": item["id"], "x": None, "y": None, "label": item["label"]} for item in SIDE_LANDMARK_DEFS]


def normalize_side_landmarks(items):
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
