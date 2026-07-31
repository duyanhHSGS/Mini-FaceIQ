"""Frozen 31-point side-profile annotation contract."""

from __future__ import annotations


SCHEMA_ID = "sir-faceiq-human-data-v1"

LANDMARKS = (
    {"index": 0, "id": "top_of_head", "label": "Top of Head", "group": "head", "color": "#3b82f6"},
    {"index": 1, "id": "occiput", "label": "Occiput", "group": "head", "color": "#3b82f6"},
    {"index": 2, "id": "nose_tip", "label": "Nose Tip", "group": "nose", "color": "#10b981"},
    {"index": 3, "id": "neck_point", "label": "Neck Point", "group": "neck", "color": "#6b7280"},
    {"index": 4, "id": "porion", "label": "Porion", "group": "ears", "color": "#ec4899"},
    {"index": 5, "id": "orbitale", "label": "Orbitale", "group": "eyes", "color": "#ef4444"},
    {"index": 6, "id": "tragus", "label": "Tragus", "group": "ears", "color": "#ec4899"},
    {"index": 7, "id": "intertragic_notch", "label": "Intertragic Notch", "group": "ears", "color": "#ec4899"},
    {"index": 8, "id": "corneal_apex", "label": "Corneal Apex", "group": "eyes", "color": "#ef4444"},
    {"index": 9, "id": "cheekbone", "label": "Cheekbone", "group": "cheeks", "color": "#ec4899"},
    {"index": 10, "id": "eyelid_end", "label": "Eyelid End", "group": "eyes", "color": "#ef4444"},
    {"index": 11, "id": "lower_eyelid", "label": "Lower Eyelid", "group": "eyes", "color": "#ef4444"},
    {"index": 12, "id": "hairline_profile", "label": "Hairline (Profile)", "group": "head", "color": "#3b82f6"},
    {"index": 13, "id": "glabella", "label": "Glabella", "group": "head", "color": "#3b82f6"},
    {"index": 14, "id": "forehead", "label": "Forehead", "group": "head", "color": "#3b82f6"},
    {"index": 15, "id": "nasal_bridge_root", "label": "Nasal Bridge Root", "group": "nose", "color": "#10b981"},
    {"index": 16, "id": "rhinion", "label": "Rhinion", "group": "nose", "color": "#10b981"},
    {"index": 17, "id": "supratip", "label": "Supratip", "group": "nose", "color": "#10b981"},
    {"index": 18, "id": "infratip", "label": "Infratip", "group": "nose", "color": "#10b981"},
    {"index": 19, "id": "columella", "label": "Columella", "group": "nose", "color": "#10b981"},
    {"index": 20, "id": "subnasale", "label": "Subnasale", "group": "nose", "color": "#10b981"},
    {"index": 21, "id": "subalare", "label": "Subalare", "group": "nose", "color": "#10b981"},
    {"index": 22, "id": "upper_lip", "label": "Upper Lip", "group": "mouth", "color": "#8b5cf6"},
    {"index": 23, "id": "mouth_corner", "label": "Mouth Corner", "group": "mouth", "color": "#8b5cf6"},
    {"index": 24, "id": "lower_lip", "label": "Lower Lip", "group": "mouth", "color": "#8b5cf6"},
    {"index": 25, "id": "labiomental_fold", "label": "Labiomental Fold", "group": "chin", "color": "#f59e0b"},
    {"index": 26, "id": "chin_point", "label": "Chin Point", "group": "chin", "color": "#f59e0b"},
    {"index": 27, "id": "chin_bottom", "label": "Chin Bottom", "group": "chin", "color": "#f59e0b"},
    {"index": 28, "id": "cervical_point", "label": "Cervical Point", "group": "neck", "color": "#6b7280"},
    {"index": 29, "id": "upper_jaw_angle", "label": "Upper Jaw Angle", "group": "jaw", "color": "#f59e0b"},
    {"index": 30, "id": "lower_jaw_angle", "label": "Lower Jaw Angle", "group": "jaw", "color": "#f59e0b"},
)

LANDMARK_COUNT = len(LANDMARKS)
LANDMARK_BY_ID = {item["id"]: item for item in LANDMARKS}
