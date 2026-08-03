from copy import deepcopy

from front_ideals import ETHNIC_FACTORS


MALE_ASIAN_SIDE = {
    "nasal_tip_angle": {"min": 106.16, "max": 166.84, "idealMin": 131.50, "idealMax": 141.50, "description": "Nasal Tip Angle (degrees)"},
    "nasal_width_to_height": {"min": -0.01, "max": 1.31, "idealMin": 0.57, "idealMax": 0.73, "description": "Nasal Width to Height Ratio"},
    "upper_lip_s_line": {"min": -8.19, "max": 7.19, "idealMin": -1.40, "idealMax": 0.40, "description": "Upper Lip S-Line Position (mm)"},
    "upper_lip_burstone": {"min": -7.70, "max": 3, "idealMin": -6.20, "idealMax": -2.80, "description": "Upper Lip Burstone Line (mm)"},
    "nasal_projection": {"min": 0.11, "max": 1.02, "idealMin": 0.53, "idealMax": 0.60, "description": "Nasal Projection ratio"},
    "nasofrontal_angle": {"min": 79.18, "max": 173.22, "idealMin": 120.20, "idealMax": 132.20, "description": "Nasofrontal Angle (degrees)"},
    "recession_frankfort": {"min": -24.08, "max": 39.05, "idealMin": 1.50, "idealMax": 15.00, "description": "Recession Frankfort Plane (mm)"},
    "holdaway_h_line": {"min": -9.79, "max": 8.79, "idealMin": -1.20, "idealMax": 0.20, "description": "Holdaway H Line (mm)"},
    "mentolabial_angle": {"min": 60.13, "max": 185.87, "idealMin": 115.00, "idealMax": 131.00, "description": "Mentolabial Angle (degrees)"},
    "upper_forehead_slope": {"min": -13.30, "max": 15.30, "idealMin": 0.00, "idealMax": 2.00, "description": "Upper Forehead Slope (degrees)"},
    "facial_convexity_nasion": {"min": 134.63, "max": 196.37, "idealMin": 164.00, "idealMax": 167.00, "description": "Facial Convexity at Nasion (degrees)"},
    "anterior_facial_depth": {"min": 36.12, "max": 103.88, "idealMin": 68.50, "idealMax": 71.50, "description": "Anterior Facial Depth (degrees)"},
    "upper_lip_e_line": {"min": -7.56, "max": 12.56, "idealMin": 0.20, "idealMax": 4.80, "description": "Upper Lip E-Line Position (mm)"},
    "submental_cervical_angle": {"min": 56.02, "max": 143.98, "idealMin": 94.00, "idealMax": 106.00, "description": "Submental Cervical Angle (degrees)"},
    "facial_depth_to_height": {"min": 0.94, "max": 1.76, "idealMin": 1.28, "idealMax": 1.42, "description": "Facial Depth to Height Ratio"},
    "browridge_inclination": {"min": -0.75, "max": 37.75, "idealMin": 15.00, "idealMax": 22.00, "description": "Browridge Inclination (degrees)"},
    "total_facial_convexity": {"min": 120.14, "max": 170.86, "idealMin": 142.00, "idealMax": 149.00, "description": "Total Facial Convexity (degrees)"},
    "facial_convexity_glabella": {"min": 153.31, "max": 193.69, "idealMin": 171.00, "idealMax": 176.00, "description": "Facial Convexity at Glabella (degrees)"},
    "orbital_vector": {"min": -10.33, "max": 19.25, "idealMin": 1.00, "idealMax": 10.00, "description": "Orbital Vector (mm)"},
    "interior_midface_projection": {"min": 35.75, "max": 85.25, "idealMin": 57.00, "idealMax": 64.00, "description": "Interior Midface Projection (degrees)"},
    "z_angle": {"min": 54.18, "max": 105.82, "idealMin": 78.00, "idealMax": 82.00, "description": "Z-Angle (degrees)"},
    "nose_tip_rotation": {"min": -15.08, "max": 48.08, "idealMin": 11.50, "idealMax": 21.50, "description": "Nose Tip Rotation (degrees)"},
    "nasolabial_angle": {"min": 55.02, "max": 145.98, "idealMin": 92.00, "idealMax": 109.00, "description": "Nasolabial Angle (degrees)"},
    "nasofacial_angle": {"min": 15.93, "max": 48.07, "idealMin": 30.00, "idealMax": 34.00, "description": "Nasofacial Angle (degrees)"},
    "nasomental_angle": {"min": 105.26, "max": 153.74, "idealMin": 127.00, "idealMax": 132.00, "description": "Nasomental Angle (degrees)"},
    "frankfort_tip_angle": {"min": 5.59, "max": 64.41, "idealMin": 32.00, "idealMax": 38.00, "description": "Frankfort-Tip Angle (degrees)"},
    "lower_lip_s_line": {"min": -8.19, "max": 7.19, "idealMin": -1.40, "idealMax": 0.40, "description": "Lower Lip S-Line (mm)"},
    "lower_lip_e_line": {"min": -7.74, "max": 11.24, "idealMin": 0.10, "idealMax": 3.40, "description": "Lower Lip E-Line (mm)"},
    "lower_lip_burstone": {"min": -9.47, "max": 3.48, "idealMin": -4.20, "idealMax": -1.80, "description": "Lower Lip Burstone Line (mm)"},
    "gonial_angle": {"min": 94.34, "max": 145.66, "idealMin": 117.00, "idealMax": 123.00, "description": "Gonial Angle (degrees)"},
    "mandibular_plane_angle": {"min": -4.68, "max": 43.68, "idealMin": 16.00, "idealMax": 23.00, "description": "Mandibular Plane Angle (degrees)"},
    "ramus_to_mandible": {"min": -0.20, "max": 1.57, "idealMin": 0.62, "idealMax": 0.75, "description": "Ramus to Mandible Ratio"},
    "gonion_to_mouth": {"min": -4.95, "max": 64.95, "idealMin": 15.00, "idealMax": 45.00, "description": "Gonion to Mouth Line (mm)"},
}


# Exact supported male plateaus captured from FaceIQ live on 2026-08-03, using
# the source-heading mapping documented beside MALE_LIVE_FRONT_PLATEAUS.
MALE_LIVE_SIDE_PLATEAUS = {
    "caucasian": {
        "nasal_tip_angle": (128.50, 138.50), "nasal_width_to_height": (0.67, 0.83),
        "upper_lip_s_line": (-0.40, 1.40), "upper_lip_burstone": (-5.20, -1.80),
        "nasal_projection": (0.58, 0.65), "nasofrontal_angle": (116.00, 128.00),
        "recession_frankfort": (1.50, 15.00), "holdaway_h_line": (-0.20, 1.20),
        "mentolabial_angle": (111.00, 127.00), "upper_forehead_slope": (0.00, 2.00),
        "facial_convexity_nasion": (163.00, 166.00), "anterior_facial_depth": (64.50, 67.50),
        "upper_lip_e_line": (1.20, 5.80), "submental_cervical_angle": (94.00, 106.00),
        "facial_depth_to_height": (1.30, 1.44), "browridge_inclination": (15.00, 22.00),
        "total_facial_convexity": (140.00, 147.00), "facial_convexity_glabella": (170.00, 175.00),
        "orbital_vector": (1.00, 10.00), "interior_midface_projection": (53.00, 60.00),
        "z_angle": (78.00, 82.00), "nose_tip_rotation": (11.50, 21.50),
        "nasolabial_angle": (97.00, 114.00), "nasofacial_angle": (31.00, 35.00),
        "nasomental_angle": (126.00, 131.00), "frankfort_tip_angle": (32.00, 38.00),
        "lower_lip_s_line": (-0.40, 1.40), "lower_lip_e_line": (1.10, 4.40),
        "lower_lip_burstone": (-3.20, -0.80), "gonial_angle": (115.00, 121.00),
        "mandibular_plane_angle": (15.00, 22.00), "ramus_to_mandible": (0.62, 0.75),
        "gonion_to_mouth": (15.00, 45.00),
    },
    "black": {
        "nasal_tip_angle": (128.50, 138.50), "nasal_width_to_height": (0.70, 0.86),
        "upper_lip_s_line": (-0.40, 1.40), "upper_lip_burstone": (-5.20, -1.80),
        "nasal_projection": (0.53, 0.60), "nasofrontal_angle": (116.00, 128.00),
        "recession_frankfort": (1.50, 15.00), "holdaway_h_line": (-0.20, 1.20),
        "mentolabial_angle": (111.00, 127.00), "upper_forehead_slope": (0.00, 2.00),
        "facial_convexity_nasion": (163.00, 166.00), "anterior_facial_depth": (64.50, 67.50),
        "upper_lip_e_line": (1.20, 5.80), "submental_cervical_angle": (94.00, 106.00),
        "facial_depth_to_height": (1.32, 1.46), "browridge_inclination": (15.00, 22.00),
        "total_facial_convexity": (140.00, 147.00), "facial_convexity_glabella": (170.00, 175.00),
        "orbital_vector": (1.00, 10.00), "interior_midface_projection": (53.00, 60.00),
        "z_angle": (78.00, 82.00), "nose_tip_rotation": (11.50, 21.50),
        "nasolabial_angle": (100.00, 117.00), "nasofacial_angle": (31.00, 35.00),
        "nasomental_angle": (126.00, 131.00), "frankfort_tip_angle": (32.00, 38.00),
        "lower_lip_s_line": (-0.40, 1.40), "lower_lip_e_line": (1.10, 4.40),
        "lower_lip_burstone": (-3.20, -0.80), "gonial_angle": (115.00, 121.00),
        "mandibular_plane_angle": (15.00, 22.00), "ramus_to_mandible": (0.62, 0.75),
        "gonion_to_mouth": (15.00, 45.00),
    },
    "hispanic": {
        "nasal_tip_angle": (129.50, 139.50), "nasal_width_to_height": (0.66, 0.82),
        "upper_lip_s_line": (-2.90, -1.10), "upper_lip_burstone": (-7.70, -4.30),
        "nasal_projection": (0.52, 0.60), "nasofrontal_angle": (114.50, 126.50),
        "recession_frankfort": (1.50, 15.00), "holdaway_h_line": (-2.45, -1.05),
        "mentolabial_angle": (112.00, 128.00), "upper_forehead_slope": (0.00, 2.00),
        "facial_convexity_nasion": (161.50, 164.50), "anterior_facial_depth": (64.00, 67.00),
        "upper_lip_e_line": (-0.80, 3.80), "submental_cervical_angle": (94.00, 106.00),
        "facial_depth_to_height": (1.33, 1.47), "browridge_inclination": (15.00, 22.00),
        "total_facial_convexity": (141.50, 148.50), "facial_convexity_glabella": (168.00, 173.00),
        "orbital_vector": (1.00, 10.00), "interior_midface_projection": (53.00, 60.00),
        "z_angle": (78.00, 82.00), "nose_tip_rotation": (14.00, 24.00),
        "nasolabial_angle": (93.00, 110.00), "nasofacial_angle": (30.50, 34.50),
        "nasomental_angle": (126.50, 131.50), "frankfort_tip_angle": (34.50, 40.50),
        "lower_lip_s_line": (-2.90, -1.10), "lower_lip_e_line": (-0.90, 2.40),
        "lower_lip_burstone": (-6.20, -3.80), "gonial_angle": (115.00, 121.00),
        "mandibular_plane_angle": (15.00, 22.00), "ramus_to_mandible": (0.62, 0.75),
        "gonion_to_mouth": (15.00, 45.00),
    },
    "middle_eastern": {
        "nasal_tip_angle": (128.50, 138.50), "nasal_width_to_height": (0.67, 0.83),
        "upper_lip_s_line": (-0.90, 0.90), "upper_lip_burstone": (-5.70, -2.30),
        "nasal_projection": (0.58, 0.65), "nasofrontal_angle": (117.00, 129.00),
        "recession_frankfort": (1.50, 15.00), "holdaway_h_line": (-0.20, 1.20),
        "mentolabial_angle": (110.00, 126.00), "upper_forehead_slope": (0.00, 2.00),
        "facial_convexity_nasion": (162.75, 165.75), "anterior_facial_depth": (64.50, 67.50),
        "upper_lip_e_line": (0.70, 5.30), "submental_cervical_angle": (94.00, 106.00),
        "facial_depth_to_height": (1.29, 1.43), "browridge_inclination": (15.00, 22.00),
        "total_facial_convexity": (139.75, 146.75), "facial_convexity_glabella": (169.75, 174.75),
        "orbital_vector": (1.00, 10.00), "interior_midface_projection": (53.00, 60.00),
        "z_angle": (78.00, 82.00), "nose_tip_rotation": (11.50, 21.50),
        "nasolabial_angle": (95.50, 112.50), "nasofacial_angle": (31.00, 35.00),
        "nasomental_angle": (126.00, 131.00), "frankfort_tip_angle": (32.00, 38.00),
        "lower_lip_s_line": (-0.90, 0.90), "lower_lip_e_line": (0.60, 3.90),
        "lower_lip_burstone": (-3.70, -1.30), "gonial_angle": (115.00, 121.00),
        "mandibular_plane_angle": (15.00, 22.00), "ramus_to_mandible": (0.62, 0.75),
        "gonion_to_mouth": (15.00, 45.00),
    },
    "south_asian": {
        "nasal_tip_angle": (128.50, 138.50), "nasal_width_to_height": (0.67, 0.83),
        "upper_lip_s_line": (-1.40, 0.40), "upper_lip_burstone": (-6.20, -2.80),
        "nasal_projection": (0.58, 0.65), "nasofrontal_angle": (118.00, 130.00),
        "recession_frankfort": (1.50, 15.00), "holdaway_h_line": (-0.20, 1.20),
        "mentolabial_angle": (109.00, 125.00), "upper_forehead_slope": (0.00, 2.00),
        "facial_convexity_nasion": (162.50, 165.50), "anterior_facial_depth": (64.50, 67.50),
        "upper_lip_e_line": (0.20, 4.80), "submental_cervical_angle": (94.00, 106.00),
        "facial_depth_to_height": (1.28, 1.42), "browridge_inclination": (15.00, 22.00),
        "total_facial_convexity": (139.50, 146.50), "facial_convexity_glabella": (169.50, 174.50),
        "orbital_vector": (1.00, 10.00), "interior_midface_projection": (53.00, 60.00),
        "z_angle": (78.00, 82.00), "nose_tip_rotation": (11.50, 21.50),
        "nasolabial_angle": (94.00, 111.00), "nasofacial_angle": (31.00, 35.00),
        "nasomental_angle": (126.00, 131.00), "frankfort_tip_angle": (32.00, 38.00),
        "lower_lip_s_line": (-1.40, 0.40), "lower_lip_e_line": (0.10, 3.40),
        "lower_lip_burstone": (-4.20, -1.80), "gonial_angle": (115.00, 121.00),
        "mandibular_plane_angle": (15.00, 22.00), "ramus_to_mandible": (0.62, 0.75),
        "gonion_to_mouth": (15.00, 45.00),
    },
    "mixed": {
        "nasal_tip_angle": (130.00, 140.00), "nasal_width_to_height": (0.62, 0.78),
        "upper_lip_s_line": (-0.90, 0.90), "upper_lip_burstone": (-5.70, -2.30),
        "nasal_projection": (0.55, 0.63), "nasofrontal_angle": (118.10, 130.10),
        "recession_frankfort": (1.50, 15.00), "holdaway_h_line": (-0.70, 0.70),
        "mentolabial_angle": (113.00, 129.00), "upper_forehead_slope": (0.00, 2.00),
        "facial_convexity_nasion": (163.50, 166.50), "anterior_facial_depth": (66.50, 69.50),
        "upper_lip_e_line": (0.70, 5.30), "submental_cervical_angle": (94.00, 106.00),
        "facial_depth_to_height": (1.29, 1.43), "browridge_inclination": (15.00, 22.00),
        "total_facial_convexity": (141.00, 148.00), "facial_convexity_glabella": (170.50, 175.50),
        "orbital_vector": (1.00, 10.00), "interior_midface_projection": (55.00, 62.00),
        "z_angle": (78.00, 82.00), "nose_tip_rotation": (11.50, 21.50),
        "nasolabial_angle": (94.50, 111.50), "nasofacial_angle": (30.50, 34.50),
        "nasomental_angle": (126.50, 131.50), "frankfort_tip_angle": (32.00, 38.00),
        "lower_lip_s_line": (-0.90, 0.90), "lower_lip_e_line": (0.60, 3.90),
        "lower_lip_burstone": (-3.70, -1.30), "gonial_angle": (116.00, 122.00),
        "mandibular_plane_angle": (15.50, 22.50), "ramus_to_mandible": (0.62, 0.75),
        "gonion_to_mouth": (15.00, 45.00),
    },
}


def _adjust(value, delta):
    return round(value + delta, 2)


def _adjust_ideal(base, delta_min, delta_max, delta_ideal_min, delta_ideal_max):
    return {
        "min": _adjust(base["min"], delta_min),
        "max": _adjust(base["max"], delta_max),
        "idealMin": _adjust(base["idealMin"], delta_ideal_min),
        "idealMax": _adjust(base["idealMax"], delta_ideal_max),
        "description": base["description"],
    }


def _apply_live_plateaus(values, plateaus):
    result = deepcopy(values)
    for key, (ideal_min, ideal_max) in plateaus.items():
        item = result[key]
        lower_margin = max(0.0, item["idealMin"] - item["min"])
        upper_margin = max(0.0, item["max"] - item["idealMax"])
        item["min"] = round(min(item["min"], ideal_min - lower_margin), 2)
        item["max"] = round(max(item["max"], ideal_max + upper_margin), 2)
        item["idealMin"] = ideal_min
        item["idealMax"] = ideal_max
    return result


def _build_side_norms(ethnicity):
    f = ETHNIC_FACTORS[ethnicity]
    np = f["nasalProjection"]
    lp = f["lipProtrusion"]
    pc = f["profileConvexity"]
    jw = f["jawWidth"]
    nw = f["noseWidth"]
    m = MALE_ASIAN_SIDE

    norms = {
        "nasal_projection": _adjust_ideal(m["nasal_projection"], np, np, np, np),
        "nasofrontal_angle": _adjust_ideal(m["nasofrontal_angle"], -np * 8, np * 8, -np * 5, np * 5),
        "nasal_tip_angle": _adjust_ideal(m["nasal_tip_angle"], -np * 15, np * 10, -np * 10, np * 8),
        "nasolabial_angle": _adjust_ideal(m["nasolabial_angle"], -np * 8, np * 5, -np * 5, np * 3),
        "nasofacial_angle": _adjust_ideal(m["nasofacial_angle"], -np * 3, np * 3, -np * 2, np * 2),
        "nasomental_angle": _adjust_ideal(m["nasomental_angle"], -np * 5, np * 5, -np * 3, np * 3),
        "nose_tip_rotation": _adjust_ideal(m["nose_tip_rotation"], -np * 3, np * 3, -np * 2, np * 2),
        "frankfort_tip_angle": _adjust_ideal(m["frankfort_tip_angle"], -np * 4, np * 4, -np * 3, np * 3),
        "nasal_width_to_height": _adjust_ideal(m["nasal_width_to_height"], nw * 0.003, nw * 0.003, nw * 0.002, nw * 0.002),
        "upper_lip_s_line": _adjust_ideal(m["upper_lip_s_line"], lp * 0.3, lp * 0.3, lp * 0.25, lp * 0.25),
        "upper_lip_e_line": _adjust_ideal(m["upper_lip_e_line"], lp * 0.35, lp * 0.35, lp * 0.3, lp * 0.3),
        "upper_lip_burstone": _adjust_ideal(m["upper_lip_burstone"], lp * 0.15, lp * 0.15, lp * 0.12, lp * 0.12),
        "lower_lip_s_line": _adjust_ideal(m["lower_lip_s_line"], lp * 0.3, lp * 0.3, lp * 0.25, lp * 0.25),
        "lower_lip_e_line": _adjust_ideal(m["lower_lip_e_line"], lp * 0.35, lp * 0.35, lp * 0.3, lp * 0.3),
        "lower_lip_burstone": _adjust_ideal(m["lower_lip_burstone"], lp * 0.15, lp * 0.15, lp * 0.12, lp * 0.12),
        "facial_convexity_nasion": _adjust_ideal(m["facial_convexity_nasion"], pc, pc, pc, pc),
        "facial_convexity_glabella": _adjust_ideal(m["facial_convexity_glabella"], pc * 0.8, pc * 0.8, pc * 0.6, pc * 0.6),
        "total_facial_convexity": _adjust_ideal(m["total_facial_convexity"], pc, pc, pc, pc),
        "gonial_angle": _adjust_ideal(m["gonial_angle"], jw, jw, jw, jw),
        "mandibular_plane_angle": _adjust_ideal(m["mandibular_plane_angle"], jw * 0.5, jw * 0.5, jw * 0.4, jw * 0.4),
        "ramus_to_mandible": _adjust_ideal(m["ramus_to_mandible"], jw * 0.003, jw * 0.003, jw * 0.002, jw * 0.002),
        "recession_frankfort": deepcopy(m["recession_frankfort"]),
        "holdaway_h_line": deepcopy(m["holdaway_h_line"]),
        "mentolabial_angle": _adjust_ideal(m["mentolabial_angle"], pc * 0.5, pc * 0.5, pc * 0.4, pc * 0.4),
        "upper_forehead_slope": deepcopy(m["upper_forehead_slope"]),
        "anterior_facial_depth": deepcopy(m["anterior_facial_depth"]),
        "submental_cervical_angle": deepcopy(m["submental_cervical_angle"]),
        "facial_depth_to_height": deepcopy(m["facial_depth_to_height"]),
        "browridge_inclination": deepcopy(m["browridge_inclination"]),
        "orbital_vector": _adjust_ideal(m["orbital_vector"], np * 0.5, np * 0.5, np * 0.4, np * 0.4),
        "interior_midface_projection": deepcopy(m["interior_midface_projection"]),
        "z_angle": deepcopy(m["z_angle"]),
        "gonion_to_mouth": _adjust_ideal(m["gonion_to_mouth"], jw * 0.3, jw * 0.3, jw * 0.25, jw * 0.25),
    }
    for key, value in m.items():
        norms.setdefault(key, deepcopy(value))
    return norms


def _build_female_side(male_values):
    result = {}
    for key, val in male_values.items():
        if key == "nasal_tip_angle":
            result[key] = _adjust_ideal(val, 4, 4, 3, 3)
        elif key == "nasolabial_angle":
            result[key] = _adjust_ideal(val, 3, 3, 2.5, 2.5)
        elif key == "mentolabial_angle":
            result[key] = _adjust_ideal(val, 4, 4, 3, 3)
        elif key == "facial_convexity_nasion":
            result[key] = _adjust_ideal(val, 3, 3, 2.5, 2.5)
        elif key == "facial_convexity_glabella":
            result[key] = _adjust_ideal(val, 2, 2, 1.5, 1.5)
        elif key == "total_facial_convexity":
            result[key] = _adjust_ideal(val, 3, 3, 2.5, 2.5)
        elif key == "gonial_angle":
            result[key] = _adjust_ideal(val, 2, 2, 1.5, 1.5)
        elif key == "mandibular_plane_angle":
            result[key] = _adjust_ideal(val, 2, 2, 1.5, 1.5)
        elif key in {"upper_lip_s_line", "upper_lip_e_line", "lower_lip_s_line", "lower_lip_e_line"}:
            result[key] = _adjust_ideal(val, -0.3, -0.3, -0.25, -0.25)
        elif key == "nasal_projection":
            result[key] = _adjust_ideal(val, -0.02, -0.02, -0.015, -0.015)
        elif key == "ramus_to_mandible":
            result[key] = _adjust_ideal(val, -0.02, -0.02, -0.015, -0.015)
        else:
            result[key] = deepcopy(val)
    return result


def build_side_ideals():
    male = {"asian": deepcopy(MALE_ASIAN_SIDE)}
    for ethnicity in ETHNIC_FACTORS:
        male[ethnicity] = _apply_live_plateaus(
            _build_side_norms(ethnicity),
            MALE_LIVE_SIDE_PLATEAUS[ethnicity],
        )

    female = {}
    for ethnicity, values in male.items():
        female[ethnicity] = _build_female_side(values)

    return {"male": male, "female": female}


SIDE_IDEALS = build_side_ideals()


def get_side_ideals(gender="male", ethnicity="asian"):
    return SIDE_IDEALS.get(gender, {}).get(ethnicity, SIDE_IDEALS["male"]["asian"])
