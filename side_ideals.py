from copy import deepcopy

from front_ideals import ETHNIC_FACTORS


MALE_ASIAN_SIDE = {
    "nasal_tip_angle": {"min": 106.16, "max": 166.84, "idealMin": 130.00, "idealMax": 142.00, "description": "Nasal Tip Angle (degrees)"},
    "nasal_width_to_height": {"min": -0.01, "max": 1.31, "idealMin": 0.55, "idealMax": 0.75, "description": "Nasal Width to Height Ratio"},
    "upper_lip_s_line": {"min": -8.19, "max": 7.19, "idealMin": -2.00, "idealMax": 1.00, "description": "Upper Lip S-Line Position (mm)"},
    "upper_lip_burstone": {"min": -2, "max": 3, "idealMin": -0.50, "idealMax": 1.50, "description": "Upper Lip Burstone Line (mm)"},
    "nasal_projection": {"min": 0.11, "max": 1.02, "idealMin": 0.50, "idealMax": 0.63, "description": "Nasal Projection ratio"},
    "nasofrontal_angle": {"min": 79.18, "max": 173.22, "idealMin": 120.00, "idealMax": 132.50, "description": "Nasofrontal Angle (degrees)"},
    "recession_frankfort": {"min": -24.08, "max": 39.05, "idealMin": 1.50, "idealMax": 15.00, "description": "Recession Frankfort Plane (mm)"},
    "holdaway_h_line": {"min": -9.79, "max": 8.79, "idealMin": -2.00, "idealMax": 1.00, "description": "Holdaway H Line (mm)"},
    "mentolabial_angle": {"min": 60.13, "max": 185.87, "idealMin": 114.00, "idealMax": 132.00, "description": "Mentolabial Angle (degrees)"},
    "upper_forehead_slope": {"min": -13.30, "max": 15.30, "idealMin": -2.00, "idealMax": 4.50, "description": "Upper Forehead Slope (degrees)"},
    "facial_convexity_nasion": {"min": 134.63, "max": 196.37, "idealMin": 160.00, "idealMax": 171.00, "description": "Facial Convexity at Nasion (degrees)"},
    "anterior_facial_depth": {"min": 36.12, "max": 103.88, "idealMin": 65.00, "idealMax": 75.00, "description": "Anterior Facial Depth (degrees)"},
    "upper_lip_e_line": {"min": -7.56, "max": 12.56, "idealMin": 1.00, "idealMax": 4.00, "description": "Upper Lip E-Line Position (mm)"},
    "submental_cervical_angle": {"min": 56.02, "max": 143.98, "idealMin": 90.00, "idealMax": 110.00, "description": "Submental Cervical Angle (degrees)"},
    "facial_depth_to_height": {"min": 0.94, "max": 1.76, "idealMin": 1.28, "idealMax": 1.42, "description": "Facial Depth to Height Ratio"},
    "browridge_inclination": {"min": -0.75, "max": 37.75, "idealMin": 14.00, "idealMax": 23.00, "description": "Browridge Inclination (degrees)"},
    "total_facial_convexity": {"min": 120.14, "max": 170.86, "idealMin": 140.00, "idealMax": 150.00, "description": "Total Facial Convexity (degrees)"},
    "facial_convexity_glabella": {"min": 153.31, "max": 193.69, "idealMin": 168.00, "idealMax": 179.00, "description": "Facial Convexity at Glabella (degrees)"},
    "orbital_vector": {"min": -10.33, "max": 19.25, "idealMin": 2.00, "idealMax": 7.00, "description": "Orbital Vector (mm)"},
    "interior_midface_projection": {"min": 35.75, "max": 85.25, "idealMin": 56.00, "idealMax": 65.00, "description": "Interior Midface Projection (degrees)"},
    "z_angle": {"min": 54.18, "max": 105.82, "idealMin": 75.00, "idealMax": 85.00, "description": "Z-Angle (degrees)"},
    "nose_tip_rotation": {"min": -15.08, "max": 48.08, "idealMin": 10.00, "idealMax": 23.00, "description": "Nose Tip Rotation (degrees)"},
    "nasolabial_angle": {"min": 55.02, "max": 145.98, "idealMin": 95.00, "idealMax": 106.00, "description": "Nasolabial Angle (degrees)"},
    "nasofacial_angle": {"min": 15.93, "max": 48.07, "idealMin": 28.00, "idealMax": 36.00, "description": "Nasofacial Angle (degrees)"},
    "nasomental_angle": {"min": 105.26, "max": 153.74, "idealMin": 124.00, "idealMax": 135.00, "description": "Nasomental Angle (degrees)"},
    "frankfort_tip_angle": {"min": 5.59, "max": 64.41, "idealMin": 30.00, "idealMax": 40.00, "description": "Frankfort-Tip Angle (degrees)"},
    "lower_lip_s_line": {"min": -8.19, "max": 7.19, "idealMin": -2.00, "idealMax": 1.00, "description": "Lower Lip S-Line (mm)"},
    "lower_lip_e_line": {"min": -7.74, "max": 11.24, "idealMin": 0.50, "idealMax": 3.00, "description": "Lower Lip E-Line (mm)"},
    "lower_lip_burstone": {"min": -9.47, "max": 3.48, "idealMin": -4.50, "idealMax": -1.50, "description": "Lower Lip Burstone Line (mm)"},
    "gonial_angle": {"min": 94.34, "max": 145.66, "idealMin": 117.00, "idealMax": 123.00, "description": "Gonial Angle (degrees)"},
    "mandibular_plane_angle": {"min": -4.68, "max": 43.68, "idealMin": 14.00, "idealMax": 24.00, "description": "Mandibular Plane Angle (degrees)"},
    "ramus_to_mandible": {"min": -0.20, "max": 1.57, "idealMin": 0.55, "idealMax": 0.80, "description": "Ramus to Mandible Ratio"},
    "gonion_to_mouth": {"min": -4.95, "max": 64.95, "idealMin": 22.00, "idealMax": 38.00, "description": "Gonion to Mouth Line (mm)"},
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
        male[ethnicity] = _build_side_norms(ethnicity)

    female = {}
    for ethnicity, values in male.items():
        female[ethnicity] = _build_female_side(values)

    return {"male": male, "female": female}


SIDE_IDEALS = build_side_ideals()


def get_side_ideals(gender="male", ethnicity="asian"):
    return SIDE_IDEALS.get(gender, {}).get(ethnicity, SIDE_IDEALS["male"]["asian"])
