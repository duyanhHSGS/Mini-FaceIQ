from copy import deepcopy


ETHNIC_FACTORS = {
    "caucasian": {
        "sizeScale": 0.95,
        "noseWidth": -8,
        "nasalProjection": 0.12,
        "lipProtrusion": -2,
        "canthalTilt": -1.5,
        "profileConvexity": -5,
        "jawWidth": -5,
        "eyeSeparation": -1,
    },
    "black": {
        "sizeScale": 1.02,
        "noseWidth": 12,
        "nasalProjection": -0.05,
        "lipProtrusion": 3,
        "canthalTilt": -2.5,
        "profileConvexity": 3,
        "jawWidth": 5,
        "eyeSeparation": 2,
    },
    "hispanic": {
        "sizeScale": 0.98,
        "noseWidth": 2,
        "nasalProjection": 0.06,
        "lipProtrusion": 1,
        "canthalTilt": -0.5,
        "profileConvexity": -2,
        "jawWidth": -2,
        "eyeSeparation": 0,
    },
    "middle_eastern": {
        "sizeScale": 0.97,
        "noseWidth": -3,
        "nasalProjection": 0.15,
        "lipProtrusion": -1,
        "canthalTilt": -1,
        "profileConvexity": -3,
        "jawWidth": -3,
        "eyeSeparation": -1,
    },
    "south_asian": {
        "sizeScale": 0.99,
        "noseWidth": 4,
        "nasalProjection": 0.04,
        "lipProtrusion": 0.5,
        "canthalTilt": 0,
        "profileConvexity": -1,
        "jawWidth": -1,
        "eyeSeparation": 1,
    },
    "mixed": {
        "sizeScale": 0.98,
        "noseWidth": 1,
        "nasalProjection": 0.05,
        "lipProtrusion": 0,
        "canthalTilt": -0.5,
        "profileConvexity": -1.5,
        "jawWidth": -2,
        "eyeSeparation": 0,
    },
}


# East-Asian ideal plateaus transcribed from docs.txt. The outer min/max values
# remain scoring falloff bounds rather than additional ideal ranges.
MALE_ASIAN_FRONT = {
    "lateral_canthal_tilt": {"min": -2.57, "max": 19.67, "idealMin": 7.70, "idealMax": 9.40, "description": "Lateral Canthal Tilt (degrees)"},
    "nose_bridge_to_width": {"min": 1.16, "max": 3.04, "idealMin": 2.10, "idealMax": 2.10, "description": "Nose Bridge to Nose Width Ratio"},
    "bitemporal_width": {"min": 75, "max": 101.30, "idealMin": 86.50, "idealMax": 92.50, "description": "Bitemporal Width (%)"},
    "cheekbone_height": {"min": 49.48, "max": 133.52, "idealMin": 83.00, "idealMax": 100.00, "description": "Cheekbone Height (%)"},
    "cupids_bow_depth": {"min": -2.15, "max": 8.53, "idealMin": 2.30, "idealMax": 4.00, "description": "Cupid's Bow Depth (mm)"},
    "bigonial_width": {"min": 68.55, "max": 110.45, "idealMin": 87.50, "idealMax": 91.50, "description": "Bigonial Width (%)"},
    "jaw_slope": {"min": 115.51, "max": 166.99, "idealMin": 140.00, "idealMax": 142.50, "description": "Jaw Slope (degrees)"},
    "middle_third": {"min": 22.74, "max": 43.06, "idealMin": 31.90, "idealMax": 33.90, "description": "Middle Third (%)"},
    "eye_aspect_ratio": {"min": 1.42, "max": 4.88, "idealMin": 2.90, "idealMax": 3.40, "description": "Eye Aspect Ratio"},
    "mouth_corner_position": {"min": -12.94, "max": 16.94, "idealMin": 0.00, "idealMax": 4.00, "description": "Mouth Corner Position (mm)"},
    "eye_separation_ratio": {"min": 37.38, "max": 54.98, "idealMin": 45.60, "idealMax": 46.70, "description": "Eye Separation Ratio (%)"},
    "eyebrow_tilt": {"min": -14.02, "max": 31.52, "idealMin": 6.50, "idealMax": 11.00, "description": "Eyebrow Tilt (degrees)"},
    "lower_third": {"min": 25.78, "max": 44.32, "idealMin": 33.50, "idealMax": 36.60, "description": "Lower Third (%)"},
    "face_width_to_height": {"min": 1.52, "max": 2.38, "idealMin": 1.90, "idealMax": 2.00, "description": "Face Width to Height Ratio (fWHR)"},
    "interpupillary_mouth_width": {"min": 37, "max": 123, "idealMin": 80.00, "idealMax": 80.00, "description": "Interpupillary-Mouth Width Ratio (%)"},
    "jaw_frontal_angle": {"min": 54.78, "max": 124.22, "idealMin": 86.50, "idealMax": 92.50, "description": "Jaw Frontal Angle (degrees)"},
    "intercanthal_nasal_width": {"min": 0.90, "max": 1.20, "idealMin": 1.00, "idealMax": 1.10, "description": "Intercanthal-Nasal Width Ratio"},
    "top_third": {"min": 20.25, "max": 42.75, "idealMin": 30.50, "idealMax": 32.50, "description": "Top Third (%)"},
    "one_eye_apart": {"min": 0.72, "max": 1.53, "idealMin": 1.10, "idealMax": 1.10, "description": "One Eye Apart Test"},
    "midface_ratio": {"min": 0.61, "max": 1.34, "idealMin": 1.00, "idealMax": 1.00, "description": "Midface Ratio"},
    "ipsilateral_alar_angle": {"min": 68.23, "max": 106.77, "idealMin": 84.50, "idealMax": 90.50, "description": "Ipsilateral Alar Angle (degrees)"},
    "mouth_width_to_nose_width": {"min": 1.04, "max": 1.80, "idealMin": 1.40, "idealMax": 1.50, "description": "Mouth Width to Nose Width Ratio"},
    "total_facial_width_to_height": {"min": 1.237, "max": 1.50, "idealMin": 1.30, "idealMax": 1.40, "description": "Total Facial Width to Height Ratio"},
    "chin_to_philtrum": {"min": 0.78, "max": 3.82, "idealMin": 2.10, "idealMax": 2.50, "description": "Chin to Philtrum Ratio"},
    "eyebrow_low_setedness": {"min": -1.96, "max": 3.21, "idealMin": 0.40, "idealMax": 0.80, "description": "Eyebrow Low Setedness"},
    "brow_length_to_face_width": {"min": 0.33, "max": 1.12, "idealMin": 0.70, "idealMax": 0.80, "description": "Brow Length to Face Width Ratio"},
    "nose_tip_position": {"min": -1, "max": 8, "idealMin": 0.00, "idealMax": 3.00, "description": "Nose Tip Position (mm)"},
    "deviation_iaa_jfa": {"min": -22.21, "max": 22.32, "idealMin": 0.00, "idealMax": 2.50, "description": "Deviation of IAA & JFA (degrees)"},
    "lower_lip_to_upper_lip": {"min": -0.44, "max": 4.04, "idealMin": 1.60, "idealMax": 1.90, "description": "Lower Lip to Upper Lip Ratio"},
    "lower_third_proportion": {"min": 26.21, "max": 38.29, "idealMin": 31.00, "idealMax": 33.50, "description": "Lower Third Proportion (%)"},
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


def _scale_ideal(base, factor):
    return {
        "min": round(base["min"] * factor, 2),
        "max": round(base["max"] * factor, 2),
        "idealMin": round(base["idealMin"] * factor, 2),
        "idealMax": round(base["idealMax"] * factor, 2),
        "description": base["description"],
    }


def _build_front_norms(ethnicity):
    f = ETHNIC_FACTORS[ethnicity]
    s = f["sizeScale"]
    nw = f["noseWidth"]
    lp = f["lipProtrusion"]
    ct = f["canthalTilt"]
    jw = f["jawWidth"]
    es = f["eyeSeparation"]
    m = MALE_ASIAN_FRONT

    norms = {
        "bitemporal_width": _scale_ideal(m["bitemporal_width"], s),
        "cheekbone_height": _scale_ideal(m["cheekbone_height"], s),
        "bigonial_width": _adjust_ideal(m["bigonial_width"], jw, jw, jw, jw),
        "middle_third": deepcopy(m["middle_third"]),
        "lower_third": deepcopy(m["lower_third"]),
        "top_third": deepcopy(m["top_third"]),
        "lower_third_proportion": deepcopy(m["lower_third_proportion"]),
        "face_width_to_height": _scale_ideal(m["face_width_to_height"], s),
        "total_facial_width_to_height": _scale_ideal(m["total_facial_width_to_height"], s),
        "nose_bridge_to_width": _adjust_ideal(m["nose_bridge_to_width"], nw * 0.015, nw * 0.015, nw * 0.012, nw * 0.012),
        "intercanthal_nasal_width": _adjust_ideal(m["intercanthal_nasal_width"], nw * 0.002, nw * 0.002, nw * 0.002, nw * 0.002),
        "mouth_width_to_nose_width": _adjust_ideal(m["mouth_width_to_nose_width"], -nw * 0.006, -nw * 0.006, -nw * 0.005, -nw * 0.005),
        "ipsilateral_alar_angle": _adjust_ideal(m["ipsilateral_alar_angle"], nw * 0.5, nw * 0.5, nw * 0.4, nw * 0.4),
        "nose_tip_position": _adjust_ideal(m["nose_tip_position"], nw * 0.02, nw * 0.02, nw * 0.015, nw * 0.015),
        "lateral_canthal_tilt": _adjust_ideal(m["lateral_canthal_tilt"], ct, ct, ct, ct),
        "eye_separation_ratio": _adjust_ideal(m["eye_separation_ratio"], es, es, es, es),
        "one_eye_apart": _adjust_ideal(m["one_eye_apart"], es * 0.01, es * 0.01, es * 0.008, es * 0.008),
        "mouth_corner_position": _adjust_ideal(m["mouth_corner_position"], lp * 0.15, lp * 0.15, lp * 0.12, lp * 0.12),
        "jaw_frontal_angle": _adjust_ideal(m["jaw_frontal_angle"], jw, jw, jw, jw),
        "jaw_slope": _adjust_ideal(m["jaw_slope"], jw * 0.5, jw * 0.5, jw * 0.4, jw * 0.4),
        "cupids_bow_depth": deepcopy(m["cupids_bow_depth"]),
        "eye_aspect_ratio": deepcopy(m["eye_aspect_ratio"]),
        "eyebrow_tilt": deepcopy(m["eyebrow_tilt"]),
        "interpupillary_mouth_width": deepcopy(m["interpupillary_mouth_width"]),
        "midface_ratio": deepcopy(m["midface_ratio"]),
        "chin_to_philtrum": deepcopy(m["chin_to_philtrum"]),
        "eyebrow_low_setedness": deepcopy(m["eyebrow_low_setedness"]),
        "brow_length_to_face_width": deepcopy(m["brow_length_to_face_width"]),
        "deviation_iaa_jfa": deepcopy(m["deviation_iaa_jfa"]),
        "lower_lip_to_upper_lip": deepcopy(m["lower_lip_to_upper_lip"]),
    }
    for key, value in m.items():
        norms.setdefault(key, deepcopy(value))
    return norms


def _build_female_front(male_values):
    result = {}
    for key, val in male_values.items():
        if key == "lateral_canthal_tilt":
            result[key] = _adjust_ideal(val, 1.5, 1.5, 1.5, 1.5)
        elif key == "jaw_slope":
            result[key] = _adjust_ideal(val, 3, 3, 2.5, 2.5)
        elif key == "jaw_frontal_angle":
            result[key] = _adjust_ideal(val, 3, 3, 2.5, 2.5)
        elif key == "eyebrow_tilt":
            result[key] = _adjust_ideal(val, 2, 2, 1.5, 1.5)
        elif key == "bigonial_width":
            result[key] = _adjust_ideal(val, -3, -3, -2.5, -2.5)
        elif key == "face_width_to_height":
            result[key] = _scale_ideal(val, 0.95)
        elif key == "total_facial_width_to_height":
            result[key] = _scale_ideal(val, 0.95)
        elif key == "eye_aspect_ratio":
            result[key] = _adjust_ideal(val, 0.2, 0.2, 0.15, 0.15)
        elif key == "chin_to_philtrum":
            result[key] = _adjust_ideal(val, -0.1, -0.1, -0.08, -0.08)
        elif key == "lower_lip_to_upper_lip":
            result[key] = _adjust_ideal(val, 0.05, 0.05, 0.04, 0.04)
        elif key == "nose_bridge_to_width":
            result[key] = _adjust_ideal(val, -0.05, -0.05, -0.04, -0.04)
        elif key == "cupids_bow_depth":
            result[key] = _adjust_ideal(val, 0.5, 0.5, 0.4, 0.4)
        elif key == "mouth_corner_position":
            result[key] = _adjust_ideal(val, 0.8, 0.8, 0.6, 0.6)
        elif key == "nose_tip_position":
            result[key] = _adjust_ideal(val, -0.5, -0.5, -0.4, -0.4)
        else:
            result[key] = deepcopy(val)
    return result


def build_front_ideals():
    male = {"asian": deepcopy(MALE_ASIAN_FRONT)}
    for ethnicity in ETHNIC_FACTORS:
        male[ethnicity] = _build_front_norms(ethnicity)

    female = {}
    for ethnicity, values in male.items():
        female[ethnicity] = _build_female_front(values)

    return {"male": male, "female": female}


FRONT_IDEALS = build_front_ideals()


def get_front_ideals(gender="male", ethnicity="asian"):
    return FRONT_IDEALS.get(gender, {}).get(ethnicity, FRONT_IDEALS["male"]["asian"])
