from copy import deepcopy


# These transformations retain the existing legacy reference bounds and female
# derivation contract. Confirmed non-Asian male ideal plateaus are applied from
# MALE_LIVE_FRONT_PLATEAUS instead of being inferred from these factors.
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


# East-Asian male ideal plateaus transcribed from the saved FaceIQ cohort notes.
# Outer min/max values remain legacy API reference bounds; exponential scoring
# uses idealMin, idealMax, and decayRate instead.
MALE_ASIAN_FRONT = {
    "lateral_canthal_tilt": {"min": -2.57, "max": 19.67, "idealMin": 7.70, "idealMax": 9.40, "description": "Lateral Canthal Tilt (degrees)"},
    "nose_bridge_to_width": {"min": 1.16, "max": 3.04, "idealMin": 2.06, "idealMax": 2.14, "description": "Nose Bridge to Nose Width Ratio"},
    "bitemporal_width": {"min": 75, "max": 101.30, "idealMin": 86.50, "idealMax": 92.50, "description": "Bitemporal Width (%)"},
    "cheekbone_height": {"min": 49.48, "max": 133.52, "idealMin": 83.00, "idealMax": 100.00, "description": "Cheekbone Height (%)"},
    "cupids_bow_depth": {"min": -2.15, "max": 8.53, "idealMin": 2.30, "idealMax": 4.00, "description": "Cupid's Bow Depth (mm)"},
    "bigonial_width": {"min": 68.55, "max": 110.45, "idealMin": 87.50, "idealMax": 91.50, "description": "Bigonial Width (%)"},
    "jaw_slope": {"min": 115.51, "max": 166.99, "idealMin": 140.00, "idealMax": 142.50, "description": "Jaw Slope (degrees)"},
    "middle_third": {"min": 22.74, "max": 43.06, "idealMin": 31.90, "idealMax": 33.90, "description": "Middle Third (%)"},
    "eye_aspect_ratio": {"min": 1.42, "max": 4.88, "idealMin": 2.90, "idealMax": 3.40, "description": "Eye Aspect Ratio"},
    "mouth_corner_position": {"min": -12.94, "max": 16.94, "idealMin": 0.00, "idealMax": 4.00, "description": "Mouth Corner Position (mm)"},
    "eye_separation_ratio": {"min": 37.38, "max": 54.98, "idealMin": 45.63, "idealMax": 46.73, "description": "Eye Separation Ratio (%)"},
    "eyebrow_tilt": {"min": -14.02, "max": 31.52, "idealMin": 6.50, "idealMax": 11.00, "description": "Eyebrow Tilt (degrees)"},
    "lower_third": {"min": 25.78, "max": 44.32, "idealMin": 33.50, "idealMax": 36.60, "description": "Lower Third (%)"},
    "face_width_to_height": {"min": 1.52, "max": 2.38, "idealMin": 1.93, "idealMax": 1.97, "description": "Face Width to Height Ratio (fWHR)"},
    "interpupillary_mouth_width": {"min": 37, "max": 123, "idealMin": 78.00, "idealMax": 82.00, "description": "Interpupillary-Mouth Width Ratio (%)"},
    "jaw_frontal_angle": {"min": 54.78, "max": 124.22, "idealMin": 86.50, "idealMax": 92.50, "description": "Jaw Frontal Angle (degrees)"},
    "intercanthal_nasal_width": {"min": 0.90, "max": 1.20, "idealMin": 1.00, "idealMax": 1.12, "description": "Intercanthal-Nasal Width Ratio"},
    "top_third": {"min": 20.25, "max": 42.75, "idealMin": 30.50, "idealMax": 32.50, "description": "Top Third (%)"},
    "one_eye_apart": {"min": 0.72, "max": 1.53, "idealMin": 1.10, "idealMax": 1.15, "description": "One Eye Apart Test"},
    "midface_ratio": {"min": 0.61, "max": 1.34, "idealMin": 0.96, "idealMax": 0.99, "description": "Midface Ratio"},
    "ipsilateral_alar_angle": {"min": 68.23, "max": 106.77, "idealMin": 84.50, "idealMax": 90.50, "description": "Ipsilateral Alar Angle (degrees)"},
    "mouth_width_to_nose_width": {"min": 1.04, "max": 1.80, "idealMin": 1.38, "idealMax": 1.46, "description": "Mouth Width to Nose Width Ratio"},
    "total_facial_width_to_height": {"min": 1.237, "max": 1.50, "idealMin": 1.34, "idealMax": 1.37, "description": "Total Facial Width to Height Ratio"},
    "chin_to_philtrum": {"min": 0.78, "max": 3.82, "idealMin": 2.15, "idealMax": 2.45, "description": "Chin to Philtrum Ratio"},
    "eyebrow_low_setedness": {"min": -1.96, "max": 3.21, "idealMin": 0.40, "idealMax": 0.85, "description": "Eyebrow Low Setedness"},
    "brow_length_to_face_width": {"min": 0.33, "max": 1.12, "idealMin": 0.69, "idealMax": 0.76, "description": "Brow Length to Face Width Ratio"},
    "nose_tip_position": {"min": -1, "max": 8, "idealMin": 0.00, "idealMax": 3.00, "description": "Nose Tip Position (mm)"},
    "deviation_iaa_jfa": {"min": -22.21, "max": 22.32, "idealMin": 0.00, "idealMax": 2.50, "description": "Deviation of IAA & JFA (degrees)"},
    "lower_lip_to_upper_lip": {"min": -0.44, "max": 4.04, "idealMin": 1.65, "idealMax": 1.95, "description": "Lower Lip to Upper Lip Ratio"},
    "lower_third_proportion": {"min": 26.21, "max": 38.29, "idealMin": 31.00, "idealMax": 33.50, "description": "Lower Third Proportion (%)"},
}


# Exact supported male plateaus captured from FaceIQ live on 2026-08-03.
# Ratio 0.83-0.87 is stored as 83-87 where the local calculator emits percent.
# Source-heading mapping: caucasian=Caucasian/African-American;
# black=African-American+Samoan; hispanic=Black+Latino/African+Brazilian;
# middle_eastern=South Asian+Caucasian; south_asian=South Asian;
# mixed=Caucasian+East Asian.
MALE_LIVE_FRONT_PLATEAUS = {
    "caucasian": {
        "lateral_canthal_tilt": (6.00, 7.70), "nose_bridge_to_width": (2.06, 2.14),
        "bitemporal_width": (86.50, 92.50), "cheekbone_height": (83.00, 100.00),
        "cupids_bow_depth": (2.30, 4.00), "bigonial_width": (87.50, 91.50),
        "jaw_slope": (140.00, 142.50), "middle_third": (31.40, 33.40),
        "eye_aspect_ratio": (3.00, 3.50), "mouth_corner_position": (0.00, 4.00),
        "eye_separation_ratio": (45.70, 46.80), "eyebrow_tilt": (6.50, 11.00),
        "lower_third": (33.90, 37.00), "face_width_to_height": (1.96, 2.00),
        "interpupillary_mouth_width": (83.00, 87.00), "jaw_frontal_angle": (86.50, 92.50),
        "intercanthal_nasal_width": (1.04, 1.16), "top_third": (30.00, 32.00),
        "one_eye_apart": (0.95, 1.00), "midface_ratio": (0.97, 1.00),
        "ipsilateral_alar_angle": (86.50, 92.50), "mouth_width_to_nose_width": (1.42, 1.50),
        "total_facial_width_to_height": (1.34, 1.37), "chin_to_philtrum": (2.15, 2.45),
        "eyebrow_low_setedness": (0.00, 0.45), "brow_length_to_face_width": (0.69, 0.76),
        "nose_tip_position": (0.00, 3.00), "deviation_iaa_jfa": (0.00, 2.50),
        "lower_lip_to_upper_lip": (1.55, 1.85), "lower_third_proportion": (31.00, 33.50),
    },
    "black": {
        "lateral_canthal_tilt": (6.00, 7.70), "nose_bridge_to_width": (2.06, 2.14),
        "bitemporal_width": (87.50, 93.50), "cheekbone_height": (83.00, 100.00),
        "cupids_bow_depth": (2.30, 4.00), "bigonial_width": (89.50, 93.50),
        "jaw_slope": (140.00, 142.50), "middle_third": (31.40, 33.40),
        "eye_aspect_ratio": (3.00, 3.50), "mouth_corner_position": (0.00, 4.00),
        "eye_separation_ratio": (45.70, 46.80), "eyebrow_tilt": (8.50, 13.00),
        "lower_third": (33.90, 37.00), "face_width_to_height": (1.98, 2.02),
        "interpupillary_mouth_width": (83.00, 87.00), "jaw_frontal_angle": (86.50, 92.50),
        "intercanthal_nasal_width": (1.10, 1.22), "top_third": (30.00, 32.00),
        "one_eye_apart": (0.95, 1.00), "midface_ratio": (0.97, 1.00),
        "ipsilateral_alar_angle": (86.50, 92.50), "mouth_width_to_nose_width": (1.36, 1.44),
        "total_facial_width_to_height": (1.36, 1.39), "chin_to_philtrum": (2.15, 2.45),
        "eyebrow_low_setedness": (0.00, 0.45), "brow_length_to_face_width": (0.69, 0.76),
        "nose_tip_position": (0.00, 3.00), "deviation_iaa_jfa": (0.00, 2.50),
        "lower_lip_to_upper_lip": (1.55, 1.85), "lower_third_proportion": (31.00, 33.50),
    },
    "hispanic": {
        "lateral_canthal_tilt": (6.60, 8.30), "nose_bridge_to_width": (2.06, 2.14),
        "bitemporal_width": (86.50, 92.50), "cheekbone_height": (83.00, 100.00),
        "cupids_bow_depth": (2.30, 4.00), "bigonial_width": (87.75, 91.75),
        "jaw_slope": (140.00, 142.50), "middle_third": (31.10, 33.10),
        "eye_aspect_ratio": (3.00, 3.50), "mouth_corner_position": (0.00, 4.00),
        "eye_separation_ratio": (46.15, 47.25), "eyebrow_tilt": (6.50, 11.00),
        "lower_third": (34.20, 37.30), "face_width_to_height": (1.98, 2.02),
        "interpupillary_mouth_width": (86.00, 90.00), "jaw_frontal_angle": (86.50, 92.50),
        "intercanthal_nasal_width": (1.14, 1.26), "top_third": (29.70, 31.70),
        "one_eye_apart": (0.98, 1.03), "midface_ratio": (1.00, 1.02),
        "ipsilateral_alar_angle": (87.00, 93.00), "mouth_width_to_nose_width": (1.41, 1.49),
        "total_facial_width_to_height": (1.34, 1.37), "chin_to_philtrum": (2.17, 2.48),
        "eyebrow_low_setedness": (0.00, 0.45), "brow_length_to_face_width": (0.69, 0.76),
        "nose_tip_position": (0.50, 3.50), "deviation_iaa_jfa": (0.00, 2.50),
        "lower_lip_to_upper_lip": (1.48, 1.78), "lower_third_proportion": (31.20, 33.70),
    },
    "middle_eastern": {
        "lateral_canthal_tilt": (6.00, 7.70), "nose_bridge_to_width": (2.06, 2.14),
        "bitemporal_width": (85.00, 91.00), "cheekbone_height": (83.00, 100.00),
        "cupids_bow_depth": (2.30, 4.00), "bigonial_width": (87.50, 91.50),
        "jaw_slope": (140.00, 142.50), "middle_third": (31.30, 33.30),
        "eye_aspect_ratio": (3.00, 3.50), "mouth_corner_position": (0.00, 4.00),
        "eye_separation_ratio": (45.70, 46.80), "eyebrow_tilt": (6.50, 11.00),
        "lower_third": (33.95, 37.05), "face_width_to_height": (1.96, 2.00),
        "interpupillary_mouth_width": (83.00, 87.00), "jaw_frontal_angle": (86.50, 92.50),
        "intercanthal_nasal_width": (1.06, 1.17), "top_third": (30.05, 32.05),
        "one_eye_apart": (0.95, 1.00), "midface_ratio": (0.97, 1.00),
        "ipsilateral_alar_angle": (86.50, 92.50), "mouth_width_to_nose_width": (1.41, 1.49),
        "total_facial_width_to_height": (1.34, 1.37), "chin_to_philtrum": (2.15, 2.45),
        "eyebrow_low_setedness": (0.00, 0.45), "brow_length_to_face_width": (0.69, 0.76),
        "nose_tip_position": (0.00, 3.00), "deviation_iaa_jfa": (0.00, 2.50),
        "lower_lip_to_upper_lip": (1.55, 1.85), "lower_third_proportion": (31.00, 33.50),
    },
    "south_asian": {
        "lateral_canthal_tilt": (6.00, 7.70), "nose_bridge_to_width": (2.06, 2.14),
        "bitemporal_width": (83.50, 89.50), "cheekbone_height": (83.00, 100.00),
        "cupids_bow_depth": (2.30, 4.00), "bigonial_width": (87.50, 91.50),
        "jaw_slope": (140.00, 142.50), "middle_third": (31.20, 33.20),
        "eye_aspect_ratio": (3.00, 3.50), "mouth_corner_position": (0.00, 4.00),
        "eye_separation_ratio": (45.70, 46.80), "eyebrow_tilt": (6.50, 11.00),
        "lower_third": (34.00, 37.10), "face_width_to_height": (1.96, 2.00),
        "interpupillary_mouth_width": (83.00, 87.00), "jaw_frontal_angle": (86.50, 92.50),
        "intercanthal_nasal_width": (1.07, 1.19), "top_third": (30.10, 32.10),
        "one_eye_apart": (0.95, 1.00), "midface_ratio": (0.97, 1.00),
        "ipsilateral_alar_angle": (86.50, 92.50), "mouth_width_to_nose_width": (1.39, 1.47),
        "total_facial_width_to_height": (1.33, 1.36), "chin_to_philtrum": (2.15, 2.45),
        "eyebrow_low_setedness": (0.00, 0.45), "brow_length_to_face_width": (0.69, 0.76),
        "nose_tip_position": (0.00, 3.00), "deviation_iaa_jfa": (0.00, 2.50),
        "lower_lip_to_upper_lip": (1.55, 1.85), "lower_third_proportion": (31.00, 33.50),
    },
    "mixed": {
        "lateral_canthal_tilt": (6.85, 8.55), "nose_bridge_to_width": (2.06, 2.14),
        "bitemporal_width": (86.50, 92.50), "cheekbone_height": (83.00, 100.00),
        "cupids_bow_depth": (2.30, 4.00), "bigonial_width": (87.50, 91.50),
        "jaw_slope": (140.00, 142.50), "middle_third": (31.65, 33.65),
        "eye_aspect_ratio": (2.95, 3.45), "mouth_corner_position": (0.00, 4.00),
        "eye_separation_ratio": (45.67, 46.77), "eyebrow_tilt": (6.50, 11.00),
        "lower_third": (33.70, 36.80), "face_width_to_height": (1.95, 1.99),
        "interpupillary_mouth_width": (81.00, 85.00), "jaw_frontal_angle": (86.50, 92.50),
        "intercanthal_nasal_width": (1.02, 1.14), "top_third": (30.25, 32.25),
        "one_eye_apart": (1.02, 1.08), "midface_ratio": (0.97, 1.00),
        "ipsilateral_alar_angle": (85.50, 91.50), "mouth_width_to_nose_width": (1.40, 1.48),
        "total_facial_width_to_height": (1.34, 1.37), "chin_to_philtrum": (2.15, 2.45),
        "eyebrow_low_setedness": (0.20, 0.65), "brow_length_to_face_width": (0.69, 0.76),
        "nose_tip_position": (0.00, 3.00), "deviation_iaa_jfa": (0.00, 2.50),
        "lower_lip_to_upper_lip": (1.60, 1.90), "lower_third_proportion": (31.00, 33.50),
    },
}


# East-Asian raw-distance exponential fits from the saved celebrity payloads.
# These are numerical estimates, not confirmed FaceIQ server constants.
# They currently apply to every demographic because no other fitted k tables
# were supplied; each demographic still uses its own ideal plateau.
FRONT_DECAY_RATES = {
    "nose_bridge_to_width": 3.63391254,
    "mouth_width_to_nose_width": 9.75152406,
    "total_facial_width_to_height": 10.91408892,
    "eyebrow_tilt": 0.16513146,
    "midface_ratio": 6.40137774,
    "lateral_canthal_tilt": 0.16815601,
    "eye_separation_ratio": 0.29852281,
    "bitemporal_width": 0.19378893,
    "one_eye_apart": 8.43829569,
    "jaw_slope": 0.10115101,
    "cupids_bow_depth": 0.42745836,
    "face_width_to_height": 5.15794334,
    "jaw_frontal_angle": 0.10786339,
    "top_third": 0.11338787,
    "lower_lip_to_upper_lip": 1.06118484,
    "brow_length_to_face_width": 10.72486905,
    "interpupillary_mouth_width": 5.27309475,
    "middle_third": 0.17187012,
    "lower_third": 0.29066025,
    "bigonial_width": 0.07526685,
    "lower_third_proportion": 0.95820067,
    "chin_to_philtrum": 2.24219367,
    "ipsilateral_alar_angle": 0.09043885,
    "deviation_iaa_jfa": 0.11876869,
    "eyebrow_low_setedness": 1.27236550,
    "eye_aspect_ratio": 1.59067658,
    "cheekbone_height": 0.06653490,
    "intercanthal_nasal_width": 3.53663547,
    "mouth_corner_position": 0.36537740,
    "nose_tip_position": 0.53211208,
}

for _key, _decay_rate in FRONT_DECAY_RATES.items():
    MALE_ASIAN_FRONT[_key]["decayRate"] = _decay_rate


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
        male[ethnicity] = _apply_live_plateaus(
            _build_front_norms(ethnicity),
            MALE_LIVE_FRONT_PLATEAUS[ethnicity],
        )

    female = {}
    for ethnicity, values in male.items():
        female[ethnicity] = _build_female_front(values)

    for gender_values in (male, female):
        for ethnicity_values in gender_values.values():
            for key, decay_rate in FRONT_DECAY_RATES.items():
                ethnicity_values[key]["decayRate"] = decay_rate

    return {"male": male, "female": female}


FRONT_IDEALS = build_front_ideals()


def get_front_ideals(gender="male", ethnicity="asian"):
    return FRONT_IDEALS.get(gender, {}).get(ethnicity, FRONT_IDEALS["male"]["asian"])
