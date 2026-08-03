import math
from collections import defaultdict

from front_ideals import get_front_ideals
from front_landmarks import normalize_front_landmarks


PENALTY_THRESHOLD = 3.5
PENALTY_MULTIPLIER = 0.25
PENALTY_CAP = 1
SCALE = 1000


def dist(a, b):
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)


def angle(p1, vertex, p2):
    v1 = {"x": p1["x"] - vertex["x"], "y": p1["y"] - vertex["y"]}
    v2 = {"x": p2["x"] - vertex["x"], "y": p2["y"] - vertex["y"]}
    dot = v1["x"] * v2["x"] + v1["y"] * v2["y"]
    cross = v1["x"] * v2["y"] - v1["y"] * v2["x"]
    return abs(math.atan2(cross, dot)) * (180 / math.pi)


def acute_angle_from_horizontal(p1, p2):
    dx = p2["x"] - p1["x"]
    dy = p2["y"] - p1["y"]
    deg = abs(math.atan2(dy, dx) * (180 / math.pi))
    return 180 - deg if deg > 90 else deg


def angle_between_lines(p1, p2, p3, p4, prefer_obtuse=False):
    v1 = {"x": p2["x"] - p1["x"], "y": p2["y"] - p1["y"]}
    v2 = {"x": p4["x"] - p3["x"], "y": p4["y"] - p3["y"]}
    dot = v1["x"] * v2["x"] + v1["y"] * v2["y"]
    cross = v1["x"] * v2["y"] - v1["y"] * v2["x"]
    acute = abs(math.atan2(cross, dot)) * (180 / math.pi)
    return acute if not prefer_obtuse else (acute if acute > 90 else 180 - acute)


def distance_to_line(point, line_start, line_end):
    a = line_end["y"] - line_start["y"]
    b = line_start["x"] - line_end["x"]
    c = line_end["x"] * line_start["y"] - line_start["x"] * line_end["y"]
    denom = math.sqrt(a * a + b * b)
    return 0 if denom == 0 else abs(a * point["x"] + b * point["y"] + c) / denom


def position_along_line(point, line_start, line_end):
    dx = line_end["x"] - line_start["x"]
    dy = line_end["y"] - line_start["y"]
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return None
    return ((point["x"] - line_start["x"]) * dx + (point["y"] - line_start["y"]) * dy) / length_squared


def midpoint(a, b):
    return {"id": "mid", "x": (a["x"] + b["x"]) / 2, "y": (a["y"] + b["y"]) / 2, "label": ""}


def hdist(a, b):
    return abs(a["x"] - b["x"])


def vdist(a, b):
    return abs(a["y"] - b["y"])


def calculate_plateau_exponential_score(value, ideal_min, ideal_max, decay_rate):
    if ideal_min <= value <= ideal_max:
        return 10.0
    if value < ideal_min:
        distance = ideal_min - value
    else:
        distance = value - ideal_max
    return 10 * math.exp(-decay_rate * distance)


def classify_deviation(value, ideal_min, ideal_max):
    if ideal_min <= value <= ideal_max:
        return "ideal"
    return "low" if value < ideal_min else "high"


def create_measurement(mid, name, value, unit, category, description, ideal):
    score = calculate_plateau_exponential_score(
        value,
        ideal["idealMin"],
        ideal["idealMax"],
        ideal["decayRate"],
    )
    deviation = classify_deviation(value, ideal["idealMin"], ideal["idealMax"])
    return {
        "id": mid,
        "name": name,
        "value": round(value, 2),
        "unit": unit,
        "score": round(score, 1),
        "idealRange": [ideal["idealMin"], ideal["idealMax"]],
        "range": [ideal["min"], ideal["max"]],
        "description": description,
        "category": category,
        "isIdeal": score >= 10.0,
        "deviation": deviation,
        "interpretation": _interpret(name, value, unit, ideal, score),
    }


def _interpret(name, value, unit, ideal, score):
    label = name.lower()
    ideal_text = f'{ideal["idealMin"]:.1f}-{ideal["idealMax"]:.1f} {unit}'
    value_text = f"{value:.1f} {unit}"
    if score >= 10.0:
        return f"Your {label} of {value_text} falls within the ideal plateau ({ideal_text}). Perfect score."
    if score >= 7.0:
        return f"Your {label} of {value_text} is slightly outside the ideal range ({ideal_text}). Minor deviation, still strong."
    if score >= 4.5:
        return f"Your {label} of {value_text} shows moderate deviation from the ideal range ({ideal_text})."
    if score >= 2.5:
        return f"Your {label} of {value_text} is significantly off from the ideal range ({ideal_text})."
    return f"Your {label} of {value_text} is far from the ideal range ({ideal_text})."


def calculate_front_measurements(front_landmarks, gender="male", ethnicity="asian", front_aspect=1):
    lm = normalize_front_landmarks(front_landmarks) if isinstance(front_landmarks, list) else front_landmarks
    lm = {key: {**value, "x": value["x"] * SCALE * front_aspect, "y": value["y"] * SCALE} for key, value in lm.items()}
    ideals = get_front_ideals(gender, ethnicity)
    results = []

    def l(*ids):
        for lid in ids:
            if lid in lm:
                return lm[lid]
        return None

    def add(mid, name, value, unit, category, description):
        ideal = ideals.get(mid)
        if ideal is not None and value is not None and math.isfinite(value):
            results.append(create_measurement(mid, name, value, unit, category, description, ideal))

    hairline = l("hairline")
    left_pupil = l("left_pupil")
    right_pupil = l("right_pupil")
    left_medial = l("left_medial_canthus")
    left_lateral = l("left_lateral_canthus")
    left_upper_eyelid = l("left_upper_eyelid")
    left_lower_eyelid = l("left_lower_eyelid")
    left_eyelid_hood_end = l("left_eyelid_hood_end")
    right_medial = l("right_medial_canthus")
    right_lateral = l("right_lateral_canthus")
    right_upper_eyelid = l("right_upper_eyelid")
    right_lower_eyelid = l("right_lower_eyelid")
    right_eyelid_hood_end = l("right_eyelid_hood_end")
    left_brow_head = l("left_brow_head")
    left_brow_inner = l("left_brow_inner_corner")
    left_brow_arch = l("left_brow_arch")
    left_brow_peak = l("left_brow_peak")
    left_brow_tail = l("left_brow_tail")
    right_brow_head = l("right_brow_head")
    right_brow_inner = l("right_brow_inner_corner")
    right_brow_arch = l("right_brow_arch")
    right_brow_peak = l("right_brow_peak")
    right_brow_tail = l("right_brow_tail")
    left_nose_side = l("left_nose_side")
    right_nose_side = l("right_nose_side")
    left_nose_bridge = l("left_nose_bridge")
    right_nose_bridge = l("right_nose_bridge")
    nasal_base = l("nasal_base")
    nose_bottom = l("nose_bottom")
    left_mouth_corner = l("left_mouth_corner")
    right_mouth_corner = l("right_mouth_corner")
    cupids_bow = l("cupids_bow")
    inner_cupids_bow = l("inner_cupids_bow")
    mouth_middle = l("mouth_middle")
    lower_lip_center = l("lower_lip_center")
    left_upper_jaw = l("left_upper_jaw_angle")
    right_upper_jaw = l("right_upper_jaw_angle")
    left_lower_jaw = l("left_lower_jaw_angle")
    right_lower_jaw = l("right_lower_jaw_angle")
    left_chin = l("left_chin")
    right_chin = l("right_chin")
    chin_bottom = l("chin_bottom")
    left_cheekbone = l("left_cheekbone")
    right_cheekbone = l("right_cheekbone")
    left_temple = l("left_temple")
    right_temple = l("right_temple")

    vals = []
    if left_medial and left_lateral:
        vals.append(acute_angle_from_horizontal(left_medial, left_lateral))
    if right_medial and right_lateral:
        vals.append(acute_angle_from_horizontal(right_medial, right_lateral))
    if vals:
        add("lateral_canthal_tilt", "Lateral Canthal Tilt", sum(vals) / len(vals), "degrees", "Eyes", "Average acute angle of medial-to-lateral canthus lines relative to horizontal.")

    if left_nose_side and right_nose_side and left_nose_bridge and right_nose_bridge:
        bridge_width = dist(left_nose_bridge, right_nose_bridge)
        if dist(left_nose_side, right_nose_side) > 0:
            add("nose_bridge_to_width", "Nose Bridge to Nose Width Ratio", dist(left_nose_side, right_nose_side) / bridge_width, "ratio", "Nose", "Ratio of nose side width to nose bridge width.")

    if left_temple and right_temple and left_cheekbone and right_cheekbone:
        face_width = dist(left_cheekbone, right_cheekbone)
        if face_width > 0:
            add("bitemporal_width", "Bitemporal Width", dist(left_temple, right_temple) / face_width * 100, "percentage", "Head", "Ratio of bitemporal width to bizygomatic width.")

    if cupids_bow and left_cheekbone and right_cheekbone and left_pupil and right_pupil:
        a = distance_to_line(cupids_bow, left_cheekbone, right_cheekbone)
        b = distance_to_line(cupids_bow, left_pupil, right_pupil)
        if b > 0:
            add("cheekbone_height", "Cheekbone Height", a / b * 100, "percentage", "Cheeks", "Cupid's Bow height to cheekbone line versus pupil line.")

    if cupids_bow and inner_cupids_bow:
        add("cupids_bow_depth", "Cupid's Bow Depth", vdist(cupids_bow, inner_cupids_bow), "mm", "Mouth", "Vertical distance between Cupid's bow and inner Cupid's bow.")

    if left_upper_jaw and right_upper_jaw and left_cheekbone and right_cheekbone:
        face_width = dist(left_cheekbone, right_cheekbone)
        if face_width > 0:
            add("bigonial_width", "Bigonial Width", dist(left_upper_jaw, right_upper_jaw) / face_width * 100, "percentage", "Jaw", "Upper jaw angle width to bizygomatic width.")

    vals = []
    if left_cheekbone and left_upper_jaw and left_lower_jaw and left_chin:
        vals.append(angle_between_lines(left_cheekbone, left_upper_jaw, left_lower_jaw, left_chin, True))
    if right_cheekbone and right_upper_jaw and right_lower_jaw and right_chin:
        vals.append(angle_between_lines(right_cheekbone, right_upper_jaw, right_lower_jaw, right_chin, True))
    if vals:
        add("jaw_slope", "Jaw Slope", sum(vals) / len(vals), "degrees", "Jaw", "Average left and right jaw slope angles.")

    if hairline and chin_bottom:
        brow_position = None
        nasal_position = position_along_line(nasal_base, hairline, chin_bottom) if nasal_base else None
        if right_brow_head and right_brow_inner and left_brow_head and left_brow_inner:
            brow_mid = midpoint(midpoint(right_brow_head, right_brow_inner), midpoint(left_brow_head, left_brow_inner))
            brow_position = position_along_line(brow_mid, hairline, chin_bottom)
        if brow_position is not None:
            add("top_third", "Top Third", brow_position * 100, "percentage", "Proportions", "Hairline-to-brow share measured along the hairline-to-chin facial axis.")
        if brow_position is not None and nasal_position is not None:
            add("middle_third", "Middle Third", (nasal_position - brow_position) * 100, "percentage", "Proportions", "Brow-to-nasal-base share measured along the hairline-to-chin facial axis.")
        if nasal_position is not None:
            add("lower_third", "Lower Third", (1 - nasal_position) * 100, "percentage", "Proportions", "Nasal-base-to-chin share measured along the hairline-to-chin facial axis.")

    vals = []
    if left_medial and left_lateral and left_upper_eyelid and left_lower_eyelid and vdist(left_upper_eyelid, left_lower_eyelid) > 0:
        vals.append(hdist(left_medial, left_lateral) / vdist(left_upper_eyelid, left_lower_eyelid))
    if right_medial and right_lateral and right_upper_eyelid and right_lower_eyelid and vdist(right_upper_eyelid, right_lower_eyelid) > 0:
        vals.append(hdist(right_medial, right_lateral) / vdist(right_upper_eyelid, right_lower_eyelid))
    if vals:
        add("eye_aspect_ratio", "Eye Aspect Ratio", sum(vals) / len(vals), "ratio", "Eyes", "Average eye width to eye height ratio.")

    if mouth_middle:
        vals = []
        if left_mouth_corner:
            vals.append(mouth_middle["y"] - left_mouth_corner["y"])
        if right_mouth_corner:
            vals.append(mouth_middle["y"] - right_mouth_corner["y"])
        if vals:
            add("mouth_corner_position", "Mouth Corner Position", sum(vals) / len(vals), "mm", "Mouth", "Average signed vertical offset of mouth corners from mouth middle.")

    if left_pupil and right_pupil and left_cheekbone and right_cheekbone:
        face_width = dist(left_cheekbone, right_cheekbone)
        if face_width > 0:
            add("eye_separation_ratio", "Eye Separation Ratio", dist(left_pupil, right_pupil) / face_width * 100, "percentage", "Eyes", "Interpupillary distance to bizygomatic width.")

    vals = []
    for bh, bi, ba, bp in ((left_brow_head, left_brow_inner, left_brow_arch, left_brow_peak), (right_brow_head, right_brow_inner, right_brow_arch, right_brow_peak)):
        if bh and bi and ba and bp:
            start = midpoint(bh, bi)
            end = midpoint(ba, bp)
            signed_deg = math.atan2(-(end["y"] - start["y"]), end["x"] - start["x"]) * (180 / math.pi)
            acute_deg = (180 - abs(signed_deg)) * (1 if signed_deg >= 0 else -1) if abs(signed_deg) > 90 else signed_deg
            vals.append(acute_deg)
    if vals:
        add("eyebrow_tilt", "Eyebrow Tilt", sum(vals) / len(vals), "degrees", "Brows", "Average signed eyebrow tilt from horizontal.")

    if left_cheekbone and right_cheekbone and cupids_bow and right_brow_head and right_brow_inner and left_brow_head and left_brow_inner:
        left_brow_mid = midpoint(left_brow_head, left_brow_inner)
        right_brow_mid = midpoint(right_brow_head, right_brow_inner)
        face_height = distance_to_line(cupids_bow, left_brow_mid, right_brow_mid)
        if face_height > 0:
            add("face_width_to_height", "Face Width to Height Ratio", dist(left_cheekbone, right_cheekbone) / face_height, "ratio", "Proportions", "Bizygomatic width divided by the perpendicular distance from Cupid's bow to the brow-midpoint line.")

    if left_mouth_corner and right_mouth_corner and left_pupil and right_pupil:
        pupil_dist = dist(left_pupil, right_pupil)
        if pupil_dist > 0:
            add("interpupillary_mouth_width", "Interpupillary-Mouth Width Ratio", dist(left_mouth_corner, right_mouth_corner) / pupil_dist * 100, "percentage", "Proportions", "Mouth width to interpupillary distance.")

    if left_lower_jaw and left_chin and right_lower_jaw and right_chin:
        jfa = angle_between_lines(left_lower_jaw, left_chin, right_lower_jaw, right_chin)
        add("jaw_frontal_angle", "Jaw Frontal Angle", 360 - jfa if jfa > 180 else jfa, "degrees", "Jaw", "Angle between left and right lower jaw-to-chin lines.")

    if left_nose_side and right_nose_side and left_medial and right_medial:
        medial_dist = dist(left_medial, right_medial)
        if medial_dist > 0:
            add("intercanthal_nasal_width", "Intercanthal-Nasal Width Ratio", dist(left_nose_side, right_nose_side) / medial_dist, "ratio", "Proportions", "Nasal width to intercanthal distance.")

    if left_medial and right_medial and left_lateral and right_lateral:
        avg_eye_width = (dist(left_medial, left_lateral) + dist(right_medial, right_lateral)) / 2
        if avg_eye_width > 0:
            add("one_eye_apart", "One Eye Apart Test", dist(left_medial, right_medial) / avg_eye_width, "ratio", "Proportions", "Intercanthal distance to average eye width.")

    if left_pupil and right_pupil and inner_cupids_bow:
        line_dist = distance_to_line(inner_cupids_bow, left_pupil, right_pupil)
        if line_dist > 0:
            add("midface_ratio", "Midface Ratio", dist(left_pupil, right_pupil) / line_dist, "ratio", "Proportions", "Interpupillary distance to inner Cupid's bow height.")

    iaa_val = None
    if nasal_base and left_eyelid_hood_end and right_eyelid_hood_end:
        iaa_val = angle(left_eyelid_hood_end, nasal_base, right_eyelid_hood_end)
        add("ipsilateral_alar_angle", "Ipsilateral Alar Angle", iaa_val, "degrees", "Nose", "Angle at nasal base between left and right eyelid hood ends.")

    if left_mouth_corner and right_mouth_corner and left_nose_side and right_nose_side:
        nose_width = dist(left_nose_side, right_nose_side)
        if nose_width > 0:
            add("mouth_width_to_nose_width", "Mouth Width to Nose Width Ratio", dist(left_mouth_corner, right_mouth_corner) / nose_width, "ratio", "Proportions", "Mouth width to nose width.")

    if hairline and chin_bottom and left_cheekbone and right_cheekbone:
        total_height = vdist(hairline, chin_bottom)
        if total_height > 0:
            add("total_facial_width_to_height", "Total Facial Width to Height Ratio", dist(left_cheekbone, right_cheekbone) / total_height, "ratio", "Proportions", "Bizygomatic width to total facial height.")

    if chin_bottom and lower_lip_center and cupids_bow and nasal_base:
        philtrum = vdist(cupids_bow, nasal_base)
        if philtrum > 0:
            add("chin_to_philtrum", "Chin to Philtrum Ratio", vdist(chin_bottom, lower_lip_center) / philtrum, "ratio", "Proportions", "Chin height to philtrum length.")

    if left_pupil and right_pupil and left_brow_inner and right_brow_inner:
        eye_heights = []
        if left_lower_eyelid and left_upper_eyelid:
            eye_heights.append(vdist(left_lower_eyelid, left_upper_eyelid))
        if right_lower_eyelid and right_upper_eyelid:
            eye_heights.append(vdist(right_lower_eyelid, right_upper_eyelid))
        if eye_heights and sum(eye_heights) > 0:
            add("eyebrow_low_setedness", "Eyebrow Low Setedness", dist(midpoint(left_pupil, right_pupil), midpoint(left_brow_inner, right_brow_inner)) / (sum(eye_heights) / len(eye_heights)), "ratio", "Brows", "Brow-to-pupil distance to average eye height.")

    if left_cheekbone and right_cheekbone:
        brow_len = 0
        brow_count = 0
        if left_brow_inner and left_brow_tail:
            brow_len += dist(left_brow_inner, left_brow_tail)
            brow_count += 1
        if right_brow_inner and right_brow_tail:
            brow_len += dist(right_brow_inner, right_brow_tail)
            brow_count += 1
        face_width = dist(left_cheekbone, right_cheekbone)
        if brow_count > 0 and face_width > 0:
            add("brow_length_to_face_width", "Brow Length to Face Width Ratio", brow_len / face_width, "ratio", "Brows", "Combined brow span divided by bizygomatic width.")

    if nasal_base and nose_bottom:
        add("nose_tip_position", "Nose Tip Position", dist(nasal_base, nose_bottom), "mm", "Nose", "Distance from nasal base to nose bottom.")

    jfa_val = None
    if left_lower_jaw and left_chin and right_lower_jaw and right_chin:
        jfa_val = angle_between_lines(left_lower_jaw, left_chin, right_lower_jaw, right_chin)
    if iaa_val is not None and jfa_val is not None:
        add("deviation_iaa_jfa", "Deviation of IAA & JFA", jfa_val - iaa_val, "degrees", "Proportions", "Difference between jaw frontal angle and ipsilateral alar angle.")

    if lower_lip_center and mouth_middle and cupids_bow:
        upper_lip = vdist(mouth_middle, cupids_bow)
        if upper_lip > 0:
            add("lower_lip_to_upper_lip", "Lower Lip to Upper Lip Ratio", vdist(lower_lip_center, mouth_middle) / upper_lip, "ratio", "Mouth", "Lower lip height to upper lip height.")

    if nasal_base and mouth_middle and chin_bottom:
        lower_face = vdist(nasal_base, chin_bottom)
        if lower_face > 0:
            add("lower_third_proportion", "Lower Third Proportion", vdist(nasal_base, mouth_middle) / lower_face * 100, "percentage", "Proportions", "Nasal base to mouth middle as a share of lower face height.")

    return results


def weighted_group_score(measurements, key_ids, std_ids):
    by_id = {m["id"]: m for m in measurements}
    weighted_sum = 0
    total_weight = 0
    for mid in key_ids:
        if mid in by_id:
            weighted_sum += by_id[mid]["score"] * 2
            total_weight += 2
    for mid in std_ids:
        if mid in by_id:
            weighted_sum += by_id[mid]["score"]
            total_weight += 1
    return 0 if total_weight == 0 else round(weighted_sum / total_weight, 2)


def calculate_penalty(measurements):
    penalty = 0
    for item in measurements:
        if item["score"] < PENALTY_THRESHOLD:
            penalty += (PENALTY_THRESHOLD - item["score"]) * PENALTY_MULTIPLIER
    return min(PENALTY_CAP, round(penalty, 2))


def clamp_score(score):
    return round(max(0, min(10, score)), 2)


def calculate_front_analysis(front_landmarks, gender="male", ethnicity="asian", front_aspect=1):
    measurements = calculate_front_measurements(front_landmarks, gender, ethnicity, front_aspect)
    f1_key = ["midface_ratio", "face_width_to_height", "lower_third"]
    f1_std = ["bitemporal_width", "bigonial_width", "jaw_slope", "middle_third", "top_third", "total_facial_width_to_height", "lower_third_proportion"]
    f2_key = ["lateral_canthal_tilt", "eye_separation_ratio"]
    f2_std = ["cheekbone_height", "eye_aspect_ratio", "eyebrow_tilt", "one_eye_apart", "eyebrow_low_setedness", "brow_length_to_face_width"]
    f3_key = ["jaw_frontal_angle", "chin_to_philtrum"]
    f3_std = ["nose_bridge_to_width", "cupids_bow_depth", "mouth_corner_position", "interpupillary_mouth_width", "intercanthal_nasal_width", "ipsilateral_alar_angle", "mouth_width_to_nose_width", "nose_tip_position", "deviation_iaa_jfa", "lower_lip_to_upper_lip"]

    g_f1 = weighted_group_score(measurements, f1_key, f1_std)
    g_f2 = weighted_group_score(measurements, f2_key, f2_std)
    g_f3 = weighted_group_score(measurements, f3_key, f3_std)
    penalty = calculate_penalty(measurements)
    side_penalty = 0
    raw_front_score = g_f1 * 0.40 + g_f2 * 0.30 + g_f3 * 0.30 - penalty
    raw_side_score = 0 - side_penalty
    raw_overall_score = raw_front_score

    category_totals = defaultdict(float)
    category_counts = defaultdict(int)
    for item in measurements:
        category_totals[item["category"]] += item["score"]
        category_counts[item["category"]] += 1
    category_scores = {cat: round(category_totals[cat] / category_counts[cat], 1) for cat in category_totals}

    sorted_items = sorted(measurements, key=lambda m: m["score"], reverse=True)
    return {
        "gender": gender,
        "ethnicity": ethnicity,
        "frontMeasurements": measurements,
        "sideMeasurements": [],
        "overallScore": clamp_score(raw_overall_score),
        "frontScore": clamp_score(raw_front_score),
        "sideScore": clamp_score(raw_side_score),
        "harmonyScore": clamp_score(raw_front_score),
        "categoryScores": category_scores,
        "topStrengths": [m["name"] for m in sorted_items[:3]],
        "topWeaknesses": [m["name"] for m in sorted_items[-3:]][::-1],
        "groups": {"G_F1": g_f1, "G_F2": g_f2, "G_F3": g_f3, "G_S1": 0, "G_S2": 0, "G_S3": 0, "P_front": penalty, "P_side": side_penalty},
        "missingCount": max(0, 30 - len(measurements)),
    }
