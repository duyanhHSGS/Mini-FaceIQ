import math
from collections import defaultdict

from front_calculator import (
    SCALE,
    angle,
    calculate_penalty,
    clamp_score,
    create_measurement,
    dist,
    distance_to_line,
    weighted_group_score,
)
from side_ideals import get_side_ideals
from side_landmarks import normalize_side_landmarks


def signed_distance_to_line(point, line_start, line_end):
    dx = line_end["x"] - line_start["x"]
    dy = line_end["y"] - line_start["y"]
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return 0
    return ((point["x"] - line_start["x"]) * dy - (point["y"] - line_start["y"]) * dx) / length


def _line_intersection(a1, a2, b1, b2):
    dx1 = a2["x"] - a1["x"]
    dy1 = a2["y"] - a1["y"]
    dx2 = b2["x"] - b1["x"]
    dy2 = b2["y"] - b1["y"]
    det = dx1 * dy2 - dy1 * dx2
    if abs(det) <= 0.001:
        return None
    t = ((b1["x"] - a1["x"]) * dy2 - (b1["y"] - a1["y"]) * dx2) / det
    return {"id": "intersection", "x": a1["x"] + dx1 * t, "y": a1["y"] + dy1 * t, "label": ""}


def _angle_at_intersection(a, intersection, b):
    v1x = a["x"] - intersection["x"]
    v1y = a["y"] - intersection["y"]
    v2x = b["x"] - intersection["x"]
    v2y = b["y"] - intersection["y"]
    dot = v1x * v2x + v1y * v2y
    cross = v1x * v2y - v1y * v2x
    return abs(math.atan2(cross, dot)) * (180 / math.pi)


def calculate_side_measurements(side_landmarks, gender="male", ethnicity="asian", side_aspect=1):
    lm = normalize_side_landmarks(side_landmarks) if isinstance(side_landmarks, list) else side_landmarks
    lm = {key: {**value, "x": value["x"] * SCALE * side_aspect, "y": value["y"] * SCALE} for key, value in lm.items()}
    ideals = get_side_ideals(gender, ethnicity)
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

    hairline = l("hairline_profile")
    forehead = l("forehead")
    glabella = l("glabella")
    nasion = l("nasal_bridge_root")
    rhinion = l("rhinion")
    supratip = l("supratip")
    nose_tip = l("nose_tip")
    infratip = l("infratip")
    columella = l("columella")
    subnasale = l("subnasale")
    subalare = l("subalare")
    upper_lip = l("upper_lip")
    mouth_corner = l("mouth_corner")
    lower_lip = l("lower_lip")
    labiomental_fold = l("labiomental_fold")
    chin_point = l("chin_point")
    chin_bottom = l("chin_bottom")
    upper_jaw_angle = l("upper_jaw_angle")
    lower_jaw_angle = l("lower_jaw_angle")
    porion = l("porion")
    tragus = l("tragus")
    intertragic_notch = l("intertragic_notch")
    orbitale = l("orbitale")
    eyelid_end = l("eyelid_end")
    lower_eyelid = l("lower_eyelid")
    cheekbone = l("cheekbone")
    cervical_point = l("cervical_point")
    neck_point = l("neck_point")

    if infratip and nose_tip and supratip:
        add("nasal_tip_angle", "Nasal Tip Angle", angle(infratip, nose_tip, supratip), "degrees", "Nose", "Angle at nose tip between infratip and supratip.")

    if nose_tip and subalare and nasion:
        nasal_height = distance_to_line(nasion, nose_tip, subalare)
        if nasal_height > 0:
            add("nasal_width_to_height", "Nasal Width to Height Ratio", dist(nose_tip, subalare) / nasal_height, "ratio", "Nose", "Ratio of nasal width to nasal height.")

    if upper_lip and columella and chin_point:
        add("upper_lip_s_line", "Upper Lip S-Line Position", signed_distance_to_line(upper_lip, columella, chin_point), "mm", "Lips", "Upper lip position relative to S-line.")

    if upper_lip and subnasale and chin_point:
        add("upper_lip_burstone", "Upper Lip Burstone Line", signed_distance_to_line(upper_lip, subnasale, chin_point), "mm", "Lips", "Upper lip position relative to Burstone line.")

    if subalare and nose_tip and nasion:
        height = dist(nose_tip, nasion)
        if height > 0:
            add("nasal_projection", "Nasal Projection", dist(subalare, nose_tip) / height, "ratio", "Nose", "Nasal projection relative to nasal height.")

    if glabella and nasion and rhinion:
        add("nasofrontal_angle", "Nasofrontal Angle", angle(glabella, nasion, rhinion), "degrees", "Nose", "Angle at nasion between glabella and rhinion.")

    if chin_point and porion and orbitale and nasion:
        perp_end = {"id": "perp_end", "x": nasion["x"] - (orbitale["y"] - porion["y"]), "y": nasion["y"] + (orbitale["x"] - porion["x"]), "label": ""}
        add("recession_frankfort", "Recession (Frankfort Plane)", signed_distance_to_line(chin_point, nasion, perp_end), "mm", "Profile", "Chin distance from nasion perpendicular to Frankfort plane.")

    if upper_lip and chin_point and lower_lip:
        add("holdaway_h_line", "Holdaway H Line", signed_distance_to_line(lower_lip, upper_lip, chin_point), "mm", "Profile", "Lower lip position relative to Holdaway H-line.")

    if lower_lip and labiomental_fold and chin_point:
        add("mentolabial_angle", "Mentolabial Angle", angle(lower_lip, labiomental_fold, chin_point), "degrees", "Chin", "Angle between lower lip and chin at the labiomental fold.")

    if glabella and forehead and hairline:
        add("upper_forehead_slope", "Upper Forehead Slope", angle(forehead, glabella, hairline), "degrees", "Forehead", "Angle at glabella between forehead and hairline.")

    if nasion and subnasale and chin_point:
        add("facial_convexity_nasion", "Facial Convexity (Nasion)", angle(nasion, subnasale, chin_point), "degrees", "Profile", "Facial convexity at subnasale using nasion and chin.")

    if tragus and subalare and orbitale:
        add("anterior_facial_depth", "Anterior Facial Depth", angle(tragus, subalare, orbitale), "degrees", "Proportions", "Angle at subalare between tragus and orbitale.")

    if upper_lip and nose_tip and chin_point:
        add("upper_lip_e_line", "Upper Lip E-Line Position", signed_distance_to_line(upper_lip, nose_tip, chin_point), "mm", "Lips", "Upper lip position relative to E-line.")

    if chin_bottom and cervical_point and neck_point:
        add("submental_cervical_angle", "Submental Cervical Angle", angle(chin_bottom, cervical_point, neck_point), "degrees", "Neck", "Angle at cervical point between chin bottom and neck point.")

    if subnasale and tragus and nasion and labiomental_fold:
        height = dist(nasion, labiomental_fold)
        if height > 0:
            add("facial_depth_to_height", "Facial Depth to Height Ratio", dist(subnasale, tragus) / height, "ratio", "Proportions", "Facial depth divided by facial height.")

    if glabella and hairline:
        angle_from_vert = abs(math.atan2(hairline["x"] - glabella["x"], -(hairline["y"] - glabella["y"])) * (180 / math.pi))
        add("browridge_inclination", "Browridge Inclination Angle", angle_from_vert, "degrees", "Brows", "Angle between vertical at glabella and line to hairline.")

    if chin_point and nose_tip and glabella:
        add("total_facial_convexity", "Total Facial Convexity", angle(chin_point, nose_tip, glabella), "degrees", "Profile", "Convexity angle at nose tip.")

    if glabella and subnasale and chin_point:
        add("facial_convexity_glabella", "Facial Convexity (Glabella)", angle(glabella, subnasale, chin_point), "degrees", "Profile", "Facial convexity at subnasale using glabella and chin.")

    if orbitale and lower_eyelid:
        add("orbital_vector", "Orbital Vector", orbitale["x"] - lower_eyelid["x"], "mm", "Eyes", "Horizontal distance from orbitale to lower eyelid.")

    if eyelid_end and subalare:
        ray = math.atan2(eyelid_end["y"] - subalare["y"], eyelid_end["x"] - subalare["x"])
        diff = ray - math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        while diff > math.pi:
            diff -= 2 * math.pi
        add("interior_midface_projection", "Interior Midface Projection Angle", abs(diff * 180 / math.pi), "degrees", "Midface", "Angle at subalare toward eyelid end.")

    if cheekbone and rhinion and chin_point and infratip:
        ix = _line_intersection(cheekbone, rhinion, chin_point, infratip)
        if ix:
            add("z_angle", "Z Angle", _angle_at_intersection(cheekbone, ix, chin_point), "degrees", "Profile", "Angle at cheekbone-rhinion and chin-infratip intersection.")

    if rhinion and cheekbone and subnasale and infratip:
        ix = _line_intersection(rhinion, cheekbone, subnasale, infratip)
        if ix:
            add("nose_tip_rotation", "Nose Tip Rotation Angle", _angle_at_intersection(rhinion, ix, subnasale), "degrees", "Nose", "Angle at rhinion-cheekbone and subnasale-infratip intersection.")

    if columella and subnasale and upper_lip:
        add("nasolabial_angle", "Nasolabial Angle", angle(columella, subnasale, upper_lip), "degrees", "Nose", "Angle at subnasale between columella and upper lip.")

    if nasion and chin_point and nose_tip:
        add("nasofacial_angle", "Nasofacial Angle", angle(chin_point, nasion, nose_tip), "degrees", "Nose", "Angle at nasion between chin point and nose tip.")

    if nasion and nose_tip and chin_point:
        add("nasomental_angle", "Nasomental Angle", angle(nasion, nose_tip, chin_point), "degrees", "Profile", "Angle at nose tip between nasion and chin point.")

    if columella and subnasale and cheekbone and rhinion:
        ix = _line_intersection(columella, subnasale, cheekbone, rhinion)
        if ix:
            add("frankfort_tip_angle", "Frankfort-Tip Angle", _angle_at_intersection(columella, ix, rhinion), "degrees", "Nose", "Angle at extended columella-subnasale and cheekbone-rhinion intersection.")

    if lower_lip and columella and chin_point:
        add("lower_lip_s_line", "Lower Lip S-Line Position", signed_distance_to_line(lower_lip, columella, chin_point), "mm", "Lips", "Lower lip position relative to S-line.")

    if lower_lip and nose_tip and chin_point:
        add("lower_lip_e_line", "Lower Lip E-Line Position", signed_distance_to_line(lower_lip, nose_tip, chin_point), "mm", "Lips", "Lower lip position relative to E-line.")

    if lower_lip and subnasale and chin_point:
        add("lower_lip_burstone", "Lower Lip Burstone Line", signed_distance_to_line(lower_lip, subnasale, chin_point), "mm", "Lips", "Lower lip position relative to Burstone line.")

    if intertragic_notch and upper_jaw_angle and chin_bottom and lower_jaw_angle:
        ix = _line_intersection(intertragic_notch, upper_jaw_angle, chin_bottom, lower_jaw_angle)
        if ix:
            add("gonial_angle", "Gonial Angle", _angle_at_intersection(intertragic_notch, ix, lower_jaw_angle), "degrees", "Jaw", "Angle between upper jaw and mandibular plane.")

    if lower_jaw_angle and chin_bottom:
        dx = chin_bottom["x"] - lower_jaw_angle["x"]
        dy = chin_bottom["y"] - lower_jaw_angle["y"]
        add("mandibular_plane_angle", "Mandibular Plane Angle", abs(math.atan2(dy, dx) * (180 / math.pi)), "degrees", "Jaw", "Mandibular plane angle relative to horizontal.")

    if chin_bottom and lower_jaw_angle and tragus and upper_jaw_angle and chin_point:
        ix = _line_intersection(chin_bottom, lower_jaw_angle, tragus, upper_jaw_angle)
        dx = lower_jaw_angle["x"] - chin_bottom["x"]
        dy = lower_jaw_angle["y"] - chin_bottom["y"]
        if ix and abs(dx) > 0.001:
            s2 = (chin_point["x"] - chin_bottom["x"]) / dx
            bx = chin_bottom["x"] + dx * s2
            by = chin_bottom["y"] + dy * s2
            ramus = math.sqrt((ix["x"] - tragus["x"]) ** 2 + (ix["y"] - tragus["y"]) ** 2)
            mandible = math.sqrt((bx - ix["x"]) ** 2 + (by - ix["y"]) ** 2)
            if mandible > 0:
                add("ramus_to_mandible", "Ramus to Mandible Ratio", ramus / mandible, "ratio", "Jaw", "Ramus height divided by mandibular body length.")

    if chin_bottom and lower_jaw_angle and tragus and upper_jaw_angle and mouth_corner:
        ix = _line_intersection(chin_bottom, lower_jaw_angle, tragus, upper_jaw_angle)
        if ix:
            add("gonion_to_mouth", "Gonion to Mouth Line", abs(ix["y"] - mouth_corner["y"]), "mm", "Jaw", "Vertical distance from gonion intersection to mouth corner.")

    return results


def calculate_side_analysis(side_landmarks, gender="male", ethnicity="asian", side_aspect=1):
    measurements = calculate_side_measurements(side_landmarks, gender, ethnicity, side_aspect)

    s1_key = ["recession_frankfort", "total_facial_convexity"]
    s1_std = ["facial_convexity_nasion", "anterior_facial_depth", "facial_depth_to_height", "facial_convexity_glabella", "interior_midface_projection", "z_angle"]
    s2_key = ["nasal_projection", "nasolabial_angle"]
    s2_std = ["nasal_tip_angle", "nasal_width_to_height", "nasofrontal_angle", "upper_forehead_slope", "browridge_inclination", "nose_tip_rotation", "nasofacial_angle", "nasomental_angle", "frankfort_tip_angle"]
    s3_key = ["gonial_angle", "lower_lip_e_line"]
    s3_std = ["upper_lip_s_line", "upper_lip_burstone", "holdaway_h_line", "mentolabial_angle", "upper_lip_e_line", "submental_cervical_angle", "orbital_vector", "lower_lip_s_line", "lower_lip_burstone", "mandibular_plane_angle", "ramus_to_mandible", "gonion_to_mouth"]

    g_s1 = weighted_group_score(measurements, s1_key, s1_std)
    g_s2 = weighted_group_score(measurements, s2_key, s2_std)
    g_s3 = weighted_group_score(measurements, s3_key, s3_std)
    penalty = calculate_penalty(measurements)
    raw_side_score = g_s1 * 0.35 + g_s2 * 0.35 + g_s3 * 0.30 - penalty

    category_totals = defaultdict(float)
    category_counts = defaultdict(int)
    for item in measurements:
        category_totals[item["category"]] += item["score"]
        category_counts[item["category"]] += 1
    category_scores = {cat: round(category_totals[cat] / category_counts[cat], 1) for cat in category_totals}

    sorted_items = sorted(measurements, key=lambda m: m["score"], reverse=True)
    side_score = clamp_score(raw_side_score)
    return {
        "gender": gender,
        "ethnicity": ethnicity,
        "frontMeasurements": [],
        "sideMeasurements": measurements,
        "overallScore": side_score,
        "frontScore": 0,
        "sideScore": side_score,
        "harmonyScore": side_score,
        "categoryScores": category_scores,
        "topStrengths": [m["name"] for m in sorted_items[:3]],
        "topWeaknesses": [m["name"] for m in sorted_items[-3:]][::-1],
        "groups": {"G_F1": 0, "G_F2": 0, "G_F3": 0, "G_S1": g_s1, "G_S2": g_s2, "G_S3": g_s3, "P_front": 0, "P_side": penalty},
        "missingCount": max(0, 33 - len(measurements)),
    }
