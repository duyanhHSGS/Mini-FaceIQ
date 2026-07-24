import os
import sys
import json
import argparse
import numpy as np
import cv2

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_3DDFA_DIR = os.path.join(_SCRIPT_DIR, "third_party", "3DDFA_V2")
if _3DDFA_DIR not in sys.path:
    sys.path.insert(0, _3DDFA_DIR)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '4'

_model_cache = {
    "face_boxes": None,
    "tddfa": None,
}


def _require_3ddfa_dir():
    if not os.path.isdir(_3DDFA_DIR):
        raise FileNotFoundError(
            "3DDFA_V2 not found. Expected it at "
            f"{_3DDFA_DIR}"
        )


def _load_models():
    if _model_cache["face_boxes"] is not None:
        return _model_cache["face_boxes"], _model_cache["tddfa"]

    _require_3ddfa_dir()
    try:
        import yaml
        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX
    except ImportError as exc:
        raise ImportError(
            "Missing 3DDFA dependency. This detector needs pyyaml, "
            "onnxruntime, torch, and 3DDFA_V2's local modules."
        ) from exc

    cfg_path = os.path.join(_3DDFA_DIR, "configs", "mb1_120x120.yml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"3DDFA config not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    for key in ["checkpoint_fp", "bfm_fp", "param_mean_std_fp"]:
        if key in cfg:
            cfg[key] = os.path.join(_3DDFA_DIR, cfg[key])
            if not os.path.exists(cfg[key]):
                raise FileNotFoundError(f"3DDFA file not found for {key}: {cfg[key]}")

    _model_cache["face_boxes"] = FaceBoxes_ONNX()
    _model_cache["tddfa"] = TDDFA_ONNX(**cfg)
    return _model_cache["face_boxes"], _model_cache["tddfa"]


SIDE_SPARSE_MAP = {
    "nose_tip":           30,
    "corneal_apex":       37,
    "lower_eyelid":       41,
    "nasal_bridge_root":  27,
    "rhinion":            28,
    "supratip":           29,
    "subnasale":          33,
    "upper_lip":          51,
    "mouth_corner":       48,
    "lower_lip":          57,
    "upper_jaw_angle":    3,
    "lower_jaw_angle":    5,
}
SIDE_DENSE_MAP = {
    "orbitale":           10855,
    "intertragic_notch":  17109,
    "cheekbone":          22478,
    "eyelid_end":         1961,
    "glabella":           31324,
    "forehead":           31172,
    "infratip":           8195,
    "columella":          8197,
    "subalare":           13727,
    "labiomental_fold":   8843,
    "chin_point":         36160,
    "chin_bottom":        36143,
    "cervical_point":     35587,
    "porion_base":        17327,
    "tragus_base":        17331,
    "hairline_line_a":    31615,
    "hairline_line_b":    31670,
}


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _relative_radius(pts_2d, fraction):
    if fraction >= 1:
        return fraction
    span_x = float(np.max(pts_2d[0, :]) - np.min(pts_2d[0, :]))
    span_y = float(np.max(pts_2d[1, :]) - np.min(pts_2d[1, :]))
    return max(1.0, max(span_x, span_y) * fraction)


def _normalized_x(px, width, mirrored=False):
    x = _clamp01(float(px) / width)
    return round(_clamp01(1.0 - x if mirrored else x), 6)


def _normalized_y(py, height):
    return round(_clamp01(float(py) / height), 6)


def _backward_direction(pts_2d):
    nose_idx = SIDE_SPARSE_MAP.get("nose_tip", 30)
    forehead_idx = SIDE_DENSE_MAP.get("forehead", 31172)
    if nose_idx >= pts_2d.shape[1] or forehead_idx >= pts_2d.shape[1]:
        return np.array([-1.0, 0.0])
    fx = pts_2d[0, nose_idx] - pts_2d[0, forehead_idx]
    fy = pts_2d[1, nose_idx] - pts_2d[1, forehead_idx]
    f_len = np.sqrt(fx * fx + fy * fy)
    if f_len < 1e-6:
        return np.array([-1.0, 0.0])
    forward = np.array([fx, fy]) / f_len
    backward = -forward
    return backward


def _point_behind(base_idx, pts_2d, search_radius=0.03, min_step=0.002):
    if base_idx >= pts_2d.shape[1]:
        return base_idx
    search_radius = _relative_radius(pts_2d, search_radius)
    min_step = _relative_radius(pts_2d, min_step)
    base_x = pts_2d[0, base_idx]
    base_y = pts_2d[1, base_idx]
    backward = _backward_direction(pts_2d)
    best_score = -float('inf')
    best_idx = base_idx
    dists = np.sqrt(
        (pts_2d[0, :] - base_x) ** 2 +
        (pts_2d[1, :] - base_y) ** 2
    )
    nearby_mask = dists < search_radius
    if np.sum(nearby_mask) < 3:
        nearby_mask = dists < search_radius * 3
    nearby_indices = np.where(nearby_mask)[0]
    if len(nearby_indices) == 0:
        return base_idx
    for idx in nearby_indices:
        if idx == base_idx:
            continue
        dx = pts_2d[0, idx] - base_x
        dy = pts_2d[1, idx] - base_y
        d_len = np.sqrt(dx * dx + dy * dy)
        if d_len < min_step:
            continue
        behind_score = (dx * backward[0] + dy * backward[1]) / d_len
        y_similarity = 1.0 - min(1.0, abs(dy) / max(d_len, 0.001))
        score = behind_score * 0.8 + y_similarity * 0.2
        if score > best_score:
            best_score = score
            best_idx = idx
    return int(best_idx)


def _find_top_of_head(ver):
    pts_2d = ver[:2, :]
    idx = np.argmin(pts_2d[1, :])
    return int(idx)


def _find_occiput(ver):
    pts_2d = ver[:2, :]
    nose_tip_idx = SIDE_SPARSE_MAP.get("nose_tip", 30)
    nose_y = pts_2d[1, nose_tip_idx] if nose_tip_idx < pts_2d.shape[1] else pts_2d[1, :].mean()
    upper_mask = pts_2d[1, :] < nose_y
    if np.sum(upper_mask) < 10:
        upper_mask = np.ones(pts_2d.shape[1], dtype=bool)
    idx = int(np.argmin(pts_2d[0, upper_mask]))
    upper_indices = np.where(upper_mask)[0]
    return int(upper_indices[idx])


def _find_hairline_profile(pts_2d_dense, face_box):
    idx_a = SIDE_DENSE_MAP.get("hairline_line_a", 31615)
    idx_b = SIDE_DENSE_MAP.get("hairline_line_b", 31670)
    if idx_a >= pts_2d_dense.shape[1] or idx_b >= pts_2d_dense.shape[1]:
        x1, y1, x2, y2 = face_box
        return (float((x1 + x2) / 2.0), float(y1))
    ax, ay = float(pts_2d_dense[0, idx_a]), float(pts_2d_dense[1, idx_a])
    bx, by = float(pts_2d_dense[0, idx_b]), float(pts_2d_dense[1, idx_b])
    x1, y1, x2, y2 = [float(v) for v in face_box[:4]]
    top_y = y1
    dy = by - ay
    if abs(dy) < 1e-6:
        return (float((x1 + x2) / 2.0), float(top_y))
    t = (top_y - ay) / dy
    ix = ax + t * (bx - ax)
    margin = (x2 - x1) * 0.3
    ix = max(x1 - margin, min(x2 + margin, ix))
    return (float(ix), float(top_y))


def _find_tragus(pts_2d_dense):
    base_idx = SIDE_DENSE_MAP.get("tragus_base", 17331)
    return _point_behind(base_idx, pts_2d_dense,
                         search_radius=0.025, min_step=0.001)


def _find_porion(pts_2d_dense):
    base_idx = SIDE_DENSE_MAP.get("porion_base", 17327)
    if base_idx >= pts_2d_dense.shape[1]:
        return base_idx
    behind_idx = _point_behind(base_idx, pts_2d_dense,
                                search_radius=0.025, min_step=0.001)
    base_x = pts_2d_dense[0, base_idx]
    base_y = pts_2d_dense[1, base_idx]
    backward = _backward_direction(pts_2d_dense)
    search_radius = _relative_radius(pts_2d_dense, 0.03)
    min_step = _relative_radius(pts_2d_dense, 0.001)
    candidates = np.where(
        (np.sqrt((pts_2d_dense[0, :] - base_x)**2 + (pts_2d_dense[1, :] - base_y)**2) < search_radius)
    )[0]
    if len(candidates) > 1:
        best_score = -float('inf')
        best_idx = behind_idx
        for idx in candidates:
            dx = pts_2d_dense[0, idx] - base_x
            dy = pts_2d_dense[1, idx] - base_y
            d_len = np.sqrt(dx*dx + dy*dy)
            if d_len < min_step:
                continue
            behind_score = (dx * backward[0] + dy * backward[1]) / d_len
            up_score = -dy / max(d_len, 0.001)
            score = behind_score * 0.7 + up_score * 0.3
            if score > best_score:
                best_score = score
                best_idx = idx
        return int(best_idx)
    return int(behind_idx)


def _find_neck_point(pts_2d_dense):
    cervical_idx = SIDE_DENSE_MAP.get("cervical_point", 35587)
    if cervical_idx >= pts_2d_dense.shape[1]:
        return cervical_idx
    cx, cy = pts_2d_dense[0, cervical_idx], pts_2d_dense[1, cervical_idx]
    narrow_x = _relative_radius(pts_2d_dense, 0.03)
    wide_x = _relative_radius(pts_2d_dense, 0.05)
    candidates = np.where(
        (pts_2d_dense[1, :] > cy) &
        (np.abs(pts_2d_dense[0, :] - cx) < narrow_x)
    )[0]
    if len(candidates) > 0:
        best = candidates[np.argmax(pts_2d_dense[1, candidates])]
        return int(best)
    candidates = np.where(np.abs(pts_2d_dense[0, :] - cx) < wide_x)[0]
    if len(candidates) > 0:
        best = candidates[np.argmax(pts_2d_dense[1, candidates])]
        return int(best)
    return cervical_idx


def _determine_facing(pts_2d_sparse):
    nose_idx = SIDE_SPARSE_MAP.get("nose_tip", 30)
    if nose_idx >= pts_2d_sparse.shape[1]:
        return "left"
    nose_x = pts_2d_sparse[0, nose_idx]
    return "left" if nose_x < 0.5 else "right"


def detect_front(image_path):
    face_boxes_model, tddfa = _load_models()
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Cannot read image: {image_path}"}
    h, w = img.shape[:2]
    boxes = face_boxes_model(img)
    if len(boxes) == 0:
        return {"error": "No face detected", "image_size": {"w": w, "h": h}}
    best_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    x1, y1, x2, y2 = [int(round(float(v))) for v in best_box[:4]]
    hairline_x = (x1 + x2) / 2.0 / w
    hairline_y = y1 / h
    param_lst, roi_box_lst = tddfa(img, boxes)
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
    mesh_points = []
    if ver_lst and len(ver_lst) > 0:
        ver = ver_lst[0]
        for i in range(ver.shape[1]):
            mesh_points.append({
                "index": i,
                "x": round(float(ver[0, i]) / w, 6),
                "y": round(float(ver[1, i]) / h, 6),
            })
    return {
        "mode": "front",
        "image_size": {"w": w, "h": h},
        "face_box": {
            "x1": round(x1 / w, 6),
            "y1": round(y1 / h, 6),
            "x2": round(x2 / w, 6),
            "y2": round(y2 / h, 6),
        },
        "hairline": {
            "x": round(hairline_x, 6),
            "y": round(hairline_y, 6),
        },
        "total_mesh_points": len(mesh_points),
        "mesh_points": mesh_points,
    }
def detect_side(image_path):
    face_boxes_model, tddfa = _load_models()
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Cannot read image: {image_path}"}
    original_h, original_w = img.shape[:2]
    was_mirrored = False
    original_facing = "unknown"
    boxes = face_boxes_model(img)
    if len(boxes) == 0:
        return {"error": "No face detected", "image_size": {"w": original_w, "h": original_h}}
    param_lst, roi_box_lst = tddfa(img, boxes)
    ver_sparse_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
    if not ver_sparse_lst or len(ver_sparse_lst) == 0:
        return {"error": "Sparse mesh reconstruction failed",
                "image_size": {"w": original_w, "h": original_h}}
    ver_sparse = ver_sparse_lst[0]
    pts_2d_sparse_initial = ver_sparse[:2, :]
    pts_sparse_norm = pts_2d_sparse_initial.copy()
    pts_sparse_norm[0, :] /= original_w
    pts_sparse_norm[1, :] /= original_h
    original_facing = _determine_facing(pts_sparse_norm)
    if original_facing == "left":
        img = cv2.flip(img, 1)
        was_mirrored = True
        boxes = face_boxes_model(img)
        if len(boxes) == 0:
            return {"error": "No face detected after mirroring",
                    "image_size": {"w": original_w, "h": original_h}}
    h, w = img.shape[:2]
    param_lst, roi_box_lst = tddfa(img, boxes)
    ver_dense_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
    if not ver_dense_lst or len(ver_dense_lst) == 0:
        return {"error": "Dense mesh reconstruction failed",
                "image_size": {"w": w, "h": h}}
    ver_dense = ver_dense_lst[0]
    num_dense = ver_dense.shape[1]
    pts_2d_dense_px = ver_dense[:2, :]
    ver_sparse_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
    if not ver_sparse_lst or len(ver_sparse_lst) == 0:
        return {"error": "Sparse mesh reconstruction failed",
                "image_size": {"w": w, "h": h}}
    ver_sparse = ver_sparse_lst[0]
    num_sparse = ver_sparse.shape[1]
    pts_2d_sparse_px = ver_sparse[:2, :]
    best_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    x1, y1, x2, y2 = [int(round(float(v))) for v in best_box[:4]]
    landmarks = {}
    def add_dense(lm_id, mesh_idx):
        if mesh_idx < num_dense:
            landmarks[lm_id] = {
                "x": _normalized_x(pts_2d_dense_px[0, mesh_idx], w, was_mirrored),
                "y": _normalized_y(pts_2d_dense_px[1, mesh_idx], h),
                "mesh_index": int(mesh_idx),
                "model_type": "2d_dense",
            }
    def add_sparse(lm_id, sparse_idx):
        if sparse_idx < num_sparse:
            landmarks[lm_id] = {
                "x": _normalized_x(pts_2d_sparse_px[0, sparse_idx], w, was_mirrored),
                "y": _normalized_y(pts_2d_sparse_px[1, sparse_idx], h),
                "mesh_index": int(sparse_idx),
                "model_type": "2d_sparse",
            }
    for lm_id, sparse_idx in SIDE_SPARSE_MAP.items():
        add_sparse(lm_id, sparse_idx)
    for lm_id, dense_idx in SIDE_DENSE_MAP.items():
        if lm_id in ("porion_base", "tragus_base", "hairline_line_a", "hairline_line_b"):
            continue
        add_dense(lm_id, dense_idx)
    add_dense("tragus", _find_tragus(pts_2d_dense_px))
    add_dense("porion", _find_porion(pts_2d_dense_px))
    add_dense("top_of_head", _find_top_of_head(ver_dense))
    add_dense("occiput", _find_occiput(ver_dense))
    add_dense("neck_point", _find_neck_point(pts_2d_dense_px))
    hairline_result = _find_hairline_profile(pts_2d_dense_px, (x1, y1, x2, y2))
    if hairline_result is not None:
        hx, hy = hairline_result
        landmarks["hairline_profile"] = {
            "x": _normalized_x(hx, w, was_mirrored),
            "y": _normalized_y(hy, h),
            "mesh_index": SIDE_DENSE_MAP.get("hairline_line_a", 31615),
            "model_type": "computed",
        }
    mesh_points = []
    for i in range(num_dense):
        mesh_points.append({
            "index": i,
            "x": _normalized_x(pts_2d_dense_px[0, i], w, was_mirrored),
            "y": _normalized_y(pts_2d_dense_px[1, i], h),
        })
    SIDE_LABELS = {
        "top_of_head": "Top of Head",
        "occiput": "Occiput",
        "hairline_profile": "Hairline (Profile)",
        "forehead": "Forehead",
        "glabella": "Glabella",
        "nasal_bridge_root": "Nasal Bridge Root",
        "rhinion": "Rhinion",
        "supratip": "Supratip",
        "nose_tip": "Nose Tip",
        "infratip": "Infratip",
        "columella": "Columella",
        "subnasale": "Subnasale",
        "subalare": "Subalare",
        "upper_lip": "Upper Lip",
        "mouth_corner": "Mouth Corner",
        "lower_lip": "Lower Lip",
        "labiomental_fold": "Labiomental Fold",
        "chin_point": "Chin Point",
        "chin_bottom": "Chin Bottom",
        "upper_jaw_angle": "Upper Jaw Angle",
        "lower_jaw_angle": "Lower Jaw Angle",
        "porion": "Porion",
        "tragus": "Tragus",
        "intertragic_notch": "Intertragic Notch",
        "orbitale": "Orbitale",
        "corneal_apex": "Corneal Apex",
        "eyelid_end": "Eyelid End",
        "lower_eyelid": "Lower Eyelid",
        "cheekbone": "Cheekbone",
        "cervical_point": "Cervical Point",
        "neck_point": "Neck Point",
    }
    result_landmarks = {}
    missing_landmarks = [lm_id for lm_id in SIDE_LABELS if lm_id not in landmarks]
    for lm_id, data in landmarks.items():
        result_landmarks[lm_id] = {
            "id": lm_id,
            "label": SIDE_LABELS.get(lm_id, lm_id),
            "x": data["x"],
            "y": data["y"],
            "mesh_index": data["mesh_index"],
            "model_type": data.get("model_type", "unknown"),
        }
    return {
        "mode": "side",
        "image_size": {"w": w, "h": h},
        "original_image_size": {"w": original_w, "h": original_h},
        "facing_direction": original_facing,
        "original_facing": original_facing,
        "was_mirrored": was_mirrored,
        "total_dense_points": num_dense,
        "total_sparse_points": num_sparse,
        "missing_landmarks": missing_landmarks,
        "landmarks": result_landmarks,
        "mesh_points": mesh_points,
    }
def main():
    parser = argparse.ArgumentParser(description="3DDFA_V2 Landmark Detection")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("mode", choices=["front", "side"], help="Detection mode")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file path")
    args = parser.parse_args()
    if not os.path.exists(args.image):
        print(json.dumps({"error": f"File not found: {args.image}"}))
        sys.exit(1)
    try:
        if args.mode == "front":
            result = detect_front(args.image)
        else:
            result = detect_side(args.image)
    except Exception as e:
        import traceback
        traceback.print_exc()
        result = {"error": str(e)}
    json_str = json.dumps(result, ensure_ascii=False)
    print(json_str)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
if __name__ == "__main__":
    main()
