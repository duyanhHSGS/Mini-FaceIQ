import os
import sys
import importlib.util
import types
import numpy as np
import cv2

_PROVIDER_DIR = os.path.dirname(os.path.abspath(__file__))
_packages_DIR = os.path.dirname(_PROVIDER_DIR)
_3DDFA_DIR = os.path.join(_packages_DIR, "3DDFA_V2")
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


def _cpu_nms_numpy(dets, thresh):
    if dets.shape[0] == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(overlap <= thresh)[0] + 1]

    return keep


def _install_faceboxes_shims():
    faceboxes_dir = os.path.join(_3DDFA_DIR, "FaceBoxes")

    package = types.ModuleType("FaceBoxes")
    package.__path__ = [faceboxes_dir]
    sys.modules["FaceBoxes"] = package

    nms_module = types.ModuleType("FaceBoxes.utils.nms.cpu_nms")
    nms_module.cpu_nms = _cpu_nms_numpy
    nms_module.cpu_soft_nms = _cpu_nms_numpy
    sys.modules["FaceBoxes.utils.nms.cpu_nms"] = nms_module


def _load_faceboxes_onnx_class():
    _install_faceboxes_shims()
    module_name = "FaceBoxes.FaceBoxes_ONNX"
    module_path = os.path.join(_3DDFA_DIR, "FaceBoxes", "FaceBoxes_ONNX.py")
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"FaceBoxes_ONNX.py not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load FaceBoxes_ONNX from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.FaceBoxes_ONNX


def _load_models():
    if _model_cache["face_boxes"] is not None:
        return _model_cache["face_boxes"], _model_cache["tddfa"]

    _require_3ddfa_dir()
    try:
        import yaml
        FaceBoxes_ONNX = _load_faceboxes_onnx_class()
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
    for key in ["bfm_fp", "param_mean_std_fp"]:
        if key in cfg:
            cfg[key] = os.path.join(_3DDFA_DIR, cfg[key])
            if not os.path.exists(cfg[key]):
                raise FileNotFoundError(f"3DDFA file not found for {key}: {cfg[key]}")
    if "checkpoint_fp" in cfg:
        checkpoint_fp = os.path.join(_3DDFA_DIR, cfg["checkpoint_fp"])
        cfg["onnx_fp"] = os.path.splitext(checkpoint_fp)[0] + ".onnx"
        if not os.path.exists(cfg["onnx_fp"]):
            raise FileNotFoundError(f"3DDFA ONNX model not found: {cfg['onnx_fp']}")

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


def detect_side(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Cannot read image: {image_path}"}
    return _detect_side_from_image(img)


def detect_side_from_upload(file_storage):
    data = np.frombuffer(file_storage.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read image file")

    result = _detect_side_from_image(img)
    return result


def _detect_side_from_image(img):
    face_boxes_model, tddfa = _load_models()
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


