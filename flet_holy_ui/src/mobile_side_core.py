"""Portable, CPU-only 3DDFA ONNX inference without PyTorch.

Uses original FaceBoxes, 3DDFA and BFM ONNX models. Outputs the original
detector's shapes so side_3ddfa landmark heuristics can be reused unchanged.
"""

from __future__ import annotations

import io
import math
import pickle
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import onnxruntime as ort

_MODEL_SUBDIR = "assets/models/side/"


def _asset_bytes(name: str) -> bytes:
    """Work with desktop source files and with Flet's Android assets/app.zip."""
    if Path(name).name != name:
        raise ValueError("Model assets must be simple filenames.")
    regular_file = Path(__file__).resolve().parent / _MODEL_SUBDIR / name
    if regular_file.is_file():
        return regular_file.read_bytes()
    module_path = str(__file__)
    if ".zip/" in module_path:
        archive = module_path.split(".zip/", 1)[0] + ".zip"
        with zipfile.ZipFile(archive) as app_zip:
            return app_zip.read(_MODEL_SUBDIR + name)
    raise FileNotFoundError(f"3DDFA model missing: {_MODEL_SUBDIR}{name}")


def _session(name: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if name == "FaceBoxesProd.onnx":
        # FaceBoxes stores tensors in an adjacent .onnx.data file. ONNX Runtime
        # cannot resolve that sidecar when given only in-memory protobuf bytes.
        with tempfile.TemporaryDirectory(prefix="miniface-side-") as temporary:
            model = Path(temporary) / name
            model.write_bytes(_asset_bytes(name))
            model.with_name(name + ".data").write_bytes(_asset_bytes(name + ".data"))
            return ort.InferenceSession(
                str(model), sess_options=options, providers=["CPUExecutionProvider"]
            )
    return ort.InferenceSession(
        _asset_bytes(name), sess_options=options, providers=["CPUExecutionProvider"]
    )


@lru_cache(maxsize=4)
def _anchors(height: int, width: int) -> np.ndarray:
    # Preserve prior_box.py ordering: for every cell, for every minimum size.
    items = []
    for step, sizes in ((32, (32, 64, 128)), (64, (256,)), (128, (512,))):
        for i in range(math.ceil(height / step)):
            for j in range(math.ceil(width / step)):
                for size in sizes:
                    offsets = (0.0, 0.25, 0.5, 0.75) if size == 32 else (
                        (0.0, 0.5) if size == 64 else (0.5,)
                    )
                    for oy in offsets:
                        for ox in offsets:
                            items.append(((j + ox) * step / width, (i + oy) * step / height,
                                          size / width, size / height))
    return np.asarray(items, dtype=np.float32)


def _nms(dets: np.ndarray, threshold: float) -> list[int]:
    if not dets.size:
        return []
    x1, y1, x2, y2, score = (dets[:, i] for i in range(5))
    areas = np.maximum(0.0, x2 - x1 + 1) * np.maximum(0.0, y2 - y1 + 1)
    order = score.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        first = int(order[0])
        keep.append(first)
        rest = order[1:]
        if not rest.size:
            break
        w = np.maximum(0.0, np.minimum(x2[first], x2[rest]) - np.maximum(x1[first], x1[rest]) + 1)
        h = np.maximum(0.0, np.minimum(y2[first], y2[rest]) - np.maximum(y1[first], y1[rest]) + 1)
        intersection = w * h
        union = areas[first] + areas[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[iou <= threshold]
    return keep


class MobileFaceBoxes:
    def __init__(self) -> None:
        self.session = _session("FaceBoxesProd.onnx")

    def __call__(self, original: np.ndarray) -> list[list[float]]:
        height, width = original.shape[:2]
        scale = min(1.0, 720 / height, 1080 / width)
        resized = (
            cv2.resize(original, (int(scale * width), int(scale * height)))
            if scale < 1 else original
        )
        h, w = resized.shape[:2]
        image = resized.astype(np.float32)
        image -= np.asarray((104, 117, 123), dtype=np.float32)
        image = np.ascontiguousarray(image.transpose(2, 0, 1)[None, ...])
        outputs = cast(list[np.ndarray], self.session.run(None, {"input": image}))
        locations = np.asarray(outputs[0][0], dtype=np.float32)
        scores = np.asarray(outputs[1][0][:, 1], dtype=np.float32)
        priors = _anchors(h, w)
        if len(priors) != len(locations):
            raise ValueError(f"FaceBoxes priors mismatch: {len(priors)} != {len(locations)}")

        centers = priors[:, :2] + locations[:, :2] * .1 * priors[:, 2:]
        size = priors[:, 2:] * np.exp(locations[:, 2:] * .2)
        boxes = np.concatenate((centers - size / 2, centers + size / 2), axis=1)
        boxes *= np.array([w, h, w, h], dtype=np.float32) / scale
        indices = np.flatnonzero(scores > 0.05)
        if indices.size == 0:
            return []
        boxes, scores = boxes[indices], scores[indices]
        order = scores.argsort()[::-1][:5000]
        dets = np.column_stack((boxes[order], scores[order])).astype(np.float32)
        dets = dets[_nms(dets, .3)][:750]
        return [
            list(map(float, detected))
            for detected in dets
            if detected[4] > 0.5
        ]


def _parse_param(param: np.ndarray):
    dimensions = {62: (40, 10), 72: (40, 20), 141: (100, 29)}
    if len(param) not in dimensions:
        raise ValueError(f"Unsupported 3DDFA param count: {len(param)}")
    shape_dim, _ = dimensions[len(param)]
    matrix = param[:12].reshape(3, 4)
    return (
        matrix[:, :3], matrix[:, 3:4],
        param[12:12 + shape_dim].reshape(-1, 1),
        param[12 + shape_dim:].reshape(-1, 1),
    )


def _similar_transform(points: np.ndarray, roi: list[float], size: int) -> np.ndarray:
    points = points.astype(np.float32, copy=True)
    points[0] -= 1
    points[2] -= 1
    points[1] = size - points[1]
    sx, sy, ex, ey = roi
    scalex = (ex - sx) / size
    scaley = (ey - sy) / size
    points[0] = points[0] * scalex + sx
    points[1] = points[1] * scaley + sy
    points[2] *= (scalex + scaley) / 2
    points[2] -= points[2].min()
    return points


def _roi_bbox(box: list[float]) -> list[float]:
    left, top, right, bottom = box[:4]
    old_size = (right - left + bottom - top) / 2
    cx = right - (right - left) / 2
    cy = bottom - (bottom - top) / 2 + old_size * .14
    width = int(old_size * 1.58)
    return [cx - width / 2, cy - width / 2, cx + width / 2, cy + width / 2]


def _crop_img(img: np.ndarray, roi: list[float]) -> np.ndarray:
    h, w = img.shape[:2]
    sx, sy, ex, ey = (round(v) for v in roi)
    dh, dw = ey - sy, ex - sx
    if dh <= 0 or dw <= 0:
        raise ValueError("Invalid face ROI dimensions.")
    output = np.zeros((dh, dw, 3), np.uint8)
    clipped_x1, clipped_y1 = max(0, sx), max(0, sy)
    clipped_x2, clipped_y2 = min(w, ex), min(h, ey)
    if clipped_x2 > clipped_x1 and clipped_y2 > clipped_y1:
        output[clipped_y1 - sy:clipped_y2 - sy, clipped_x1 - sx:clipped_x2 - sx] = (
            img[clipped_y1:clipped_y2, clipped_x1:clipped_x2]
        )
    return output


class MobileTDDFA:
    def __init__(self) -> None:
        self.session = _session("mb1_120x120.onnx")
        self.bfm_session = _session("bfm_noneck_v3.onnx")
        stats = pickle.loads(_asset_bytes("param_mean_std_62d_120x120.pkl"))  # bundled trusted local file
        self.mean = np.asarray(stats["mean"])
        self.std = np.asarray(stats["std"])
        with np.load(io.BytesIO(_asset_bytes("sparse_bfm.npz")), allow_pickle=False) as sparse:
            self.u_base = sparse["u_base"]
            self.w_shp_base = sparse["w_shp_base"]
            self.w_exp_base = sparse["w_exp_base"]

    def __call__(self, image: np.ndarray, boxes: list[list[float]], **options):
        _ = options
        parameters, rois = [], []
        for box in boxes:
            roi = _roi_bbox(box)
            crop = cv2.resize(_crop_img(image, roi), (120, 120)).astype(np.float32)
            prepared = np.ascontiguousarray((crop.transpose(2, 0, 1)[None, ...] - 127.5) / 128.0)
            param = np.asarray(self.session.run(None, {"input": prepared})[0])
            param = param.flatten().astype(np.float32) * self.std + self.mean
            parameters.append(param)
            rois.append(roi)
        return parameters, rois

    def recon_vers(self, parameters, rois, **options):
        dense = options.get("dense_flag", False)
        outputs = []
        for param, roi in zip(parameters, rois):
            r, offset, shape, exp = _parse_param(param)
            if dense:
                points = np.asarray(self.bfm_session.run(
                    None, {"R": r, "offset": offset, "alpha_shp": shape, "alpha_exp": exp}
                )[0], dtype=np.float32)
            else:
                # Match desktop 3DDFA's sparse BFM path exactly.
                points = r @ (
                    self.u_base + self.w_shp_base @ shape + self.w_exp_base @ exp
                ).reshape(3, -1, order="F") + offset
            outputs.append(_similar_transform(points, roi, 120))
        return outputs


@lru_cache(maxsize=1)
def get_models() -> tuple[MobileFaceBoxes, MobileTDDFA]:
    return MobileFaceBoxes(), MobileTDDFA()

