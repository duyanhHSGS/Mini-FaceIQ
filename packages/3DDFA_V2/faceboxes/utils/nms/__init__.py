"""Pure NumPy non-maximum suppression used by the ONNX face detector."""

from __future__ import annotations

import numpy as np


def cpu_nms(detections: np.ndarray, threshold: float) -> list[int]:
    """Return indices kept after greedy intersection-over-union suppression."""
    if detections.size == 0:
        return []
    x1, y1, x2, y2, scores = (detections[:, index] for index in range(5))
    areas = np.maximum(0.0, x2 - x1 + 1) * np.maximum(0.0, y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        overlap_width = np.maximum(
            0.0,
            np.minimum(x2[current], x2[remaining])
            - np.maximum(x1[current], x1[remaining])
            + 1,
        )
        overlap_height = np.maximum(
            0.0,
            np.minimum(y2[current], y2[remaining])
            - np.maximum(y1[current], y1[remaining])
            + 1,
        )
        intersection = overlap_width * overlap_height
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = remaining[iou <= threshold]
    return keep


def cpu_soft_nms(
    detections: np.ndarray,
    sigma: float = 0.5,
    threshold: float = 0.3,
) -> list[int]:
    """Compatibility entrypoint; the production provider uses hard NMS."""
    # TODO: implement score-decay Soft-NMS if a future provider needs it.
    _ = sigma
    return cpu_nms(detections, threshold)
