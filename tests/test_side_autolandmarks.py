import importlib
import os

import numpy as np
import pytest

import packages.adapter_side as side_autolandmarks
from side_landmarks import SIDE_LANDMARK_DEFS

cpu_nms = importlib.import_module(
    "packages.3DDFA_V2.faceboxes.utils.nms"
).cpu_nms


def test_cpu_nms_returns_empty_for_empty_input():
    assert cpu_nms(np.empty((0, 5), dtype=np.float32), 0.3) == []


def test_cpu_nms_keeps_highest_score_from_overlapping_boxes():
    detections = np.array(
        [[0, 0, 10, 10, 0.7], [1, 1, 11, 11, 0.9]],
        dtype=np.float32,
    )
    assert cpu_nms(detections, 0.3) == [1]


def test_cpu_nms_keeps_disjoint_boxes_in_score_order():
    detections = np.array(
        [[0, 0, 2, 2, 0.2], [10, 10, 12, 12, 0.8]],
        dtype=np.float32,
    )
    assert cpu_nms(detections, 0.3) == [1, 0]


def test_cpu_nms_handles_zero_area_boxes_without_dividing_by_zero():
    detections = np.array(
        [[2, 2, 1, 1, 0.9], [5, 5, 4, 4, 0.8]],
        dtype=np.float32,
    )
    assert cpu_nms(detections, 0.3) == [0, 1]


def _side_smoke_image_path():
    env_path = os.environ.get("FACEIQ_SIDE_SMOKE_IMAGE")
    candidates = [
        env_path,
        os.path.join(os.path.dirname(__file__), "fixtures", "side_profile.jpg"),
        os.path.join(os.path.dirname(__file__), "fixtures", "side_profile.png"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def test_side_landmark_list_returns_app_fields_in_definition_order():
    result = {
        "landmarks": {
            "nose_tip": {
                "id": "nose_tip",
                "label": "Nose Tip",
                "x": 0.55,
                "y": 0.35,
                "mesh_index": 30,
                "model_type": "2d_sparse",
            },
            "top_of_head": {
                "id": "top_of_head",
                "label": "Top of Head",
                "x": 0.45,
                "y": 0.05,
                "mesh_index": 123,
                "model_type": "2d_dense",
            },
        }
    }

    landmarks = side_autolandmarks._side_landmark_list(result)

    assert landmarks == [
        {"id": "top_of_head", "label": "Top of Head", "x": 0.45, "y": 0.05},
        {"id": "nose_tip", "label": "Nose Tip", "x": 0.55, "y": 0.35},
    ]


def test_side_detector_real_smoke_returns_normalized_app_landmarks():
    image_path = _side_smoke_image_path()
    if image_path is None:
        pytest.skip("Set FACEIQ_SIDE_SMOKE_IMAGE or add tests/fixtures/side_profile.jpg to run real 3DDFA smoke.")

    result = side_autolandmarks.detect_side(image_path)

    assert "error" not in result
    assert result["mode"] == "side"
    assert isinstance(result["missing_landmarks"], list)

    detected = result["landmarks"]
    assert isinstance(detected, dict)
    expected_ids = {item["id"] for item in SIDE_LANDMARK_DEFS}
    missing_ids = set(result["missing_landmarks"])
    assert expected_ids <= set(detected) | missing_ids

    for landmark in detected.values():
        assert {"id", "label", "x", "y"} <= set(landmark)
        assert 0.0 <= landmark["x"] <= 1.0
        assert 0.0 <= landmark["y"] <= 1.0
