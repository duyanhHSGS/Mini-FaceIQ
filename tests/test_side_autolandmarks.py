import os

import pytest

import side_autolandmarks
from side_landmarks import SIDE_LANDMARK_DEFS


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
    expected_ids = {item["id"] for item in SIDE_LANDMARK_DEFS}
    missing_ids = set(result["missing_landmarks"])
    assert expected_ids <= set(detected) | missing_ids

    for landmark in detected.values():
        assert {"id", "label", "x", "y"} <= set(landmark)
        assert 0.0 <= landmark["x"] <= 1.0
        assert 0.0 <= landmark["y"] <= 1.0
