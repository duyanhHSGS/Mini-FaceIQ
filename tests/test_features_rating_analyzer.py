import numpy as np
import pytest
import torch
from PIL import Image

from conftest import load_feature_analyzer


face_analyzer = load_feature_analyzer()


def test_module_uses_supported_device_and_aux_input_is_deterministic():
    assert face_analyzer.device.type in {"cpu", "cuda"}
    first = face_analyzer.make_aux_input()
    second = face_analyzer.make_aux_input()
    assert first.shape == (1, 7)
    assert torch.equal(first, second)


def test_region_mask_ignores_invalid_points_and_requires_three_points():
    landmarks = [(10, 10), (30, 10), (30, 30), (10, 30)]
    mask = face_analyzer.region_mask_from_landmarks(landmarks, [0, 1, 2, 99], 40, 40)
    assert mask.dtype == np.uint8
    assert mask.shape == (40, 40)
    assert mask.max() == 255

    empty = face_analyzer.region_mask_from_landmarks([(10, 10)], [0, 1], 40, 40)
    assert np.count_nonzero(empty) == 0


def test_region_mask_stays_inside_image_bounds():
    landmarks = [(-100, -100), (1000, -100), (1000, 1000), (-100, 1000)]
    mask = face_analyzer.region_mask_from_landmarks(landmarks, [0, 1, 2, 3], 40, 40)
    assert mask.shape == (40, 40)
    assert np.count_nonzero(mask) <= 40 * 40


def test_create_all_masks_returns_all_regions_with_requested_shape(monkeypatch):
    image = np.zeros((50, 80, 3), dtype=np.uint8)
    landmarks = [(10, 10)] * 500
    monkeypatch.setattr(face_analyzer, "get_landmarks_mp", lambda _: landmarks)
    masks = face_analyzer.create_all_masks(image, target=32)

    assert set(masks) == {
        "Left Eye", "Right Eye", "Nose", "Mouth", "L Eyebrow", "R Eyebrow",
        "Skin", "Hair",
    }
    assert all(mask.shape == (32, 32) for mask in masks.values())
    assert all(mask.dtype == np.uint8 for mask in masks.values())


def test_create_all_masks_falls_back_when_mediapipe_fails(monkeypatch):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    monkeypatch.setattr(face_analyzer, "get_landmarks_mp", lambda _: None)
    masks = face_analyzer.create_all_masks(image, target=24)
    assert set(masks) == {
        "Left Eye", "Right Eye", "Nose", "Mouth", "L Eyebrow", "R Eyebrow",
        "Skin", "Hair",
    }
    assert all(mask.shape == (24, 24) for mask in masks.values())


def test_get_landmarker_reports_missing_task_model(monkeypatch):
    monkeypatch.setattr(face_analyzer, "_landmarker", None)
    monkeypatch.setattr(face_analyzer.os.path, "exists", lambda _: False)
    with pytest.raises(FileNotFoundError, match="MediaPipe model not found"):
        face_analyzer.get_landmarker()


def test_blur_occlude_zero_mask_is_unchanged_and_full_mask_preserves_shape():
    image = torch.arange(1 * 3 * 16 * 16, dtype=torch.float32).reshape(1, 3, 16, 16)
    zero = np.zeros((16, 16), dtype=np.uint8)
    full = np.full((16, 16), 255, dtype=np.uint8)
    assert torch.equal(face_analyzer.blur_occlude(image, zero, k=3), image)
    blurred = face_analyzer.blur_occlude(image, full, k=3)
    assert blurred.shape == image.shape
    assert blurred.dtype == image.dtype
    assert not torch.equal(blurred, image)


@pytest.mark.parametrize(
    "mask,k,error",
    [
        (np.zeros((8, 8), dtype=np.uint8), 2, "positive odd"),
        (np.zeros((8, 8), dtype=np.uint8), 0, "positive odd"),
        (np.zeros((7, 8), dtype=np.uint8), 3, "mask must have shape"),
    ],
)
def test_blur_occlude_rejects_invalid_inputs(mask, k, error):
    image = torch.zeros((1, 3, 8, 8))
    with pytest.raises(ValueError, match=error):
        face_analyzer.blur_occlude(image, mask, k=k)


class SequenceModel:
    def __init__(self, scores):
        self.scores = iter(scores)

    def to(self, _device):
        return self

    def eval(self):
        return self

    def load_state_dict(self, _state, strict=False):
        return self

    def __call__(self, _image, _parsing, _aux):
        return torch.tensor([next(self.scores)], dtype=torch.float32)


@pytest.fixture
def analyze_dependencies(monkeypatch, tmp_path):
    image_path = tmp_path / "face.png"
    Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8)).save(image_path)
    masks = {
        "Left Eye": np.full((224, 224), 255, dtype=np.uint8),
        "Right Eye": np.zeros((224, 224), dtype=np.uint8),
    }
    monkeypatch.setattr(face_analyzer, "Net", lambda: SequenceModel([3.0, 2.0, 4.0]))
    monkeypatch.setattr(face_analyzer, "create_all_masks", lambda *_args: masks)
    monkeypatch.setattr(
        face_analyzer,
        "get_region_polygons",
        lambda *_args: {"Left Eye": [{"x": 0.1, "y": 0.2}], "Right Eye": []},
    )
    monkeypatch.setattr(face_analyzer.os.path, "exists", lambda path: path == str(image_path))
    return image_path


def test_analyze_face_reports_score_contract_and_delta_direction(analyze_dependencies):
    result = face_analyzer.analyze_face(str(analyze_dependencies))

    assert result["score"] == 3.0
    assert result["score_10"] == 5.0
    assert isinstance(result["heatmap"], Image.Image)
    assert result["heatmap"].size == (224, 244)
    assert result["deltas"] == {"Left Eye": 1.0, "Right Eye": -1.0}
    assert "Left Eye" in result["summary"]
    assert "GOOD" in result["summary"]
    assert "BAD" in result["summary"]


@pytest.mark.parametrize(
    "raw,expected",
    [(0.0, 0.0), (1.0, 0.0), (3.0, 5.0), (5.0, 10.0), (9.0, 10.0)],
)
def test_analyze_face_clamps_and_maps_raw_score(monkeypatch, tmp_path, raw, expected):
    image_path = tmp_path / "face.png"
    Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8)).save(image_path)
    monkeypatch.setattr(face_analyzer, "Net", lambda: SequenceModel([raw, raw]))
    monkeypatch.setattr(
        face_analyzer,
        "create_all_masks",
        lambda *_args: {"Nose": np.zeros((224, 224), dtype=np.uint8)},
    )
    monkeypatch.setattr(face_analyzer, "get_region_polygons", lambda *_args: {})
    monkeypatch.setattr(face_analyzer.os.path, "exists", lambda path: path == str(image_path))
    result = face_analyzer.analyze_face(str(image_path))
    assert result["score_10"] == expected
    assert 0.0 <= result["score_10"] <= 10.0


def test_analyze_face_is_repeatable_for_same_image(monkeypatch, tmp_path):
    image_path = tmp_path / "face.png"
    Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8)).save(image_path)
    monkeypatch.setattr(face_analyzer, "Net", lambda: SequenceModel([3.0, 2.5]))
    monkeypatch.setattr(
        face_analyzer,
        "create_all_masks",
        lambda *_args: {"Nose": np.zeros((224, 224), dtype=np.uint8)},
    )
    monkeypatch.setattr(face_analyzer, "get_region_polygons", lambda *_args: {})
    monkeypatch.setattr(face_analyzer.os.path, "exists", lambda path: path == str(image_path))
    first = face_analyzer.analyze_face(str(image_path))
    second = face_analyzer.analyze_face(str(image_path))
    assert first["score"] == second["score"]
    assert first["deltas"] == second["deltas"]
    assert first["summary"] == second["summary"]


def test_analyze_face_rejects_invalid_parsing_map(monkeypatch, tmp_path):
    image_path = tmp_path / "face.png"
    parsing_path = tmp_path / "bad.npy"
    Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(image_path)
    np.save(parsing_path, np.zeros((3, 16, 16), dtype=np.float32))
    monkeypatch.setattr(face_analyzer, "Net", lambda: SequenceModel([3.0]))
    monkeypatch.setattr(
        face_analyzer.os.path,
        "exists",
        lambda path: path in {str(image_path), str(parsing_path)},
    )
    with pytest.raises(ValueError, match="Parsing map"):
        face_analyzer.analyze_face(str(image_path), str(parsing_path))


def test_analyze_face_rejects_missing_image():
    with pytest.raises(FileNotFoundError):
        face_analyzer.analyze_face("does-not-exist.png")
