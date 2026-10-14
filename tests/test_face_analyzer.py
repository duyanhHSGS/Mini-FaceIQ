import importlib
import sys
import types
from typing import ClassVar

import pytest


@pytest.fixture(autouse=True)
def clean_face_analyzer_module():
    sys.modules.pop("packages.providers.front_mediapipe", None)
    yield
    sys.modules.pop("packages.providers.front_mediapipe", None)


class FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class FakeFaceLandmarks:
    landmarks: ClassVar[list[FakeLandmark]] = [
        FakeLandmark(0.25, 0.5),
        FakeLandmark(0.75, 0.1),
    ]


class FakeFaceLandmarker:
    created_options = None
    detect_calls: ClassVar[list[object]] = []
    face_landmarks: ClassVar[list[list[FakeLandmark]]] = [
        FakeFaceLandmarks.landmarks
    ]

    @classmethod
    def create_from_options(cls, options):
        cls.created_options = options
        return cls()

    def detect(self, image):
        type(self).detect_calls.append(image)
        return types.SimpleNamespace(face_landmarks=type(self).face_landmarks)


class FakeImage:
    def __init__(self, *, image_format, data):
        self.image_format = image_format
        self.data = data


def _reload_face_analyzer(monkeypatch, fake_mediapipe):
    fake_cv2 = types.SimpleNamespace(
        COLOR_BGR2RGB="bgr_to_rgb",
        cvtColor=lambda image, mode: ("rgb", image, mode),
    )

    sys.modules.pop("packages.providers.front_mediapipe", None)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mediapipe)
    return importlib.import_module("packages.providers.front_mediapipe")


def _fake_mediapipe():
    return types.SimpleNamespace(
        Image=FakeImage,
        ImageFormat=types.SimpleNamespace(SRGB="srgb"),
        tasks=types.SimpleNamespace(
            BaseOptions=lambda **kwargs: types.SimpleNamespace(**kwargs),
            vision=types.SimpleNamespace(
                FaceLandmarker=FakeFaceLandmarker,
                FaceLandmarkerOptions=lambda **kwargs: types.SimpleNamespace(**kwargs),
                RunningMode=types.SimpleNamespace(IMAGE="image"),
            ),
        ),
    )


def test_face_analyzer_uses_tasks_landmarker(monkeypatch):
    FakeFaceLandmarker.created_options = None
    FakeFaceLandmarker.detect_calls = []
    FakeFaceLandmarker.face_landmarks = [FakeFaceLandmarks.landmarks]
    fake_mediapipe = _fake_mediapipe()

    face_analyzer = _reload_face_analyzer(monkeypatch, fake_mediapipe)

    image = types.SimpleNamespace(shape=(200, 400, 3))
    landmarks = face_analyzer.get_landmarks_mp(image)

    options = FakeFaceLandmarker.created_options
    assert isinstance(options, types.SimpleNamespace)
    assert options.base_options.model_asset_path.endswith("face_landmarker.task")
    assert options.running_mode == "image"
    assert options.num_faces == 1
    assert options.min_face_detection_confidence == 0.5
    assert options.min_face_presence_confidence == 0.5
    assert len(FakeFaceLandmarker.detect_calls) == 1
    detected_image = FakeFaceLandmarker.detect_calls[0]
    assert isinstance(detected_image, FakeImage)
    assert detected_image.image_format == "srgb"
    assert landmarks == [(100, 100), (300, 20)]


def test_face_analyzer_caches_tasks_landmarker(monkeypatch):
    FakeFaceLandmarker.created_options = None
    fake_mediapipe = types.SimpleNamespace(
        **vars(_fake_mediapipe())
    )
    face_analyzer = _reload_face_analyzer(monkeypatch, fake_mediapipe)

    first = face_analyzer.get_face_landmarker()
    second = face_analyzer.get_face_landmarker()

    assert first is second


def test_face_analyzer_returns_none_when_no_face_is_detected(monkeypatch):
    FakeFaceLandmarker.face_landmarks = []
    fake_mediapipe = _fake_mediapipe()
    face_analyzer = _reload_face_analyzer(monkeypatch, fake_mediapipe)

    image = types.SimpleNamespace(shape=(200, 400, 3))
    assert face_analyzer.get_landmarks_mp(image) is None


def test_face_analyzer_reports_missing_task_model(monkeypatch, tmp_path):
    fake_mediapipe = _fake_mediapipe()
    face_analyzer = _reload_face_analyzer(monkeypatch, fake_mediapipe)
    monkeypatch.setattr(face_analyzer, "_MODEL_PATH", tmp_path / "missing.task")

    try:
        face_analyzer.get_face_landmarker()
    except FileNotFoundError as exc:
        assert "MediaPipe task model not found" in str(exc)
    else:
        raise AssertionError("Expected a missing task model to raise FileNotFoundError")
