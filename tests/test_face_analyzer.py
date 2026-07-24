import importlib
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def clean_face_analyzer_module():
    sys.modules.pop("face_analyzer", None)
    yield
    sys.modules.pop("face_analyzer", None)


class FakeImage:
    shape = (200, 400, 3)


class FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class FakeFaceLandmarks:
    landmark = [FakeLandmark(0.25, 0.5), FakeLandmark(0.75, 0.1)]


class FakeFaceMesh:
    created_with = None
    process_calls = []

    def __init__(self, **kwargs):
        type(self).created_with = kwargs

    def process(self, image):
        type(self).process_calls.append(image)
        return types.SimpleNamespace(multi_face_landmarks=[FakeFaceLandmarks()])


def _reload_face_analyzer(monkeypatch, fake_mediapipe):
    fake_cv2 = types.SimpleNamespace(
        COLOR_BGR2RGB="bgr_to_rgb",
        cvtColor=lambda image, mode: ("rgb", image, mode),
    )

    sys.modules.pop("face_analyzer", None)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mediapipe)
    return importlib.import_module("face_analyzer")


def test_face_analyzer_uses_solutions_face_mesh_without_task_file(monkeypatch):
    FakeFaceMesh.created_with = None
    FakeFaceMesh.process_calls = []
    fake_mediapipe = types.SimpleNamespace(
        solutions=types.SimpleNamespace(
            face_mesh=types.SimpleNamespace(FaceMesh=FakeFaceMesh)
        )
    )

    face_analyzer = _reload_face_analyzer(monkeypatch, fake_mediapipe)

    image = FakeImage()
    landmarks = face_analyzer.get_landmarks_mp(image)

    assert FakeFaceMesh.created_with == {
        "static_image_mode": True,
        "max_num_faces": 1,
        "refine_landmarks": True,
        "min_detection_confidence": 0.5,
    }
    assert FakeFaceMesh.process_calls == [("rgb", image, "bgr_to_rgb")]
    assert landmarks == [(100, 100), (300, 20)]
    assert not hasattr(face_analyzer, "_MODEL_PATH")


def test_face_analyzer_caches_face_mesh_instance(monkeypatch):
    FakeFaceMesh.created_with = None
    fake_mediapipe = types.SimpleNamespace(
        solutions=types.SimpleNamespace(
            face_mesh=types.SimpleNamespace(FaceMesh=FakeFaceMesh)
        )
    )
    face_analyzer = _reload_face_analyzer(monkeypatch, fake_mediapipe)

    first = face_analyzer.get_face_mesh()
    second = face_analyzer.get_face_mesh()

    assert first is second


def test_face_analyzer_returns_none_when_no_face_is_detected(monkeypatch):
    class EmptyFaceMesh:
        def __init__(self, **kwargs):
            pass

        def process(self, image):
            return types.SimpleNamespace(multi_face_landmarks=[])

    fake_mediapipe = types.SimpleNamespace(
        solutions=types.SimpleNamespace(
            face_mesh=types.SimpleNamespace(FaceMesh=EmptyFaceMesh)
        )
    )
    face_analyzer = _reload_face_analyzer(monkeypatch, fake_mediapipe)

    assert face_analyzer.get_landmarks_mp(FakeImage()) is None


def test_face_analyzer_exposes_mediapipe_install_shape_bug(monkeypatch):
    fake_mediapipe = types.SimpleNamespace()
    face_analyzer = _reload_face_analyzer(monkeypatch, fake_mediapipe)

    try:
        face_analyzer.get_face_mesh()
    except AttributeError as exc:
        assert "solutions" in str(exc)
    else:
        raise AssertionError("Expected missing mediapipe.solutions to raise AttributeError")
