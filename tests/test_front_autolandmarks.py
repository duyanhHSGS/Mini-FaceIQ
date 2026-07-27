import types
from io import BytesIO

import third_party.adapter_front as front_autolandmarks
from front_landmarks import FRONT_LANDMARK_DEFS


class FakeImage:
    shape = (100, 200, 3)


def _points(count, x=50, y=25):
    return [(x, y) for _ in range(count)]


def test_direct_point_rejects_missing_and_out_of_range_indices():
    points = [(10, 20)]

    assert front_autolandmarks._direct_point(points, None) is None
    assert front_autolandmarks._direct_point(points, 5) is None
    assert front_autolandmarks._direct_point(points, 0) == (10, 20)


def test_average_point_uses_available_indices_only():
    points = [(0, 0), (10, 20), (30, 40)]

    assert front_autolandmarks._average_point(points, [1, 2, 99]) == (20, 30)
    assert front_autolandmarks._average_point(points, [99]) is None


def test_detect_front_landmarks_maps_direct_and_average_points(monkeypatch):
    detected = _points(476)
    detected[10] = (-20, 10)
    detected[468] = (120, 40)
    detected[105] = (10, 20)
    detected[63] = (30, 60)

    fake_cv2 = types.SimpleNamespace(
        IMREAD_COLOR=1,
        imdecode=lambda data, mode: FakeImage(),
    )
    fake_numpy = types.SimpleNamespace(
        uint8="uint8",
        frombuffer=lambda data, dtype: data,
    )

    monkeypatch.setattr(front_autolandmarks, "cv2", fake_cv2)
    monkeypatch.setattr(front_autolandmarks, "np", fake_numpy)
    monkeypatch.setattr(front_autolandmarks.front_mediapipe, "get_landmarks_mp", lambda image: detected)

    landmarks = front_autolandmarks.detect_front_landmarks_from_upload(BytesIO(b"image bytes"))
    by_id = {item["id"]: item for item in landmarks}

    assert len(landmarks) == len(FRONT_LANDMARK_DEFS)
    assert by_id["hairline"]["x"] == 0.0
    assert by_id["hairline"]["y"] == 0.1
    assert by_id["left_pupil"]["x"] == 0.6
    assert by_id["left_pupil"]["y"] == 0.4
    assert by_id["left_brow_peak"]["x"] == 0.1
    assert by_id["left_brow_peak"]["y"] == 0.4
    assert by_id["left_pupil"]["label"] == "Left Pupil"


def test_detect_front_landmarks_skips_points_that_need_unavailable_indices(monkeypatch):
    detected = _points(20)
    fake_cv2 = types.SimpleNamespace(
        IMREAD_COLOR=1,
        imdecode=lambda data, mode: FakeImage(),
    )
    fake_numpy = types.SimpleNamespace(
        uint8="uint8",
        frombuffer=lambda data, dtype: data,
    )

    monkeypatch.setattr(front_autolandmarks, "cv2", fake_cv2)
    monkeypatch.setattr(front_autolandmarks, "np", fake_numpy)
    monkeypatch.setattr(front_autolandmarks.front_mediapipe, "get_landmarks_mp", lambda image: detected)

    landmarks = front_autolandmarks.detect_front_landmarks_from_upload(BytesIO(b"image bytes"))
    ids = {item["id"] for item in landmarks}

    assert "hairline" in ids
    assert "left_pupil" not in ids
    assert "right_pupil" not in ids


def test_detect_front_landmarks_raises_clear_errors(monkeypatch):
    fake_numpy = types.SimpleNamespace(
        uint8="uint8",
        frombuffer=lambda data, dtype: data,
    )
    monkeypatch.setattr(front_autolandmarks, "np", fake_numpy)
    monkeypatch.setattr(front_autolandmarks.front_mediapipe, "get_landmarks_mp", lambda image: [])

    monkeypatch.setattr(front_autolandmarks, "cv2", types.SimpleNamespace(IMREAD_COLOR=1, imdecode=lambda data, mode: None))

    try:
        front_autolandmarks.detect_front_landmarks_from_upload(BytesIO(b"bad image"))
    except ValueError as exc:
        assert str(exc) == "Could not read image file"
    else:
        raise AssertionError("Expected unreadable image to raise ValueError")

    monkeypatch.setattr(front_autolandmarks, "cv2", types.SimpleNamespace(IMREAD_COLOR=1, imdecode=lambda data, mode: FakeImage()))
    monkeypatch.setattr(front_autolandmarks.front_mediapipe, "get_landmarks_mp", lambda image: None)

    try:
        front_autolandmarks.detect_front_landmarks_from_upload(BytesIO(b"face image"))
    except ValueError as exc:
        assert str(exc) == "No face detected"
    else:
        raise AssertionError("Expected missing face to raise ValueError")
