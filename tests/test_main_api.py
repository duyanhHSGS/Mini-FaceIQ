from io import BytesIO

import main
from front_landmarks import FRONT_LANDMARK_DEFS
from side_landmarks import SIDE_LANDMARK_DEFS


def _placed_landmarks(defs):
    return [
        {"id": item["id"], "x": 0.25 + (index % 7) * 0.06, "y": 0.2 + (index // 7) * 0.08}
        for index, item in enumerate(defs)
    ]


def test_allowed_file_accepts_supported_image_extensions():
    assert main.allowed_file("face.jpg")
    assert main.allowed_file("face.JPEG")
    assert main.allowed_file("face.png")
    assert main.allowed_file("face.webp")
    assert not main.allowed_file("face.gif")
    assert not main.allowed_file("face")


def test_landmark_endpoints_return_front_and_side_definitions():
    client = main.app.test_client()

    front_response = client.get("/api/front-landmarks")
    side_response = client.get("/api/side-landmarks")

    assert front_response.status_code == 200
    assert side_response.status_code == 200
    assert front_response.get_json()["landmarks"] == FRONT_LANDMARK_DEFS
    assert side_response.get_json()["landmarks"] == SIDE_LANDMARK_DEFS


def test_front_metrics_reports_missing_landmarks():
    client = main.app.test_client()

    response = client.post(
        "/api/front-metrics",
        json={"landmarks": [{"id": FRONT_LANDMARK_DEFS[0]["id"], "x": 0.5, "y": 0.5}]},
    )
    payload = response.get_json()

    assert response.status_code == 400
    assert payload["success"] is False
    assert FRONT_LANDMARK_DEFS[0]["id"] not in payload["missing"]
    assert len(payload["missing"]) == len(FRONT_LANDMARK_DEFS) - 1


def test_side_metrics_ignores_unknown_and_invalid_landmarks():
    client = main.app.test_client()

    response = client.post(
        "/api/side-metrics",
        json={
            "landmarks": [
                {"id": SIDE_LANDMARK_DEFS[0]["id"], "x": "nope", "y": 0.5},
                {"id": "not_a_real_landmark", "x": 0.5, "y": 0.5},
            ]
        },
    )
    payload = response.get_json()

    assert response.status_code == 400
    assert payload["success"] is False
    assert len(payload["missing"]) == len(SIDE_LANDMARK_DEFS)


def test_front_metrics_calls_calculator_with_full_payload(monkeypatch):
    client = main.app.test_client()
    captured = {}

    def fake_calculate_front_analysis(landmarks, gender, ethnicity, front_aspect):
        captured["landmarks"] = landmarks
        captured["gender"] = gender
        captured["ethnicity"] = ethnicity
        captured["front_aspect"] = front_aspect
        return {"frontScore": 8.75, "frontMeasurements": []}

    monkeypatch.setattr(main, "calculate_front_analysis", fake_calculate_front_analysis)

    response = client.post(
        "/api/front-metrics",
        json={
            "gender": "female",
            "ethnicity": "mixed",
            "frontAspect": "1.25",
            "landmarks": _placed_landmarks(FRONT_LANDMARK_DEFS),
        },
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload == {"success": True, "data": {"frontScore": 8.75, "frontMeasurements": []}}
    assert captured["gender"] == "female"
    assert captured["ethnicity"] == "mixed"
    assert captured["front_aspect"] == 1.25
    assert len(captured["landmarks"]) == len(FRONT_LANDMARK_DEFS)


def test_front_metrics_uses_default_demographics_and_aspect(monkeypatch):
    client = main.app.test_client()
    captured = {}

    def fake_calculate_front_analysis(landmarks, gender, ethnicity, front_aspect):
        captured["gender"] = gender
        captured["ethnicity"] = ethnicity
        captured["front_aspect"] = front_aspect
        return {"frontScore": 6.5, "frontMeasurements": []}

    monkeypatch.setattr(main, "calculate_front_analysis", fake_calculate_front_analysis)

    response = client.post(
        "/api/front-metrics",
        json={"landmarks": _placed_landmarks(FRONT_LANDMARK_DEFS)},
    )

    assert response.status_code == 200
    assert captured == {"gender": "male", "ethnicity": "asian", "front_aspect": 1.0}


def test_front_metrics_returns_calculator_errors_as_json(monkeypatch):
    client = main.app.test_client()

    def fake_calculate_front_analysis(landmarks, gender, ethnicity, front_aspect):
        raise RuntimeError("math exploded")

    monkeypatch.setattr(main, "calculate_front_analysis", fake_calculate_front_analysis)

    response = client.post(
        "/api/front-metrics",
        json={"landmarks": _placed_landmarks(FRONT_LANDMARK_DEFS)},
    )

    assert response.status_code == 500
    assert response.get_json() == {"success": False, "error": "math exploded"}


def test_side_metrics_calls_calculator_with_full_payload(monkeypatch):
    client = main.app.test_client()
    captured = {}

    def fake_calculate_side_analysis(landmarks, gender, ethnicity, side_aspect):
        captured["landmarks"] = landmarks
        captured["gender"] = gender
        captured["ethnicity"] = ethnicity
        captured["side_aspect"] = side_aspect
        return {"sideScore": 7.5, "sideMeasurements": []}

    monkeypatch.setattr(main, "calculate_side_analysis", fake_calculate_side_analysis)

    response = client.post(
        "/api/side-metrics",
        json={
            "gender": "male",
            "ethnicity": "south_asian",
            "sideAspect": "0.8",
            "landmarks": _placed_landmarks(SIDE_LANDMARK_DEFS),
        },
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload == {"success": True, "data": {"sideScore": 7.5, "sideMeasurements": []}}
    assert captured["gender"] == "male"
    assert captured["ethnicity"] == "south_asian"
    assert captured["side_aspect"] == 0.8
    assert len(captured["landmarks"]) == len(SIDE_LANDMARK_DEFS)


def test_side_metrics_uses_default_demographics_and_aspect(monkeypatch):
    client = main.app.test_client()
    captured = {}

    def fake_calculate_side_analysis(landmarks, gender, ethnicity, side_aspect):
        captured["gender"] = gender
        captured["ethnicity"] = ethnicity
        captured["side_aspect"] = side_aspect
        return {"sideScore": 5.5, "sideMeasurements": []}

    monkeypatch.setattr(main, "calculate_side_analysis", fake_calculate_side_analysis)

    response = client.post(
        "/api/side-metrics",
        json={"landmarks": _placed_landmarks(SIDE_LANDMARK_DEFS)},
    )

    assert response.status_code == 200
    assert captured == {"gender": "male", "ethnicity": "asian", "side_aspect": 1.0}


def test_side_metrics_returns_calculator_errors_as_json(monkeypatch):
    client = main.app.test_client()

    def fake_calculate_side_analysis(landmarks, gender, ethnicity, side_aspect):
        raise RuntimeError("profile math exploded")

    monkeypatch.setattr(main, "calculate_side_analysis", fake_calculate_side_analysis)

    response = client.post(
        "/api/side-metrics",
        json={"landmarks": _placed_landmarks(SIDE_LANDMARK_DEFS)},
    )

    assert response.status_code == 500
    assert response.get_json() == {"success": False, "error": "profile math exploded"}


def test_front_autolandmarks_rejects_unsupported_upload_extension():
    client = main.app.test_client()

    response = client.post(
        "/api/front-autolandmarks",
        data={"image": (BytesIO(b"not an image"), "face.gif")},
        content_type="multipart/form-data",
    )
    payload = response.get_json()

    assert response.status_code == 400
    assert payload == {"success": False, "error": "Use jpg, jpeg, png, or webp"}


def test_front_autolandmarks_rejects_missing_upload():
    client = main.app.test_client()

    response = client.post("/api/front-autolandmarks", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.get_json() == {"success": False, "error": "No image file uploaded"}


def test_front_autolandmarks_returns_detector_errors_as_json(monkeypatch):
    client = main.app.test_client()

    def fake_detect_front_landmarks_from_upload(image):
        raise ValueError("No face detected")

    monkeypatch.setattr(main, "detect_front_landmarks_from_upload", fake_detect_front_landmarks_from_upload)

    response = client.post(
        "/api/front-autolandmarks",
        data={"image": (BytesIO(b"pretend image bytes"), "face.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 500
    assert response.get_json() == {"success": False, "error": "No face detected"}


def test_front_autolandmarks_returns_detector_output(monkeypatch):
    client = main.app.test_client()
    detected = [{"id": "left_pupil", "x": 0.4, "y": 0.45, "label": "Left Pupil"}]

    monkeypatch.setattr(main, "detect_front_landmarks_from_upload", lambda image: detected)

    response = client.post(
        "/api/front-autolandmarks",
        data={"image": (BytesIO(b"pretend image bytes"), "face.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json() == {"success": True, "landmarks": detected}


def test_side_autolandmarks_rejects_unsupported_upload_extension():
    client = main.app.test_client()

    response = client.post(
        "/api/side-autolandmarks",
        data={"image": (BytesIO(b"not an image"), "face.gif")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json() == {"success": False, "error": "Use jpg, jpeg, png, or webp"}


def test_side_autolandmarks_rejects_missing_upload():
    client = main.app.test_client()

    response = client.post("/api/side-autolandmarks", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.get_json() == {"success": False, "error": "No image file uploaded"}


def test_side_autolandmarks_returns_detector_errors_as_json(monkeypatch):
    client = main.app.test_client()

    def fake_detect_side_landmarks_from_upload(image):
        raise ValueError("No face detected")

    monkeypatch.setattr(main, "detect_side_landmarks_from_upload", fake_detect_side_landmarks_from_upload)

    response = client.post(
        "/api/side-autolandmarks",
        data={"image": (BytesIO(b"pretend image bytes"), "face.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 500
    assert response.get_json() == {"success": False, "error": "No face detected"}


def test_side_autolandmarks_returns_detector_output(monkeypatch):
    client = main.app.test_client()
    detected = [{"id": "nose_tip", "x": 0.55, "y": 0.32, "label": "Nose Tip"}]

    monkeypatch.setattr(main, "detect_side_landmarks_from_upload", lambda image: detected)

    response = client.post(
        "/api/side-autolandmarks",
        data={"image": (BytesIO(b"pretend image bytes"), "face.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json() == {"success": True, "landmarks": detected}


def test_features_rating_uses_local_bundle_path():
    import os

    import third_party.features_rating as feature_adapter

    assert feature_adapter._BUNDLE_DIR.endswith(os.path.join("third_party", "features_rating_bundle"))


def test_features_rating_endpoint_formats_analyzer_output(monkeypatch):
    client = main.app.test_client()

    formatted = {
        "rawScore": 3.0,
        "score10": 5.0,
        "summary": "SCORE: 5.0/10",
        "regions": [{"name": "Nose", "polygon": [{"x": 0.5, "y": 0.4}]}],
        "heatmapPng": "data:image/png;base64,abc",
    }

    monkeypatch.setattr(main, "analyze_features_from_upload", lambda image, suffix: formatted)

    response = client.post(
        "/api/features-rating",
        data={"image": (BytesIO(b"pretend image bytes"), "face.png")},
        content_type="multipart/form-data",
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    data = payload["data"]
    assert data["rawScore"] == 3.0
    assert data["score10"] == 5.0
    assert data["summary"] == "SCORE: 5.0/10"
    assert data["heatmapPng"].startswith("data:image/png;base64,")
    assert data["regions"][0]["name"] == "Nose"
    assert data["regions"][0]["polygon"] == [{"x": 0.5, "y": 0.4}]
