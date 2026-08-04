from flask import Flask, jsonify, render_template, request

from front_calculator import calculate_front_analysis
from front_landmarks import FRONT_LANDMARK_DEFS
from side_calculator import calculate_side_analysis
from side_landmarks import SIDE_LANDMARK_DEFS
from packages import (
    analyze_features_from_upload,
    detect_front_landmarks_from_upload,
    detect_side_landmarks_from_upload,
)


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

app = Flask(__name__, template_folder="web")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/front-landmarks")
def front_landmarks():
    return jsonify({"success": True, "landmarks": FRONT_LANDMARK_DEFS})


@app.route("/api/side-landmarks")
def side_landmarks():
    return jsonify({"success": True, "landmarks": SIDE_LANDMARK_DEFS})


@app.route("/api/front-autolandmarks", methods=["POST"])
def front_autolandmarks():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file uploaded"}), 400

    image = request.files["image"]
    if not image.filename:
        return jsonify({"success": False, "error": "No selected file"}), 400

    if not allowed_file(image.filename):
        return jsonify({"success": False, "error": "Use jpg, jpeg, png, or webp"}), 400

    try:
        landmarks = detect_front_landmarks_from_upload(image)
        return jsonify({"success": True, "landmarks": landmarks})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/side-autolandmarks", methods=["POST"])
def side_autolandmarks():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file uploaded"}), 400

    image = request.files["image"]
    if not image.filename:
        return jsonify({"success": False, "error": "No selected file"}), 400

    if not allowed_file(image.filename):
        return jsonify({"success": False, "error": "Use jpg, jpeg, png, or webp"}), 400

    try:
        landmarks = detect_side_landmarks_from_upload(image)
        return jsonify({"success": True, "landmarks": landmarks})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/features-rating", methods=["POST"])
def features_rating():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file uploaded"}), 400

    image = request.files["image"]
    if not image.filename:
        return jsonify({"success": False, "error": "No selected file"}), 400

    if not allowed_file(image.filename):
        return jsonify({"success": False, "error": "Use jpg, jpeg, png, or webp"}), 400

    suffix = "." + image.filename.rsplit(".", 1)[1].lower()
    try:
        result = analyze_features_from_upload(image, suffix)
        return jsonify({"success": True, "data": result})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


def _has_valid_coordinates(item):
    try:
        float(item.get("x"))
        float(item.get("y"))
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _missing_landmark_ids(landmarks, landmark_defs):
    required_ids = {item["id"] for item in landmark_defs}
    placed_ids = {
        item.get("id")
        for item in landmarks
        if isinstance(item, dict) and item.get("id") in required_ids and _has_valid_coordinates(item)
    }
    return sorted(required_ids - placed_ids)


@app.route("/api/front-metrics", methods=["POST"])
def front_metrics():
    payload = request.get_json(silent=True) or {}
    landmarks = payload.get("landmarks", [])
    missing_ids = _missing_landmark_ids(landmarks, FRONT_LANDMARK_DEFS)

    if missing_ids:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Place all front landmarks first. Missing {len(missing_ids)}.",
                    "missing": missing_ids,
                }
            ),
            400,
        )

    try:
        data = calculate_front_analysis(
            landmarks,
            gender=payload.get("gender", "male"),
            ethnicity=payload.get("ethnicity", "asian"),
            front_aspect=float(payload.get("frontAspect", 1)),
        )
        return jsonify({"success": True, "data": data})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/side-metrics", methods=["POST"])
def side_metrics():
    payload = request.get_json(silent=True) or {}
    landmarks = payload.get("landmarks", [])
    missing_ids = _missing_landmark_ids(landmarks, SIDE_LANDMARK_DEFS)

    if missing_ids:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Place all side landmarks first. Missing {len(missing_ids)}.",
                    "missing": missing_ids,
                }
            ),
            400,
        )

    try:
        data = calculate_side_analysis(
            landmarks,
            gender=payload.get("gender", "male"),
            ethnicity=payload.get("ethnicity", "asian"),
            side_aspect=float(payload.get("sideAspect", 1)),
        )
        return jsonify({"success": True, "data": data})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
