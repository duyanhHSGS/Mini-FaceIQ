import os
import traceback
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from feature_scorer import analyze
from front_autolandmarks import detect_front_landmarks_from_upload
from front_calculator import calculate_front_analysis
from front_landmarks import FRONT_LANDMARK_DEFS
from side_calculator import calculate_side_analysis
from side_landmarks import SIDE_LANDMARK_DEFS


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(ROOT_DIR, "outputs", "uploads")
HEATMAP_DIR = os.path.join(ROOT_DIR, "outputs", "heatmaps")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

app = Flask(__name__, template_folder="web")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_output_dirs():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(HEATMAP_DIR, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(os.path.join(ROOT_DIR, "outputs"), filename)


@app.route("/api/analyze", methods=["POST"])
def analyze_upload():
    ensure_output_dirs()

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file uploaded"}), 400

    image = request.files["image"]
    if not image.filename:
        return jsonify({"success": False, "error": "No selected file"}), 400

    if not allowed_file(image.filename):
        return jsonify({"success": False, "error": "Use jpg, jpeg, png, or webp"}), 400

    ext = image.filename.rsplit(".", 1)[1].lower()
    safe_stem = os.path.splitext(secure_filename(image.filename))[0] or "face"
    run_id = uuid.uuid4().hex
    upload_name = f"{safe_stem}_{run_id}.{ext}"
    heatmap_name = f"{safe_stem}_{run_id}.png"
    upload_path = os.path.join(UPLOAD_DIR, upload_name)
    heatmap_path = os.path.join(HEATMAP_DIR, heatmap_name)

    try:
        image.save(upload_path)
        data = analyze(upload_path, heatmap_path=heatmap_path)
        data["image_url"] = f"/outputs/uploads/{upload_name}"
        data["heatmap_url"] = f"/outputs/heatmaps/{heatmap_name}"
        return jsonify({"success": True, "data": data})
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc()[:1200],
                }
            ),
            500,
        )


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
    ensure_output_dirs()
    app.run(host="127.0.0.1", port=7860, debug=False)
