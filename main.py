import os
import base64
import importlib.util
import io
import sys
import tempfile

from flask import Flask, jsonify, render_template, request

from front_autolandmarks import detect_front_landmarks_from_upload
from front_calculator import calculate_front_analysis
from front_landmarks import FRONT_LANDMARK_DEFS
from side_autolandmarks import detect_side_landmarks_from_upload
from side_calculator import calculate_side_analysis
from side_landmarks import SIDE_LANDMARK_DEFS


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_RATING_DIR = os.path.join(ROOT_DIR, "features_rating")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
_features_rating_analyzer = None

app = Flask(__name__, template_folder="web")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _load_features_rating_analyzer():
    global _features_rating_analyzer
    if _features_rating_analyzer is not None:
        return _features_rating_analyzer

    analyzer_path = os.path.join(FEATURES_RATING_DIR, "face_analyzer.py")
    if not os.path.exists(analyzer_path):
        raise FileNotFoundError(f"Features-rating analyzer not found: {analyzer_path}")

    scut_path = os.path.join(FEATURES_RATING_DIR, "code", "scut")
    if scut_path not in sys.path:
        sys.path.insert(0, scut_path)

    spec = importlib.util.spec_from_file_location("features_rating_analyzer", analyzer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Features-rating analyzer from {analyzer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _features_rating_analyzer = module
    return module


def _feature_label(delta):
    if delta > 0.003:
        return "helps prediction"
    if delta < -0.003:
        return "lowers prediction"
    return "neutral"


def _format_features_rating_result(result):
    heatmap_buffer = io.BytesIO()
    result["heatmap"].save(heatmap_buffer, format="PNG")
    deltas = result.get("deltas", {})
    regions = []
    for name, delta in sorted(deltas.items(), key=lambda item: item[1], reverse=True):
        regions.append(
            {
                "name": name,
                "delta": round(float(delta), 4),
                "score": round(max(0, min(100, 50 + float(delta) * 200)), 1),
                "effect": _feature_label(float(delta)),
                "polygon": result.get("region_polygons", {}).get(name, []),
            }
        )
    return {
        "rawScore": round(float(result["score"]), 4),
        "score10": round(float(result["score_10"]), 2),
        "summary": result.get("summary", ""),
        "regions": regions,
        "heatmapPng": "data:image/png;base64," + base64.b64encode(heatmap_buffer.getvalue()).decode("ascii"),
    }


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
    temp_path = None
    try:
        analyzer = _load_features_rating_analyzer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            image.save(tmp)
            temp_path = tmp.name
        result = analyzer.analyze_face(temp_path)
        return jsonify({"success": True, "data": _format_features_rating_result(result)})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


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
