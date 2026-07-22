import argparse
import json
import os
import traceback


REGION_ORDER = [
    "Left Eye",
    "Right Eye",
    "L Eyebrow",
    "R Eyebrow",
    "Nose",
    "Mouth",
    "Skin",
    "Hair",
]

FEATURE_CENTERS = {
    "Left Eye": {"x": 0.33, "y": 0.35},
    "Right Eye": {"x": 0.67, "y": 0.35},
    "Nose": {"x": 0.50, "y": 0.48},
    "Mouth": {"x": 0.50, "y": 0.62},
    "L Eyebrow": {"x": 0.33, "y": 0.25},
    "R Eyebrow": {"x": 0.67, "y": 0.25},
    "Skin": {"x": 0.50, "y": 0.52},
    "Hair": {"x": 0.50, "y": 0.10},
}


def _region_group(region_name):
    mapping = {
        "Left Eye": "Eyes",
        "Right Eye": "Eyes",
        "Nose": "Nose",
        "Mouth": "Mouth",
        "L Eyebrow": "Eyebrows",
        "R Eyebrow": "Eyebrows",
        "Skin": "Face Shape",
        "Hair": "Forehead",
    }
    return mapping.get(region_name, region_name)


def format_results(result):
    if result is None:
        return {"error": "No face detected or analysis failed"}

    deltas = result.get("deltas", {})
    score_10 = result.get("score_10", 5.0)
    features = []

    for region_name in REGION_ORDER:
        delta = float(deltas.get(region_name, 0.0))
        center = FEATURE_CENTERS.get(region_name, {"x": 0.5, "y": 0.5})
        norm_score = max(0, min(100, 50 + delta * 200))
        features.append(
            {
                "name": region_name.lower().replace(" ", "_"),
                "display_name": region_name,
                "score": round(norm_score, 1),
                "value": round(delta, 4),
                "region": _region_group(region_name),
                "landmark_center": center,
            }
        )

    return {
        "overall_score": round(score_10 * 10, 1),
        "score_10": round(score_10, 2),
        "score_raw": result.get("score", 0),
        "features": features,
        "summary": result.get("summary", ""),
        "region_polygons": result.get("region_polygons", {}),
    }


def analyze(image_path, heatmap_path=None):
    from face_analyzer import analyze_face

    result = analyze_face(image_path, mat_path=None, cb=None)
    formatted = format_results(result)

    if heatmap_path and result and result.get("heatmap"):
        os.makedirs(os.path.dirname(os.path.abspath(heatmap_path)), exist_ok=True)
        result["heatmap"].save(heatmap_path)
        formatted["heatmap_path"] = heatmap_path

    return formatted


def main():
    parser = argparse.ArgumentParser(description="Mini FaceIQ feature scorer")
    parser.add_argument("image", help="Path to a face image")
    parser.add_argument("--heatmap", help="Optional output path for heatmap PNG")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(json.dumps({"success": False, "error": f"Image not found: {args.image}"}))
        raise SystemExit(1)

    try:
        data = analyze(args.image, heatmap_path=args.heatmap)
        print(json.dumps({"success": True, "data": data}, indent=2))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "success": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc()[:1000],
                },
                indent=2,
            )
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
