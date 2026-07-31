"""
Features Rating API Wrapper.

Public function:
    analyze_features_from_upload(file_storage, suffix) -> dict
        Takes a Flask FileStorage object and a file suffix (e.g., '.jpg'),
        runs the beauty rating deep model, and returns a formatted dict:
          - rawScore (float)
          - score10 (float)
          - summary (str)
          - regions (list of {name, delta, score, effect, polygon})
          - heatmapPng (base64-encoded PNG)

Internal:
    _load_features_rating_analyzer() -> module
        Lazy-loads the analyzer from `features_rating_bundle/face_analyzer.py`.
"""
import base64
import importlib.util
import io
import os
import sys
import tempfile


_BUNDLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features_rating_bundle")
_features_rating_analyzer = None


def analyze_features_from_upload(file_storage, suffix):
    temp_path = None
    try:
        analyzer = _load_features_rating_analyzer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file_storage.save(tmp)
            temp_path = tmp.name
        result = analyzer.analyze_face(temp_path)
        return _format_features_rating_result(result)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _load_features_rating_analyzer():
    global _features_rating_analyzer
    if _features_rating_analyzer is not None:
        return _features_rating_analyzer

    analyzer_path = os.path.join(_BUNDLE_DIR, "face_analyzer.py")
    if not os.path.exists(analyzer_path):
        raise FileNotFoundError(f"Features-rating analyzer not found: {analyzer_path}")

    scut_path = os.path.join(_BUNDLE_DIR, "code", "scut")
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