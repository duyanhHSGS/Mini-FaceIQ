import importlib.util
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_RATING_ROOT = os.path.join(PROJECT_ROOT, "features_rating")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_feature_analyzer():
    """Load bundled Features-rating analyzer without shadowing mini-faceIQ.face_analyzer."""
    module_name = "features_rating_face_analyzer"
    if module_name in sys.modules:
        return sys.modules[module_name]

    scut_root = os.path.join(FEATURES_RATING_ROOT, "code", "scut")
    if scut_root not in sys.path:
        sys.path.insert(0, scut_root)

    source = os.path.join(FEATURES_RATING_ROOT, "face_analyzer.py")
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Features-rating analyzer from {source}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
