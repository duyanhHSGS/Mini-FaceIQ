import math
import os

import numpy as np
import pytest
from PIL import Image

from conftest import load_feature_analyzer


face_analyzer = load_feature_analyzer()


# @pytest.mark.skipif(
#     os.environ.get("FEATURES_RATING_REAL_SMOKE") != "1",
#     reason="Set FEATURES_RATING_REAL_SMOKE=1 to run the heavyweight checkpoint smoke test",
# )
def test_real_checkpoint_smoke(tmp_path):
    image_path = tmp_path / "face.png"
    Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8)).save(image_path)
    result = face_analyzer.analyze_face(str(image_path))
    assert {"score", "score_10", "heatmap", "deltas", "summary", "region_polygons"} <= result.keys()
    assert math.isfinite(result["score"])
    assert 0.0 <= result["score_10"] <= 10.0
    assert all(math.isfinite(value) for value in result["deltas"].values())
