"""Guard against cached neural sessions lingering between Android analysis jobs."""

from __future__ import annotations

import gc
import io
import sys
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
try:
    import mobile_features
    import mobile_side_core
    import mobile_side_provider
finally:
    sys.path.remove(str(SRC))


def _image() -> bytes:
    data = io.BytesIO()
    Image.new("RGB", (280, 350), (110, 90, 80)).save(data, format="PNG")
    return data.getvalue()


class ModelLifecycleTests(unittest.TestCase):
    def test_side_models_are_new_for_each_analysis(self) -> None:
        with (
            patch.object(mobile_side_core, "MobileFaceBoxes", side_effect=object),
            patch.object(mobile_side_core, "MobileTDDFA", side_effect=object),
        ):
            first = mobile_side_core.get_models()
            second = mobile_side_core.get_models()
        self.assertIsNot(first[0], second[0])
        self.assertIsNot(first[1], second[1])
        self.assertFalse(hasattr(mobile_side_core.get_models, "cache_info"))

    def test_side_releases_models_on_early_no_face_result(self) -> None:
        references: list[weakref.ReferenceType] = []

        class NoFaceModel:
            def __init__(self) -> None:
                references.append(weakref.ref(self))

            def __call__(self, _image):
                return []

        class OtherModel:
            def __init__(self) -> None:
                references.append(weakref.ref(self))

        with patch.object(
            mobile_side_provider, "_load_models",
            side_effect=lambda: (NoFaceModel(), OtherModel()),
        ):
            response = mobile_side_provider._detect_side_from_image(
                np.zeros((200, 200, 3), dtype=np.uint8)
            )

        self.assertEqual(response["error"], "No face detected")
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_scut_keeps_one_session_for_nine_passes_then_releases_it(self) -> None:
        references: list[weakref.ReferenceType] = []
        run_counts: list[int] = []

        class Session:
            def __init__(self) -> None:
                self.count = 0
                references.append(weakref.ref(self))

            def run(self, _outputs, _inputs):
                self.count += 1
                return [np.asarray([[2.0]], dtype=np.float32)]

            def __del__(self) -> None:
                run_counts.append(self.count)

        with patch.object(mobile_features, "_model_session", side_effect=Session):
            result = mobile_features.analyze_mobile_features(_image(), [])

        self.assertEqual(len(result["regions"]), 8)
        gc.collect()
        self.assertEqual(run_counts, [9])
        self.assertIsNone(references[0]())

    def test_scut_session_released_when_inference_raises(self) -> None:
        references: list[weakref.ReferenceType] = []

        class FailingSession:
            def __init__(self) -> None:
                references.append(weakref.ref(self))

            def run(self, _outputs, _inputs):
                raise RuntimeError("inference exploded")

        with (
            patch.object(mobile_features, "_model_session", side_effect=FailingSession),
            self.assertRaisesRegex(RuntimeError, "inference exploded"),
        ):
            mobile_features.analyze_mobile_features(_image(), [])

        gc.collect()
        self.assertIsNone(references[0]())


if __name__ == "__main__":
    unittest.main()
