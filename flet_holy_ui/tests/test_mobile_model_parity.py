"""Cross-platform parity tests for the original front landmark mapper and SCUT ONNX."""

from __future__ import annotations

import base64
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "flet_holy_ui" / "src"
sys.path.insert(0, str(SRC))
try:
    from local_engine import _MemoryUpload
    from mobile_features import analyze_mobile_features
    from mobile_front import map_mediapipe_front

    from front_landmarks import FRONT_LANDMARK_DEFS
finally:
    sys.path.remove(str(SRC))


def _photo(width: int = 250, height: int = 360) -> bytes:
    y, x = np.indices((height, width), dtype=np.uint8)
    array = np.stack((x, y, (x // 2 + y // 2).astype(np.uint8)), axis=2)
    buf = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


class MobileModelParityTests(unittest.TestCase):
    @patch("packages.adapter_front.front_mediapipe.get_landmarks_mp")
    def test_front_mapping_identical_to_linux(self, original_detector) -> None:
        from packages.adapter_front import detect_front_landmarks_from_upload

        photo = _photo()
        # 478 landmarks includes iris positions 468 and 473.
        coords = [[((i * 37) % 997) / 997, ((i * 61) % 1009) / 1009]
                  for i in range(478)]
        original_detector.return_value = [
            (int(x * 250), int(y * 360)) for x, y in coords
        ]
        desktop = detect_front_landmarks_from_upload(_MemoryUpload(photo, "test.png"))
        portable = map_mediapipe_front(coords, photo)
        self.assertEqual(portable, desktop)
        self.assertEqual(len(portable), len(FRONT_LANDMARK_DEFS))

    def test_exported_scut_onnx_yields_full_region_report(self) -> None:
        response = analyze_mobile_features(_photo(), [])
        self.assertIsInstance(response["rawScore"], float)
        self.assertEqual(len(response["regions"]), 8)
        self.assertTrue(response["heatmapPng"].startswith("data:image/png;base64,"))
        image = Image.open(io.BytesIO(base64.b64decode(
            response["heatmapPng"].partition(",")[2]
        )))
        self.assertEqual(image.size, (224, 244))
        self.assertIn("SCORE:", response["summary"])


if __name__ == "__main__":
    unittest.main()

