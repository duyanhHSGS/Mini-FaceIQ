"""Fast offline tests: patch ML functions rather than load large model weights."""

import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_engine import (
    MAX_IMAGE_BYTES,
    AnalysisError,
    _MemoryUpload,
    analyze_photo,
    front_autolandmarks,
    image_aspect_ratio,
    side_autolandmarks,
)


class LocalEngineTests(unittest.TestCase):
    def test_upload_adapter_works_without_flask(self):
        upload = _MemoryUpload(b"raw photo", "face.jpg")
        self.assertEqual(upload.read(), b"raw photo")
        stream = io.BytesIO()
        upload.save(stream)
        self.assertEqual(stream.getvalue(), b"raw photo")

    def test_auto_landmark_wrappers_call_python_functions_not_flask(self):
        fake_packages = types.ModuleType("packages")
        fake_packages.detect_front_landmarks_from_upload = Mock(return_value=[{"id": "front"}])
        fake_packages.detect_side_landmarks_from_upload = Mock(return_value=[{"id": "side"}])
        with patch.dict(sys.modules, {"packages": fake_packages}):
            self.assertEqual(front_autolandmarks(b"front image", "front.png"), [{"id": "front"}])
            self.assertEqual(side_autolandmarks(b"side image", "side.png"), [{"id": "side"}])
        front_upload = fake_packages.detect_front_landmarks_from_upload.call_args.args[0]
        side_upload = fake_packages.detect_side_landmarks_from_upload.call_args.args[0]
        self.assertEqual(front_upload.read(), b"front image")
        self.assertEqual(side_upload.read(), b"side image")

    @patch("local_engine.front_metrics", return_value={"frontScore": 8.2})
    @patch("local_engine._missing_landmarks", return_value=[])
    @patch("local_engine.front_autolandmarks", return_value=[{"id": "eye", "x": 0.5, "y": 0.5}])
    def test_front_analysis_direct(self, detect, missing, metrics):
        png = (
            bytes.fromhex("89504e470d0a1a0a") + bytes(8)
            + (300).to_bytes(4, "big") + (600).to_bytes(4, "big")
        )
        result = analyze_photo("front", "portrait.png", png)
        self.assertEqual(result["data"]["frontScore"], 8.2)
        self.assertEqual(result["landmark_count"], 1)
        detect.assert_called_once_with(png, "portrait.png")
        missing.assert_called_once()
        self.assertEqual(metrics.call_args.args[3], 0.5)

    @patch("local_engine.side_metrics")
    @patch("local_engine._missing_landmarks", return_value=["tragus", "gonion"])
    @patch("local_engine.side_autolandmarks", return_value=[{"id": "eye", "x": 0.5, "y": 0.5}])
    def test_missing_landmarks_prevents_wrong_scores(self, _detect, _missing, metrics):
        result = analyze_photo("side", "side.jpg", b"photo")
        self.assertIsNone(result["data"])
        self.assertEqual(result["missing_ids"], ["tragus", "gonion"])
        metrics.assert_not_called()

    @patch("local_engine.features_rating", return_value={"rating": 9.1})
    def test_feature_rating_direct(self, features):
        result = analyze_photo("features", "face.webp", b"photo")
        self.assertEqual(result["data"], {"rating": 9.1})
        features.assert_called_once_with(b"photo", "face.webp")

    def test_png_jpeg_webp_dimensions(self):
        png = (
            bytes.fromhex("89504e470d0a1a0a") + bytes(8)
            + (200).to_bytes(4, "big") + (400).to_bytes(4, "big")
        )
        jpg = bytes.fromhex("ffd8ffc0000b08006400c801010111")
        webp = (
            b"RIFF" + bytes(4) + b"WEBP"
            + b"VP8X" + (10).to_bytes(4, "little") + bytes(4)
            + (199).to_bytes(3, "little") + (99).to_bytes(3, "little")
        )
        self.assertEqual(image_aspect_ratio(png), 0.5)
        self.assertEqual(image_aspect_ratio(jpg), 2.0)
        self.assertEqual(image_aspect_ratio(webp), 2.0)
        self.assertEqual(image_aspect_ratio(b"not an image"), 1.0)

    def test_input_validation(self):
        for mode, name, data in (
            ("invalid", "a.jpg", b"ok"),
            ("front", "a.txt", b"ok"),
            ("front", "a.png", b""),
            ("front", "a.png", b"x" * (MAX_IMAGE_BYTES + 1)),
        ):
            with self.subTest(mode=mode, name=name), self.assertRaises(AnalysisError):
                analyze_photo(mode, name, data)


if __name__ == "__main__":
    unittest.main()
