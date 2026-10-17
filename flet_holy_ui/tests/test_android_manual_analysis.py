"""Android-ready manual geometry path uses the exact desktop calculators."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
try:
    from local_engine import AnalysisError, analyze_manual_landmarks

    from front_landmarks import FRONT_LANDMARK_DEFS
    from side_landmarks import SIDE_LANDMARK_DEFS
finally:
    sys.path.remove(str(SRC))


def _png() -> bytes:
    return (
        bytes.fromhex("89504e470d0a1a0a") + bytes(8)
        + (200).to_bytes(4, "big") + (400).to_bytes(4, "big")
    )


def _points(definitions: list[dict]) -> list[dict]:
    return [
        {"id": definition["id"], "x": (index + 1) / (len(definitions) + 2),
         "y": ((index * 7) % len(definitions) + 1) / (len(definitions) + 2)}
        for index, definition in enumerate(definitions)
    ]


class AndroidManualAnalysisTests(unittest.TestCase):
    @patch("local_engine.front_metrics", return_value={"overallScore": 7.3})
    def test_front_calculates_without_detector(self, metrics):
        points = _points(FRONT_LANDMARK_DEFS)
        result = analyze_manual_landmarks("front", _png(), points, "female", "mixed")
        self.assertEqual(result["data"]["overallScore"], 7.3)
        self.assertEqual(result["landmark_count"], len(points))
        metrics.assert_called_once_with(points, "female", "mixed", 0.5)

    @patch("local_engine.side_metrics", return_value={"overallScore": 6.1})
    def test_side_calculates_without_detector(self, metrics):
        points = _points(SIDE_LANDMARK_DEFS)
        result = analyze_manual_landmarks("side", _png(), points)
        self.assertEqual(result["data"]["overallScore"], 6.1)
        metrics.assert_called_once_with(points, "male", "asian", 0.5)

    @patch("local_engine.front_metrics")
    def test_incomplete_input_never_scores(self, metrics):
        result = analyze_manual_landmarks("front", _png(), _points(FRONT_LANDMARK_DEFS)[:2])
        self.assertIsNone(result["data"])
        self.assertGreater(len(result["missing_ids"]), 0)
        metrics.assert_not_called()

    def test_out_of_range_or_nonfinite_points_are_rejected(self):
        points = _points(FRONT_LANDMARK_DEFS)
        points[0]["x"] = float("nan")
        points[1]["y"] = 1.5
        result = analyze_manual_landmarks("front", _png(), points)
        self.assertIsNone(result["data"])
        self.assertEqual(len(result["missing_ids"]), 2)

    def test_rejects_unsupported_modes(self):
        with self.assertRaises(AnalysisError):
            analyze_manual_landmarks("features", _png(), [])


if __name__ == "__main__":
    unittest.main()
