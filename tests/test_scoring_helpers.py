from front_calculator import (
    SCORING_FLOOR,
    calculate_plateau_gaussian_score,
    classify_deviation,
    clamp_score,
)
from front_landmarks import normalize_front_landmarks


def test_plateau_gaussian_score_is_perfect_inside_ideal_range():
    assert calculate_plateau_gaussian_score(5, 0, 10, 4, 6) == 10.0


def test_plateau_gaussian_score_uses_floor_outside_scored_range():
    assert calculate_plateau_gaussian_score(-1, 0, 10, 4, 6) == SCORING_FLOOR
    assert calculate_plateau_gaussian_score(11, 0, 10, 4, 6) == SCORING_FLOOR


def test_classify_deviation_marks_low_ideal_and_high_values():
    assert classify_deviation(3, 4, 6) == "low"
    assert classify_deviation(5, 4, 6) == "ideal"
    assert classify_deviation(7, 4, 6) == "high"


def test_clamp_score_keeps_score_between_zero_and_ten():
    assert clamp_score(-5) == 0
    assert clamp_score(4.567) == 4.57
    assert clamp_score(15) == 10


def test_normalize_front_landmarks_skips_bad_items():
    normalized = normalize_front_landmarks(
        [
            {"id": "left_pupil", "x": "0.4", "y": "0.5", "label": "Left Pupil"},
            {"id": "right_pupil", "x": "bad", "y": "0.5"},
            {"x": 0.1, "y": 0.2},
        ]
    )

    assert normalized == {
        "left_pupil": {"id": "left_pupil", "x": 0.4, "y": 0.5, "label": "Left Pupil"}
    }

