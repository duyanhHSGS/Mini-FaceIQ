from front_calculator import (
    PENALTY_CAP,
    SCORING_FLOOR,
    calculate_penalty,
    calculate_plateau_gaussian_score,
    classify_deviation,
    clamp_score,
    create_measurement,
    weighted_group_score,
)
from front_landmarks import normalize_front_landmarks
from side_landmarks import normalize_side_landmarks


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


def test_normalize_side_landmarks_keeps_latest_valid_duplicate():
    normalized = normalize_side_landmarks(
        [
            {"id": "nose_tip", "x": 0.1, "y": 0.2, "label": "Old Nose Tip"},
            {"id": "nose_tip", "x": "0.3", "y": "0.4", "label": "New Nose Tip"},
            {"id": "chin_point", "x": None, "y": 0.9},
        ]
    )

    assert normalized == {
        "nose_tip": {"id": "nose_tip", "x": 0.3, "y": 0.4, "label": "New Nose Tip"}
    }


def test_weighted_group_score_prioritizes_key_metrics():
    measurements = [
        {"id": "key_metric", "score": 10},
        {"id": "standard_metric", "score": 4},
        {"id": "ignored_metric", "score": 1},
    ]

    score = weighted_group_score(
        measurements,
        key_ids=["key_metric"],
        std_ids=["standard_metric", "missing_standard"],
    )

    assert score == 8


def test_weighted_group_score_returns_zero_without_matching_metrics():
    assert weighted_group_score([{"id": "other", "score": 9}], ["key"], ["std"]) == 0


def test_calculate_penalty_caps_many_low_scores():
    measurements = [{"score": 1.0}, {"score": 1.0}, {"score": 2.0}, {"score": 10.0}]

    assert calculate_penalty(measurements) == PENALTY_CAP


def test_create_measurement_marks_low_high_and_ideal_states():
    ideal = {"min": 0, "max": 10, "idealMin": 4, "idealMax": 6}

    low = create_measurement("metric", "Metric", 2, "ratio", "Group", "Description", ideal)
    perfect = create_measurement("metric", "Metric", 5, "ratio", "Group", "Description", ideal)
    high = create_measurement("metric", "Metric", 8, "ratio", "Group", "Description", ideal)

    assert low["deviation"] == "low"
    assert low["isIdeal"] is False
    assert perfect["score"] == 10.0
    assert perfect["isIdeal"] is True
    assert high["deviation"] == "high"
