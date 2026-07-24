from front_ideals import FRONT_IDEALS, MALE_ASIAN_FRONT, get_front_ideals
from side_ideals import SIDE_IDEALS, MALE_ASIAN_SIDE, get_side_ideals


def test_front_ideals_exist_for_every_gender_ethnicity_pair():
    expected_ethnicities = {
        "asian",
        "caucasian",
        "black",
        "hispanic",
        "middle_eastern",
        "south_asian",
        "mixed",
    }

    assert set(FRONT_IDEALS) == {"male", "female"}
    for gender in FRONT_IDEALS:
        assert set(FRONT_IDEALS[gender]) == expected_ethnicities
        for values in FRONT_IDEALS[gender].values():
            assert set(values) == set(MALE_ASIAN_FRONT)


def test_side_ideals_exist_for_every_gender_ethnicity_pair():
    expected_ethnicities = {
        "asian",
        "caucasian",
        "black",
        "hispanic",
        "middle_eastern",
        "south_asian",
        "mixed",
    }

    assert set(SIDE_IDEALS) == {"male", "female"}
    for gender in SIDE_IDEALS:
        assert set(SIDE_IDEALS[gender]) == expected_ethnicities
        for values in SIDE_IDEALS[gender].values():
            assert set(values) == set(MALE_ASIAN_SIDE)


def test_unknown_front_and_side_demographics_fall_back_to_male_asian():
    assert get_front_ideals("unknown", "unknown") == MALE_ASIAN_FRONT
    assert get_side_ideals("unknown", "unknown") == MALE_ASIAN_SIDE


def test_demographic_adjustments_are_separate_from_baseline_tables():
    female_black_front = get_front_ideals("female", "black")
    female_middle_eastern_side = get_side_ideals("female", "middle_eastern")

    assert female_black_front is not MALE_ASIAN_FRONT
    assert female_black_front["lateral_canthal_tilt"] is not MALE_ASIAN_FRONT["lateral_canthal_tilt"]
    assert female_middle_eastern_side is not MALE_ASIAN_SIDE
    assert female_middle_eastern_side["nasal_tip_angle"] is not MALE_ASIAN_SIDE["nasal_tip_angle"]
