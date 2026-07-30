import random
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
from PIL import Image
import pytest
import torch

from third_party.my_side_profile_savior.benchmark import (
    FAILURE_PENALTY,
    normalized_landmark_error,
    per_landmark_wins,
)
from third_party.my_side_profile_savior.dataset import (
    AugmentationSettings,
    ProfileAnnotation,
    _augment_crop,
    _select_model_landmarks,
    make_subject_split,
)
from third_party.my_side_profile_savior.factory_config import (
    FactoryConfig,
    atomic_torch_save,
    encoded_seed,
    resolve_device,
)
from third_party.my_side_profile_savior.factory_state import (
    latest_best_checkpoint,
    request_graceful_stop,
    status_progress,
)
from third_party.my_side_profile_savior.inference import (
    _output_layout_from_checkpoint,
)
from third_party.my_side_profile_savior.mapping import load_landmark_mapping
from third_party.my_side_profile_savior.model import (
    ProfileLandmarkModel,
    masked_landmark_loss,
)
from third_party.my_side_profile_savior.train import _validate_resume
from third_party.my_side_profile_savior.ui import (
    _default_landmark_color,
    _draw_points,
    _hover_label_text,
    _landmark_color,
    _nearest_landmark_name,
    _panned_view_limits,
    _synchronized_view_limits,
    _zoomed_view_limits,
)


PACKAGE_DIR = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "my_side_profile_savior"
)


def _mapping_file(tmp_path, rows):
    path = tmp_path / "mapping.txt"
    header = "# name | source | reference | dataset index\n"
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _fake_record(subject_id):
    return ProfileAnnotation(
        image_path=Path(f"{subject_id}.jpg"),
        relative_image_path=f"image/profile/{subject_id}.jpg",
        bbox_xyxy=np.asarray([0, 0, 100, 100], dtype=np.float32),
        auxiliary_points_xy=np.zeros((5, 2), dtype=np.float32),
        landmarks_xy=np.zeros((39, 2), dtype=np.float32),
        subject_id=subject_id,
        camera_code="090",
    )


def test_user_mapping_is_authoritative_and_masks_nonintegers():
    mapping = load_landmark_mapping(PACKAGE_DIR / "user-custom.txt")

    assert mapping.confirmed_count == 15
    assert mapping.names_by_dataset_index[0] == "porion"
    assert mapping.names_by_dataset_index[19] == "nose_tip"
    assert "infratip" not in mapping.names_by_dataset_index.values()
    assert "lower_eyelid" not in mapping.names_by_dataset_index.values()
    assert len(mapping.entries) == 31
    assert mapping.entries[2].name == "nose_tip"
    assert mapping.entries[2].model_index == 2
    assert int(mapping.active_mask().sum().item()) == 15
    assert mapping.active_mask().shape == (31,)
    layout = mapping.output_layout()
    assert len(layout) == 31
    assert sum(bool(item["active"]) for item in layout) == 15
    assert layout[2]["dataset_index"] == 19


def test_only_confirmed_multipie_points_enter_31_model_slots():
    mapping = load_landmark_mapping(PACKAGE_DIR / "user-custom.txt")
    dataset_points = np.arange(39 * 2, dtype=np.float32).reshape(39, 2)

    selected = _select_model_landmarks(dataset_points, mapping)

    assert selected.shape == (31, 2)
    for entry in mapping.entries:
        if entry.confirmed:
            np.testing.assert_array_equal(
                selected[entry.model_index],
                dataset_points[int(entry.dataset_index)],
            )
        else:
            np.testing.assert_array_equal(
                selected[entry.model_index],
                np.zeros(2, dtype=np.float32),
            )


def test_checkpoint_layout_is_strictly_31_output_only():
    mapping = load_landmark_mapping(PACKAGE_DIR / "user-custom.txt")
    checkpoint = {
        "mapping": mapping.snapshot(),
        "output_layout": mapping.output_layout(),
    }

    layout = _output_layout_from_checkpoint(checkpoint)

    assert len(layout) == 15
    assert layout[0] == {
        "model_index": 2,
        "dataset_index": 19,
        "name": "nose_tip",
    }

    broken = {
        "mapping": mapping.snapshot(),
        "output_layout": mapping.output_layout(),
    }
    del broken["output_layout"][2]["model_index"]
    with pytest.raises(ValueError, match="require model_index and name"):
        _output_layout_from_checkpoint(broken)

    broken_mapping = {
        "mapping": mapping.snapshot(),
        "output_layout": mapping.output_layout(),
    }
    del broken_mapping["mapping"]["entries"][2]["model_index"]
    with pytest.raises(ValueError, match="require model_index and name"):
        _output_layout_from_checkpoint(broken_mapping)


def test_mapping_rejects_duplicate_dataset_indices(tmp_path):
    path = _mapping_file(
        tmp_path,
        [
            "nose_tip | sparse | 30 | 19",
            "another | dense | 42 | 19",
        ],
    )

    with pytest.raises(ValueError, match="already assigned"):
        load_landmark_mapping(path)


def test_subject_split_is_seeded_and_disjoint():
    records = [_fake_record(f"{index:03d}") for index in range(20)]

    first = make_subject_split(records, seed_text="Mini-FaceIQ")
    second = make_subject_split(records, seed_text="Mini-FaceIQ")

    assert first == second
    assert first.seed == encoded_seed("Mini-FaceIQ")
    assert set(first.train).isdisjoint(first.validation)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.validation).isdisjoint(first.test)
    assert set(first.train) | set(first.validation) | set(first.test) == {
        record.subject_id for record in records
    }


def test_augmentation_moves_image_targets_and_auxiliary_points_together():
    image = Image.new("RGB", (256, 256), color=(128, 128, 128))
    points = np.asarray([[0.25, 0.40], [0.75, 0.60]], dtype=np.float32)
    settings = AugmentationSettings(
        enabled=True,
        rotation_degrees=8,
        translation_fraction=0.04,
        scale_jitter=0.08,
        brightness_jitter=0,
        contrast_jitter=0,
        blur_probability=0,
    )
    random.seed(123)

    _, transformed_landmarks, transformed_auxiliary = _augment_crop(
        image,
        points,
        points.copy(),
        settings,
    )

    np.testing.assert_allclose(transformed_landmarks, transformed_auxiliary)
    assert not np.allclose(transformed_landmarks, points)


def test_model_keeps_31_output_slots_and_quarter_resolution():
    model = ProfileLandmarkModel(pretrained=False)

    output = model(torch.zeros(1, 3, 64, 64))

    assert output.shape == (1, 31, 16, 16)


def test_masked_loss_ignores_unconfirmed_slots():
    mapping = load_landmark_mapping(PACKAGE_DIR / "user-custom.txt")
    logits = torch.zeros(1, 31, 16, 16, requires_grad=True)
    target = torch.full((1, 31, 2), 0.5)
    visibility = torch.ones(1, 31, dtype=torch.bool)
    active = mapping.active_mask()
    inactive_index = next(
        entry.model_index for entry in mapping.entries if not entry.confirmed
    )

    first = masked_landmark_loss(logits, target, active, visibility)
    changed = target.clone()
    changed[:, inactive_index, :] = 0.99
    second = masked_landmark_loss(logits, changed, active, visibility)

    assert torch.allclose(first["total"], second["total"])
    first["total"].backward()
    assert logits.grad is not None


def test_explicit_cuda_fails_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device("auto").type == "cpu"
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(RuntimeError, match="explicitly requested"):
        resolve_device("cuda")


def test_config_rejects_resume_and_initial_checkpoint_together():
    config = FactoryConfig(
        annotation_path="annotations.txt",
        mapping_path="mapping.txt",
        runs_root="runs",
        resume_checkpoint="last.pt",
        initial_checkpoint="old-best.pt",
    )

    with pytest.raises(ValueError, match="either resume_checkpoint"):
        config.validate()


def test_resume_accepts_mapping_snapshot_from_another_path(tmp_path):
    current_path = _mapping_file(
        tmp_path,
        ["nose_tip | sparse | 30 | 19"],
    )
    mapping = load_landmark_mapping(current_path)
    saved = mapping.snapshot()
    saved["source_path"] = "/another/machine/user-custom.txt"
    config = FactoryConfig(
        annotation_path="annotations.txt",
        mapping_path=str(current_path),
        runs_root="runs",
    )
    checkpoint = {
        "mapping": saved,
        "output_layout": mapping.output_layout(),
        "config": config.to_dict(),
    }

    _validate_resume(checkpoint, config, mapping)


def test_atomic_checkpoint_and_factory_state_helpers(tmp_path):
    checkpoint = tmp_path / "best.pt"
    atomic_torch_save({"value": torch.tensor([1])}, checkpoint)

    assert latest_best_checkpoint(tmp_path) == checkpoint.resolve()
    stop_path = request_graceful_stop(tmp_path)
    assert stop_path.is_file()
    percent, text = status_progress(
        {
            "state": "training",
            "epoch": 4,
            "epochs": 10,
            "device": "cuda",
            "best_validation_nme": 0.02,
        }
    )
    assert percent == 50.0
    assert "epoch 5/10" in text


def test_invalid_prediction_receives_full_failure_penalty():
    error = normalized_landmark_error(
        None,
        np.asarray([0.5, 0.5]),
        crop_xyxy=np.asarray([0, 0, 125, 125]),
        bbox_xyxy=np.asarray([10, 10, 110, 110]),
    )

    assert error == FAILURE_PENALTY


def test_graduation_requires_winning_every_landmark():
    custom = {
        "per_landmark": {
            "nose_tip": {"mean_nme": 0.01},
            "chin_point": {"mean_nme": 0.04},
        }
    }
    legacy = {
        "per_landmark": {
            "nose_tip": {"mean_nme": 0.02},
            "chin_point": {"mean_nme": 0.03},
        }
    }

    wins = per_landmark_wins(
        custom,
        legacy,
        ["nose_tip", "chin_point"],
    )

    assert wins == {"nose_tip": True, "chin_point": False}
    assert not all(wins.values())


def test_qa_palette_keeps_porion_blue_and_honors_session_override():
    automatic = _default_landmark_color(0)
    overrides = {}

    assert automatic == "#0066ff"
    assert _landmark_color("porion", 0, overrides) == automatic

    overrides["porion"] = "#ff00aa"
    assert _landmark_color("porion", 0, overrides) == "#ff00aa"
    assert _landmark_color("porion", 0, overrides) == _landmark_color(
        "porion",
        0,
        overrides,
    )

    overrides.pop("porion")
    assert _landmark_color("porion", 0, overrides) == automatic


def test_qa_points_start_with_hidden_labels_and_live_opacity():
    image = Image.new("RGB", (100, 100), color=(128, 128, 128))
    points = [
        {
            "name": "porion",
            "dataset_index": 0,
            "x": 0.25,
            "y": 0.40,
        }
    ]
    figure, axes = plt.subplots(1, 2)
    try:
        first = _draw_points(
            axes[0],
            image,
            points,
            color_overrides={},
            point_alpha=0.42,
            label_alpha=0.31,
            title="Truth",
        )
        second = _draw_points(
            axes[1],
            image,
            points,
            color_overrides={},
            point_alpha=0.42,
            label_alpha=0.31,
            title="Custom",
        )

        first_entry = first["entries"]["porion"]
        second_entry = second["entries"]["porion"]
        assert not first_entry["annotation"].get_visible()
        assert first_entry["artist"].get_alpha() == pytest.approx(0.42)
        assert (
            first_entry["annotation"].get_bbox_patch().get_alpha()
            == pytest.approx(0.31)
        )
        assert to_hex(first_entry["artist"].get_facecolor()[0]) == "#0066ff"
        assert to_hex(second_entry["artist"].get_facecolor()[0]) == "#0066ff"
    finally:
        plt.close(figure)


def test_qa_hover_helpers_find_points_and_report_missing_predictions():
    points = [
        {
            "name": "porion",
            "dataset_index": 0,
            "x": 0.25,
            "y": 0.40,
        }
    ]

    assert (
        _nearest_landmark_name(points, 26.0, 39.0, (100, 100))
        == "porion"
    )
    assert _nearest_landmark_name(points, 90.0, 90.0, (100, 100)) is None
    assert _hover_label_text("porion", 0, None) == (
        "porion [0]\nnot predicted"
    )


def test_qa_zoom_pan_and_reset_limits_can_stay_synchronized():
    original_x = (-0.5, 99.5)
    original_y = (99.5, -0.5)
    zoomed_x, zoomed_y = _zoomed_view_limits(
        original_x,
        original_y,
        (50.0, 50.0),
        0.5,
    )
    panned_x, panned_y = _panned_view_limits(
        zoomed_x,
        zoomed_y,
        (0.1, -0.2),
    )

    synchronized = _synchronized_view_limits(
        3,
        panned_x,
        panned_y,
    )
    reset = _synchronized_view_limits(3, original_x, original_y)

    assert len(synchronized) == 3
    assert all(limits == synchronized[0] for limits in synchronized)
    assert all(limits == (original_x, original_y) for limits in reset)
