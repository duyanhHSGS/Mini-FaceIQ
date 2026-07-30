import random
from pathlib import Path

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
from third_party.my_side_profile_savior.mapping import load_landmark_mapping
from third_party.my_side_profile_savior.model import (
    ProfileLandmarkModel,
    masked_landmark_loss,
)
from third_party.my_side_profile_savior.train import _validate_resume


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
    assert int(mapping.active_mask().sum().item()) == 15


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


def test_model_keeps_39_output_slots_and_quarter_resolution():
    model = ProfileLandmarkModel(pretrained=False)

    output = model(torch.zeros(1, 3, 64, 64))

    assert output.shape == (1, 39, 16, 16)


def test_masked_loss_ignores_unconfirmed_slots():
    logits = torch.zeros(1, 39, 16, 16, requires_grad=True)
    target = torch.full((1, 39, 2), 0.5)
    visibility = torch.ones(1, 39, dtype=torch.bool)
    active = torch.zeros(39)
    active[19] = 1

    first = masked_landmark_loss(logits, target, active, visibility)
    changed = target.clone()
    changed[:, 18, :] = 0.99
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
