import json
from pathlib import Path

from PIL import Image
import pytest

from third_party.my_side_profile_savior.annotator.domain import (
    consensus,
    crop_to_source,
    deterministic_split,
    source_to_crop,
    source_to_view,
    square_crop,
    validate_annotation,
    view_to_source,
)
from third_party.my_side_profile_savior.annotator.projects import (
    create_arbitrary_project,
    export_project,
    project_store,
)
from third_party.my_side_profile_savior.annotator.store import (
    StaleRevisionError,
    WriterBusyError,
)
from third_party.my_side_profile_savior.annotator.app import create_app
from third_party.my_side_profile_savior.human_dataset import (
    HumanExportLandmarkDataset,
)
from third_party.my_side_profile_savior.annotator.suggestions import cache_key


def _manifest(tmp_path: Path, *, count: int = 6) -> Path:
    rows = []
    for index in range(count):
        image = tmp_path / f"source-{index}.jpg"
        Image.new("RGB", (100, 80), (index * 10, 30, 40)).save(image)
        rows.append(
            {
                "image_path": str(image),
                "subject_id": f"subject-{index}",
                "facing": "left" if index % 2 else "right",
                "bbox": [20, 10, 80, 70],
            }
        )
    manifest = tmp_path / "images.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def test_coordinate_round_trips_include_mirror_crop_and_quarter_pixel():
    crop = square_crop([20, 10, 80, 70], 100, 80, scale=1.5)
    source = (27.25, 19.75)
    normalized = source_to_crop(*source, crop)

    assert crop_to_source(*normalized, crop) == pytest.approx(source)
    mirrored = source_to_view(*source, width=100, mirrored=True)
    assert view_to_source(*mirrored, width=100, mirrored=True) == pytest.approx(
        source
    )
    nudged = (source[0] + 0.25, source[1] - 0.25)
    assert crop_to_source(*source_to_crop(*nudged, crop), crop) == pytest.approx(
        nudged
    )


def test_annotation_states_are_exact_and_nonplaced_coordinates_are_rejected():
    assert validate_annotation(
        {"state": "placed", "x": 1.25, "y": 2.5}
    )["origin"] == "manual"
    for state in ("unreviewed", "occluded", "out_of_frame"):
        assert validate_annotation({"state": state})["x"] is None
    with pytest.raises(ValueError, match="cannot contain coordinates"):
        validate_annotation({"state": "occluded", "x": 1, "y": 2})


def test_split_is_frozen_seeded_and_subject_disjoint():
    first = deterministic_split(f"subject-{index}" for index in range(20))
    second = deterministic_split(f"subject-{index}" for index in range(20))

    assert first == second
    assert first["seed_text"] == "Mini-FaceIQ"
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])


def test_consensus_threshold_and_state_disagreement():
    bbox = [0, 0, 100, 100]
    close = consensus(
        {"state": "placed", "x": 50.0, "y": 50.0},
        {"state": "placed", "x": 51.0, "y": 50.0},
        bbox,
    )
    assert not close.requires_adjudication
    assert close.x == pytest.approx(50.5)
    far = consensus(
        {"state": "placed", "x": 20.0, "y": 20.0},
        {"state": "placed", "x": 30.0, "y": 30.0},
        bbox,
    )
    assert far.requires_adjudication
    mismatch = consensus(
        {"state": "occluded", "x": None, "y": None},
        {"state": "out_of_frame", "x": None, "y": None},
        bbox,
    )
    assert mismatch.reason == "state_disagreement"


def test_project_import_autosave_lock_stale_revision_and_immutable_export(tmp_path):
    projects_root = tmp_path / "projects"
    project_dir = create_arbitrary_project(
        "people",
        _manifest(tmp_path),
        projects_root=projects_root,
    )
    store = project_store("people", projects_root)
    project = store.project()
    assert project["crop_scale"] == 1.5
    assert len(project["mapping"]["entries"]) == 31
    assert len(store.queue()) == 6
    assert {item["split_name"] for item in store.queue()} == {
        "train",
        "validation",
        "test",
    }

    store.register_profile("alice")
    token = store.acquire_writer("alice")
    with pytest.raises(WriterBusyError):
        store.acquire_writer("bob")
    saved = store.mutate_label(
        image_id=1,
        pass_number=1,
        landmark_index=0,
        annotation={"state": "placed", "x": 24.25, "y": 14.75},
        expected_revision=0,
        annotator_id="alice",
        writer_token=token,
    )
    assert saved["revision"] == 1
    assert store.queue()[0]["review_state"] == "partial"
    with pytest.raises(StaleRevisionError):
        store.mutate_label(
            image_id=1,
            pass_number=1,
            landmark_index=0,
            annotation={"state": "occluded"},
            expected_revision=0,
            annotator_id="alice",
            writer_token=token,
        )
    undone = store.undo(
        event_id=saved["event_id"],
        annotator_id="alice",
        writer_token=token,
    )
    assert undone["state"] == "unreviewed"
    assert any(event["action"] == "undo" for event in store.history(1))

    first = export_project(store, project_dir)
    second = export_project(store, project_dir)
    assert first != second
    assert (first / "labels.jsonl").read_bytes() == (
        second / "labels.jsonl"
    ).read_bytes()
    assert (first / "labels.csv").read_bytes() == (
        second / "labels.csv"
    ).read_bytes()


def test_arbitrary_import_rejects_duplicate_content(tmp_path):
    source = tmp_path / "same.jpg"
    Image.new("RGB", (100, 80), "black").save(source)
    rows = [
        {
            "image_path": str(source),
            "subject_id": f"subject-{index}",
            "facing": "left",
            "bbox": [20, 10, 80, 70],
        }
        for index in range(3)
    ]
    manifest = tmp_path / "duplicates.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate image"):
        create_arbitrary_project(
            "duplicates",
            manifest,
            projects_root=tmp_path / "projects",
        )


def test_exif_orientation_is_normalized_and_facing_is_validated(tmp_path):
    source = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (20, 40), "white")
    exif = image.getexif()
    exif[274] = 6
    image.save(source, exif=exif)
    manifest = tmp_path / "rotated.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "image_path": str(source),
                "subject_id": "one",
                "facing": "left",
                "bbox": [1, 1, 19, 19],
            }
        )
        + "\n"
        + json.dumps(
            {
                "image_path": str(tmp_path / "other.jpg"),
                "subject_id": "two",
                "facing": "right",
                "bbox": [1, 1, 19, 19],
            }
        )
        + "\n"
        + json.dumps(
            {
                "image_path": str(tmp_path / "third.jpg"),
                "subject_id": "three",
                "facing": "right",
                "bbox": [1, 1, 19, 19],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    Image.new("RGB", (20, 40), "red").save(tmp_path / "other.jpg")
    Image.new("RGB", (20, 40), "blue").save(tmp_path / "third.jpg")
    create_arbitrary_project(
        "exif",
        manifest,
        projects_root=tmp_path / "projects",
    )
    first = project_store("exif", tmp_path / "projects").image(1)
    assert (first["width"], first["height"]) == (40, 20)

    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(
            {
                "image_path": str(source),
                "subject_id": "one",
                "facing": "upside_down",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="facing"):
        create_arbitrary_project(
            "bad-facing",
            bad,
            projects_root=tmp_path / "projects",
        )


def test_project_id_containment_and_suggestion_cache_key_are_deterministic(tmp_path):
    with pytest.raises(ValueError, match="project_id"):
        project_store("../escape", tmp_path / "projects")
    first = cache_key("image-hash", [1, 2, 3, 4], "provider")
    second = cache_key("image-hash", [1, 2, 3, 4], "provider")
    changed = cache_key("image-hash", [1, 2, 3, 5], "provider")
    assert first == second
    assert first != changed


def test_flask_project_queue_image_annotation_history_and_export_apis(tmp_path):
    projects_root = tmp_path / "projects"
    create_arbitrary_project(
        "api-project",
        _manifest(tmp_path),
        projects_root=projects_root,
    )
    client = create_app(projects_root=projects_root).test_client()

    assert client.get("/api/projects").status_code == 200
    writer = client.post(
        "/api/projects/api-project/writer",
        json={"annotator_id": "alice"},
    ).get_json()
    queue = client.get("/api/projects/api-project/queue").get_json()["images"]
    image_id = queue[0]["id"]
    image = client.get(
        f"/api/projects/api-project/images/{image_id}"
    ).get_json()
    response = client.put(
        f"/api/projects/api-project/images/{image_id}/passes/1/landmarks/0",
        json={
            "annotator_id": "alice",
            "writer_token": writer["writer_token"],
            "expected_revision": 0,
            "annotation": {"state": "placed", "x": 30.0, "y": 20.0},
        },
    )
    assert response.status_code == 200
    stale = client.put(
        f"/api/projects/api-project/images/{image_id}/passes/1/landmarks/0",
        json={
            "annotator_id": "alice",
            "writer_token": writer["writer_token"],
            "expected_revision": 0,
            "annotation": {"state": "occluded"},
        },
    )
    assert stale.status_code == 409
    history = client.get(
        f"/api/projects/api-project/images/{image_id}/history"
    ).get_json()["events"]
    assert history[0]["action"] == "set_label"
    exported = client.post(
        "/api/projects/api-project/export",
        json={
            "annotator_id": "alice",
            "writer_token": writer["writer_token"],
        },
    )
    assert exported.status_code == 201
    assert Path(exported.get_json()["path"], "manifest.json").is_file()
    assert image["bbox_state"] == "needs_confirmation"


def test_blind_pass_rejects_all_aid_exposure(tmp_path):
    projects_root = tmp_path / "projects"
    create_arbitrary_project(
        "blind",
        _manifest(tmp_path),
        projects_root=projects_root,
    )
    store = project_store("blind", projects_root)
    store.register_profile("reviewer")
    token = store.acquire_writer("reviewer")

    with pytest.raises(ValueError, match="Blind"):
        store.mark_exposure(
            image_id=1,
            pass_number=2,
            source="model",
            writer_token=token,
            annotator_id="reviewer",
        )


def test_human_dataset_uses_only_exported_visibility_without_multipie_fallback(
    tmp_path,
):
    projects_root = tmp_path / "projects"
    project_dir = create_arbitrary_project(
        "human",
        _manifest(tmp_path, count=9),
        projects_root=projects_root,
    )
    store = project_store("human", projects_root)
    store.register_profile("alice")
    token = store.acquire_writer("alice")
    train_image = next(item for item in store.queue() if item["split_name"] == "train")
    store.mutate_label(
        image_id=train_image["id"],
        pass_number=1,
        landmark_index=30,
        annotation={"state": "placed", "x": 30.0, "y": 20.0},
        expected_revision=0,
        annotator_id="alice",
        writer_token=token,
    )
    export_dir = export_project(store, project_dir)

    dataset = HumanExportLandmarkDataset(export_dir)
    sample_index = next(
        index
        for index, record in enumerate(dataset.records)
        if record.relative_image_path == train_image["relative_path"]
    )
    sample = dataset[sample_index]

    assert sample["visibility"].shape == (31,)
    assert int(sample["visibility"].sum().item()) == 1
    assert bool(sample["visibility"][30])
    assert not bool(sample["visibility"][0])
