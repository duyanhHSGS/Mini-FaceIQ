import ast
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest
from werkzeug.datastructures import FileStorage

from packages.human_data_factory.app import create_app
from packages.human_data_factory.domain import (
    advance_position,
    source_to_view,
    validate_label,
    view_to_source,
)
from packages.human_data_factory.projects import (
    create_project,
    export_project,
    import_uploaded_files,
    project_store,
)
from packages.human_data_factory.schema import LANDMARKS
from packages.human_data_factory.store import StaleRevisionError


PACKAGE_DIR = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "human_data_factory"
)


def _image_bytes(color="navy", size=(80, 60)):
    stream = BytesIO()
    Image.new("RGB", size, color).save(stream, format="JPEG")
    stream.seek(0)
    return stream


def _upload(name, *, color="navy", size=(80, 60)):
    return FileStorage(
        stream=_image_bytes(color=color, size=size),
        filename=name,
        content_type="image/jpeg",
    )


def _custom_project(tmp_path, project_id="people"):
    root = tmp_path / "projects"
    create_project(
        project_id=project_id,
        annotator="alice",
        source_kind="custom",
        projects_root=root,
    )
    report = import_uploaded_files(
        project_id=project_id,
        uploads=[_upload("face.jpg")],
        original_relative_paths=["folder/face.jpg"],
        projects_root=root,
    )
    assert report["imported"] == 1
    return root


def test_schema_is_frozen_to_31_side_profile_landmarks():
    assert len(LANDMARKS) == 31
    assert [item["index"] for item in LANDMARKS] == list(range(31))
    assert LANDMARKS[4]["id"] == "porion"


def test_all_mode_and_focused_mode_advance_differently():
    assert advance_position(
        mode="all",
        image_index=0,
        landmark_index=4,
        image_count=3,
    ) == (0, 5, False)
    assert advance_position(
        mode="all",
        image_index=0,
        landmark_index=30,
        image_count=3,
    ) == (1, 0, False)
    assert advance_position(
        mode="focused",
        image_index=0,
        landmark_index=4,
        image_count=3,
    ) == (1, 4, False)
    assert advance_position(
        mode="focused",
        image_index=2,
        landmark_index=4,
        image_count=3,
    ) == (2, 4, True)


def test_mirror_round_trip_and_fractional_coordinates_are_preserved():
    source = (12.25, 19.75)
    view = source_to_view(
        *source,
        image_width=80,
        mirrored=True,
    )
    restored = view_to_source(
        *view,
        image_width=80,
        mirrored=True,
    )
    assert restored == pytest.approx(source)
    assert validate_label(
        {"state": "placed", "x": 12.25, "y": 19.75},
        image_width=80,
        image_height=60,
    ) == {"state": "placed", "x": 12.25, "y": 19.75}


def test_unavailable_has_no_coordinates_and_unreviewed_is_valid():
    assert validate_label(
        {"state": "unavailable", "x": None, "y": None},
        image_width=80,
        image_height=60,
    )["state"] == "unavailable"
    assert validate_label(
        {"state": "unreviewed"},
        image_width=80,
        image_height=60,
    )["state"] == "unreviewed"
    with pytest.raises(ValueError, match="cannot contain coordinates"):
        validate_label(
            {"state": "unavailable", "x": 1, "y": 2},
            image_width=80,
            image_height=60,
        )


def test_autosave_revision_conflict_and_undo_are_transactional(tmp_path):
    root = _custom_project(tmp_path)
    store = project_store("people", root)
    saved = store.mutate_label(
        image_id=1,
        landmark_index=4,
        annotation={"state": "placed", "x": 20.25, "y": 10.75},
        expected_revision=0,
        annotator="alice",
    )
    assert saved["revision"] == 1
    assert store.image(1)["labels"][4]["x"] == pytest.approx(20.25)
    with pytest.raises(StaleRevisionError):
        store.mutate_label(
            image_id=1,
            landmark_index=4,
            annotation={"state": "unavailable"},
            expected_revision=0,
            annotator="alice",
        )
    undone = store.undo_last(image_id=1, annotator="alice")
    assert undone["state"] == "unreviewed"
    assert store.image(1)["labels"][4]["state"] == "unreviewed"
    assert store.history(1)[0]["action"] == "undo"


def test_custom_import_normalizes_images_and_rejects_duplicate_pixels(tmp_path):
    root = tmp_path / "projects"
    create_project(
        project_id="duplicates",
        annotator="alice",
        source_kind="custom",
        projects_root=root,
    )
    report = import_uploaded_files(
        project_id="duplicates",
        uploads=[
            _upload("first.jpg", color="red"),
            _upload("second.jpg", color="red"),
        ],
        original_relative_paths=["a/first.jpg", "b/second.jpg"],
        projects_root=root,
    )
    assert report["imported"] == 1
    assert len(report["duplicates"]) == 1
    image = project_store("duplicates", root).image(1)
    assert image["relative_path"].endswith(".png")
    assert image["original_relative_path"] == "a/first.jpg"


def test_builtin_multipie_import_reads_images_without_any_label_file(tmp_path):
    source = tmp_path / "unlabeled"
    source.mkdir()
    Image.new("RGB", (32, 24), "white").save(source / "one.jpg")
    Image.new("RGB", (32, 24), "black").save(source / "two.jpg")
    root = tmp_path / "projects"
    _, report = create_project(
        project_id="multipie",
        annotator="alice",
        source_kind="multipie",
        projects_root=root,
        multipie_source=source,
    )
    assert report["imported"] == 2
    for queued in project_store("multipie", root).queue():
        labels = project_store("multipie", root).image(queued["id"])["labels"]
        assert len(labels) == 31
        assert {item["state"] for item in labels} == {"unreviewed"}


def test_partial_export_contains_images_and_all_31_states(tmp_path):
    root = _custom_project(tmp_path)
    store = project_store("people", root)
    store.mutate_label(
        image_id=1,
        landmark_index=4,
        annotation={"state": "placed", "x": 20.25, "y": 10.75},
        expected_revision=0,
        annotator="alice",
    )
    destination = export_project("people", projects_root=root)
    assert (destination / "images" / "00000001.png").is_file()
    assert (destination / "landmarks.json").is_file()
    with (destination / "labels.jsonl").open(encoding="utf-8") as handle:
        record = json.loads(handle.readline())
    assert len(record["landmarks"]) == 31
    assert record["landmarks"][4]["id"] == "porion"
    assert record["landmarks"][4]["x_px"] == pytest.approx(20.25)
    assert record["landmarks"][0]["state"] == "unreviewed"
    manifest = json.loads(
        (destination / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["partial_exports_allowed"]
    assert manifest["counts"]["label_states"] == {
        "placed": 1,
        "unavailable": 0,
        "unreviewed": 30,
    }


def test_local_api_supports_focused_queue_autosave_undo_and_export(tmp_path):
    root = tmp_path / "projects"
    client = create_app(
        projects_root=root,
        multipie_source=tmp_path / "unused",
    ).test_client()
    created = client.post(
        "/api/projects",
        json={
            "project_id": "api",
            "annotator": "alice",
            "source_kind": "custom",
        },
    )
    assert created.status_code == 201
    uploaded = client.post(
        "/api/projects/api/images",
        data={
            "relative_paths": json.dumps(["folder/face.jpg"]),
            "files": (_image_bytes(), "face.jpg"),
        },
        content_type="multipart/form-data",
    )
    assert uploaded.status_code == 200
    focused = client.get(
        "/api/projects/api/queue?mode=focused&landmark_index=4"
    ).get_json()
    assert focused["images"][0]["focused_state"] == "unreviewed"
    saved = client.put(
        "/api/projects/api/images/1/landmarks/4",
        json={
            "annotator": "alice",
            "expected_revision": 0,
            "annotation": {"state": "placed", "x": 10.5, "y": 11.25},
        },
    )
    assert saved.status_code == 200
    assert client.post(
        "/api/projects/api/images/1/undo",
        json={"annotator": "alice"},
    ).status_code == 200
    assert client.post("/api/projects/api/export").status_code == 201


def test_browser_workspace_exposes_both_modes_and_auto_continue():
    html = (PACKAGE_DIR / "static" / "index.html").read_text(encoding="utf-8")
    assert '<option value="all">All landmarks</option>' in html
    assert '<option value="focused">One focused landmark</option>' in html
    assert 'id="autoContinue"' in html
    assert 'id="continueButton"' in html
    assert "webkitdirectory" in html
    assert 'id="canvas"' in html


def test_package_has_no_forbidden_runtime_imports():
    forbidden = {
        "torch",
        "torchvision",
        "cv2",
        "onnxruntime",
        "mediapipe",
        "numpy",
        "scipy",
    }
    imported = set()
    for path in PACKAGE_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
    assert forbidden.isdisjoint(imported)
