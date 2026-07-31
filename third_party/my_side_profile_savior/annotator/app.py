"""Flask API for the factory-only Sir FaceIQ browser annotator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file
from PIL import Image

from .domain import square_crop, validate_annotation
from .projects import (
    PROJECTS_ROOT,
    create_arbitrary_project,
    create_multipie_project,
    export_project,
    image_path,
    list_projects,
    project_store,
    review_consensus,
)
from .store import AnnotationStore, StaleRevisionError, WriterBusyError
from .suggestions import (
    cache_key,
    custom_suggestions,
    detect_bbox,
    legacy_suggestions,
    provider_fingerprint,
)


PACKAGE_DIR = Path(__file__).resolve().parent


def _payload() -> dict[str, Any]:
    value = request.get_json(silent=False)
    if not isinstance(value, dict):
        raise ValueError("JSON request body must be an object")
    return value


def _writer_fields(payload: dict[str, Any]) -> tuple[str, str]:
    annotator_id = str(payload.get("annotator_id", "")).strip()
    writer_token = str(payload.get("writer_token", "")).strip()
    if not annotator_id or not writer_token:
        raise ValueError("annotator_id and writer_token are required")
    return annotator_id, writer_token


def create_app(*, projects_root: str | Path = PROJECTS_ROOT) -> Flask:
    root = Path(projects_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for item in list_projects(root):
        project_store(item["project_id"], root).reset_writer_after_server_restart()
    app = Flask(
        __name__,
        static_folder=str(PACKAGE_DIR / "static"),
        static_url_path="/static",
    )
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

    @app.errorhandler(ValueError)
    @app.errorhandler(KeyError)
    @app.errorhandler(FileNotFoundError)
    @app.errorhandler(FileExistsError)
    def bad_request(error):
        return jsonify({"error": str(error)}), 400

    @app.errorhandler(StaleRevisionError)
    def stale(error):
        return jsonify({"error": str(error), "code": "stale_revision"}), 409

    @app.errorhandler(WriterBusyError)
    def writer_busy(error):
        return jsonify({"error": str(error), "code": "writer_busy"}), 423

    @app.get("/")
    def index():
        return send_file(PACKAGE_DIR / "static" / "index.html")

    @app.get("/api/projects")
    def projects():
        return jsonify({"projects": list_projects(root)})

    @app.post("/api/projects")
    def create_project():
        values = _payload()
        kind = values.get("kind")
        project_id = str(values.get("project_id", ""))
        if kind == "multipie":
            directory = create_multipie_project(
                project_id,
                annotation_path=values.get("annotation_path") or None
                or str(
                    Path(__file__).resolve().parents[3]
                    / "git-plz-ignore"
                    / "MultiPIE"
                    / "MultiPIE_profile_train.txt"
                ),
                projects_root=root,
            )
        elif kind == "arbitrary":
            directory = create_arbitrary_project(
                project_id,
                values.get("manifest_path", ""),
                projects_root=root,
            )
        else:
            raise ValueError("kind must be multipie or arbitrary")
        return jsonify({"project_id": directory.name}), 201

    @app.get("/api/projects/<project_id>")
    def project(project_id: str):
        store = project_store(project_id, root)
        return jsonify(store.project())

    @app.post("/api/projects/<project_id>/writer")
    def acquire_writer(project_id: str):
        values = _payload()
        annotator_id = str(values.get("annotator_id", "")).strip()
        if not annotator_id:
            raise ValueError("annotator_id is required")
        store = project_store(project_id, root)
        store.register_profile(annotator_id)
        token = store.acquire_writer(annotator_id, values.get("writer_token"))
        return jsonify({"writer_token": token, "annotator_id": annotator_id})

    @app.delete("/api/projects/<project_id>/writer")
    def release_writer(project_id: str):
        values = _payload()
        project_store(project_id, root).release_writer(
            str(values.get("writer_token", ""))
        )
        return "", 204

    @app.post("/api/projects/<project_id>/writer/release")
    def release_writer_beacon(project_id: str):
        values = _payload()
        project_store(project_id, root).release_writer(
            str(values.get("writer_token", ""))
        )
        return "", 204

    @app.get("/api/projects/<project_id>/queue")
    def queue(project_id: str):
        pass_number = int(request.args.get("pass", "1"))
        split_name = request.args.get("split")
        return jsonify(
            {
                "images": project_store(project_id, root).queue(
                    pass_number=pass_number, split_name=split_name
                )
            }
        )

    @app.get("/api/projects/<project_id>/images/<int:image_id>")
    def image(project_id: str, image_id: int):
        pass_number = int(request.args.get("pass", "1"))
        return jsonify(
            project_store(project_id, root).image(
                image_id, pass_number=pass_number
            )
        )

    @app.get("/api/projects/<project_id>/images/<int:image_id>/pixels")
    def pixels(project_id: str, image_id: int):
        store = project_store(project_id, root)
        image_record = store.image(image_id)
        path = image_path(project_id, image_record, root)
        if hashlib.sha256(path.read_bytes()).hexdigest() != image_record["sha256"]:
            raise ValueError("Image hash no longer matches the frozen project record")
        return send_file(path)

    @app.put(
        "/api/projects/<project_id>/images/<int:image_id>/passes/"
        "<int:pass_number>/landmarks/<int:landmark_index>"
    )
    def set_label(
        project_id: str,
        image_id: int,
        pass_number: int,
        landmark_index: int,
    ):
        values = _payload()
        annotator_id, writer_token = _writer_fields(values)
        result = project_store(project_id, root).mutate_label(
            image_id=image_id,
            pass_number=pass_number,
            landmark_index=landmark_index,
            annotation=values.get("annotation", {}),
            expected_revision=int(values.get("expected_revision", -1)),
            annotator_id=annotator_id,
            writer_token=writer_token,
        )
        return jsonify(result)

    @app.post(
        "/api/projects/<project_id>/images/<int:image_id>/passes/"
        "<int:pass_number>/exposure"
    )
    def expose(project_id: str, image_id: int, pass_number: int):
        values = _payload()
        annotator_id, writer_token = _writer_fields(values)
        project_store(project_id, root).mark_exposure(
            image_id=image_id,
            pass_number=pass_number,
            source=str(values.get("source", "")),
            annotator_id=annotator_id,
            writer_token=writer_token,
        )
        return "", 204

    @app.get("/api/projects/<project_id>/images/<int:image_id>/history")
    def history(project_id: str, image_id: int):
        return jsonify(
            {"events": project_store(project_id, root).history(image_id)}
        )

    @app.post("/api/projects/<project_id>/history/<int:event_id>/undo")
    def undo(project_id: str, event_id: int):
        values = _payload()
        annotator_id, writer_token = _writer_fields(values)
        return jsonify(
            project_store(project_id, root).undo(
                event_id=event_id,
                annotator_id=annotator_id,
                writer_token=writer_token,
            )
        )

    @app.put("/api/projects/<project_id>/images/<int:image_id>/bbox")
    def set_bbox(project_id: str, image_id: int):
        values = _payload()
        annotator_id, writer_token = _writer_fields(values)
        store = project_store(project_id, root)
        image_record = store.image(image_id)
        bbox = [float(value) for value in values.get("bbox", [])]
        crop = square_crop(
            bbox,
            image_record["width"],
            image_record["height"],
            scale=store.project()["crop_scale"],
        )
        revision = store.update_bbox(
            image_id=image_id,
            bbox=bbox,
            crop=crop,
            expected_revision=int(values.get("expected_revision", -1)),
            annotator_id=annotator_id,
            writer_token=writer_token,
            bbox_state="confirmed",
        )
        return jsonify({"bbox": bbox, "crop": crop, "revision": revision})

    @app.post("/api/projects/<project_id>/images/<int:image_id>/detect-bbox")
    def facebox(project_id: str, image_id: int):
        values = _payload()
        annotator_id, writer_token = _writer_fields(values)
        store = project_store(project_id, root)
        image_record = store.image(image_id)
        with Image.open(image_path(project_id, image_record, root)) as source:
            bbox = detect_bbox(source.convert("RGB"))
        crop = square_crop(
            bbox,
            image_record["width"],
            image_record["height"],
            scale=store.project()["crop_scale"],
        )
        revision = store.update_bbox(
            image_id=image_id,
            bbox=bbox,
            crop=crop,
            expected_revision=int(values.get("expected_revision", image_record["revision"])),
            annotator_id=annotator_id,
            writer_token=writer_token,
            bbox_state="needs_confirmation",
        )
        return jsonify(
            {
                "bbox": bbox,
                "crop": crop,
                "bbox_state": "needs_confirmation",
                "revision": revision,
            }
        )

    @app.get("/api/projects/<project_id>/images/<int:image_id>/consensus")
    def image_consensus(project_id: str, image_id: int):
        return jsonify(
            {"landmarks": review_consensus(project_store(project_id, root), image_id)}
        )

    @app.put(
        "/api/projects/<project_id>/images/<int:image_id>/adjudication/"
        "<int:landmark_index>"
    )
    def adjudicate(project_id: str, image_id: int, landmark_index: int):
        values = _payload()
        annotator_id, writer_token = _writer_fields(values)
        return jsonify(
            project_store(project_id, root).adjudicate(
                image_id=image_id,
                landmark_index=landmark_index,
                annotation=values.get("annotation", {}),
                annotator_id=annotator_id,
                writer_token=writer_token,
            )
        )

    @app.get(
        "/api/projects/<project_id>/images/<int:image_id>/suggestions/"
        "<provider>/<int:landmark_index>"
    )
    def suggestion(
        project_id: str,
        image_id: int,
        provider: str,
        landmark_index: int,
    ):
        store = project_store(project_id, root)
        record = store.image(image_id)
        if record["bbox"] is None:
            raise ValueError("Confirm a bbox before requesting suggestions")
        checkpoint = request.args.get("checkpoint")
        fingerprint = provider_fingerprint(provider, checkpoint)
        key = cache_key(record["sha256"], record["bbox"], fingerprint)
        cached = store.suggestion(key, landmark_index)
        if cached is not None:
            return jsonify({"cached": True, **cached})
        project = store.project()
        name = project["mapping"]["entries"][landmark_index]["name"]
        with Image.open(image_path(project_id, record, root)) as source:
            rgb = source.convert("RGB")
            if provider == "legacy":
                item = legacy_suggestions(rgb).get(name)
            else:
                item = custom_suggestions(
                    rgb,
                    str(checkpoint),
                    selected_landmark_index=landmark_index,
                ).get(landmark_index)
        if item is None:
            raise ValueError(f"{provider} did not predict {name}")
        payload = {"landmark_index": landmark_index, "name": name, **item}
        store.cache_suggestion(key, landmark_index, fingerprint, payload)
        return jsonify(
            {"cached": False, "provider_fingerprint": fingerprint, **payload}
        )

    @app.post("/api/projects/<project_id>/export")
    def export(project_id: str):
        values = _payload()
        _writer_fields(values)
        store = project_store(project_id, root)
        directory = export_project(store, root / project_id)
        return jsonify({"export_id": directory.name, "path": str(directory)}), 201

    return app
