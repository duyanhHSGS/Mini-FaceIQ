"""Loopback Flask application for Sir FaceIQ human annotation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file

from .domain import validate_landmark_index
from .projects import (
    MULTIPIE_SOURCE,
    PROJECTS_ROOT,
    create_project,
    export_project,
    image_path,
    import_uploaded_files,
    list_projects,
    project_store,
)
from .schema import LANDMARKS, SCHEMA_ID
from .store import StaleRevisionError


def _payload() -> dict[str, Any]:
    value = request.get_json(silent=True)
    if not isinstance(value, dict):
        raise ValueError("Request body must be a JSON object")
    return value


def create_app(
    *,
    projects_root: Path = PROJECTS_ROOT,
    multipie_source: Path = MULTIPIE_SOURCE,
) -> Flask:
    static_dir = Path(__file__).resolve().parent / "static"
    app = Flask(
        __name__,
        static_folder=str(static_dir),
        static_url_path="/static",
    )
    app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024

    @app.errorhandler(ValueError)
    def invalid(error):
        return jsonify({"error": str(error)}), 400

    @app.errorhandler(KeyError)
    def missing(error):
        return jsonify({"error": str(error).strip("'")}), 404

    @app.errorhandler(FileNotFoundError)
    def missing_file(error):
        return jsonify({"error": str(error)}), 404

    @app.errorhandler(FileExistsError)
    def conflict(error):
        return jsonify({"error": str(error)}), 409

    @app.errorhandler(StaleRevisionError)
    def stale(error):
        return jsonify({"error": str(error), "code": "stale_revision"}), 409

    @app.get("/")
    def index():
        return send_file(static_dir / "index.html")

    @app.get("/api/schema")
    def schema():
        return jsonify(
            {
                "schema_id": SCHEMA_ID,
                "landmarks": list(LANDMARKS),
            }
        )

    @app.get("/api/projects")
    def projects():
        return jsonify({"projects": list_projects(projects_root)})

    @app.post("/api/projects")
    def new_project():
        values = _payload()
        directory, report = create_project(
            project_id=str(values.get("project_id", "")),
            annotator=str(values.get("annotator", "")),
            source_kind=str(values.get("source_kind", "")),
            projects_root=projects_root,
            multipie_source=multipie_source,
        )
        project = project_store(directory.name, projects_root).project()
        return jsonify({"project": project, "import_report": report}), 201

    @app.get("/api/projects/<project_id>")
    def project(project_id: str):
        return jsonify(project_store(project_id, projects_root).project())

    @app.post("/api/projects/<project_id>/images")
    def upload_images(project_id: str):
        uploads = request.files.getlist("files")
        if not uploads:
            raise ValueError("Upload batch contains no files")
        raw_paths = request.form.get("relative_paths", "[]")
        try:
            relative_paths = json.loads(raw_paths)
        except json.JSONDecodeError as exc:
            raise ValueError("relative_paths must be valid JSON") from exc
        if not isinstance(relative_paths, list) or not all(
            isinstance(item, str) for item in relative_paths
        ):
            raise ValueError("relative_paths must be a JSON string array")
        return jsonify(
            import_uploaded_files(
                project_id=project_id,
                uploads=uploads,
                original_relative_paths=relative_paths,
                projects_root=projects_root,
            )
        )

    @app.get("/api/projects/<project_id>/queue")
    def queue(project_id: str):
        mode = request.args.get("mode", "all")
        if mode not in {"all", "focused"}:
            raise ValueError("mode must be all or focused")
        focused_index = None
        if mode == "focused":
            focused_index = validate_landmark_index(
                int(request.args.get("landmark_index", "0"))
            )
        images = project_store(project_id, projects_root).queue(
            focused_landmark_index=focused_index
        )
        return jsonify(
            {
                "mode": mode,
                "focused_landmark_index": focused_index,
                "images": images,
            }
        )

    @app.get("/api/projects/<project_id>/images/<int:image_id>")
    def image(project_id: str, image_id: int):
        return jsonify(
            project_store(project_id, projects_root).image(image_id)
        )

    @app.get("/api/projects/<project_id>/images/<int:image_id>/pixels")
    def pixels(project_id: str, image_id: int):
        store = project_store(project_id, projects_root)
        record = store.image(image_id)
        return send_file(
            image_path(project_id, record, projects_root),
            mimetype="image/png",
            max_age=0,
        )

    @app.put(
        "/api/projects/<project_id>/images/<int:image_id>/"
        "landmarks/<int:landmark_index>"
    )
    def set_landmark(
        project_id: str,
        image_id: int,
        landmark_index: int,
    ):
        values = _payload()
        saved = project_store(project_id, projects_root).mutate_label(
            image_id=image_id,
            landmark_index=landmark_index,
            annotation=values.get("annotation", {}),
            expected_revision=int(values.get("expected_revision", -1)),
            annotator=str(values.get("annotator", "")),
        )
        return jsonify(saved)

    @app.get("/api/projects/<project_id>/images/<int:image_id>/history")
    def history(project_id: str, image_id: int):
        return jsonify(
            {
                "events": project_store(
                    project_id,
                    projects_root,
                ).history(image_id)
            }
        )

    @app.post("/api/projects/<project_id>/images/<int:image_id>/undo")
    def undo(project_id: str, image_id: int):
        values = _payload()
        return jsonify(
            project_store(project_id, projects_root).undo_last(
                image_id=image_id,
                annotator=str(values.get("annotator", "")),
            )
        )

    @app.post("/api/projects/<project_id>/export")
    def export(project_id: str):
        directory = export_project(
            project_id,
            projects_root=projects_root,
        )
        return jsonify(
            {
                "export_id": directory.name,
                "path": str(directory),
            }
        ), 201

    return app
