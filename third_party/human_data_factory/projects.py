"""Project creation, image importing, safe lookup, and portable exports."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable
import uuid

from PIL import Image, ImageOps, UnidentifiedImageError

from .domain import validate_project_id
from .schema import LANDMARKS, SCHEMA_ID
from .store import AnnotationStore, utc_now


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECTS_ROOT = PACKAGE_DIR / "projects"
MULTIPIE_SOURCE = PACKAGE_DIR / "source_images" / "multipie"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contained(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Path escapes the human-data project root") from exc
    return candidate


def project_directory(
    project_id: str,
    root: Path = PROJECTS_ROOT,
) -> Path:
    project_id = validate_project_id(project_id)
    return _contained(root, root / project_id)


def project_store(
    project_id: str,
    root: Path = PROJECTS_ROOT,
) -> AnnotationStore:
    directory = project_directory(project_id, root)
    database = directory / "annotations.sqlite3"
    if not database.is_file():
        raise KeyError(f"Unknown project: {project_id}")
    return AnnotationStore(database)


def list_projects(root: Path = PROJECTS_ROOT) -> list[dict[str, Any]]:
    root = root.resolve()
    if not root.is_dir():
        return []
    projects = []
    for directory in sorted(root.iterdir()):
        database = directory / "annotations.sqlite3"
        if not directory.is_dir() or not database.is_file():
            continue
        try:
            projects.append(AnnotationStore(database).project())
        except (ValueError, OSError):
            continue
    return projects


def _normalize_image(
    source: Path,
    temporary_destination: Path,
) -> tuple[int, int, str]:
    try:
        with Image.open(source) as opened:
            opened.load()
            normalized = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = normalized.size
            if width < 2 or height < 2:
                raise ValueError("Image must be at least 2x2 pixels")
            temporary_destination.parent.mkdir(parents=True, exist_ok=True)
            normalized.save(
                temporary_destination,
                format="PNG",
                optimize=False,
            )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Corrupt or unsupported image: {source.name}") from exc
    return width, height, sha256_file(temporary_destination)


def _import_path(
    *,
    source: Path,
    original_relative_path: str,
    project_dir: Path,
    store: AnnotationStore,
    source_kind: str,
) -> dict[str, Any]:
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return {
            "status": "unsupported",
            "name": source.name,
            "detail": f"Unsupported extension: {suffix or '(none)'}",
        }
    temporary = project_dir / "images" / f".import-{uuid.uuid4().hex}.png"
    try:
        width, height, digest = _normalize_image(source, temporary)
        if store.image_hash_exists(digest):
            temporary.unlink(missing_ok=True)
            return {
                "status": "duplicate",
                "name": source.name,
                "sha256": digest,
            }
        destination = project_dir / "images" / f"{digest}.png"
        temporary.replace(destination)
        image_id = store.add_image(
            relative_path=destination.relative_to(project_dir).as_posix(),
            original_name=source.name,
            original_relative_path=original_relative_path,
            sha256=digest,
            width=width,
            height=height,
            source_kind=source_kind,
        )
        return {
            "status": "imported",
            "image_id": image_id,
            "name": source.name,
            "sha256": digest,
            "width": width,
            "height": height,
        }
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def create_project(
    *,
    project_id: str,
    annotator: str,
    source_kind: str,
    projects_root: Path = PROJECTS_ROOT,
    multipie_source: Path = MULTIPIE_SOURCE,
) -> tuple[Path, dict[str, Any]]:
    project_id = validate_project_id(project_id)
    projects_root = projects_root.resolve()
    projects_root.mkdir(parents=True, exist_ok=True)
    destination = project_directory(project_id, projects_root)
    if destination.exists():
        raise FileExistsError(f"Project already exists: {project_id}")
    staging = projects_root / f".creating-{project_id}-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    report = {
        "source_kind": source_kind,
        "imported": 0,
        "duplicates": [],
        "unsupported": [],
        "errors": [],
    }
    try:
        store = AnnotationStore(staging / "annotations.sqlite3")
        store.initialize_project(
            project_id=project_id,
            annotator=annotator,
            source_kind=source_kind,
        )
        (staging / "images").mkdir()
        (staging / "exports").mkdir()
        if source_kind == "multipie":
            source_root = multipie_source.resolve()
            if not source_root.is_dir():
                raise FileNotFoundError(
                    "Built-in unlabeled MultiPIE images are missing from "
                    f"{source_root}"
                )
            candidates = sorted(
                path
                for path in source_root.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
            )
            if not candidates:
                raise ValueError("Built-in MultiPIE image folder is empty")
            for source in candidates:
                try:
                    item = _import_path(
                        source=source,
                        original_relative_path=source.relative_to(
                            source_root
                        ).as_posix(),
                        project_dir=staging,
                        store=store,
                        source_kind="multipie",
                    )
                    _record_import_result(report, item)
                except ValueError as exc:
                    report["errors"].append(
                        {"name": source.name, "detail": str(exc)}
                    )
            if report["imported"] == 0:
                raise ValueError("No valid MultiPIE images could be imported")
        elif source_kind != "custom":
            raise ValueError("source_kind must be multipie or custom")
        staging.replace(destination)
        return destination, report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _record_import_result(
    report: dict[str, Any],
    item: dict[str, Any],
) -> None:
    status = item["status"]
    if status == "imported":
        report["imported"] += 1
    elif status == "duplicate":
        report["duplicates"].append(item)
    elif status == "unsupported":
        report["unsupported"].append(item)


def import_uploaded_files(
    *,
    project_id: str,
    uploads: Iterable[Any],
    original_relative_paths: list[str],
    projects_root: Path = PROJECTS_ROOT,
) -> dict[str, Any]:
    project_dir = project_directory(project_id, projects_root)
    store = project_store(project_id, projects_root)
    if store.project()["source_kind"] != "custom":
        raise ValueError("Only custom projects accept browser uploads")
    report = {
        "source_kind": "custom",
        "imported": 0,
        "duplicates": [],
        "unsupported": [],
        "errors": [],
    }
    staging_dir = project_dir / ".uploads"
    staging_dir.mkdir(exist_ok=True)
    for offset, upload in enumerate(uploads):
        raw_name = Path(str(getattr(upload, "filename", "") or "")).name
        relative_name = (
            original_relative_paths[offset]
            if offset < len(original_relative_paths)
            else raw_name
        )
        suffix = Path(raw_name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            report["unsupported"].append(
                {
                    "status": "unsupported",
                    "name": raw_name,
                    "detail": f"Unsupported extension: {suffix or '(none)'}",
                }
            )
            continue
        staged = staging_dir / f"{uuid.uuid4().hex}{suffix}"
        try:
            upload.save(staged)
            item = _import_path(
                source=staged,
                original_relative_path=relative_name,
                project_dir=project_dir,
                store=store,
                source_kind="custom",
            )
            _record_import_result(report, item)
        except (OSError, ValueError) as exc:
            report["errors"].append(
                {"name": raw_name, "detail": str(exc)}
            )
        finally:
            staged.unlink(missing_ok=True)
    try:
        staging_dir.rmdir()
    except OSError:
        pass
    report["project"] = store.project()
    return report


def image_path(
    project_id: str,
    image: dict[str, Any],
    projects_root: Path = PROJECTS_ROOT,
) -> Path:
    directory = project_directory(project_id, projects_root)
    path = _contained(directory, directory / image["relative_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Project image is missing: {path.name}")
    return path


def export_project(
    project_id: str,
    *,
    projects_root: Path = PROJECTS_ROOT,
) -> Path:
    project_dir = project_directory(project_id, projects_root)
    store = project_store(project_id, projects_root)
    project = store.project()
    export_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    staging = project_dir / "exports" / f".{export_id}.tmp"
    destination = project_dir / "exports" / export_id
    if staging.exists() or destination.exists():
        raise FileExistsError(f"Export already exists: {export_id}")
    staging.mkdir(parents=True)
    output_images = staging / "images"
    output_images.mkdir()
    state_counts = {
        "unreviewed": 0,
        "placed": 0,
        "unavailable": 0,
    }
    image_manifest: list[dict[str, Any]] = []
    labels_path = staging / "labels.jsonl"
    try:
        with labels_path.open("w", encoding="utf-8", newline="\n") as handle:
            for queued in store.queue():
                image = store.image(int(queued["id"]))
                source = image_path(project_id, image, projects_root)
                output_name = f"{int(image['id']):08d}.png"
                output_path = output_images / output_name
                shutil.copy2(source, output_path)
                output_digest = sha256_file(output_path)
                labels = []
                for label in image["labels"]:
                    state_counts[label["state"]] += 1
                    placed = label["state"] == "placed"
                    labels.append(
                        {
                            "index": int(label["index"]),
                            "id": label["id"],
                            "state": label["state"],
                            "x_px": label["x"] if placed else None,
                            "y_px": label["y"] if placed else None,
                            "x_norm": (
                                float(label["x"]) / (int(image["width"]) - 1)
                                if placed
                                else None
                            ),
                            "y_norm": (
                                float(label["y"]) / (int(image["height"]) - 1)
                                if placed
                                else None
                            ),
                            "annotator": label["annotator"],
                            "updated_at": label["updated_at"],
                            "revision": int(label["revision"]),
                        }
                    )
                record = {
                    "image_id": int(image["id"]),
                    "image_path": f"images/{output_name}",
                    "width": int(image["width"]),
                    "height": int(image["height"]),
                    "sha256": output_digest,
                    "source_kind": image["source_kind"],
                    "original_name": image["original_name"],
                    "original_relative_path": image[
                        "original_relative_path"
                    ],
                    "landmarks": labels,
                }
                handle.write(
                    json.dumps(
                        record,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                image_manifest.append(
                    {
                        "image_id": int(image["id"]),
                        "path": f"images/{output_name}",
                        "sha256": output_digest,
                    }
                )
        landmarks_path = staging / "landmarks.json"
        _write_json(
            landmarks_path,
            {
                "schema_id": SCHEMA_ID,
                "landmarks": list(LANDMARKS),
            },
        )
        labels_digest = sha256_file(labels_path)
        _write_json(
            staging / "manifest.json",
            {
                "schema_id": SCHEMA_ID,
                "project_id": project["project_id"],
                "export_id": export_id,
                "annotator": project["annotator"],
                "source_kind": project["source_kind"],
                "project_created_at": project["created_at"],
                "exported_at": utc_now(),
                "partial_exports_allowed": True,
                "counts": {
                    "images": len(image_manifest),
                    "landmarks_per_image": len(LANDMARKS),
                    "label_states": state_counts,
                },
                "checksums": {
                    "labels_jsonl_sha256": labels_digest,
                    "landmarks_json_sha256": sha256_file(landmarks_path),
                },
                "images": image_manifest,
            },
        )
        staging.replace(destination)
        store.record_export(
            export_id=export_id,
            relative_path=destination.relative_to(project_dir).as_posix(),
            labels_sha256=labels_digest,
        )
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)
