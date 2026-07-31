"""Project creation, safe imports, image serving, review, and immutable exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable
import uuid

from PIL import Image, ImageOps, UnidentifiedImageError

from ..dataset import load_profile_annotations
from ..mapping import LandmarkMapping, load_landmark_mapping
from .domain import (
    ANATOMICAL_SWEEP,
    CROP_SCALE,
    FACINGS,
    consensus,
    deterministic_split,
    point_in_crop,
    square_crop,
    validate_annotation,
    validate_project_id,
)
from .store import AnnotationStore, SCHEMA_ID, utc_now


PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]
PROJECTS_ROOT = REPOSITORY_ROOT / "git-plz-ignore" / "profile_annotation_projects"
DEFAULT_MAPPING_PATH = PACKAGE_DIR / "user-custom.txt"
DEFAULT_MULTIPIE_PATH = (
    REPOSITORY_ROOT / "git-plz-ignore" / "MultiPIE" / "MultiPIE_profile_train.txt"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mapping_snapshot(mapping: LandmarkMapping) -> dict[str, Any]:
    if len(mapping.entries) != 31:
        raise ValueError("Annotator projects require exactly 31 worksheet landmarks")
    if tuple(entry.name for entry in mapping.entries) != ANATOMICAL_SWEEP:
        # Worksheet order is the model contract. The UX sweep is stored separately.
        worksheet_names = [entry.name for entry in mapping.entries]
    else:
        worksheet_names = list(ANATOMICAL_SWEEP)
    return {
        **mapping.snapshot(),
        "entries": [
            {
                "model_index": entry.model_index,
                "name": entry.name,
                "dataset_index": entry.dataset_index,
            }
            for entry in mapping.entries
        ],
        "worksheet_names": worksheet_names,
        "anatomical_sweep": list(ANATOMICAL_SWEEP),
    }


def _split_lookup(split: dict[str, Any]) -> dict[str, str]:
    return {
        subject: name
        for name in ("train", "validation", "test")
        for subject in split[name]
    }


def _contained(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes allowed root: {candidate}") from exc
    return candidate


def _safe_manifest_path(manifest_path: str | Path) -> Path:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Import manifest not found: {path}")
    if path.suffix.lower() not in {".csv", ".jsonl"}:
        raise ValueError("Arbitrary import manifest must be CSV or JSONL")
    return path


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    else:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected an object")
                rows.append(value)
    if not rows:
        raise ValueError("Import manifest is empty")
    return rows


def _bbox_from_row(row: dict[str, Any]) -> list[float] | None:
    if row.get("bbox") not in (None, ""):
        value = row["bbox"]
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, list) or len(value) != 4:
            raise ValueError("bbox must be a four-value JSON array")
        return [float(item) for item in value]
    keys = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
    if any(row.get(key) not in (None, "") for key in keys):
        if not all(row.get(key) not in (None, "") for key in keys):
            raise ValueError("bbox_x1/y1/x2/y2 must be supplied together")
        return [float(row[key]) for key in keys]
    return None


def _normalized_copy(source: Path, destination: Path) -> tuple[int, int, str]:
    try:
        with Image.open(source) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            normalized.load()
            width, height = normalized.size
            if width <= 0 or height <= 0:
                raise ValueError("Image has invalid dimensions")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            normalized.save(temporary, format="PNG", optimize=True)
            os.replace(temporary, destination)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError(f"Corrupt or unsupported image: {source}") from exc
    return width, height, sha256_file(destination)


def _inspect_reference(source: Path) -> tuple[int, int, str]:
    try:
        with Image.open(source) as image:
            image.load()
            width, height = image.size
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Corrupt or unsupported image: {source}") from exc
    return width, height, sha256_file(source)


def list_projects(root: Path = PROJECTS_ROOT) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    projects = []
    for child in sorted(root.iterdir()):
        database = child / "annotations.sqlite3"
        if not database.is_file():
            continue
        try:
            metadata = AnnotationStore(database).project()
        except (OSError, ValueError):
            continue
        projects.append(
            {
                "project_id": metadata["project_id"],
                "created_at": metadata["created_at"],
                "path": str(child),
            }
        )
    return projects


def create_multipie_project(
    project_id: str,
    *,
    annotation_path: str | Path = DEFAULT_MULTIPIE_PATH,
    mapping_path: str | Path = DEFAULT_MAPPING_PATH,
    projects_root: Path = PROJECTS_ROOT,
) -> Path:
    project_id = validate_project_id(project_id)
    project_dir = projects_root.resolve() / project_id
    if project_dir.exists():
        raise FileExistsError(f"Project already exists: {project_id}")
    mapping = load_landmark_mapping(mapping_path)
    records = load_profile_annotations(annotation_path)
    split = deterministic_split(record.subject_id for record in records)
    lookup = _split_lookup(split)
    image_rows = []
    hashes: set[str] = set()
    for record in records:
        width, height, digest = _inspect_reference(record.image_path)
        if digest in hashes:
            raise ValueError(f"Duplicate image content: {record.image_path}")
        hashes.add(digest)
        bbox = [float(value) for value in record.bbox_xyxy]
        crop = square_crop(bbox, width, height, scale=CROP_SCALE)
        image_rows.append(
            {
                "source_kind": "multipie",
                "relative_path": os.path.relpath(record.image_path, REPOSITORY_ROOT).replace("\\", "/"),
                "source_path": str(record.image_path),
                "sha256": digest,
                "width": width,
                "height": height,
                "subject_id": record.subject_id,
                "facing": "left" if record.camera_code in {"110", "120"} else "right",
                "camera": record.camera_code,
                "session": None,
                "bbox_json": json.dumps(bbox),
                "bbox_state": "confirmed",
                "crop_json": json.dumps(crop),
                "split_name": lookup[record.subject_id],
                "raw_points_json": json.dumps(record.landmarks_xy.tolist()),
                "created_at": utc_now(),
            }
        )
    project_dir.mkdir(parents=True)
    try:
        store = AnnotationStore(project_dir / "annotations.sqlite3")
        store.initialize_project(
            project_id=project_id,
            mapping=mapping_snapshot(mapping),
            split=split,
            source={
                "kind": "multipie",
                "annotation_path": os.path.relpath(
                    Path(annotation_path).resolve(), REPOSITORY_ROOT
                ).replace("\\", "/"),
                "annotation_sha256": sha256_file(Path(annotation_path).resolve()),
            },
            crop_scale=CROP_SCALE,
        )
        store.add_images(image_rows)
    except Exception:
        shutil.rmtree(project_dir)
        raise
    return project_dir


def create_arbitrary_project(
    project_id: str,
    manifest_path: str | Path,
    *,
    mapping_path: str | Path = DEFAULT_MAPPING_PATH,
    projects_root: Path = PROJECTS_ROOT,
) -> Path:
    project_id = validate_project_id(project_id)
    manifest = _safe_manifest_path(manifest_path)
    rows = _manifest_rows(manifest)
    required = {"image_path", "subject_id", "facing"}
    for index, row in enumerate(rows, 1):
        missing = [key for key in required if not str(row.get(key, "")).strip()]
        if missing:
            raise ValueError(f"Manifest row {index} missing: {', '.join(missing)}")
        if str(row["facing"]).lower() not in FACINGS:
            raise ValueError(f"Manifest row {index}: facing must be left or right")
    split = deterministic_split(str(row["subject_id"]).strip() for row in rows)
    lookup = _split_lookup(split)
    mapping = load_landmark_mapping(mapping_path)
    project_dir = projects_root.resolve() / project_id
    if project_dir.exists():
        raise FileExistsError(f"Project already exists: {project_id}")
    project_dir.mkdir(parents=True)
    images_dir = project_dir / "images"
    prepared = []
    hashes: set[str] = set()
    try:
        for index, row in enumerate(rows, 1):
            source = Path(str(row["image_path"])).expanduser()
            if not source.is_absolute():
                source = manifest.parent / source
            source = source.resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Manifest row {index}: image not found: {source}")
            destination = images_dir / f"{index:08d}.png"
            width, height, digest = _normalized_copy(source, destination)
            if digest in hashes:
                raise ValueError(f"Manifest row {index}: duplicate image content")
            hashes.add(digest)
            bbox = _bbox_from_row(row)
            bbox_state = "needs_confirmation"
            if bbox is None:
                from .suggestions import detect_bbox

                try:
                    with Image.open(destination) as normalized:
                        bbox = detect_bbox(normalized.convert("RGB"))
                except RuntimeError:
                    bbox_state = "needs_redraw"
            crop = square_crop(bbox, width, height, scale=CROP_SCALE) if bbox else None
            prepared.append(
                {
                    "source_kind": "arbitrary",
                    "relative_path": destination.relative_to(project_dir).as_posix(),
                    "source_path": str(source),
                    "sha256": digest,
                    "width": width,
                    "height": height,
                    "subject_id": str(row["subject_id"]).strip(),
                    "facing": str(row["facing"]).lower(),
                    "camera": row.get("camera") or None,
                    "session": row.get("session") or None,
                    "bbox_json": json.dumps(bbox) if bbox else None,
                    "bbox_state": bbox_state,
                    "crop_json": json.dumps(crop) if crop else None,
                    "split_name": lookup[str(row["subject_id"]).strip()],
                    "raw_points_json": None,
                    "created_at": utc_now(),
                }
            )
        store = AnnotationStore(project_dir / "annotations.sqlite3")
        store.initialize_project(
            project_id=project_id,
            mapping=mapping_snapshot(mapping),
            split=split,
            source={
                "kind": "arbitrary",
                "manifest_name": manifest.name,
                "manifest_sha256": sha256_file(manifest),
            },
            crop_scale=CROP_SCALE,
        )
        store.add_images(prepared)
    except Exception:
        shutil.rmtree(project_dir)
        raise
    return project_dir


def project_store(project_id: str, root: Path = PROJECTS_ROOT) -> AnnotationStore:
    project_id = validate_project_id(project_id)
    directory = _contained(root, root / project_id)
    database = directory / "annotations.sqlite3"
    if not database.is_file():
        raise KeyError(f"Unknown project: {project_id}")
    return AnnotationStore(database)


def image_path(project_id: str, image: dict[str, Any], root: Path = PROJECTS_ROOT) -> Path:
    if image["source_kind"] == "arbitrary":
        return _contained(root / project_id, root / project_id / image["relative_path"])
    source = Path(image["source_path"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Referenced Multi-PIE image is missing: {source}")
    return source


def review_consensus(store: AnnotationStore, image_id: int) -> list[dict[str, Any]]:
    image_first = store.image(image_id, pass_number=1)
    image_second = store.image(image_id, pass_number=2)
    if image_first["bbox"] is None:
        raise ValueError("A confirmed bbox is required for consensus")
    output = []
    for index in range(31):
        first = image_first["labels"][index]
        second = image_second["labels"][index]
        decision = consensus(first, second, image_first["bbox"])
        output.append(
            {
                "landmark_index": index,
                "state": decision.state,
                "x": decision.x,
                "y": decision.y,
                "requires_adjudication": decision.requires_adjudication,
                "reason": decision.reason,
            }
        )
    return output


def _training_truth(
    store: AnnotationStore,
    image: dict[str, Any],
    landmark_index: int,
) -> tuple[dict[str, Any], str]:
    with store.connect() as db:
        adjudicated = db.execute(
            "SELECT * FROM adjudications WHERE image_id=? AND landmark_index=?",
            (image["id"], landmark_index),
        ).fetchone()
    if adjudicated:
        return dict(adjudicated), "adjudicated"
    first = image["labels"][landmark_index]
    if image["split_name"] != "test":
        return first, "single_pass"
    second_image = store.image(image["id"], pass_number=2)
    second = second_image["labels"][landmark_index]
    decision = consensus(first, second, image["bbox"]) if image["bbox"] else None
    if decision and not decision.requires_adjudication:
        return {
            "state": decision.state,
            "x": decision.x,
            "y": decision.y,
            "suggestion_exposed": bool(
                first.get("suggestion_exposed") or second.get("suggestion_exposed")
            ),
        }, "blind_consensus"
    return {"state": "unreviewed", "x": None, "y": None}, "unavailable"


def export_project(store: AnnotationStore, project_dir: Path) -> Path:
    project = store.project()
    export_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    export_dir = project_dir / "exports" / export_id
    export_dir.mkdir(parents=True)
    queue = store.queue()
    labels_records: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    state_counts = {state: 0 for state in ("unreviewed", "placed", "occluded", "out_of_frame")}
    per_landmark = {
        item["name"]: {
            "placed_truth": 0,
            "test_subjects": set(),
            "unexposed_test_subjects": set(),
            "suggestion_exposed": 0,
        }
        for item in project["mapping"]["entries"]
    }
    crop_overflows = []
    disagreements = 0
    source_hashes = []
    for queued in queue:
        image = store.image(queued["id"], pass_number=1)
        second_pass_image = store.image(queued["id"], pass_number=2)
        labels = []
        source_hashes.append(
            {"image_id": image["id"], "path": image["relative_path"], "sha256": image["sha256"]}
        )
        consensus_rows = review_consensus(store, image["id"])
        disagreements += sum(item["requires_adjudication"] for item in consensus_rows)
        for mapping_entry in project["mapping"]["entries"]:
            index = mapping_entry["model_index"]
            raw = image["labels"][index]
            second_raw = second_pass_image["labels"][index]
            truth, truth_source = _training_truth(store, image, index)
            state_counts[raw["state"]] += 1
            outside_crop = bool(
                raw["state"] == "placed"
                and image["crop"]
                and not point_in_crop(raw["x"], raw["y"], image["crop"])
            )
            if outside_crop:
                crop_overflows.append(
                    {"image_id": image["id"], "landmark": mapping_entry["name"]}
                )
            training_visible = bool(
                truth["state"] == "placed"
                and not outside_crop
                and (image["split_name"] != "test" or truth_source in {"blind_consensus", "adjudicated"})
            )
            exposed = bool(raw.get("suggestion_exposed") or image["exposures"])
            if truth["state"] == "placed" and training_visible:
                per_landmark[mapping_entry["name"]]["placed_truth"] += 1
                if image["split_name"] == "test" and truth_source in {"blind_consensus", "adjudicated"}:
                    per_landmark[mapping_entry["name"]]["test_subjects"].add(image["subject_id"])
                    if not exposed:
                        per_landmark[mapping_entry["name"]]["unexposed_test_subjects"].add(
                            image["subject_id"]
                        )
            if exposed:
                per_landmark[mapping_entry["name"]]["suggestion_exposed"] += 1
            exported = {
                "model_index": index,
                "name": mapping_entry["name"],
                "state": raw["state"],
                "x": raw["x"],
                "y": raw["y"],
                "confidence": raw["confidence"],
                "origin": raw["origin"],
                "provider_fingerprint": raw["provider_fingerprint"],
                "suggestion_exposed": exposed,
                "annotator_id": raw.get("annotator_id"),
                "timestamp": raw.get("updated_at"),
                "revision": raw["revision"],
                "review_pass": 1,
                "reviews": [
                    {
                        "review_pass": pass_number,
                        "state": review["state"],
                        "x": review["x"],
                        "y": review["y"],
                        "confidence": review["confidence"],
                        "origin": review["origin"],
                        "provider_fingerprint": review["provider_fingerprint"],
                        "suggestion_exposed": bool(
                            review.get("suggestion_exposed")
                        ),
                        "annotator_id": review.get("annotator_id"),
                        "timestamp": review.get("updated_at"),
                        "revision": review["revision"],
                    }
                    for pass_number, review in ((1, raw), (2, second_raw))
                    if review["revision"] > 0
                ],
                "outside_crop": outside_crop,
                "training_visible": training_visible,
                "truth_source": truth_source,
                "truth_state": truth["state"],
                "truth_x": truth.get("x"),
                "truth_y": truth.get("y"),
            }
            labels.append(exported)
            long_rows.append({"image_id": image["id"], **exported})
        labels_records.append(
            {
                "image_id": image["id"],
                "image_path": image["relative_path"],
                "sha256": image["sha256"],
                "width": image["width"],
                "height": image["height"],
                "subject_id": image["subject_id"],
                "facing": image["facing"],
                "camera": image["camera"],
                "session": image["session"],
                "split": image["split_name"],
                "bbox": image["bbox"],
                "bbox_state": image["bbox_state"],
                "crop": image["crop"],
                "crop_scale": project["crop_scale"],
                "review_state": queued["review_state"],
                "landmarks": labels,
            }
        )
    qa_landmarks = {}
    for name, values in per_landmark.items():
        subject_count = len(values.pop("test_subjects"))
        unexposed_subject_count = len(values.pop("unexposed_test_subjects"))
        qa_landmarks[name] = {
            **values,
            "blind_test_subjects": subject_count,
            "unexposed_blind_test_subjects": unexposed_subject_count,
            "graduation_eligible": unexposed_subject_count >= 50,
        }
    qa = {
        "state_counts": state_counts,
        "per_landmark": qa_landmarks,
        "disagreements_requiring_adjudication": disagreements,
        "crop_overflows": crop_overflows,
    }
    manifest = {
        "schema_id": SCHEMA_ID,
        "project_id": project["project_id"],
        "export_id": export_id,
        "created_at": utc_now(),
        "landmark_snapshot": project["mapping"],
        "source": project["source"],
        "source_hashes": source_hashes,
        "split": project["split"],
        "crop_policy": {"scale": project["crop_scale"], "outside_points_masked": True},
        "counts": {"images": len(labels_records), "rows": len(long_rows)},
        "provenance": {"editable_source": "annotations.sqlite3", "export": "immutable"},
    }
    _write_json(export_dir / "manifest.json", manifest)
    _write_json(export_dir / "split.json", project["split"])
    _write_json(export_dir / "qa.json", qa)
    with (export_dir / "labels.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in labels_records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    fieldnames = list(long_rows[0]) if long_rows else ["image_id"]
    with (export_dir / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(long_rows)
    snapshot_hash = sha256_file(export_dir / "labels.jsonl")
    with store.transaction() as db:
        db.execute(
            "INSERT INTO exports(export_id, path, sha256, created_at) VALUES(?, ?, ?, ?)",
            (export_id, str(export_dir), snapshot_hash, utc_now()),
        )
    return export_dir


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
