"""Transactional SQLite source of truth for Sir FaceIQ Annotator."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator
import uuid

from .domain import consensus, validate_annotation


SCHEMA_ID = "sir-faceiq-annotator-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StaleRevisionError(RuntimeError):
    pass


class WriterBusyError(RuntimeError):
    pass


class AnnotationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS project (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    schema_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    crop_scale REAL NOT NULL,
                    annotation_started INTEGER NOT NULL DEFAULT 0,
                    mapping_json TEXT NOT NULL,
                    split_json TEXT NOT NULL,
                    source_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS profiles (
                    annotator_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY,
                    source_kind TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    source_path TEXT,
                    sha256 TEXT NOT NULL UNIQUE,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    subject_id TEXT NOT NULL,
                    facing TEXT NOT NULL,
                    camera TEXT,
                    session TEXT,
                    bbox_json TEXT,
                    bbox_state TEXT NOT NULL,
                    crop_json TEXT,
                    split_name TEXT NOT NULL,
                    raw_points_json TEXT,
                    revision INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_images_queue
                    ON images(split_name, subject_id, id);
                CREATE TABLE IF NOT EXISTS labels (
                    image_id INTEGER NOT NULL REFERENCES images(id),
                    pass_number INTEGER NOT NULL,
                    landmark_index INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    x REAL,
                    y REAL,
                    confidence TEXT,
                    origin TEXT,
                    provider_fingerprint TEXT,
                    suggestion_exposed INTEGER NOT NULL DEFAULT 0,
                    annotator_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    PRIMARY KEY(image_id, pass_number, landmark_index)
                );
                CREATE TABLE IF NOT EXISTS pass_exposure (
                    image_id INTEGER NOT NULL REFERENCES images(id),
                    pass_number INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    exposed_at TEXT NOT NULL,
                    PRIMARY KEY(image_id, pass_number, source)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_uuid TEXT NOT NULL UNIQUE,
                    image_id INTEGER REFERENCES images(id),
                    pass_number INTEGER,
                    landmark_index INTEGER,
                    action TEXT NOT NULL,
                    before_json TEXT,
                    after_json TEXT,
                    annotator_id TEXT NOT NULL,
                    review_pass INTEGER,
                    created_at TEXT NOT NULL,
                    inverse_of INTEGER REFERENCES events(id)
                );
                CREATE INDEX IF NOT EXISTS idx_events_image ON events(image_id, id DESC);
                CREATE TABLE IF NOT EXISTS writer (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    token TEXT NOT NULL,
                    annotator_id TEXT NOT NULL,
                    acquired_at TEXT NOT NULL,
                    heartbeat_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS suggestions (
                    cache_key TEXT NOT NULL,
                    landmark_index INTEGER NOT NULL,
                    provider_fingerprint TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(cache_key, landmark_index)
                );
                CREATE TABLE IF NOT EXISTS adjudications (
                    image_id INTEGER NOT NULL REFERENCES images(id),
                    landmark_index INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    x REAL,
                    y REAL,
                    annotator_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(image_id, landmark_index)
                );
                CREATE TABLE IF NOT EXISTS exports (
                    export_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def initialize_project(
        self,
        *,
        project_id: str,
        mapping: dict[str, Any],
        split: dict[str, Any],
        source: dict[str, Any],
        crop_scale: float,
    ) -> None:
        with self.transaction() as db:
            if db.execute("SELECT 1 FROM project").fetchone():
                raise ValueError("Project database is already initialized")
            db.execute(
                """
                INSERT INTO project(
                    singleton, schema_id, project_id, created_at, crop_scale,
                    mapping_json, split_json, source_json
                ) VALUES(1, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    SCHEMA_ID,
                    project_id,
                    utc_now(),
                    crop_scale,
                    json.dumps(mapping, sort_keys=True),
                    json.dumps(split, sort_keys=True),
                    json.dumps(source, sort_keys=True),
                ),
            )

    def add_images(self, images: list[dict[str, Any]]) -> None:
        with self.transaction() as db:
            db.executemany(
                """
                INSERT INTO images(
                    source_kind, relative_path, source_path, sha256, width, height,
                    subject_id, facing, camera, session, bbox_json, bbox_state,
                    crop_json, split_name, raw_points_json, created_at
                ) VALUES(
                    :source_kind, :relative_path, :source_path, :sha256, :width,
                    :height, :subject_id, :facing, :camera, :session, :bbox_json,
                    :bbox_state, :crop_json, :split_name, :raw_points_json, :created_at
                )
                """,
                images,
            )

    def project(self) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM project WHERE singleton = 1").fetchone()
            if row is None:
                raise ValueError("Project database is not initialized")
            result = dict(row)
            for key in ("mapping_json", "split_json", "source_json"):
                result[key.removesuffix("_json")] = json.loads(result.pop(key))
            result["annotation_started"] = bool(result["annotation_started"])
            return result

    def register_profile(self, annotator_id: str) -> None:
        normalized = annotator_id.strip()
        if not normalized or len(normalized) > 80:
            raise ValueError("annotator_id must contain 1-80 characters")
        with self.transaction() as db:
            db.execute(
                "INSERT OR IGNORE INTO profiles(annotator_id, created_at) VALUES(?, ?)",
                (normalized, utc_now()),
            )

    def acquire_writer(self, annotator_id: str, token: str | None = None) -> str:
        token = token or uuid.uuid4().hex
        with self.transaction() as db:
            row = db.execute("SELECT * FROM writer WHERE singleton = 1").fetchone()
            if (
                row is not None
                and row["token"] != token
                and row["annotator_id"] != annotator_id
            ):
                raise WriterBusyError(
                    f"Project already has an active writer: {row['annotator_id']}"
                )
            db.execute(
                """
                INSERT INTO writer(singleton, token, annotator_id, acquired_at, heartbeat_at)
                VALUES(1, ?, ?, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    token=excluded.token, annotator_id=excluded.annotator_id,
                    acquired_at=excluded.acquired_at,
                    heartbeat_at=excluded.heartbeat_at
                """,
                (token, annotator_id, utc_now(), utc_now()),
            )
        return token

    def assert_writer(self, db: sqlite3.Connection, token: str) -> None:
        row = db.execute("SELECT token FROM writer WHERE singleton = 1").fetchone()
        if row is None or row["token"] != token:
            raise WriterBusyError("This browser is not the active project writer")
        db.execute("UPDATE writer SET heartbeat_at = ? WHERE singleton = 1", (utc_now(),))

    def release_writer(self, token: str) -> None:
        with self.transaction() as db:
            db.execute("DELETE FROM writer WHERE singleton = 1 AND token = ?", (token,))

    def reset_writer_after_server_restart(self) -> None:
        """Server-process leases never survive a local annotator restart."""

        with self.transaction() as db:
            db.execute("DELETE FROM writer WHERE singleton = 1")

    def queue(self, *, pass_number: int = 1, split_name: str | None = None) -> list[dict[str, Any]]:
        params: list[Any] = [pass_number]
        where = ""
        if split_name:
            where = "WHERE i.split_name = ?"
            params.append(split_name)
        with self.connect() as db:
            rows = db.execute(
                f"""
                SELECT i.*, COUNT(l.landmark_index) AS touched,
                       SUM(CASE WHEN l.state != 'unreviewed' THEN 1 ELSE 0 END) AS reviewed
                FROM images i
                LEFT JOIN labels l ON l.image_id = i.id AND l.pass_number = ?
                {where}
                GROUP BY i.id ORDER BY i.id
                """,
                params,
            ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            count = int(item.pop("reviewed") or 0)
            item["review_state"] = (
                "unstarted" if count == 0 else "reviewed" if count == 31 else "partial"
            )
            item["reviewed_count"] = count
            item.pop("touched", None)
            output.append(item)
        return output

    def image(self, image_id: int, *, pass_number: int = 1) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown image id: {image_id}")
            labels = {
                item["landmark_index"]: dict(item)
                for item in db.execute(
                    "SELECT * FROM labels WHERE image_id = ? AND pass_number = ?",
                    (image_id, pass_number),
                )
            }
            exposures = [
                item["source"]
                for item in db.execute(
                    "SELECT source FROM pass_exposure WHERE image_id = ? AND pass_number = ?",
                    (image_id, pass_number),
                )
            ]
        output = dict(row)
        for key in ("bbox_json", "crop_json", "raw_points_json"):
            output[key.removesuffix("_json")] = (
                json.loads(output.pop(key)) if output[key] is not None else None
            )
        output["labels"] = [
            labels.get(
                index,
                {
                    "image_id": image_id,
                    "pass_number": pass_number,
                    "landmark_index": index,
                    "state": "unreviewed",
                    "x": None,
                    "y": None,
                    "confidence": None,
                    "origin": None,
                    "provider_fingerprint": None,
                    "suggestion_exposed": 0,
                    "revision": 0,
                },
            )
            for index in range(31)
        ]
        output["exposures"] = exposures
        return output

    def mutate_label(
        self,
        *,
        image_id: int,
        pass_number: int,
        landmark_index: int,
        annotation: dict[str, Any],
        expected_revision: int,
        annotator_id: str,
        writer_token: str,
        action: str = "set_label",
        inverse_of: int | None = None,
    ) -> dict[str, Any]:
        if pass_number not in (1, 2):
            raise ValueError("pass_number must be 1 or 2")
        if not 0 <= landmark_index < 31:
            raise ValueError("landmark_index must be 0-30")
        clean = validate_annotation(annotation)
        with self.transaction() as db:
            self.assert_writer(db, writer_token)
            image = db.execute(
                "SELECT revision FROM images WHERE id = ?", (image_id,)
            ).fetchone()
            if image is None:
                raise KeyError(f"Unknown image id: {image_id}")
            if pass_number == 2:
                first_annotator = db.execute(
                    """
                    SELECT annotator_id FROM labels
                    WHERE image_id = ? AND pass_number = 1
                    LIMIT 1
                    """,
                    (image_id,),
                ).fetchone()
                if (
                    first_annotator is not None
                    and first_annotator["annotator_id"] == annotator_id
                ):
                    raise ValueError(
                        "Blind pass 2 requires a different annotator profile"
                    )
            current = db.execute(
                """
                SELECT * FROM labels
                WHERE image_id = ? AND pass_number = ? AND landmark_index = ?
                """,
                (image_id, pass_number, landmark_index),
            ).fetchone()
            current_revision = int(current["revision"]) if current else 0
            if expected_revision != current_revision:
                raise StaleRevisionError(
                    f"Stale label revision {expected_revision}; current is {current_revision}"
                )
            next_revision = current_revision + 1
            before = dict(current) if current else {
                "state": "unreviewed", "x": None, "y": None, "confidence": None,
                "origin": None, "provider_fingerprint": None,
                "suggestion_exposed": False, "revision": 0,
            }
            after = {**clean, "revision": next_revision}
            now = utc_now()
            db.execute(
                """
                INSERT INTO labels(
                    image_id, pass_number, landmark_index, state, x, y, confidence,
                    origin, provider_fingerprint, suggestion_exposed, annotator_id,
                    updated_at, revision
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id, pass_number, landmark_index) DO UPDATE SET
                    state=excluded.state, x=excluded.x, y=excluded.y,
                    confidence=excluded.confidence, origin=excluded.origin,
                    provider_fingerprint=excluded.provider_fingerprint,
                    suggestion_exposed=excluded.suggestion_exposed,
                    annotator_id=excluded.annotator_id, updated_at=excluded.updated_at,
                    revision=excluded.revision
                """,
                (
                    image_id, pass_number, landmark_index, clean["state"], clean["x"],
                    clean["y"], clean["confidence"], clean["origin"],
                    clean["provider_fingerprint"], int(clean["suggestion_exposed"]),
                    annotator_id, now, next_revision,
                ),
            )
            db.execute(
                "UPDATE images SET revision = revision + 1 WHERE id = ?", (image_id,)
            )
            db.execute("UPDATE project SET annotation_started = 1 WHERE singleton = 1")
            event_id = db.execute(
                """
                INSERT INTO events(
                    event_uuid, image_id, pass_number, landmark_index, action,
                    before_json, after_json, annotator_id, review_pass, created_at,
                    inverse_of
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex, image_id, pass_number, landmark_index, action,
                    json.dumps(before, sort_keys=True), json.dumps(after, sort_keys=True),
                    annotator_id, pass_number, now, inverse_of,
                ),
            ).lastrowid
        return {**after, "event_id": event_id, "updated_at": now}

    def mark_exposure(
        self,
        *,
        image_id: int,
        pass_number: int,
        source: str,
        writer_token: str,
        annotator_id: str,
    ) -> None:
        if pass_number == 2:
            raise ValueError("Blind second passes cannot expose annotation aids")
        if source not in {"model", "previous_image", "multipie"}:
            raise ValueError("Unknown exposure source")
        with self.transaction() as db:
            self.assert_writer(db, writer_token)
            db.execute(
                """
                INSERT OR IGNORE INTO pass_exposure(image_id, pass_number, source, exposed_at)
                VALUES(?, ?, ?, ?)
                """,
                (image_id, pass_number, source, utc_now()),
            )
            db.execute(
                """
                INSERT INTO events(
                    event_uuid, image_id, pass_number, action, annotator_id,
                    review_pass, created_at, after_json
                ) VALUES(?, ?, ?, 'exposure', ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex, image_id, pass_number, annotator_id,
                    pass_number, utc_now(), json.dumps({"source": source}),
                ),
            )

    def history(self, image_id: int, limit: int = 200) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM events WHERE image_id = ? ORDER BY id DESC LIMIT ?",
                (image_id, min(max(limit, 1), 1000)),
            ).fetchall()
        return [dict(row) for row in rows]

    def undo(
        self,
        *,
        event_id: int,
        annotator_id: str,
        writer_token: str,
    ) -> dict[str, Any]:
        with self.connect() as db:
            event = db.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        if event is None or event["landmark_index"] is None or not event["before_json"]:
            raise ValueError("Event cannot be undone")
        current = self.image(event["image_id"], pass_number=event["pass_number"])[
            "labels"
        ][event["landmark_index"]]
        return self.mutate_label(
            image_id=event["image_id"],
            pass_number=event["pass_number"],
            landmark_index=event["landmark_index"],
            annotation=json.loads(event["before_json"]),
            expected_revision=current["revision"],
            annotator_id=annotator_id,
            writer_token=writer_token,
            action="redo" if event["action"] == "undo" else "undo",
            inverse_of=event_id,
        )

    def update_bbox(
        self,
        *,
        image_id: int,
        bbox: list[float],
        crop: list[float],
        expected_revision: int,
        annotator_id: str,
        writer_token: str,
        bbox_state: str = "confirmed",
    ) -> int:
        if bbox_state not in {"confirmed", "needs_confirmation", "needs_redraw"}:
            raise ValueError("Invalid bbox_state")
        with self.transaction() as db:
            self.assert_writer(db, writer_token)
            row = db.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown image id: {image_id}")
            if row["revision"] != expected_revision:
                raise StaleRevisionError("Stale image revision")
            next_revision = expected_revision + 1
            db.execute(
                """
                UPDATE images SET bbox_json=?, crop_json=?, bbox_state=?,
                    revision=? WHERE id=?
                """,
                (
                    json.dumps(bbox), json.dumps(crop), bbox_state,
                    next_revision, image_id,
                ),
            )
            db.execute(
                """
                INSERT INTO events(
                    event_uuid, image_id, action, before_json, after_json,
                    annotator_id, created_at
                ) VALUES(?, ?, 'set_bbox', ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex, image_id,
                    json.dumps({"bbox": json.loads(row["bbox_json"]) if row["bbox_json"] else None}),
                    json.dumps({"bbox": bbox, "crop": crop}),
                    annotator_id, utc_now(),
                ),
            )
        return next_revision

    def adjudicate(
        self,
        *,
        image_id: int,
        landmark_index: int,
        annotation: dict[str, Any],
        annotator_id: str,
        writer_token: str,
    ) -> dict[str, Any]:
        clean = validate_annotation(annotation)
        if clean["state"] == "unreviewed":
            raise ValueError("Adjudication must choose a definitive state")
        if not 0 <= landmark_index < 31:
            raise ValueError("landmark_index must be 0-30")
        with self.transaction() as db:
            self.assert_writer(db, writer_token)
            if db.execute(
                "SELECT 1 FROM images WHERE id = ?", (image_id,)
            ).fetchone() is None:
                raise KeyError(f"Unknown image id: {image_id}")
            image = db.execute(
                "SELECT bbox_json FROM images WHERE id = ?", (image_id,)
            ).fetchone()
            labels = db.execute(
                """
                SELECT * FROM labels
                WHERE image_id = ? AND landmark_index = ? AND pass_number IN (1, 2)
                ORDER BY pass_number
                """,
                (image_id, landmark_index),
            ).fetchall()
            if len(labels) != 2:
                raise ValueError(
                    "Adjudication requires completed first and blind second labels"
                )
            if annotator_id in {row["annotator_id"] for row in labels}:
                raise ValueError("Adjudication requires a third annotator profile")
            bbox = json.loads(image["bbox_json"]) if image["bbox_json"] else None
            if bbox is None:
                raise ValueError("Adjudication requires a confirmed bbox")
            if not consensus(dict(labels[0]), dict(labels[1]), bbox).requires_adjudication:
                raise ValueError("These labels already have automatic consensus")
            now = utc_now()
            db.execute(
                """
                INSERT INTO adjudications(
                    image_id, landmark_index, state, x, y, annotator_id, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id, landmark_index) DO UPDATE SET
                    state=excluded.state, x=excluded.x, y=excluded.y,
                    annotator_id=excluded.annotator_id, created_at=excluded.created_at
                """,
                (
                    image_id, landmark_index, clean["state"], clean["x"], clean["y"],
                    annotator_id, now,
                ),
            )
            db.execute(
                """
                INSERT INTO events(
                    event_uuid, image_id, landmark_index, action, after_json,
                    annotator_id, review_pass, created_at
                ) VALUES(?, ?, ?, 'adjudicate', ?, ?, 3, ?)
                """,
                (
                    uuid.uuid4().hex, image_id, landmark_index,
                    json.dumps(clean, sort_keys=True), annotator_id, now,
                ),
            )
        return {**clean, "annotator_id": annotator_id, "created_at": now}

    def cache_suggestion(
        self,
        cache_key: str,
        landmark_index: int,
        fingerprint: str,
        payload: dict[str, Any],
    ) -> None:
        with self.transaction() as db:
            db.execute(
                """
                INSERT OR REPLACE INTO suggestions(
                    cache_key, landmark_index, provider_fingerprint, payload_json, created_at
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (cache_key, landmark_index, fingerprint, json.dumps(payload), utc_now()),
            )

    def suggestion(self, cache_key: str, landmark_index: int) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                """
                SELECT * FROM suggestions WHERE cache_key = ? AND landmark_index = ?
                """,
                (cache_key, landmark_index),
            ).fetchone()
        if row is None:
            return None
        return {
            "provider_fingerprint": row["provider_fingerprint"],
            **json.loads(row["payload_json"]),
        }
