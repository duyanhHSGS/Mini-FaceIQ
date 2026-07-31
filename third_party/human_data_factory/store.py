"""Transactional SQLite source of truth for human landmark labels."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator
import uuid

from .domain import (
    validate_annotator,
    validate_label,
    validate_landmark_index,
)
from .schema import LANDMARKS, SCHEMA_ID


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StaleRevisionError(RuntimeError):
    """Raised when a browser edits an older label revision."""


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
                    annotator TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    relative_path TEXT NOT NULL UNIQUE,
                    original_name TEXT NOT NULL,
                    original_relative_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    source_kind TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS labels (
                    image_id INTEGER NOT NULL REFERENCES images(id),
                    landmark_index INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    x REAL,
                    y REAL,
                    annotator TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    PRIMARY KEY(image_id, landmark_index)
                );
                CREATE INDEX IF NOT EXISTS idx_labels_progress
                    ON labels(landmark_index, state, image_id);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_uuid TEXT NOT NULL UNIQUE,
                    image_id INTEGER NOT NULL REFERENCES images(id),
                    landmark_index INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    before_json TEXT NOT NULL,
                    after_json TEXT NOT NULL,
                    annotator TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    undone INTEGER NOT NULL DEFAULT 0,
                    inverse_of INTEGER REFERENCES events(id)
                );
                CREATE INDEX IF NOT EXISTS idx_events_latest
                    ON events(image_id, id DESC);
                CREATE TABLE IF NOT EXISTS exports (
                    export_id TEXT PRIMARY KEY,
                    relative_path TEXT NOT NULL,
                    labels_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def initialize_project(
        self,
        *,
        project_id: str,
        annotator: str,
        source_kind: str,
    ) -> None:
        annotator = validate_annotator(annotator)
        if source_kind not in {"multipie", "custom"}:
            raise ValueError("source_kind must be multipie or custom")
        with self.transaction() as db:
            if db.execute("SELECT 1 FROM project").fetchone():
                raise ValueError("Project database is already initialized")
            db.execute(
                """
                INSERT INTO project(
                    singleton, schema_id, project_id, annotator,
                    source_kind, created_at
                ) VALUES(1, ?, ?, ?, ?, ?)
                """,
                (
                    SCHEMA_ID,
                    project_id,
                    annotator,
                    source_kind,
                    utc_now(),
                ),
            )

    def project(self) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM project WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise ValueError("Project database is not initialized")
            result = dict(row)
            result["image_count"] = int(
                db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            )
            result["reviewed_count"] = int(
                db.execute(
                    "SELECT COUNT(*) FROM labels WHERE state != 'unreviewed'"
                ).fetchone()[0]
            )
            result["total_label_count"] = (
                result["image_count"] * len(LANDMARKS)
            )
            return result

    def image_hash_exists(self, digest: str) -> bool:
        with self.connect() as db:
            return (
                db.execute(
                    "SELECT 1 FROM images WHERE sha256 = ?",
                    (digest,),
                ).fetchone()
                is not None
            )

    def add_image(
        self,
        *,
        relative_path: str,
        original_name: str,
        original_relative_path: str,
        sha256: str,
        width: int,
        height: int,
        source_kind: str,
    ) -> int:
        with self.transaction() as db:
            cursor = db.execute(
                """
                INSERT INTO images(
                    relative_path, original_name, original_relative_path,
                    sha256, width, height, source_kind, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    relative_path,
                    original_name,
                    original_relative_path,
                    sha256,
                    int(width),
                    int(height),
                    source_kind,
                    utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def queue(
        self,
        *,
        focused_landmark_index: int | None = None,
    ) -> list[dict[str, Any]]:
        if focused_landmark_index is not None:
            focused_landmark_index = validate_landmark_index(
                focused_landmark_index
            )
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT
                    images.*,
                    COUNT(
                        CASE WHEN labels.state != 'unreviewed' THEN 1 END
                    ) AS reviewed_count
                FROM images
                LEFT JOIN labels ON labels.image_id = images.id
                GROUP BY images.id
                ORDER BY images.id
                """
            ).fetchall()
            result = [dict(row) for row in rows]
            if focused_landmark_index is not None:
                focused = {
                    int(row["image_id"]): row["state"]
                    for row in db.execute(
                        """
                        SELECT image_id, state
                        FROM labels
                        WHERE landmark_index = ?
                        """,
                        (focused_landmark_index,),
                    ).fetchall()
                }
                for item in result:
                    item["focused_state"] = focused.get(
                        int(item["id"]),
                        "unreviewed",
                    )
            return result

    def image(self, image_id: int) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM images WHERE id = ?",
                (int(image_id),),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown image: {image_id}")
            labels = [
                {
                    **item,
                    "state": "unreviewed",
                    "x": None,
                    "y": None,
                    "annotator": None,
                    "updated_at": None,
                    "revision": 0,
                }
                for item in LANDMARKS
            ]
            for saved in db.execute(
                "SELECT * FROM labels WHERE image_id = ?",
                (int(image_id),),
            ).fetchall():
                index = int(saved["landmark_index"])
                labels[index].update(dict(saved))
            return {**dict(row), "labels": labels}

    def mutate_label(
        self,
        *,
        image_id: int,
        landmark_index: int,
        annotation: dict[str, Any],
        expected_revision: int,
        annotator: str,
    ) -> dict[str, Any]:
        landmark_index = validate_landmark_index(landmark_index)
        annotator = validate_annotator(annotator)
        with self.transaction() as db:
            image = db.execute(
                "SELECT * FROM images WHERE id = ?",
                (int(image_id),),
            ).fetchone()
            if image is None:
                raise KeyError(f"Unknown image: {image_id}")
            clean = validate_label(
                annotation,
                image_width=int(image["width"]),
                image_height=int(image["height"]),
            )
            current_row = db.execute(
                """
                SELECT * FROM labels
                WHERE image_id = ? AND landmark_index = ?
                """,
                (int(image_id), landmark_index),
            ).fetchone()
            current = (
                dict(current_row)
                if current_row is not None
                else {
                    "state": "unreviewed",
                    "x": None,
                    "y": None,
                    "annotator": None,
                    "updated_at": None,
                    "revision": 0,
                }
            )
            if int(current["revision"]) != int(expected_revision):
                raise StaleRevisionError(
                    "Label changed in another browser; reload before saving"
                )
            revision = int(current["revision"]) + 1
            after = {
                **clean,
                "annotator": annotator,
                "updated_at": utc_now(),
                "revision": revision,
            }
            db.execute(
                """
                INSERT INTO labels(
                    image_id, landmark_index, state, x, y,
                    annotator, updated_at, revision
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id, landmark_index) DO UPDATE SET
                    state = excluded.state,
                    x = excluded.x,
                    y = excluded.y,
                    annotator = excluded.annotator,
                    updated_at = excluded.updated_at,
                    revision = excluded.revision
                """,
                (
                    int(image_id),
                    landmark_index,
                    after["state"],
                    after["x"],
                    after["y"],
                    annotator,
                    after["updated_at"],
                    revision,
                ),
            )
            event = db.execute(
                """
                INSERT INTO events(
                    event_uuid, image_id, landmark_index, action,
                    before_json, after_json, annotator, created_at
                ) VALUES(?, ?, ?, 'set_label', ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    int(image_id),
                    landmark_index,
                    json.dumps(current, sort_keys=True),
                    json.dumps(after, sort_keys=True),
                    annotator,
                    utc_now(),
                ),
            )
            return {
                "image_id": int(image_id),
                "landmark_index": landmark_index,
                **after,
                "event_id": int(event.lastrowid),
            }

    def history(self, image_id: int, *, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    """
                    SELECT id, image_id, landmark_index, action,
                           annotator, created_at, undone, inverse_of
                    FROM events
                    WHERE image_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (int(image_id), int(limit)),
                ).fetchall()
            ]

    def undo_last(self, *, image_id: int, annotator: str) -> dict[str, Any]:
        annotator = validate_annotator(annotator)
        with self.transaction() as db:
            event = db.execute(
                """
                SELECT * FROM events
                WHERE image_id = ?
                  AND action = 'set_label'
                  AND undone = 0
                ORDER BY id DESC
                LIMIT 1
                """,
                (int(image_id),),
            ).fetchone()
            if event is None:
                raise ValueError("Nothing to undo for this image")
            current = json.loads(event["after_json"])
            before = json.loads(event["before_json"])
            saved = db.execute(
                """
                SELECT * FROM labels
                WHERE image_id = ? AND landmark_index = ?
                """,
                (int(image_id), int(event["landmark_index"])),
            ).fetchone()
            current_matches = (
                saved is not None
                and saved["state"] == current["state"]
                and saved["x"] == current["x"]
                and saved["y"] == current["y"]
            )
            if not current_matches:
                raise StaleRevisionError(
                    "Only the latest unchanged edit can be undone"
                )
            revision = int(saved["revision"]) + 1
            restored = {
                "state": before["state"],
                "x": before["x"],
                "y": before["y"],
                "annotator": annotator,
                "updated_at": utc_now(),
                "revision": revision,
            }
            db.execute(
                """
                UPDATE labels
                SET state = ?, x = ?, y = ?, annotator = ?,
                    updated_at = ?, revision = ?
                WHERE image_id = ? AND landmark_index = ?
                """,
                (
                    restored["state"],
                    restored["x"],
                    restored["y"],
                    restored["annotator"],
                    restored["updated_at"],
                    restored["revision"],
                    int(image_id),
                    int(event["landmark_index"]),
                ),
            )
            db.execute(
                "UPDATE events SET undone = 1 WHERE id = ?",
                (int(event["id"]),),
            )
            inverse = db.execute(
                """
                INSERT INTO events(
                    event_uuid, image_id, landmark_index, action,
                    before_json, after_json, annotator, created_at, inverse_of
                ) VALUES(?, ?, ?, 'undo', ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    int(image_id),
                    int(event["landmark_index"]),
                    json.dumps(current, sort_keys=True),
                    json.dumps(restored, sort_keys=True),
                    annotator,
                    utc_now(),
                    int(event["id"]),
                ),
            )
            return {
                "image_id": int(image_id),
                "landmark_index": int(event["landmark_index"]),
                **restored,
                "event_id": int(inverse.lastrowid),
            }

    def record_export(
        self,
        *,
        export_id: str,
        relative_path: str,
        labels_sha256: str,
    ) -> None:
        with self.transaction() as db:
            db.execute(
                """
                INSERT INTO exports(
                    export_id, relative_path, labels_sha256, created_at
                ) VALUES(?, ?, ?, ?)
                """,
                (
                    export_id,
                    relative_path,
                    labels_sha256,
                    utc_now(),
                ),
            )
