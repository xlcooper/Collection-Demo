"""SQLite initialization, transactions, and basic status queries."""

from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from typing import Iterator

from .assets import MemoryPaths


SCHEMA_VERSION = 2
CORE_TABLES = (
    "runs",
    "source_images",
    "proposals",
    "objects",
    "observations",
    "decisions",
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (
        status IN ('running', 'completed', 'completed_with_errors', 'failed')
    ),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    config_digest TEXT NOT NULL,
    sam_model_id TEXT NOT NULL,
    qwen_model_id TEXT NOT NULL,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS source_images (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    sha256 TEXT NOT NULL UNIQUE CHECK (length(sha256) = 64),
    relative_path TEXT NOT NULL,
    width INTEGER NOT NULL CHECK (width > 0),
    height INTEGER NOT NULL CHECK (height > 0),
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'processing', 'completed', 'failed')
    ),
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS proposals (
    id TEXT PRIMARY KEY,
    source_image_id TEXT NOT NULL REFERENCES source_images(id),
    raw_candidate_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    score REAL NOT NULL CHECK (score >= 0.0 AND score <= 1.0),
    bbox_x_min REAL NOT NULL CHECK (bbox_x_min >= 0.0),
    bbox_y_min REAL NOT NULL CHECK (bbox_y_min >= 0.0),
    bbox_x_max REAL NOT NULL CHECK (bbox_x_max > bbox_x_min),
    bbox_y_max REAL NOT NULL CHECK (bbox_y_max > bbox_y_min),
    mask_area_pixels INTEGER NOT NULL CHECK (mask_area_pixels >= 0),
    mask_area_ratio REAL NOT NULL CHECK (
        mask_area_ratio >= 0.0 AND mask_area_ratio <= 1.0
    ),
    mask_path TEXT,
    crop_path TEXT,
    overlay_path TEXT,
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'filtered', 'decided', 'failed')
    ),
    filter_reason TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (source_image_id, raw_candidate_id)
);

CREATE TABLE IF NOT EXISTS objects (
    id TEXT PRIMARY KEY,
    coarse_category TEXT NOT NULL,
    fine_category TEXT NOT NULL,
    material_json TEXT NOT NULL,
    color_json TEXT NOT NULL,
    shape TEXT NOT NULL,
    description TEXT NOT NULL,
    annotation_confidence REAL NOT NULL CHECK (
        annotation_confidence >= 0.0 AND annotation_confidence <= 1.0
    ),
    status TEXT NOT NULL CHECK (status IN ('active', 'archived')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    object_id TEXT NOT NULL REFERENCES objects(id),
    proposal_id TEXT NOT NULL UNIQUE REFERENCES proposals(id),
    source_image_id TEXT NOT NULL REFERENCES source_images(id),
    crop_path TEXT NOT NULL,
    mask_path TEXT NOT NULL,
    overlay_path TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL REFERENCES proposals(id),
    decision TEXT NOT NULL CHECK (
        decision IN ('new', 'existing', 'ignored', 'uncertain')
    ),
    matched_object_id TEXT REFERENCES objects(id),
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    reason_code TEXT NOT NULL,
    short_reason TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    raw_response_path TEXT,
    attempt INTEGER NOT NULL CHECK (attempt >= 1),
    created_at TEXT NOT NULL,
    CHECK (
        (decision = 'existing' AND matched_object_id IS NOT NULL)
        OR (decision <> 'existing' AND matched_object_id IS NULL)
    ),
    UNIQUE (proposal_id, attempt)
);

CREATE INDEX IF NOT EXISTS idx_source_images_status
    ON source_images(status);
CREATE INDEX IF NOT EXISTS idx_proposals_source
    ON proposals(source_image_id);
CREATE INDEX IF NOT EXISTS idx_proposals_status
    ON proposals(status);
CREATE INDEX IF NOT EXISTS idx_objects_status
    ON objects(status);
CREATE INDEX IF NOT EXISTS idx_observations_object
    ON observations(object_id, created_at);
CREATE INDEX IF NOT EXISTS idx_decisions_proposal
    ON decisions(proposal_id, attempt);
"""


MIGRATION_V1_TO_V2_SQL = """
BEGIN;
ALTER TABLE proposals
    ADD COLUMN prompt TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE proposals
    ADD COLUMN mask_area_pixels INTEGER NOT NULL DEFAULT 0
    CHECK (mask_area_pixels >= 0);
ALTER TABLE proposals
    ADD COLUMN mask_area_ratio REAL NOT NULL DEFAULT 0.0
    CHECK (mask_area_ratio >= 0.0 AND mask_area_ratio <= 1.0);
PRAGMA user_version = 2;
COMMIT;
"""


class MemoryStoreError(RuntimeError):
    """Raised when the on-disk store is missing or incompatible."""


@dataclass(frozen=True, slots=True)
class StoreStatus:
    database_path: str
    schema_version: int
    counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "database_path": self.database_path,
            "schema_version": self.schema_version,
            "counts": self.counts,
        }


class MemoryStore:
    """Own the SQLite connection boundary for the application."""

    def __init__(self, paths: MemoryPaths) -> None:
        self.paths = paths

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.paths.database, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self) -> StoreStatus:
        """Create or safely reopen the schema and fixed asset directories."""

        self.paths.ensure_layout()
        with closing(self._connect()) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version < 0 or version > SCHEMA_VERSION:
                raise MemoryStoreError(
                    f"Unsupported schema version {version}; expected {SCHEMA_VERSION}."
                )
            if version == 1:
                connection.executescript(MIGRATION_V1_TO_V2_SQL)
            connection.executescript(SCHEMA_SQL)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.commit()
        return self.status()

    def status(self) -> StoreStatus:
        """Return schema version and row counts for the six core tables."""

        if not self.paths.database.is_file():
            raise MemoryStoreError(
                f"Memory database is not initialized: {self.paths.database}"
            )

        with closing(self._connect()) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version != SCHEMA_VERSION:
                raise MemoryStoreError(
                    f"Unsupported schema version {version}; expected {SCHEMA_VERSION}."
                )
            existing_tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            missing = sorted(set(CORE_TABLES) - existing_tables)
            if missing:
                raise MemoryStoreError(
                    f"Memory database is missing core tables: {', '.join(missing)}"
                )
            counts = {
                table: int(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
                for table in CORE_TABLES
            }

        return StoreStatus(
            database_path=str(self.paths.database),
            schema_version=version,
            counts=counts,
        )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Provide one explicit transaction for later pipeline write operations."""

        if not self.paths.database.is_file():
            raise MemoryStoreError(
                f"Memory database is not initialized: {self.paths.database}"
            )
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
