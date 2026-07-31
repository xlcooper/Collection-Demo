"""SQLite initialization, transactions, and object-memory persistence."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from typing import Iterator

from .assets import MemoryPaths
from .schemas import (
    Decision,
    DecisionType,
    MemoryObject,
    ObjectAnnotation,
    ObjectCard,
    Observation,
    Proposal,
    ProposalStatus,
    Run,
    RunStatus,
    SourceImage,
    SourceImageStatus,
    utc_now,
)


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


@dataclass(frozen=True, slots=True)
class SourceRegistration:
    """Result of the SHA-256 idempotency gate for one source image."""

    source_id: str
    status: SourceImageStatus
    duplicate: bool
    resumed: bool


@dataclass(frozen=True, slots=True)
class DecisionWriteResult:
    """Identifiers created by one committed proposal decision."""

    proposal_id: str
    decision_id: str
    decision: DecisionType
    proposal_status: ProposalStatus
    object_id: str | None
    observation_id: str | None


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Compact database-backed summary for one completed or running run."""

    run_id: str
    status: RunStatus
    source_counts: dict[str, int]
    proposal_counts: dict[str, int]
    decision_counts: dict[str, int]
    observations_added: int
    active_objects_total: int
    duplicate_sources_skipped: int = 0
    external_errors: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "source_counts": self.source_counts,
            "proposal_counts": self.proposal_counts,
            "decision_counts": self.decision_counts,
            "observations_added": self.observations_added,
            "active_objects_total": self.active_objects_total,
            "duplicate_sources_skipped": self.duplicate_sources_skipped,
            "external_errors": self.external_errors,
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

    def begin_run(self, run: Run) -> None:
        """Insert one running batch before source images are registered."""

        if run.status is not RunStatus.RUNNING:
            raise MemoryStoreError("A new run must start with status='running'.")
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    id, status, started_at, completed_at, config_digest,
                    sam_model_id, qwen_model_id, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.status.value,
                    run.started_at.isoformat(),
                    run.completed_at.isoformat() if run.completed_at else None,
                    run.config_digest,
                    run.sam_model_id,
                    run.qwen_model_id,
                    run.error_message,
                ),
            )

    def register_source(self, source: SourceImage) -> SourceRegistration:
        """Apply the source-hash gate and return the canonical source id.

        A completed hash is a true duplicate and is skipped. A failed or
        interrupted row can only be resumed inside its original run.
        """

        now = utc_now().isoformat()
        with self.transaction() as connection:
            existing = connection.execute(
                """
                SELECT id, run_id, status
                FROM source_images
                WHERE sha256 = ?
                """,
                (source.sha256,),
            ).fetchone()
            if existing is not None:
                existing_status = SourceImageStatus(str(existing["status"]))
                if existing_status is SourceImageStatus.COMPLETED or (
                    str(existing["run_id"]) == source.run_id
                    and existing_status
                    in {SourceImageStatus.PENDING, SourceImageStatus.PROCESSING}
                ):
                    return SourceRegistration(
                        source_id=str(existing["id"]),
                        status=existing_status,
                        duplicate=True,
                        resumed=False,
                    )
                if str(existing["run_id"]) != source.run_id:
                    raise MemoryStoreError(
                        "An unfinished source hash already belongs to run "
                        f"{existing['run_id']}; resume that run before starting "
                        "another one."
                    )
                connection.execute(
                    """
                    UPDATE source_images
                    SET relative_path = ?, width = ?, height = ?,
                        status = ?, error_message = NULL, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        source.relative_path,
                        source.width,
                        source.height,
                        SourceImageStatus.PROCESSING.value,
                        now,
                        str(existing["id"]),
                    ),
                )
                return SourceRegistration(
                    source_id=str(existing["id"]),
                    status=SourceImageStatus.PROCESSING,
                    duplicate=False,
                    resumed=True,
                )

            connection.execute(
                """
                INSERT INTO source_images (
                    id, run_id, sha256, relative_path, width, height, status,
                    error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source.id,
                    source.run_id,
                    source.sha256,
                    source.relative_path,
                    source.width,
                    source.height,
                    SourceImageStatus.PROCESSING.value,
                    None,
                    source.created_at.isoformat(),
                    now,
                ),
            )
            return SourceRegistration(
                source_id=source.id,
                status=SourceImageStatus.PROCESSING,
                duplicate=False,
                resumed=False,
            )

    def record_filtered_proposal(self, proposal: Proposal) -> None:
        """Persist a filtered candidate without sending it to Qwen."""

        if proposal.status is not ProposalStatus.FILTERED:
            raise MemoryStoreError("Filtered proposals require status='filtered'.")
        if not proposal.filter_reason:
            raise MemoryStoreError("Filtered proposals require filter_reason.")
        with self.transaction() as connection:
            self._require_processing_source(connection, proposal.source_image_id)
            self._insert_proposal(connection, proposal, ProposalStatus.FILTERED)

    def commit_decision(
        self,
        *,
        proposal: Proposal,
        decision: Decision,
        memory_object: MemoryObject | None = None,
        object_annotation: ObjectAnnotation | None = None,
        observation: Observation | None = None,
    ) -> DecisionWriteResult:
        """Atomically write one validated Qwen decision and its side effects."""

        self._validate_decision_records(
            proposal=proposal,
            decision=decision,
            memory_object=memory_object,
            object_annotation=object_annotation,
            observation=observation,
        )
        proposal_status = (
            ProposalStatus.PENDING
            if decision.decision is DecisionType.UNCERTAIN
            else ProposalStatus.DECIDED
        )
        with self.transaction() as connection:
            self._prepare_proposal_for_decision(
                connection,
                proposal,
                decision,
            )

            if memory_object is not None:
                connection.execute(
                    """
                    INSERT INTO objects (
                        id, coarse_category, fine_category, material_json,
                        color_json, shape, description, annotation_confidence,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_object.id,
                        memory_object.coarse_category,
                        memory_object.fine_category,
                        json.dumps(memory_object.material, ensure_ascii=False),
                        json.dumps(memory_object.color, ensure_ascii=False),
                        memory_object.shape,
                        memory_object.description,
                        memory_object.annotation_confidence,
                        memory_object.status.value,
                        memory_object.created_at.isoformat(),
                        memory_object.updated_at.isoformat(),
                    ),
                )
            elif decision.decision is DecisionType.EXISTING:
                existing_object = connection.execute(
                    """
                    SELECT id
                    FROM objects
                    WHERE id = ? AND status = 'active'
                    """,
                    (decision.matched_object_id,),
                ).fetchone()
                if existing_object is None:
                    raise MemoryStoreError(
                        "existing decision references a missing or archived object: "
                        f"{decision.matched_object_id}"
                    )
                assert object_annotation is not None
                connection.execute(
                    """
                    UPDATE objects
                    SET coarse_category = ?, fine_category = ?,
                        material_json = ?, color_json = ?, shape = ?,
                        description = ?, annotation_confidence = ?,
                        updated_at = ?
                    WHERE id = ? AND status = 'active'
                    """,
                    (
                        object_annotation.coarse_category,
                        object_annotation.fine_category,
                        json.dumps(
                            object_annotation.material,
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            object_annotation.color,
                            ensure_ascii=False,
                        ),
                        object_annotation.shape,
                        object_annotation.description,
                        object_annotation.annotation_confidence,
                        utc_now().isoformat(),
                        decision.matched_object_id,
                    ),
                )

            if observation is not None:
                connection.execute(
                    """
                    INSERT INTO observations (
                        id, object_id, proposal_id, source_image_id, crop_path,
                        mask_path, overlay_path, description, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        observation.id,
                        observation.object_id,
                        observation.proposal_id,
                        observation.source_image_id,
                        observation.crop_path,
                        observation.mask_path,
                        observation.overlay_path,
                        observation.description,
                        observation.created_at.isoformat(),
                    ),
                )

            connection.execute(
                """
                INSERT INTO decisions (
                    id, proposal_id, decision, matched_object_id, confidence,
                    reason_code, short_reason, prompt_version,
                    raw_response_path, attempt, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.id,
                    decision.proposal_id,
                    decision.decision.value,
                    decision.matched_object_id,
                    decision.confidence,
                    decision.reason_code,
                    decision.short_reason,
                    decision.prompt_version,
                    decision.raw_response_path,
                    decision.attempt,
                    decision.created_at.isoformat(),
                ),
            )
            connection.execute(
                """
                UPDATE proposals
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (proposal_status.value, utc_now().isoformat(), proposal.id),
            )

        object_id = observation.object_id if observation is not None else None
        return DecisionWriteResult(
            proposal_id=proposal.id,
            decision_id=decision.id,
            decision=decision.decision,
            proposal_status=proposal_status,
            object_id=object_id,
            observation_id=observation.id if observation is not None else None,
        )

    def record_proposal_failure(self, proposal: Proposal, error_message: str) -> None:
        """Record a candidate whose processing or validation failed."""

        if proposal.status is not ProposalStatus.PENDING:
            raise MemoryStoreError("Only pending input proposals can fail.")
        message = error_message.strip()
        if not message:
            raise ValueError("error_message must not be empty")
        failed = proposal.model_copy(
            update={
                "status": ProposalStatus.FAILED,
                "error_message": message,
                "updated_at": utc_now(),
            }
        )
        with self.transaction() as connection:
            existing = connection.execute(
                """
                SELECT source_image_id, raw_candidate_id, status
                FROM proposals
                WHERE id = ?
                """,
                (proposal.id,),
            ).fetchone()
            if existing is None:
                self._require_processing_source(connection, proposal.source_image_id)
                self._insert_proposal(connection, failed, ProposalStatus.FAILED)
            else:
                if (
                    str(existing["source_image_id"]) != proposal.source_image_id
                    or str(existing["raw_candidate_id"]) != proposal.raw_candidate_id
                ):
                    raise MemoryStoreError(
                        "Stored proposal identity does not match failure record."
                    )
                if str(existing["status"]) != ProposalStatus.PENDING.value:
                    raise MemoryStoreError(
                        f"Only pending proposals can become failed: {proposal.id}"
                    )
                connection.execute(
                    """
                    UPDATE proposals
                    SET status = ?, error_message = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        ProposalStatus.FAILED.value,
                        message,
                        utc_now().isoformat(),
                        proposal.id,
                    ),
                )

    def complete_source(self, source_id: str) -> None:
        """Mark one image complete after all of its proposals are accounted for."""

        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE source_images
                SET status = ?, error_message = NULL, updated_at = ?
                WHERE id = ? AND status = ?
                """,
                (
                    SourceImageStatus.COMPLETED.value,
                    utc_now().isoformat(),
                    source_id,
                    SourceImageStatus.PROCESSING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise MemoryStoreError(
                    f"Source image is not in processing state: {source_id}"
                )

    def fail_source(self, source_id: str, error_message: str) -> None:
        """Mark an image failed when processing cannot reach candidate writes."""

        message = error_message.strip()
        if not message:
            raise ValueError("error_message must not be empty")
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE source_images
                SET status = ?, error_message = ?, updated_at = ?
                WHERE id = ? AND status IN ('pending', 'processing')
                """,
                (
                    SourceImageStatus.FAILED.value,
                    message,
                    utc_now().isoformat(),
                    source_id,
                ),
            )
            if cursor.rowcount != 1:
                raise MemoryStoreError(f"Source image cannot be failed: {source_id}")

    def complete_run(
        self,
        run_id: str,
        *,
        error_message: str | None = None,
    ) -> RunSummary:
        """Close a run, treating pending or failed work as completed-with-errors."""

        with self.transaction() as connection:
            exists = connection.execute(
                "SELECT id FROM runs WHERE id = ? AND status = 'running'",
                (run_id,),
            ).fetchone()
            if exists is None:
                raise MemoryStoreError(f"Run is not active: {run_id}")
            incomplete_sources = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM source_images
                    WHERE run_id = ? AND status <> 'completed'
                    """,
                    (run_id,),
                ).fetchone()[0]
            )
            pending_or_failed_proposals = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM proposals AS p
                    JOIN source_images AS s ON s.id = p.source_image_id
                    WHERE s.run_id = ? AND p.status IN ('pending', 'failed')
                    """,
                    (run_id,),
                ).fetchone()[0]
            )
            final_status = (
                RunStatus.COMPLETED_WITH_ERRORS
                if incomplete_sources or pending_or_failed_proposals or error_message
                else RunStatus.COMPLETED
            )
            connection.execute(
                """
                UPDATE runs
                SET status = ?, completed_at = ?, error_message = ?
                WHERE id = ?
                """,
                (
                    final_status.value,
                    utc_now().isoformat(),
                    error_message,
                    run_id,
                ),
            )
        return self.run_summary(run_id)

    def list_object_cards(self, *, max_reference_views: int = 2) -> list[ObjectCard]:
        """Read every active object card with its recent reference views."""

        if max_reference_views <= 0:
            raise ValueError("max_reference_views must be positive")
        with closing(self._connect()) as connection:
            object_rows = connection.execute(
                """
                SELECT *
                FROM objects
                WHERE status = 'active'
                ORDER BY created_at, id
                """
            ).fetchall()
            cards: list[ObjectCard] = []
            for row in object_rows:
                view_rows = connection.execute(
                    """
                    SELECT crop_path
                    FROM observations
                    WHERE object_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (str(row["id"]), max_reference_views),
                ).fetchall()
                cards.append(
                    ObjectCard(
                        object_id=str(row["id"]),
                        coarse_category=str(row["coarse_category"]),
                        fine_category=str(row["fine_category"]),
                        material=json.loads(str(row["material_json"])),
                        color=json.loads(str(row["color_json"])),
                        shape=str(row["shape"]),
                        description=str(row["description"]),
                        representative_view_paths=[
                            str(view["crop_path"]) for view in view_rows
                        ],
                    )
                )
        return cards

    def run_summary(self, run_id: str) -> RunSummary:
        """Return counts for one run without exposing SQL to the pipeline."""

        with closing(self._connect()) as connection:
            run_row = connection.execute(
                "SELECT status FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if run_row is None:
                raise MemoryStoreError(f"Run not found: {run_id}")
            source_counts = self._group_counts(
                connection,
                """
                SELECT status, COUNT(*) AS count
                FROM source_images
                WHERE run_id = ?
                GROUP BY status
                """,
                run_id,
                tuple(status.value for status in SourceImageStatus),
            )
            proposal_counts = self._group_counts(
                connection,
                """
                SELECT p.status, COUNT(*) AS count
                FROM proposals AS p
                JOIN source_images AS s ON s.id = p.source_image_id
                WHERE s.run_id = ?
                GROUP BY p.status
                """,
                run_id,
                tuple(status.value for status in ProposalStatus),
            )
            decision_counts = self._group_counts(
                connection,
                """
                SELECT d.decision AS status, COUNT(*) AS count
                FROM decisions AS d
                JOIN proposals AS p ON p.id = d.proposal_id
                JOIN source_images AS s ON s.id = p.source_image_id
                WHERE s.run_id = ?
                GROUP BY d.decision
                """,
                run_id,
                tuple(decision.value for decision in DecisionType),
            )
            observations_added = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM observations AS o
                    JOIN source_images AS s ON s.id = o.source_image_id
                    WHERE s.run_id = ?
                    """,
                    (run_id,),
                ).fetchone()[0]
            )
            active_objects_total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM objects WHERE status = 'active'"
                ).fetchone()[0]
            )
        return RunSummary(
            run_id=run_id,
            status=RunStatus(str(run_row["status"])),
            source_counts=source_counts,
            proposal_counts=proposal_counts,
            decision_counts=decision_counts,
            observations_added=observations_added,
            active_objects_total=active_objects_total,
        )

    @staticmethod
    def _group_counts(
        connection: sqlite3.Connection,
        statement: str,
        run_id: str,
        expected_keys: tuple[str, ...],
    ) -> dict[str, int]:
        counts = {key: 0 for key in expected_keys}
        for row in connection.execute(statement, (run_id,)):
            counts[str(row["status"])] = int(row["count"])
        return counts

    @staticmethod
    def _require_processing_source(
        connection: sqlite3.Connection,
        source_image_id: str,
    ) -> None:
        source = connection.execute(
            "SELECT status FROM source_images WHERE id = ?",
            (source_image_id,),
        ).fetchone()
        if source is None or str(source["status"]) != SourceImageStatus.PROCESSING.value:
            raise MemoryStoreError(
                f"Source image is not registered for processing: {source_image_id}"
            )

    @staticmethod
    def _prepare_proposal_for_decision(
        connection: sqlite3.Connection,
        proposal: Proposal,
        decision: Decision,
    ) -> None:
        """Insert the first or an explicit later decision for a pending proposal."""

        if proposal.status is not ProposalStatus.PENDING:
            raise MemoryStoreError("Qwen decisions require a pending proposal.")

        source = connection.execute(
            "SELECT status FROM source_images WHERE id = ?",
            (proposal.source_image_id,),
        ).fetchone()
        if source is None or str(source["status"]) not in {
            SourceImageStatus.PROCESSING.value,
            SourceImageStatus.COMPLETED.value,
        }:
            raise MemoryStoreError(
                f"Source image cannot accept decisions: {proposal.source_image_id}"
            )
        existing = connection.execute(
            """
            SELECT p.source_image_id, p.raw_candidate_id, p.status,
                   COALESCE(MAX(d.attempt), 0) AS last_attempt
            FROM proposals AS p
            LEFT JOIN decisions AS d ON d.proposal_id = p.id
            WHERE p.id = ?
            GROUP BY p.id
            """,
            (proposal.id,),
        ).fetchone()
        if existing is None:
            if str(source["status"]) != SourceImageStatus.PROCESSING.value:
                raise MemoryStoreError(
                    "New proposals require a source in processing state."
                )
            if decision.attempt != 1:
                raise MemoryStoreError("A proposal must start with attempt=1.")
            MemoryStore._insert_proposal(
                connection,
                proposal,
                ProposalStatus.PENDING,
            )
            return
        if (
            str(existing["source_image_id"]) != proposal.source_image_id
            or str(existing["raw_candidate_id"]) != proposal.raw_candidate_id
        ):
            raise MemoryStoreError(
                "Stored proposal identity does not match the decision request."
            )
        if str(existing["status"]) != ProposalStatus.PENDING.value:
            raise MemoryStoreError(
                f"Only pending proposals can receive another decision: {proposal.id}"
            )
        expected_attempt = int(existing["last_attempt"]) + 1
        if decision.attempt != expected_attempt:
            raise MemoryStoreError(
                f"Proposal {proposal.id} requires attempt={expected_attempt}, "
                f"received {decision.attempt}."
            )

    @staticmethod
    def _insert_proposal(
        connection: sqlite3.Connection,
        proposal: Proposal,
        status: ProposalStatus,
    ) -> None:
        connection.execute(
            """
            INSERT INTO proposals (
                id, source_image_id, raw_candidate_id, prompt, score,
                bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max,
                mask_area_pixels, mask_area_ratio, mask_path, crop_path,
                overlay_path, status, filter_reason, error_message,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                proposal.id,
                proposal.source_image_id,
                proposal.raw_candidate_id,
                proposal.prompt,
                proposal.score,
                proposal.bbox.x_min,
                proposal.bbox.y_min,
                proposal.bbox.x_max,
                proposal.bbox.y_max,
                proposal.mask_area_pixels,
                proposal.mask_area_ratio,
                proposal.mask_path,
                proposal.crop_path,
                proposal.overlay_path,
                status.value,
                proposal.filter_reason,
                proposal.error_message,
                proposal.created_at.isoformat(),
                proposal.updated_at.isoformat(),
            ),
        )

    @staticmethod
    def _validate_decision_records(
        *,
        proposal: Proposal,
        decision: Decision,
        memory_object: MemoryObject | None,
        object_annotation: ObjectAnnotation | None,
        observation: Observation | None,
    ) -> None:
        if decision.proposal_id != proposal.id:
            raise MemoryStoreError("Decision and proposal ids do not agree.")
        if observation is not None and (
            observation.proposal_id != proposal.id
            or observation.source_image_id != proposal.source_image_id
        ):
            raise MemoryStoreError("Observation does not belong to the proposal.")
        if decision.decision is DecisionType.NEW:
            if (
                memory_object is None
                or object_annotation is not None
                or observation is None
            ):
                raise MemoryStoreError("new requires an object and first observation.")
            if observation.object_id != memory_object.id:
                raise MemoryStoreError("New observation must reference the new object.")
        elif decision.decision is DecisionType.EXISTING:
            if (
                memory_object is not None
                or object_annotation is None
                or observation is None
            ):
                raise MemoryStoreError(
                    "existing requires an annotation update and new observation."
                )
            if observation.object_id != decision.matched_object_id:
                raise MemoryStoreError("Observation must use matched_object_id.")
        elif (
            memory_object is not None
            or object_annotation is not None
            or observation is not None
        ):
            raise MemoryStoreError(
                "ignored and uncertain must not create objects or observations."
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
