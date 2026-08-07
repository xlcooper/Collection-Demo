"""FastAPI service for the single-machine object-memory experiment UI."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import warnings
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Sequence
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, ConfigDict

from .config import load_config, resolve_memory_root
from .memory_store import CORE_TABLES, SCHEMA_VERSION
from .pipeline import SUPPORTED_IMAGE_SUFFIXES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = Path(__file__).resolve().parent / "web_static"
UPLOAD_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
IMAGE_FORMATS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
}
MEMORY_IMAGE_PREFIXES = frozenset({"sources", "proposals", "objects"})
AUDIT_JSON_PREFIXES = frozenset({"raw_responses", "run_reports"})
ACTIVE_RUN_STATUSES = frozenset({"starting", "running"})
PROCESS_LOG_TAIL_BYTES = 16 * 1024
MEMORY_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MEMORY_DIRECTORY_PREFIX = "memory_"
MEMORY_METADATA_FILENAME = "memory_info.json"
MEMORY_SELECTION_FILENAME = "selected_memory.json"
MAX_MEMORY_LABEL_LENGTH = 40


class SafePathError(ValueError):
    """Raised when a requested file is outside an allowed application root."""


class InputValidationError(ValueError):
    """Raised when an uploaded input is unsafe or not a supported image."""


class ExperimentBusyError(RuntimeError):
    """Raised when a second mutation conflicts with an active experiment."""


class MemoryReadError(RuntimeError):
    """Raised when an initialized memory database cannot be read safely."""


class MemoryLibraryError(ValueError):
    """Raised when a managed memory library request is invalid."""


class MemoryMutationError(RuntimeError):
    """Raised when a library mutation cannot preserve its storage invariants."""


class MemoryCreateRequest(BaseModel):
    """Human-facing label for a new server-generated memory library."""

    model_config = ConfigDict(extra="forbid")
    label: str = ""


class RunStartRequest(BaseModel):
    """Only the opaque managed library ID may be selected by the browser."""

    model_config = ConfigDict(extra="forbid")
    memory_id: str = "default"


@dataclass(frozen=True, slots=True)
class WebSettings:
    """Fixed server paths and limits; none are accepted from browser requests."""

    project_root: Path
    input_root: Path
    memory_root: Path
    database_filename: str
    report_path: Path
    run_state_root: Path
    static_root: Path
    python_executable: Path
    basic_username: str = "object-memory"
    basic_password: str | None = None
    max_upload_bytes: int = 20 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    max_audit_json_bytes: int = 16 * 1024 * 1024

    def resolved(self) -> "WebSettings":
        """Return a settings copy with canonical filesystem roots."""

        return replace(
            self,
            project_root=self.project_root.expanduser().resolve(),
            input_root=self.input_root.expanduser().resolve(),
            memory_root=self.memory_root.expanduser().resolve(),
            report_path=self.report_path.expanduser().resolve(),
            run_state_root=self.run_state_root.expanduser().resolve(),
            static_root=self.static_root.expanduser().resolve(),
            python_executable=self.python_executable.expanduser().resolve(),
        )


def default_web_settings(
    *,
    basic_username: str = "object-memory",
    basic_password: str | None = None,
) -> WebSettings:
    """Build the server configuration from the repository's canonical config."""

    config = load_config()
    return WebSettings(
        project_root=PROJECT_ROOT,
        input_root=PROJECT_ROOT / "data" / "input",
        memory_root=resolve_memory_root(config, base_dir=PROJECT_ROOT),
        database_filename=config.storage.database_filename,
        report_path=PROJECT_ROOT / "environment" / "run_report.json",
        run_state_root=PROJECT_ROOT / "temp" / "web_runs",
        static_root=STATIC_ROOT,
        python_executable=Path(sys.executable),
        basic_username=basic_username,
        basic_password=basic_password,
    ).resolved()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MemoryReadError(f"Cannot read JSON artifact {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MemoryReadError(f"JSON artifact {path.name} must contain an object")
    return payload


def _validated_memory_id(value: str) -> str:
    memory_id = str(value or "").strip()
    if not MEMORY_ID.fullmatch(memory_id):
        raise MemoryLibraryError(
            "Memory library ID must use lowercase letters, digits, underscores, "
            "or hyphens"
        )
    return memory_id


def _validated_memory_label(value: str) -> str:
    label = " ".join(str(value or "").strip().split())
    if not label:
        return datetime.now().astimezone().strftime("实验记忆 %Y-%m-%d %H:%M")
    if len(label) > MAX_MEMORY_LABEL_LENGTH:
        raise MemoryLibraryError(
            f"Memory library name must not exceed {MAX_MEMORY_LABEL_LENGTH} characters"
        )
    if any(ord(character) < 32 for character in label):
        raise MemoryLibraryError("Memory library name contains control characters")
    return label


def resolve_memory_library_root(settings: WebSettings, memory_id: str) -> Path:
    """Resolve one opaque library ID to an exact direct child of data/."""

    safe_id = _validated_memory_id(memory_id)
    project_root = settings.project_root.expanduser().resolve()
    catalog_root = project_root / "data"
    if catalog_root.is_symlink():
        raise SafePathError("The managed data directory must not be a symbolic link")
    if catalog_root.resolve() != catalog_root.absolute():
        raise SafePathError("The managed data directory must resolve inside the project")
    if catalog_root.exists() and not catalog_root.is_dir():
        raise SafePathError("The managed data path must be a directory")
    if catalog_root.parent != project_root:
        raise SafePathError("The managed data directory is outside the project root")
    default_root = catalog_root / "memory"
    configured_root = settings.memory_root.expanduser()
    if not configured_root.is_absolute():
        configured_root = settings.project_root / configured_root
    if configured_root.absolute() != default_root:
        raise SafePathError("The Web default memory library must be project data/memory")
    if default_root.is_symlink():
        raise SafePathError("The default memory library must not be a symbolic link")
    unresolved = (
        default_root
        if safe_id == "default"
        else catalog_root / f"{MEMORY_DIRECTORY_PREFIX}{safe_id}"
    )
    if unresolved.is_symlink():
        raise SafePathError("Symbolic-link memory libraries are not allowed")
    resolved = unresolved.resolve()
    if resolved == catalog_root or resolved.parent != catalog_root:
        raise SafePathError("Memory library is outside the managed data directory")
    if safe_id != "default" and resolved == default_root:
        raise SafePathError("Memory library ID collides with the default library")
    return resolved


def _memory_library_metadata(root: Path) -> dict[str, Any]:
    metadata_path = root / MEMORY_METADATA_FILENAME
    if metadata_path.is_symlink() or not metadata_path.is_file():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _memory_library_has_partial_content(root: Path) -> bool:
    if not root.is_dir():
        return False
    try:
        return any(path.name != MEMORY_METADATA_FILENAME for path in root.iterdir())
    except OSError:
        return True


def _memory_library_item(
    settings: WebSettings,
    memory_id: str,
    *,
    active_id: str | None = None,
) -> dict[str, Any]:
    """Return a small read-only health summary for one managed library."""

    safe_id = _validated_memory_id(memory_id)
    root = resolve_memory_library_root(settings, safe_id)
    metadata = _memory_library_metadata(root)
    label_value = metadata.get("label")
    label = (
        str(label_value).strip()
        if isinstance(label_value, str) and str(label_value).strip()
        else ("默认记忆库" if safe_id == "default" else f"实验记忆 {safe_id}")
    )
    database = root / settings.database_filename
    counts = {"active_objects": 0, "observations": 0, "runs": 0}
    latest_run_at: str | None = None
    schema_version: int | None = None
    issue: str | None = None
    issue_code: str | None = None
    status = "empty"
    continuable = True

    if root.exists() and not root.is_dir():
        status = "unreadable"
        continuable = False
        issue_code = "invalid_library_root"
        issue = "Memory library root must be a directory"
    elif database.is_symlink():
        status = "unreadable"
        continuable = False
        issue_code = "database_symlink"
        issue = "memory.sqlite must not be a symbolic link"
    elif database.is_file():
        try:
            with closing(_open_read_only_database(database)) as connection:
                schema_version = int(
                    connection.execute("PRAGMA user_version").fetchone()[0]
                )
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                missing = set(CORE_TABLES) - tables
                if schema_version not in {2, SCHEMA_VERSION}:
                    raise MemoryReadError(
                        f"Unsupported schema version {schema_version}"
                    )
                if missing:
                    raise MemoryReadError(
                        f"Memory database is missing tables: {sorted(missing)}"
                    )
                counts = {
                    "active_objects": int(
                        connection.execute(
                            "SELECT COUNT(*) FROM objects WHERE status = 'active'"
                        ).fetchone()[0]
                    ),
                    "observations": int(
                        connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
                    ),
                    "runs": int(
                        connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
                    ),
                }
                latest = connection.execute(
                    "SELECT started_at FROM runs ORDER BY started_at DESC, id DESC LIMIT 1"
                ).fetchone()
                latest_run_at = str(latest[0]) if latest and latest[0] else None
                incomplete_runs = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM runs WHERE status != 'completed'"
                    ).fetchone()[0]
                )
                incomplete_sources = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM source_images WHERE status != 'completed'"
                    ).fetchone()[0]
                )
                incomplete_proposals = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM proposals WHERE status IN ('pending', 'failed')"
                    ).fetchone()[0]
                )
                if incomplete_runs or incomplete_sources or incomplete_proposals:
                    status = "review_only"
                    continuable = False
                    issue_code = "incomplete_run"
                    issue = (
                        "This library contains an incomplete or failed run; review it "
                        "or start a new blank library"
                    )
                else:
                    status = "ready"
                    required_directories = (
                        "sources",
                        "proposals",
                        "objects",
                        "raw_responses",
                        "run_reports",
                    )
                    invalid_directories = [
                        name
                        for name in required_directories
                        if (root / name).is_symlink() or not (root / name).is_dir()
                    ]
                    if invalid_directories:
                        status = "review_only"
                        continuable = False
                        issue_code = "missing_asset_directories"
                        issue = (
                            "This library is missing required asset directories: "
                            f"{invalid_directories}"
                        )
                    elif schema_version < SCHEMA_VERSION:
                        status = "review_only"
                        continuable = False
                        issue_code = "legacy_read_only"
                        issue = (
                            "This library records the legacy two-Qwen workflow and "
                            "is preserved for read-only review; create a blank library "
                            "for the DINOv3 workflow"
                        )
        except (MemoryReadError, sqlite3.Error, OSError) as exc:
            status = "unreadable"
            continuable = False
            issue_code = "database_unreadable"
            issue = str(exc)
    elif _memory_library_has_partial_content(root):
        status = "review_only"
        continuable = False
        issue_code = "partial_without_database"
        issue = "This directory contains partial assets but no memory.sqlite"

    modified_at: str | None = None
    try:
        if root.exists():
            modified_at = datetime.fromtimestamp(
                max(
                    [root.stat().st_mtime]
                    + [path.stat().st_mtime for path in root.iterdir() if not path.is_symlink()]
                ),
                timezone.utc,
            ).isoformat()
    except OSError:
        modified_at = None

    if active_id == safe_id:
        status = "running"
        continuable = False
        issue_code = None
        issue = None

    return {
        "id": safe_id,
        "label": label,
        "status": status,
        "continuable": continuable,
        "deletable": active_id != safe_id,
        "active": active_id == safe_id,
        "initialized": database.is_file() and status != "unreadable",
        "schema_version": schema_version,
        "counts": counts,
        "latest_run_at_utc": latest_run_at,
        "modified_at_utc": modified_at,
        "issue_code": issue_code,
        "issue": issue,
    }


def list_memory_libraries(
    settings: WebSettings,
    *,
    active_id: str | None = None,
) -> list[dict[str, Any]]:
    """Enumerate only the canonical default and safe direct-child memory roots."""

    ids = {"default"}
    resolve_memory_library_root(settings, "default")
    catalog_root = settings.project_root.expanduser().resolve() / "data"
    if catalog_root.is_dir():
        for path in catalog_root.iterdir():
            if path.is_symlink() or not path.is_dir():
                continue
            if not path.name.startswith(MEMORY_DIRECTORY_PREFIX):
                continue
            memory_id = path.name[len(MEMORY_DIRECTORY_PREFIX) :]
            if memory_id != "default" and MEMORY_ID.fullmatch(memory_id):
                ids.add(memory_id)
    items = [
        _memory_library_item(settings, memory_id, active_id=active_id)
        for memory_id in ids
    ]
    default_items = [item for item in items if item["id"] == "default"]
    managed_items = sorted(
        (item for item in items if item["id"] != "default"),
        key=lambda item: (
            str(item.get("modified_at_utc") or ""),
            str(item["id"]),
        ),
        reverse=True,
    )
    return default_items + managed_items


def read_memory_report_for_run(
    memory_root: Path,
    run_id: str,
) -> dict[str, Any] | None:
    """Read the exact internal report whose filename and payload match a run ID."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise MemoryReadError("Run ID is not safe for an internal report filename")
    unresolved_root = memory_root.expanduser()
    if unresolved_root.is_symlink():
        raise MemoryReadError("Memory root must not be a symbolic link")
    root = unresolved_root.resolve()
    report_root = root / "run_reports"
    report_path = report_root / f"{run_id}.json"
    if report_root.is_symlink() or report_path.is_symlink():
        raise MemoryReadError("Internal report paths must not be symbolic links")
    if report_path.resolve().parent != report_root:
        raise MemoryReadError("Internal report is outside its memory library")
    report = _read_json_object(report_path)
    if report is None:
        return None
    report_run = report.get("run") if isinstance(report.get("run"), dict) else {}
    if str(report_run.get("run_id") or "") != run_id:
        raise MemoryReadError("Internal report run ID does not match its filename")
    return report


def read_latest_memory_report(
    memory_root: Path,
    database_filename: str = "memory.sqlite",
) -> dict[str, Any] | None:
    """Read the report for the newest SQLite run, never by file mtime."""

    unresolved_root = memory_root.expanduser()
    if unresolved_root.is_symlink():
        raise MemoryReadError("Memory root must not be a symbolic link")
    root = unresolved_root.resolve()
    database = root / database_filename
    if database.is_symlink():
        raise MemoryReadError("memory.sqlite must not be a symbolic link")
    if not database.is_file():
        return None
    try:
        with closing(_open_read_only_database(database)) as connection:
            latest = connection.execute(
                "SELECT id FROM runs ORDER BY started_at DESC, id DESC LIMIT 1"
            ).fetchone()
    except sqlite3.Error as exc:
        raise MemoryReadError(f"Cannot find latest memory run: {exc}") from exc
    if latest is None:
        return None
    return read_memory_report_for_run(root, str(latest[0]))


def read_memory_run_timing(
    memory_root: Path,
    run_id: str,
    database_filename: str = "memory.sqlite",
) -> dict[str, Any]:
    """Recover stable timing fields for a historical library report."""

    unresolved_root = memory_root.expanduser()
    if unresolved_root.is_symlink():
        raise MemoryReadError("Memory root must not be a symbolic link")
    root = unresolved_root.resolve()
    database = root / database_filename
    if database.is_symlink():
        raise MemoryReadError("memory.sqlite must not be a symbolic link")
    if not database.is_file():
        return {}
    try:
        with closing(_open_read_only_database(database)) as connection:
            row = connection.execute(
                "SELECT started_at, completed_at FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
    except sqlite3.Error as exc:
        raise MemoryReadError(f"Cannot read memory run timing: {exc}") from exc
    if row is None:
        return {}
    started_at = str(row[0]) if row[0] else None
    completed_at = str(row[1]) if row[1] else None
    elapsed_seconds: float | None = None
    if started_at and completed_at:
        try:
            started = datetime.fromisoformat(started_at)
            completed = datetime.fromisoformat(completed_at)
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            if completed.tzinfo is None:
                completed = completed.replace(tzinfo=timezone.utc)
            elapsed_seconds = round(max(0.0, (completed - started).total_seconds()), 3)
        except ValueError:
            elapsed_seconds = None
    return {
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "elapsed_seconds": elapsed_seconds,
    }


def _is_formal_failure_report(report: dict[str, Any] | None) -> bool:
    return bool(
        report
        and report.get("status") in {"failed", "completed_with_errors"}
    )


def _safe_relative_path(value: str) -> PurePosixPath:
    if not value or "\\" in value or "\x00" in value:
        raise SafePathError("Path must be a non-empty POSIX relative path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise SafePathError("Absolute paths and parent traversal are not allowed")
    if any(part in {"", "."} for part in candidate.parts):
        raise SafePathError("Path contains an invalid segment")
    return candidate


def _resolve_allowed_file(
    root: Path,
    relative_path: str,
    *,
    suffixes: frozenset[str] | set[str],
    prefixes: frozenset[str] | None = None,
) -> Path:
    relative = _safe_relative_path(relative_path)
    if prefixes is not None and relative.parts[0] not in prefixes:
        raise SafePathError("Path is outside the allowed asset collections")
    if relative.suffix.lower() not in suffixes:
        raise SafePathError("File type is not allowed")

    canonical_root = root.expanduser().resolve()
    unresolved = canonical_root.joinpath(*relative.parts)
    cursor = canonical_root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise SafePathError("Symbolic links are not allowed")
    resolved = unresolved.resolve()
    if not resolved.is_relative_to(canonical_root):
        raise SafePathError("Path escapes its allowed root")
    if not resolved.is_file():
        raise FileNotFoundError(relative_path)
    return resolved


def resolve_input_file(input_root: Path, relative_path: str) -> Path:
    return _resolve_allowed_file(
        input_root,
        relative_path,
        suffixes=set(SUPPORTED_IMAGE_SUFFIXES),
    )


def resolve_memory_image(memory_root: Path, relative_path: str) -> Path:
    return _resolve_allowed_file(
        memory_root,
        relative_path,
        suffixes=set(SUPPORTED_IMAGE_SUFFIXES),
        prefixes=MEMORY_IMAGE_PREFIXES,
    )


def resolve_audit_json(memory_root: Path, relative_path: str) -> Path:
    return _resolve_allowed_file(
        memory_root,
        relative_path,
        suffixes={".json"},
        prefixes=AUDIT_JSON_PREFIXES,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def list_input_images(input_root: Path) -> list[dict[str, Any]]:
    """Return deterministic metadata for every safe input image."""

    root = input_root.expanduser().resolve()
    if not root.exists():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            continue
        try:
            with Image.open(resolved) as image:
                width, height = image.size
        except (OSError, ValueError):
            width, height = 0, 0
        stat = resolved.stat()
        relative = resolved.relative_to(root).as_posix()
        items.append(
            {
                "path": relative,
                "name": resolved.name,
                "size_bytes": stat.st_size,
                "width": width,
                "height": height,
                "sha256": _sha256(resolved),
                "modified_at_utc": datetime.fromtimestamp(
                    stat.st_mtime,
                    timezone.utc,
                ).isoformat(),
                "asset_url": f"/api/input-asset?path={quote(relative, safe='')}",
            }
        )
    first_path_by_hash: dict[str, str] = {}
    for item in items:
        digest = str(item["sha256"])
        duplicate_of = first_path_by_hash.get(digest)
        item["is_duplicate"] = duplicate_of is not None
        item["duplicate_of"] = duplicate_of
        if duplicate_of is None:
            first_path_by_hash[digest] = str(item["path"])
    return items


def input_listing_payload(
    input_root: Path,
    *,
    locked: bool,
) -> dict[str, Any]:
    """Return the input collection with deterministic content-dedup counts."""

    items = list_input_images(input_root)
    duplicates = sum(bool(item["is_duplicate"]) for item in items)
    unique = len(items) - duplicates
    return {
        "total": len(items),
        "unique": unique,
        "duplicates": duplicates,
        "locked": locked,
        "items": items,
        # Compatibility aliases for early clients.
        "count": len(items),
        "inputs": items,
    }


def _validated_upload_name(filename: str | None) -> str:
    if filename is None or not UPLOAD_NAME.fullmatch(filename):
        raise InputValidationError(
            "Filenames may contain only letters, numbers, dot, underscore, and dash"
        )
    if Path(filename).name != filename:
        raise InputValidationError("Uploaded files cannot contain directories")
    if Path(filename).suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise InputValidationError("Only JPG, JPEG, PNG, and WebP images are allowed")
    return filename


def _copy_upload(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    written = 0
    with destination.open("xb") as handle:
        while True:
            block = upload.file.read(1024 * 1024)
            if not block:
                break
            written += len(block)
            if written > max_bytes:
                raise InputValidationError(
                    f"{upload.filename!r} exceeds the upload size limit"
                )
            handle.write(block)
    if written == 0:
        raise InputValidationError(f"{upload.filename!r} is empty")
    return written


def _validate_image_file(path: Path, filename: str, max_pixels: int) -> None:
    expected_format = IMAGE_FORMATS[Path(filename).suffix.lower()]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                width, height = image.size
                image_format = image.format
                if width <= 0 or height <= 0 or width * height > max_pixels:
                    raise InputValidationError(
                        f"{filename!r} exceeds the decoded pixel limit"
                    )
                image.verify()
    except InputValidationError:
        raise
    except (
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise InputValidationError(f"{filename!r} is not a valid image") from exc
    if image_format != expected_format:
        raise InputValidationError(
            f"{filename!r} content does not match its file extension"
        )


def save_input_uploads(
    input_root: Path,
    uploads: Sequence[UploadFile],
    *,
    max_bytes: int,
    max_pixels: int,
) -> list[str]:
    """Validate a multipart batch and atomically promote its images."""

    if not uploads:
        raise InputValidationError("Select at least one image")
    root = input_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    names = [_validated_upload_name(upload.filename) for upload in uploads]
    folded = [name.casefold() for name in names]
    if len(folded) != len(set(folded)):
        raise InputValidationError("The upload contains duplicate filenames")
    existing = {
        path.name.casefold()
        for path in root.iterdir()
        if path.is_file() or path.is_symlink()
    }
    conflict = next((name for name in names if name.casefold() in existing), None)
    if conflict is not None:
        raise FileExistsError(conflict)

    staged: list[tuple[Path, Path]] = []
    promoted: list[Path] = []
    try:
        for upload, name in zip(uploads, names, strict=True):
            temporary = root / f".upload-{uuid4().hex}.tmp"
            destination = root / name
            _copy_upload(upload, temporary, max_bytes)
            _validate_image_file(temporary, name, max_pixels)
            staged.append((temporary, destination))
        for temporary, destination in staged:
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination.name)
            os.replace(temporary, destination)
            promoted.append(destination)
        return [path.name for path in promoted]
    except Exception:
        for path in promoted:
            path.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        for path in root.glob(".upload-*.tmp"):
            if path.is_file() and not path.is_symlink():
                path.unlink(missing_ok=True)


def delete_input_image(input_root: Path, relative_path: str) -> None:
    path = resolve_input_file(input_root, relative_path)
    path.unlink()


def _read_progress_events(path: Path, after_sequence: int = 0) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                sequence = event.get("sequence")
                if isinstance(sequence, int) and sequence > after_sequence:
                    events.append(event)
    except OSError:
        return []
    return sorted(events, key=lambda event: int(event["sequence"]))


def _pid_is_alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _elapsed_since(started_at: object) -> float:
    if not isinstance(started_at, str):
        return 0.0
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return 0.0
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())


class ExperimentManager:
    """Own one durable subprocess-backed experiment at a time."""

    def __init__(self, settings: WebSettings) -> None:
        self.settings = settings.resolved()
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any = None
        self._current_run_dir = self._find_latest_run_dir()
        self._selected_memory_id = self._load_selected_memory_id()

    def _selection_path(self) -> Path:
        return self.settings.run_state_root / MEMORY_SELECTION_FILENAME

    def _load_selected_memory_id(self) -> str:
        """Restore the last explicit UI selection without trusting browser paths."""

        path = self._selection_path()
        if path.is_symlink():
            raise SafePathError("The memory selection file must not be a symbolic link")
        candidate: object = None
        try:
            payload = _read_json_object(path)
        except MemoryReadError:
            payload = None
        if payload is not None:
            candidate = payload.get("memory_id")
        if candidate is None and self._current_run_dir is not None:
            try:
                state = _read_json_object(self._state_path(self._current_run_dir))
            except MemoryReadError:
                state = None
            if state is not None:
                candidate = state.get("memory_id")
        try:
            memory_id = _validated_memory_id(str(candidate or "default"))
        except MemoryLibraryError:
            memory_id = "default"
        root = resolve_memory_library_root(self.settings, memory_id)
        if memory_id != "default" and not root.is_dir():
            return "default"
        return memory_id

    def _set_selected_memory_id_locked(self, memory_id: str) -> str:
        safe_id = _validated_memory_id(memory_id)
        root = resolve_memory_library_root(self.settings, safe_id)
        if safe_id != "default" and not root.is_dir():
            raise FileNotFoundError(safe_id)
        selection_path = self._selection_path()
        if selection_path.is_symlink():
            raise SafePathError("The memory selection file must not be a symbolic link")
        _write_json_atomic(
            selection_path,
            {"memory_id": safe_id, "updated_at_utc": _utc_now()},
        )
        self._selected_memory_id = safe_id
        return safe_id

    def _memory_for_read_locked(
        self,
        memory_id: str | None,
    ) -> tuple[str, Path]:
        """Resolve an explicit ID or the durable UI selection under one lock."""

        explicit = memory_id is not None
        safe_id = _validated_memory_id(
            memory_id if explicit else self._selected_memory_id
        )
        root = resolve_memory_library_root(self.settings, safe_id)
        if safe_id != "default" and not root.is_dir():
            if explicit:
                raise FileNotFoundError(safe_id)
            safe_id = self._set_selected_memory_id_locked("default")
            root = resolve_memory_library_root(self.settings, safe_id)
        return safe_id, root

    def selected_memory_id(self) -> str:
        """Return the persisted selection, repairing a vanished managed root."""

        with self._lock:
            safe_id, _ = self._memory_for_read_locked(None)
            return safe_id

    def memory_for_read(
        self,
        memory_id: str | None = None,
    ) -> tuple[str, Path]:
        """Resolve one readable managed root without accepting browser paths."""

        with self._lock:
            return self._memory_for_read_locked(memory_id)

    def _find_latest_run_dir(self) -> Path | None:
        root = self.settings.run_state_root
        if not root.is_dir():
            return None
        directories = sorted(
            (
                path
                for path in root.iterdir()
                if path.is_dir() and path.name.startswith("web_run_")
            ),
            key=lambda path: path.name,
        )
        return directories[-1] if directories else None

    @staticmethod
    def _state_path(run_dir: Path) -> Path:
        return run_dir / "state.json"

    @staticmethod
    def _events_path(run_dir: Path) -> Path:
        return run_dir / "events.jsonl"

    @staticmethod
    def _process_log_path(run_dir: Path) -> Path:
        return run_dir / "process.log"

    def _read_current_process_log_tail_locked(
        self,
        run_dir: Path,
        *,
        max_bytes: int = PROCESS_LOG_TAIL_BYTES,
    ) -> str:
        """Read only the bounded tail of the fixed current run's process log."""

        if self._current_run_dir != run_dir or max_bytes <= 0:
            return ""
        path = self._process_log_path(run_dir)
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - max_bytes), os.SEEK_SET)
                return handle.read(max_bytes).decode("utf-8", errors="replace")
        except OSError:
            return ""

    def _unexpected_process_error_locked(
        self,
        run_dir: Path,
        *,
        exit_code: int | None,
        kind: str,
    ) -> dict[str, Any]:
        return {
            "kind": kind,
            "exit_code": exit_code,
            "log_tail": self._read_current_process_log_tail_locked(run_dir),
        }

    def _read_state_locked(self) -> dict[str, Any] | None:
        if self._current_run_dir is None:
            return None
        return _read_json_object(self._state_path(self._current_run_dir))

    def _apply_latest_event_locked(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        assert self._current_run_dir is not None
        events = _read_progress_events(self._events_path(self._current_run_dir))
        latest_sequence = int(state.get("last_sequence") or 0)
        if events and int(events[-1]["sequence"]) > latest_sequence:
            latest = events[-1]
            state["last_sequence"] = latest.get("sequence", latest_sequence)
            state["last_event"] = latest.get("event")
            state["last_event_timestamp_utc"] = latest.get("timestamp_utc")
            state["run_id"] = latest.get("run_id") or state.get("run_id")
            state["stage"] = latest.get("stage") or state.get("stage")
            state["stage_status"] = latest.get("status")
            state["current"] = latest.get("current")
            state["total"] = latest.get("total")
            state["overall_percent"] = latest.get(
                "overall_percent",
                state.get("overall_percent", 0.0),
            )
            state["message"] = latest.get("message") or state.get("message")
            state["elapsed_seconds"] = latest.get(
                "elapsed_seconds",
                state.get("elapsed_seconds", 0.0),
            )
            state["updated_at_utc"] = _utc_now()
        if state.get("status") in ACTIVE_RUN_STATUSES:
            state["status"] = "running"
            state["elapsed_seconds"] = round(
                max(
                    float(state.get("elapsed_seconds") or 0.0),
                    _elapsed_since(state.get("started_at_utc")),
                ),
                3,
            )
            state["updated_at_utc"] = _utc_now()
        return state

    def _refresh_locked(self) -> dict[str, Any] | None:
        state = self._read_state_locked()
        if state is None:
            return None
        before = json.dumps(state, ensure_ascii=False, sort_keys=True)
        state = self._apply_latest_event_locked(state)
        if (
            state.get("status") in ACTIVE_RUN_STATUSES
            and self._process is None
            and not _pid_is_alive(state.get("pid"))
        ):
            completed_event = state.get("last_event") == "cli_completed"
            try:
                report = self._current_report_locked(state)
            except MemoryReadError:
                report = None
            report_status = report.get("status") if report else None
            has_failure_report = _is_formal_failure_report(report)
            if completed_event:
                result_status = state.get("stage_status")
            elif has_failure_report:
                result_status = report_status
            else:
                result_status = None
            completed = completed_event and result_status == "passed"
            unexpected_exit = not completed and not has_failure_report
            if completed:
                message = "Experiment completed."
            elif unexpected_exit:
                message = (
                    "Experiment process disappeared after service recovery; inspect "
                    "process_error.log_tail."
                )
            else:
                message = (
                    "Experiment finished with a failed result; inspect the report "
                    "and audit trail."
                )
            state.update(
                {
                    "status": "completed" if completed else "failed",
                    "stage_status": "completed" if completed else "failed",
                    "result_status": result_status,
                    "exit_code": None,
                    "completed_at_utc": (
                        state.get("last_event_timestamp_utc")
                        if completed_event
                        else _utc_now()
                    ),
                    "updated_at_utc": _utc_now(),
                    "message": message,
                }
            )
            if unexpected_exit and self._current_run_dir is not None:
                state["process_error"] = self._unexpected_process_error_locked(
                    self._current_run_dir,
                    exit_code=None,
                    kind="unexpected_recovery_exit",
                )
            else:
                state.pop("process_error", None)
        after = json.dumps(state, ensure_ascii=False, sort_keys=True)
        if before != after:
            _write_json_atomic(self._state_path(self._current_run_dir), state)
        return state

    def _running_locked(self) -> bool:
        state = self._refresh_locked()
        if state is None:
            return False
        return state.get("status") in ACTIVE_RUN_STATUSES

    @contextmanager
    def input_mutation(self) -> Iterator[None]:
        """Serialize input changes against experiment startup."""

        with self._lock:
            if self._running_locked():
                raise ExperimentBusyError(
                    "Inputs are locked while an experiment is running"
                )
            yield

    def inputs_locked(self) -> bool:
        with self._lock:
            return self._running_locked()

    def _active_memory_id_locked(self) -> str | None:
        state = self._refresh_locked()
        if not state or state.get("status") not in ACTIVE_RUN_STATUSES:
            return None
        try:
            return _validated_memory_id(str(state.get("memory_id") or "default"))
        except MemoryLibraryError:
            return "default"

    def memory_libraries(self) -> dict[str, Any]:
        """List server-managed libraries and the immutable active selection."""

        with self._lock:
            active_id = self._active_memory_id_locked()
            items = list_memory_libraries(self.settings, active_id=active_id)
            known_ids = {str(item["id"]) for item in items}
            selected_id = active_id or self._selected_memory_id
            if selected_id not in known_ids:
                selected_id = self._set_selected_memory_id_locked("default")
            return {
                "selected_id": selected_id,
                "active_id": active_id,
                "locked": active_id is not None,
                "items": items,
            }

    def select_memory_library(self, memory_id: str) -> dict[str, Any]:
        """Persist one existing library as the UI selection between page loads."""

        with self._lock:
            if self._running_locked():
                raise ExperimentBusyError(
                    "Memory libraries are locked while an experiment is running"
                )
            safe_id = self._set_selected_memory_id_locked(memory_id)
            return _memory_library_item(self.settings, safe_id)

    def memory_root_for_read(self, memory_id: str | None = None) -> Path:
        """Resolve a listed library without permitting browser paths."""

        _, root = self.memory_for_read(memory_id)
        return root

    def create_memory_library(self, label: str) -> dict[str, Any]:
        """Create one empty Git-manageable root with a server-generated ID."""

        with self._lock:
            if self._running_locked():
                raise ExperimentBusyError(
                    "Memory libraries are locked while an experiment is running"
                )
            safe_label = _validated_memory_label(label)
            existing_labels = {
                str(item["label"]).casefold()
                for item in list_memory_libraries(self.settings)
            }
            if safe_label.casefold() in existing_labels:
                raise FileExistsError(safe_label)
            for _ in range(8):
                timestamp = datetime.now(timezone.utc).strftime(
                    "%Y%m%dT%H%M%S%fZ"
                ).lower()
                memory_id = f"{timestamp}_{secrets.token_hex(3)}"
                root = resolve_memory_library_root(self.settings, memory_id)
                try:
                    root.mkdir(parents=False, exist_ok=False)
                except FileExistsError:
                    continue
                previous_selection = self._selected_memory_id
                try:
                    _write_json_atomic(
                        root / MEMORY_METADATA_FILENAME,
                        {
                            "schema_version": 1,
                            "id": memory_id,
                            "label": safe_label,
                            "created_at_utc": _utc_now(),
                        },
                    )
                    self._set_selected_memory_id_locked(memory_id)
                except Exception:
                    self._selected_memory_id = previous_selection
                    shutil.rmtree(root)
                    raise
                return _memory_library_item(self.settings, memory_id)
            raise FileExistsError("Could not allocate a unique memory library ID")

    def delete_memory_library(self, memory_id: str) -> dict[str, Any]:
        """Transactionally remove one library from the managed catalog."""

        with self._lock:
            if self._running_locked():
                raise ExperimentBusyError(
                    "Memory libraries are locked while an experiment is running"
                )
            safe_id = _validated_memory_id(memory_id)
            root = resolve_memory_library_root(self.settings, safe_id)
            if safe_id != "default" and not root.is_dir():
                raise FileNotFoundError(safe_id)
            if root.is_symlink():
                raise SafePathError("Symbolic-link memory libraries are not allowed")
            if root.exists() and not root.is_dir():
                raise MemoryLibraryError("Memory library root must be a directory")
            owned_run_id: str | None = None
            try:
                owned_report = read_latest_memory_report(
                    root,
                    self.settings.database_filename,
                )
                owned_run = (
                    owned_report.get("run")
                    if isinstance(owned_report, dict)
                    and isinstance(owned_report.get("run"), dict)
                    else {}
                )
                owned_run_id = str(owned_run.get("run_id") or "") or None
            except (MemoryReadError, OSError):
                owned_run_id = None
            state = self._read_state_locked()
            state_belongs_to_library = bool(
                state is not None
                and str(state.get("memory_id") or "default") == safe_id
                and self._current_run_dir is not None
            )
            original_state = dict(state) if state_belongs_to_library and state else None
            original_selection = self._selected_memory_id
            deletion_root = self.settings.run_state_root / "deletions"
            if deletion_root.is_symlink():
                raise SafePathError("Deletion staging directory must not be a symbolic link")
            deletion_root.mkdir(parents=True, exist_ok=True)
            tombstone = deletion_root / f"{root.name}.{uuid4().hex}.deleting"
            if tombstone.resolve().parent != deletion_root.resolve():
                raise SafePathError("Deletion staging path is outside the Web run state")

            moved_root = False
            recreated_default = False
            selection_changed = False
            state_changed = False
            try:
                if root.exists():
                    os.replace(root, tombstone)
                    moved_root = True
                if safe_id == "default":
                    root.mkdir(parents=False, exist_ok=False)
                    recreated_default = True
                if self._selected_memory_id == safe_id and safe_id != "default":
                    self._set_selected_memory_id_locked("default")
                    selection_changed = True
                if state_belongs_to_library and state is not None:
                    updated_state = {**state, "memory_deleted_at_utc": _utc_now()}
                    assert self._current_run_dir is not None
                    _write_json_atomic(
                        self._state_path(self._current_run_dir),
                        updated_state,
                    )
                    state_changed = True
            except Exception as exc:
                rollback_errors: list[str] = []
                if recreated_default:
                    try:
                        root.rmdir()
                    except OSError as rollback_exc:
                        rollback_errors.append(f"default root: {rollback_exc}")
                if moved_root and tombstone.exists():
                    try:
                        os.replace(tombstone, root)
                    except OSError as rollback_exc:
                        rollback_errors.append(f"library root: {rollback_exc}")
                if selection_changed:
                    try:
                        self._set_selected_memory_id_locked(original_selection)
                    except Exception as rollback_exc:  # noqa: BLE001
                        rollback_errors.append(f"selection: {rollback_exc}")
                else:
                    self._selected_memory_id = original_selection
                if state_changed and original_state is not None:
                    try:
                        assert self._current_run_dir is not None
                        _write_json_atomic(
                            self._state_path(self._current_run_dir),
                            original_state,
                        )
                    except Exception as rollback_exc:  # noqa: BLE001
                        rollback_errors.append(f"run state: {rollback_exc}")
                if rollback_errors:
                    raise MemoryMutationError(
                        "Memory deletion failed and rollback needs manual review: "
                        + "; ".join(rollback_errors)
                    ) from exc
                raise

            cleanup_pending = False
            if moved_root:
                try:
                    shutil.rmtree(tombstone)
                except OSError:
                    cleanup_pending = True
            if owned_run_id and self.settings.report_path.is_file():
                try:
                    external_report = _read_json_object(self.settings.report_path)
                    external_run = (
                        external_report.get("run")
                        if isinstance(external_report, dict)
                        and isinstance(external_report.get("run"), dict)
                        else {}
                    )
                    if str(external_run.get("run_id") or "") == owned_run_id:
                        self.settings.report_path.unlink()
                except (MemoryReadError, OSError):
                    pass

            response: dict[str, Any] = {
                "deleted_id": safe_id,
                "action": "cleared" if safe_id == "default" else "deleted",
                "selected_id": self._selected_memory_id,
                "cleanup_pending": cleanup_pending,
            }
            if safe_id == "default":
                response["item"] = _memory_library_item(self.settings, "default")
            return response

    def _new_run_dir(self) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = self.settings.run_state_root / f"web_run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _command(
        self,
        events_path: Path,
        memory_root: Path | None = None,
        *,
        validate_demo: bool = True,
    ) -> list[str]:
        selected_root = (memory_root or self.settings.memory_root).resolve()
        command = [
            str(self.settings.python_executable),
            str(self.settings.project_root / "scripts" / "run_object_memory.py"),
            "--input-dir",
            str(self.settings.input_root),
            "--memory-root",
            str(selected_root),
        ]
        if validate_demo:
            command.append("--validate-demo")
        command.extend(
            [
                "--report",
                str(self.settings.report_path),
                "--progress-file",
                str(events_path),
            ]
        )
        return command

    def start(self, memory_id: str = "default") -> dict[str, Any]:
        """Start the canonical CLI with fixed paths and no browser-supplied flags."""

        with self._lock:
            if self._running_locked():
                raise ExperimentBusyError("An experiment is already running")
            if not list_input_images(self.settings.input_root):
                raise InputValidationError("Upload at least one input image first")
            safe_memory_id = _validated_memory_id(memory_id)
            memory_root = resolve_memory_library_root(self.settings, safe_memory_id)
            if safe_memory_id != "default" and not memory_root.is_dir():
                raise FileNotFoundError(safe_memory_id)
            library = _memory_library_item(self.settings, safe_memory_id)
            if not library["continuable"]:
                raise MemoryLibraryError(
                    str(library.get("issue") or "This memory library cannot be continued")
                )
            self._set_selected_memory_id_locked(safe_memory_id)
            validate_demo = int(library["counts"].get("runs") or 0) == 0

            run_dir = self._new_run_dir()
            self._current_run_dir = run_dir
            events_path = self._events_path(run_dir)
            log_path = self._process_log_path(run_dir)
            state = {
                "web_run_id": run_dir.name,
                "run_id": None,
                "status": "starting",
                "stage": "startup",
                "stage_status": "starting",
                "current": 0,
                "total": None,
                "overall_percent": 0.0,
                "message": "Starting the end-to-end experiment.",
                "started_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
                "completed_at_utc": None,
                "elapsed_seconds": 0.0,
                "last_sequence": 0,
                "pid": None,
                "exit_code": None,
                "result_status": None,
                "memory_id": safe_memory_id,
                "memory_label": library["label"],
                "validation_mode": (
                    "fresh_demo" if validate_demo else "incremental"
                ),
                "memory_root_relative": memory_root.relative_to(
                    self.settings.project_root
                ).as_posix(),
                "report_mtime_before_ns": (
                    self.settings.report_path.stat().st_mtime_ns
                    if self.settings.report_path.is_file()
                    else None
                ),
            }
            state["started_at"] = state["started_at_utc"]
            _write_json_atomic(self._state_path(run_dir), state)
            self._log_handle = log_path.open("wb")
            child_environment = os.environ.copy()
            child_environment.pop("OBJECT_MEMORY_WEB_PASSWORD", None)
            try:
                process = subprocess.Popen(
                    self._command(
                        events_path,
                        memory_root,
                        validate_demo=validate_demo,
                    ),
                    cwd=self.settings.project_root,
                    env=child_environment,
                    stdin=subprocess.DEVNULL,
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception as exc:
                events_path.touch(exist_ok=True)
                self._log_handle.close()
                self._log_handle = None
                state.update(
                    {
                        "status": "failed",
                        "completed_at_utc": _utc_now(),
                        "updated_at_utc": _utc_now(),
                        "message": (
                            "Experiment startup failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
                )
                _write_json_atomic(self._state_path(run_dir), state)
                raise

            self._process = process
            state.update(
                {
                    "status": "running",
                    "stage_status": "running",
                    "pid": process.pid,
                    "updated_at_utc": _utc_now(),
                }
            )
            _write_json_atomic(self._state_path(run_dir), state)
            watcher = threading.Thread(
                target=self._watch_process,
                args=(run_dir, process),
                daemon=True,
                name=f"object-memory-{run_dir.name}",
            )
            watcher.start()
            return self.current(after_sequence=0)

    def _release_watched_process_locked(
        self,
        run_dir: Path,
        process: subprocess.Popen[bytes],
    ) -> None:
        """Clear only the handles owned by one watcher while holding ``_lock``."""

        owns_process = self._process is process
        owns_run_without_process = (
            self._process is None and self._current_run_dir == run_dir
        )
        if owns_process or owns_run_without_process:
            log_handle = self._log_handle
            self._log_handle = None
            if log_handle is not None:
                try:
                    log_handle.close()
                except OSError:
                    pass
        if owns_process:
            self._process = None

    def _watch_process(
        self,
        run_dir: Path,
        process: subprocess.Popen[bytes],
    ) -> None:
        last_state: dict[str, Any] = {}
        watcher_error: Exception | None = None
        exit_code: int | None = None
        try:
            while process.poll() is None:
                with self._lock:
                    if self._current_run_dir == run_dir:
                        try:
                            refreshed = self._refresh_locked()
                            if refreshed is not None:
                                last_state = dict(refreshed)
                        except Exception as exc:  # noqa: BLE001 - persist evidence
                            watcher_error = watcher_error or exc
                time.sleep(0.5)
            exit_code = process.wait()
            with self._lock:
                if self._current_run_dir != run_dir:
                    self._release_watched_process_locked(run_dir, process)
                    return
                try:
                    state = self._read_state_locked() or dict(last_state)
                    state = self._apply_latest_event_locked(state)
                    last_state = dict(state)
                except Exception as exc:  # noqa: BLE001 - repair broken state JSON
                    watcher_error = watcher_error or exc
                    state = dict(last_state)

                report: dict[str, Any] | None = None
                if watcher_error is None:
                    try:
                        report = self._current_report_locked(state)
                    except Exception as exc:  # noqa: BLE001 - persist report failure
                        watcher_error = exc

                if watcher_error is not None:
                    now = _utc_now()
                    state.setdefault("web_run_id", run_dir.name)
                    state.setdefault("memory_id", self._selected_memory_id)
                    state.setdefault("status", "running")
                    state.setdefault("stage", "watcher")
                    state.setdefault("overall_percent", 0.0)
                    state.setdefault("last_sequence", 0)
                    state.setdefault("started_at_utc", now)
                    process_error = self._unexpected_process_error_locked(
                        run_dir,
                        exit_code=exit_code,
                        kind="watcher_artifact_error",
                    )
                    process_error.update(
                        {
                            "error_type": type(watcher_error).__name__,
                            "message": str(watcher_error),
                        }
                    )
                    state.update(
                        {
                            "status": "failed",
                            "stage_status": "failed",
                            "completed_at_utc": now,
                            "updated_at_utc": now,
                            "elapsed_seconds": round(
                                _elapsed_since(state.get("started_at_utc")),
                                3,
                            ),
                            "exit_code": exit_code,
                            "result_status": None,
                            "message": (
                                "Experiment watcher could not read its state or report; "
                                "inspect process_error."
                            ),
                            "process_error": process_error,
                        }
                    )
                    self._release_watched_process_locked(run_dir, process)
                    _write_json_atomic(self._state_path(run_dir), state)
                    return

                result_status = report.get("status") if report else None
                completed = (
                    exit_code == 0
                    and report is not None
                    and result_status == "passed"
                )
                has_failure_report = _is_formal_failure_report(report)
                unexpected_exit = not completed and not has_failure_report
                if completed:
                    message = "Experiment completed."
                elif unexpected_exit:
                    message = (
                        "Experiment process exited unexpectedly; inspect "
                        "process_error.log_tail."
                    )
                else:
                    message = (
                        "Experiment finished with a failed result; inspect "
                        "the report and audit trail."
                    )
                state.update(
                    {
                        "status": "completed" if completed else "failed",
                        "stage_status": "completed" if completed else "failed",
                        "completed_at_utc": _utc_now(),
                        "updated_at_utc": _utc_now(),
                        "elapsed_seconds": round(
                            _elapsed_since(state.get("started_at_utc")),
                            3,
                        ),
                        "exit_code": exit_code,
                        "result_status": result_status,
                        "message": message,
                    }
                )
                if unexpected_exit:
                    state["process_error"] = self._unexpected_process_error_locked(
                        run_dir,
                        exit_code=exit_code,
                        kind="unexpected_process_exit",
                    )
                else:
                    state.pop("process_error", None)
                if completed:
                    state["overall_percent"] = 100.0
                self._release_watched_process_locked(run_dir, process)
                _write_json_atomic(self._state_path(run_dir), state)
        finally:
            with self._lock:
                self._release_watched_process_locked(run_dir, process)

    def _current_report_locked(
        self,
        state: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not self.settings.report_path.is_file():
            return None
        current = self.settings.report_path.stat().st_mtime_ns
        if state is None:
            return _read_json_object(self.settings.report_path)
        before = state.get("report_mtime_before_ns")
        if before == current:
            return None
        return _read_json_object(self.settings.report_path)

    def current(self, *, after_sequence: int = 0) -> dict[str, Any]:
        with self._lock:
            state = self._refresh_locked()
            if state is None or self._current_run_dir is None:
                return {
                    "active": False,
                    "input_locked": False,
                    "state": {
                        "status": "idle",
                        "stage": None,
                        "overall_percent": 0.0,
                        "last_sequence": 0,
                    },
                    "events": [],
                }
            events = _read_progress_events(
                self._events_path(self._current_run_dir),
                after_sequence=after_sequence,
            )
            active = state.get("status") in ACTIVE_RUN_STATUSES
            return {
                "active": active,
                "input_locked": active,
                "state": state,
                "events": events,
            }

    def result_for_memory(
        self,
        memory_id: str | None = None,
    ) -> dict[str, Any]:
        """Read one library's report and ownership state under one manager lock."""

        with self._lock:
            current_state = self._refresh_locked()
            selected_id, memory_root = self._memory_for_read_locked(memory_id)
            library = _memory_library_item(self.settings, selected_id)
            belongs_to_current = bool(
                current_state
                and current_state.get("web_run_id")
                and not current_state.get("memory_deleted_at_utc")
                and str(current_state.get("memory_id") or "default") == selected_id
            )
            if belongs_to_current:
                assert current_state is not None
                try:
                    report = self._current_report_locked(current_state)
                except (MemoryReadError, OSError):
                    process_error = current_state.get("process_error")
                    if not (
                        isinstance(process_error, dict)
                        and process_error.get("kind") == "watcher_artifact_error"
                    ):
                        raise
                    report = None
                expected_run_id = str(current_state.get("run_id") or "")
                if expected_run_id:
                    report_run = (
                        report.get("run")
                        if isinstance(report, dict)
                        and isinstance(report.get("run"), dict)
                        else {}
                    )
                    report_run_id = str(report_run.get("run_id") or "")
                    if report_run_id != expected_run_id:
                        exact_report = read_memory_report_for_run(
                            memory_root,
                            expected_run_id,
                        )
                        if exact_report is not None:
                            report = exact_report
                        elif report_run_id or not _is_formal_failure_report(report):
                            report = None
                state = {
                    **current_state,
                    "memory_id": selected_id,
                    "memory_label": current_state.get("memory_label")
                    or library["label"],
                }
                is_current = report is not None
                is_latest_for_library = report is not None
            else:
                report = read_latest_memory_report(
                    memory_root,
                    self.settings.database_filename,
                )
                report_run = (
                    report.get("run")
                    if isinstance(report, dict)
                    and isinstance(report.get("run"), dict)
                    else {}
                )
                run_id = str(report_run.get("run_id") or "")
                timing = (
                    read_memory_run_timing(
                        memory_root,
                        run_id,
                        self.settings.database_filename,
                    )
                    if run_id
                    else {}
                )
                state = {
                    "status": report.get("status") if report else "idle",
                    "stage": "completed" if report else None,
                    "overall_percent": 100.0 if report else 0.0,
                    "last_sequence": 0,
                    "memory_id": selected_id,
                    "memory_label": library["label"],
                    **timing,
                }
                is_current = False
                is_latest_for_library = report is not None

            return {
                "available": report is not None,
                "is_current_run": is_current,
                "is_latest_for_library": is_latest_for_library,
                "memory_id": selected_id,
                "state": state,
                "report": report,
                "summary": (
                    deterministic_result_summary(report, state)
                    if report is not None
                    else None
                ),
            }


def _open_read_only_database(path: Path) -> sqlite3.Connection:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def _json_list(value: object) -> list[Any]:
    try:
        decoded = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    return decoded if isinstance(decoded, list) else []


def _json_object(value: object) -> dict[str, Any]:
    try:
        decoded = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _object_observations(
    connection: sqlite3.Connection,
    object_id: str,
    schema_version: int,
) -> list[dict[str, Any]]:
    if schema_version >= 3:
        statement = """
            SELECT
                o.*, p.crop_path, p.mask_path, p.overlay_path,
                s.relative_path AS source_path,
                p.prompt AS sam_text_prompt, p.score AS sam_score,
                d.decision, d.matched_object_id,
                d.confidence AS decision_confidence,
                d.reason_code, d.short_reason, d.raw_response_path,
                d.qwen_hypothesis, d.qwen_matched_object_id,
                d.visual_evidence_json
            FROM observations AS o
            JOIN source_images AS s ON s.id = o.source_image_id
            JOIN proposals AS p ON p.id = o.proposal_id
            LEFT JOIN decisions AS d ON d.proposal_id = p.id
            WHERE o.object_id = ?
            ORDER BY o.created_at, o.id
        """
    else:
        statement = """
            SELECT
                o.*, s.relative_path AS source_path,
                p.prompt AS sam_text_prompt, p.score AS sam_score,
                d.decision, d.matched_object_id,
                d.confidence AS decision_confidence,
                d.reason_code, d.short_reason, d.raw_response_path
            FROM observations AS o
            JOIN source_images AS s ON s.id = o.source_image_id
            JOIN proposals AS p ON p.id = o.proposal_id
            LEFT JOIN decisions AS d
              ON d.proposal_id = p.id
             AND d.attempt = (
                 SELECT MAX(d2.attempt) FROM decisions AS d2
                 WHERE d2.proposal_id = p.id
             )
            WHERE o.object_id = ?
            ORDER BY o.created_at, o.id
        """
    rows = connection.execute(statement, (object_id,)).fetchall()
    if schema_version >= 3:
        observations = []
        for row in rows:
            item = dict(row)
            item["visual_evidence"] = _json_object(
                item.pop("visual_evidence_json", "{}")
            )
            observations.append(item)
        return observations
    return [dict(row) for row in rows]


def _read_objects(
    connection: sqlite3.Connection,
    schema_version: int,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            o.*,
            COUNT(obs.id) AS observation_count,
            MAX(obs.created_at) AS last_observed_at
        FROM objects AS o
        LEFT JOIN observations AS obs ON obs.object_id = o.id
        GROUP BY o.id
        ORDER BY o.created_at, o.id
        """
    ).fetchall()
    objects: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if schema_version >= 3:
            summary = _json_object(item.pop("summary_json", "{}"))
            item.update(summary)
            item["description"] = summary.get("stable_description")
            item["annotation_confidence"] = summary.get("summary_confidence")
            item["material"] = []
            item["color"] = []
        else:
            materials = _json_list(item.pop("material_json", "[]"))
            colors = _json_list(item.pop("color_json", "[]"))
            item["material"] = materials
            item["color"] = colors
            item["stable_description"] = item.get("description")
            item["stable_identity_features"] = []
            item["brand_or_markings"] = []
            item["part_appearance"] = (
                [{"part": "整体（旧版）", "color": colors, "material": materials}]
                if colors or materials
                else []
            )
            item["summary_confidence"] = item.get("annotation_confidence")
        item["observations"] = _object_observations(
            connection,
            str(item["id"]),
            schema_version,
        )
        objects.append(item)
    return objects


def _read_candidates(
    connection: sqlite3.Connection,
    schema_version: int,
) -> list[dict[str, Any]]:
    if schema_version >= 3:
        statement = """
            SELECT
                p.*, s.run_id, s.relative_path AS source_path,
                s.status AS source_status, d.decision, d.matched_object_id,
                d.confidence AS decision_confidence, d.reason_code,
                d.short_reason, d.raw_response_path, d.qwen_hypothesis,
                d.qwen_matched_object_id, d.visual_evidence_json,
                o.id AS observation_id, o.object_id AS observation_object_id
            FROM proposals AS p
            JOIN source_images AS s ON s.id = p.source_image_id
            LEFT JOIN decisions AS d ON d.proposal_id = p.id
            LEFT JOIN observations AS o ON o.proposal_id = p.id
            ORDER BY p.created_at, p.id
        """
    else:
        statement = """
            SELECT
                p.*, s.run_id, s.relative_path AS source_path,
                s.status AS source_status, d.decision, d.matched_object_id,
                d.confidence AS decision_confidence, d.reason_code,
                d.short_reason, d.raw_response_path,
                o.id AS observation_id, o.object_id AS observation_object_id
            FROM proposals AS p
            JOIN source_images AS s ON s.id = p.source_image_id
            LEFT JOIN decisions AS d
              ON d.proposal_id = p.id
             AND d.attempt = (
                 SELECT MAX(d2.attempt) FROM decisions AS d2
                 WHERE d2.proposal_id = p.id
             )
            LEFT JOIN observations AS o ON o.proposal_id = p.id
            ORDER BY p.created_at, p.id
        """
    rows = connection.execute(statement).fetchall()
    candidates: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["bbox"] = {
            "x_min": item.pop("bbox_x_min"),
            "y_min": item.pop("bbox_y_min"),
            "x_max": item.pop("bbox_x_max"),
            "y_max": item.pop("bbox_y_max"),
        }
        item["object_id"] = (
            item.get("observation_object_id") or item.get("matched_object_id")
        )
        if schema_version >= 3:
            item["target_anchor"] = _json_object(
                item.pop("target_anchor_json", "{}")
            )
            item["fingerprint"] = _json_object(
                item.pop("fingerprint_json", "{}")
            )
            item["visual_evidence"] = _json_object(
                item.pop("visual_evidence_json", "{}")
            )
        candidates.append(item)
    return candidates


def _read_runs(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            r.*,
            (SELECT COUNT(*) FROM source_images AS s WHERE s.run_id = r.id)
                AS source_count,
            (
                SELECT COUNT(*)
                FROM proposals AS p
                JOIN source_images AS s ON s.id = p.source_image_id
                WHERE s.run_id = r.id
            ) AS proposal_count,
            (
                SELECT COUNT(*)
                FROM observations AS o
                JOIN source_images AS s ON s.id = o.source_image_id
                WHERE s.run_id = r.id
            ) AS observation_count
        FROM runs AS r
        ORDER BY r.started_at DESC, r.id DESC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _read_sources(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT *
        FROM source_images
        ORDER BY created_at, id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def read_memory_snapshot(
    memory_root: Path,
    database_filename: str = "memory.sqlite",
) -> dict[str, Any]:
    """Read a short, query-only snapshot for cards, timelines, and lineage."""

    unresolved_root = memory_root.expanduser()
    if unresolved_root.is_symlink():
        raise MemoryReadError("Memory root must not be a symbolic link")
    root = unresolved_root.resolve()
    database = root / database_filename
    if database.is_symlink():
        raise MemoryReadError("memory.sqlite must not be a symbolic link")
    if database.resolve().parent != root:
        raise MemoryReadError("Memory database is outside its library root")
    empty_counts = {table: 0 for table in CORE_TABLES}
    empty_counts["active_objects"] = 0
    if not database.is_file():
        return {
            "initialized": False,
            "schema_version": None,
            "counts": empty_counts,
            "objects": [],
            "candidates": [],
            "runs": [],
            "sources": [],
        }
    try:
        with closing(_open_read_only_database(database)) as connection:
            schema_version = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if schema_version not in {2, SCHEMA_VERSION}:
                raise MemoryReadError(
                    f"Unsupported schema version {schema_version}"
                )
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            missing = set(CORE_TABLES) - tables
            if missing:
                raise MemoryReadError(
                    f"Memory database is missing tables: {sorted(missing)}"
                )
            counts = {
                table: int(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
                for table in CORE_TABLES
            }
            counts["active_objects"] = int(
                connection.execute(
                    "SELECT COUNT(*) FROM objects WHERE status = 'active'"
                ).fetchone()[0]
            )
            return {
                "initialized": True,
                "schema_version": schema_version,
                "read_only": schema_version < SCHEMA_VERSION,
                "counts": counts,
                "objects": _read_objects(connection, schema_version),
                "candidates": _read_candidates(connection, schema_version),
                "runs": _read_runs(connection),
                "sources": _read_sources(connection),
            }
    except MemoryReadError:
        raise
    except sqlite3.Error as exc:
        raise MemoryReadError(f"Cannot read memory database: {exc}") from exc


def deterministic_result_summary(
    report: dict[str, Any],
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a stable, non-model-authored summary from report counters."""

    images = report.get("images")
    image_items = images if isinstance(images, list) else []
    run = report.get("run") if isinstance(report.get("run"), dict) else {}
    source_counts = (
        run.get("source_counts") if isinstance(run.get("source_counts"), dict) else {}
    )
    proposal_counts = (
        run.get("proposal_counts")
        if isinstance(run.get("proposal_counts"), dict)
        else {}
    )
    decision_counts = (
        run.get("decision_counts")
        if isinstance(run.get("decision_counts"), dict)
        else {}
    )

    scene_targets = 0
    sam_prompts = 0
    zero_prompts = 0
    raw_candidates = 0
    kept_candidates = 0
    filtered_candidates = 0
    for image in image_items:
        if not isinstance(image, dict):
            continue
        guidance = image.get("scene_guidance")
        if isinstance(guidance, dict):
            targets = guidance.get("targets")
            if isinstance(targets, list):
                scene_targets += len(targets)
            else:
                target_count = guidance.get("target_count")
                if isinstance(target_count, int):
                    scene_targets += target_count
        sam = image.get("sam")
        if isinstance(sam, dict):
            prompt_counts = sam.get("prompt_detection_counts")
            if isinstance(prompt_counts, dict):
                sam_prompts += len(prompt_counts)
            zero = sam.get("zero_candidate_prompts")
            if isinstance(zero, list):
                zero_prompts += len(zero)
            raw_candidates += int(sam.get("above_confidence_threshold_candidates") or 0)
            kept_candidates += int(sam.get("kept") or 0)
            filtered_candidates += int(sam.get("filtered") or 0)

    summary_errors: list[dict[str, Any]] = []
    seen_error_messages: set[str] = set()

    def append_error(source: str, error: object) -> None:
        if not error:
            return
        error_type: str | None = None
        if isinstance(error, dict):
            message_value = error.get("message") or error
            message = (
                str(message_value)
                if isinstance(message_value, str)
                else json.dumps(message_value, ensure_ascii=False, sort_keys=True)
            )
            if error.get("type"):
                error_type = str(error["type"])
        else:
            message = str(error)
        if message in seen_error_messages:
            return
        seen_error_messages.add(message)
        item: dict[str, Any] = {"source": source, "message": message}
        if error_type is not None:
            item["type"] = error_type
        summary_errors.append(item)

    for field in ("error", "progress_error"):
        append_error(f"report.{field}", report.get(field))
    external_errors = report.get("external_errors")
    if isinstance(external_errors, list):
        for error in external_errors:
            append_error("external_errors", error)
    for image_index, image in enumerate(image_items):
        if not isinstance(image, dict):
            continue
        append_error(f"images[{image_index}].error", image.get("error"))
        candidate_reasoning = image.get("candidate_reasoning")
        if isinstance(candidate_reasoning, dict):
            candidate_errors = candidate_reasoning.get("errors")
            if isinstance(candidate_errors, list):
                for error in candidate_errors:
                    append_error(
                        f"images[{image_index}].candidate_reasoning.errors",
                        error,
                    )
        decisions = image.get("decisions")
        if isinstance(decisions, list):
            for decision_index, decision in enumerate(decisions):
                if not isinstance(decision, dict):
                    continue
                decision_errors = decision.get("errors")
                if isinstance(decision_errors, list):
                    for error in decision_errors:
                        append_error(
                            f"images[{image_index}].decisions[{decision_index}].errors",
                            error,
                        )
    elapsed = state.get("elapsed_seconds") if isinstance(state, dict) else None
    sam_metrics = (
        report.get("models", {}).get("sam3", {})
        if isinstance(report.get("models"), dict)
        else {}
    )
    return {
        "status": report.get("status"),
        "run_id": run.get("run_id"),
        "memory_id": state.get("memory_id") if isinstance(state, dict) else None,
        "memory_label": state.get("memory_label") if isinstance(state, dict) else None,
        "input_files": len(image_items),
        "unique_sources": sum(int(value or 0) for value in source_counts.values()),
        "duplicate_sources_skipped": int(run.get("duplicate_sources_skipped") or 0),
        "source_counts": source_counts,
        "scene_targets": scene_targets,
        "sam_prompts": sam_prompts,
        "sam_zero_candidate_prompts": zero_prompts,
        "sam_above_threshold_candidates": raw_candidates,
        "sam_kept": kept_candidates,
        "sam_filtered": filtered_candidates,
        "sam_confidence_threshold": (
            sam_metrics.get("confidence_threshold")
            if isinstance(sam_metrics, dict)
            else None
        ),
        "proposal_counts": proposal_counts,
        "decision_counts": decision_counts,
        "observations_added": int(run.get("observations_added") or 0),
        "active_objects_total": int(run.get("active_objects_total") or 0),
        "elapsed_seconds": elapsed,
        "error_count": len(summary_errors),
        "errors": summary_errors,
        "manual_review_required": True,
        "review_notice": (
            "A passed structure report does not establish semantic object-memory "
            "accuracy; inspect targets, candidates, decisions, and object timelines."
        ),
    }


def _basic_authorized(
    authorization: str | None,
    *,
    expected_username: str,
    expected_password: str,
) -> bool:
    if authorization is None:
        return False
    try:
        scheme, token = authorization.split(" ", 1)
        if scheme.casefold() != "basic":
            return False
        decoded = base64.b64decode(token, validate=True).decode("utf-8")
        username, password = decoded.split(":", 1)
    except (ValueError, UnicodeDecodeError, binascii.Error):
        return False
    return secrets.compare_digest(
        username.encode("utf-8"),
        expected_username.encode("utf-8"),
    ) and secrets.compare_digest(
        password.encode("utf-8"),
        expected_password.encode("utf-8"),
    )


def _input_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ExperimentBusyError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, FileExistsError):
        return HTTPException(
            status_code=409,
            detail=f"An input named {exc.args[0]!r} already exists",
        )
    if isinstance(exc, (InputValidationError, SafePathError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail="Input image not found")
    return HTTPException(status_code=500, detail="Input operation failed")


def _memory_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ExperimentBusyError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, FileExistsError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, (MemoryLibraryError, SafePathError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail="Memory library not found")
    if isinstance(exc, MemoryReadError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, MemoryMutationError):
        return HTTPException(status_code=500, detail=str(exc))
    return HTTPException(status_code=500, detail="Memory library operation failed")


def _asset_response(
    path: Path,
    *,
    cache_control: str = "private, max-age=300, immutable",
) -> FileResponse:
    return FileResponse(
        path,
        headers={"Cache-Control": cache_control},
    )


def create_app(settings: WebSettings | None = None) -> FastAPI:
    """Create the single-worker application without starting model runtimes."""

    resolved = (settings or default_web_settings()).resolved()
    resolved.input_root.mkdir(parents=True, exist_ok=True)
    manager = ExperimentManager(resolved)
    app = FastAPI(
        title="Object Memory Experiment",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.web_settings = resolved
    app.state.experiment_manager = manager

    if resolved.basic_password is not None:

        @app.middleware("http")
        async def require_basic_auth(request: Request, call_next: Any) -> Any:
            if not _basic_authorized(
                request.headers.get("authorization"),
                expected_username=resolved.basic_username,
                expected_password=resolved.basic_password or "",
            ):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Authentication required"},
                    headers={
                        "WWW-Authenticate": (
                            'Basic realm="Object Memory Lab", charset="UTF-8"'
                        )
                    },
                )
            return await call_next(request)

    @app.middleware("http")
    async def disable_api_cache(request: Request, call_next: Any) -> Any:
        if (
            request.url.path.startswith("/api/")
            and request.method in {"POST", "PUT", "PATCH", "DELETE"}
            and request.headers.get("x-object-memory-request") != "web-ui"
        ):
            return JSONResponse(
                status_code=403,
                content={"detail": "Mutating API requests require the Web UI header"},
            )
        response = await call_next(request)
        if request.url.path.startswith("/api/") and request.url.path not in {
            "/api/input-asset",
            "/api/memory-asset",
        }:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/inputs")
    def get_inputs() -> dict[str, Any]:
        return input_listing_payload(
            resolved.input_root,
            locked=manager.inputs_locked(),
        )

    @app.post("/api/inputs", status_code=201)
    def upload_inputs(files: list[UploadFile] = File(...)) -> dict[str, Any]:
        try:
            with manager.input_mutation():
                uploaded = save_input_uploads(
                    resolved.input_root,
                    files,
                    max_bytes=resolved.max_upload_bytes,
                    max_pixels=resolved.max_image_pixels,
                )
                payload = input_listing_payload(resolved.input_root, locked=False)
        except Exception as exc:
            raise _input_http_error(exc) from exc
        return {"uploaded": uploaded, **payload}

    @app.delete("/api/inputs")
    def delete_input(path: str = Query(..., min_length=1)) -> dict[str, Any]:
        try:
            with manager.input_mutation():
                delete_input_image(resolved.input_root, path)
                payload = input_listing_payload(resolved.input_root, locked=False)
        except Exception as exc:
            raise _input_http_error(exc) from exc
        return {"deleted": path, **payload}

    @app.get("/api/memories")
    def get_memories() -> dict[str, Any]:
        try:
            return manager.memory_libraries()
        except (MemoryReadError, MemoryLibraryError, SafePathError) as exc:
            raise _memory_http_error(exc) from exc

    @app.post("/api/memories", status_code=201)
    def create_memory(payload: MemoryCreateRequest | None = None) -> dict[str, Any]:
        try:
            item = manager.create_memory_library((payload or MemoryCreateRequest()).label)
        except Exception as exc:
            raise _memory_http_error(exc) from exc
        return {"created": True, "selected_id": item["id"], "item": item}

    @app.post("/api/memories/{memory_id}/select")
    def select_memory(memory_id: str) -> dict[str, Any]:
        try:
            item = manager.select_memory_library(memory_id)
        except Exception as exc:
            raise _memory_http_error(exc) from exc
        return {"selected_id": item["id"], "item": item}

    @app.delete("/api/memories/{memory_id}")
    def delete_memory(
        memory_id: str,
        confirm: str = Query(..., min_length=1),
    ) -> dict[str, Any]:
        try:
            safe_id = _validated_memory_id(memory_id)
            safe_confirmation = _validated_memory_id(confirm)
        except MemoryLibraryError as exc:
            raise _memory_http_error(exc) from exc
        if safe_id != safe_confirmation:
            raise HTTPException(
                status_code=400,
                detail="Deletion confirmation does not match the memory library ID",
            )
        try:
            return manager.delete_memory_library(safe_id)
        except Exception as exc:
            raise _memory_http_error(exc) from exc

    @app.post("/api/runs", status_code=202)
    def start_run(payload: RunStartRequest | None = None) -> dict[str, Any]:
        try:
            return manager.start((payload or RunStartRequest()).memory_id)
        except (ExperimentBusyError, InputValidationError) as exc:
            raise _input_http_error(exc) from exc
        except (MemoryLibraryError, SafePathError, FileNotFoundError) as exc:
            raise _memory_http_error(exc) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Experiment process could not start: {exc}",
            ) from exc

    @app.get("/api/runs/current")
    def current_run(
        after_sequence: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        return manager.current(after_sequence=after_sequence)

    @app.get("/api/results")
    def get_results(
        memory_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        try:
            return manager.result_for_memory(memory_id)
        except Exception as exc:
            raise _memory_http_error(exc) from exc

    @app.get("/api/memory")
    def get_memory(
        memory_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        try:
            safe_id, root = manager.memory_for_read(memory_id)
            library = _memory_library_item(resolved, safe_id)
            snapshot = read_memory_snapshot(
                root,
                resolved.database_filename,
            )
        except Exception as exc:
            raise _memory_http_error(exc) from exc
        return {
            "memory_id": safe_id,
            "memory_label": library["label"],
            "library": library,
            **snapshot,
        }

    @app.get("/api/input-asset")
    def get_input_asset(path: str = Query(..., min_length=1)) -> FileResponse:
        try:
            return _asset_response(
                resolve_input_file(resolved.input_root, path),
                cache_control="no-store",
            )
        except SafePathError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except MemoryLibraryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="Input image not found",
            ) from exc

    @app.get("/api/memory-asset")
    def get_memory_asset(
        path: str = Query(..., min_length=1),
        memory_id: str | None = Query(default=None),
    ) -> FileResponse:
        try:
            _, root = manager.memory_for_read(memory_id)
            return _asset_response(resolve_memory_image(root, path))
        except SafePathError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except MemoryLibraryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="Memory image not found",
            ) from exc

    @app.get("/api/audit-json")
    def get_audit_json(
        path: str = Query(..., min_length=1),
        memory_id: str | None = Query(default=None),
    ) -> JSONResponse:
        try:
            _, root = manager.memory_for_read(memory_id)
            artifact = resolve_audit_json(root, path)
            if artifact.stat().st_size > resolved.max_audit_json_bytes:
                raise HTTPException(status_code=413, detail="Audit JSON is too large")
            payload = json.loads(artifact.read_text(encoding="utf-8"))
        except SafePathError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except MemoryLibraryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Audit JSON not found") from exc
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=503,
                detail="Audit JSON is invalid",
            ) from exc
        return JSONResponse(content=payload)

    app.mount(
        "/static",
        StaticFiles(directory=str(resolved.static_root), check_dir=False),
        name="static",
    )

    @app.get("/")
    def index() -> FileResponse:
        index_path = resolved.static_root / "index.html"
        if not index_path.is_file():
            raise HTTPException(
                status_code=503,
                detail="Web interface is not installed",
            )
        return FileResponse(index_path, headers={"Cache-Control": "no-store"})

    return app


__all__ = [
    "ExperimentManager",
    "InputValidationError",
    "MemoryLibraryError",
    "MemoryReadError",
    "SafePathError",
    "WebSettings",
    "create_app",
    "default_web_settings",
    "delete_input_image",
    "deterministic_result_summary",
    "input_listing_payload",
    "list_input_images",
    "list_memory_libraries",
    "read_latest_memory_report",
    "read_memory_report_for_run",
    "read_memory_run_timing",
    "read_memory_snapshot",
    "resolve_audit_json",
    "resolve_input_file",
    "resolve_memory_library_root",
    "resolve_memory_image",
    "save_input_uploads",
]
