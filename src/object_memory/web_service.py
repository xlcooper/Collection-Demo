"""FastAPI service for the single-machine object-memory experiment UI."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import secrets
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

from .config import load_config, resolve_memory_root
from .memory_store import CORE_TABLES
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


class SafePathError(ValueError):
    """Raised when a requested file is outside an allowed application root."""


class InputValidationError(ValueError):
    """Raised when an uploaded input is unsafe or not a supported image."""


class ExperimentBusyError(RuntimeError):
    """Raised when a second mutation conflicts with an active experiment."""


class MemoryReadError(RuntimeError):
    """Raised when an initialized memory database cannot be read safely."""


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

    def _new_run_dir(self) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = self.settings.run_state_root / f"web_run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _command(self, events_path: Path) -> list[str]:
        return [
            str(self.settings.python_executable),
            str(self.settings.project_root / "scripts" / "run_object_memory.py"),
            "--input-dir",
            str(self.settings.input_root),
            "--memory-root",
            str(self.settings.memory_root),
            "--validate-demo",
            "--report",
            str(self.settings.report_path),
            "--progress-file",
            str(events_path),
        ]

    def start(self) -> dict[str, Any]:
        """Start the canonical CLI with fixed paths and no browser-supplied flags."""

        with self._lock:
            if self._running_locked():
                raise ExperimentBusyError("An experiment is already running")
            if not list_input_images(self.settings.input_root):
                raise InputValidationError("Upload at least one input image first")

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
                    self._command(events_path),
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

    def _watch_process(
        self,
        run_dir: Path,
        process: subprocess.Popen[bytes],
    ) -> None:
        while process.poll() is None:
            with self._lock:
                if self._current_run_dir == run_dir:
                    self._refresh_locked()
            time.sleep(0.5)
        exit_code = process.wait()
        with self._lock:
            if self._current_run_dir != run_dir:
                return
            state = self._read_state_locked() or {}
            state = self._apply_latest_event_locked(state)
            report = self._current_report_locked(state)
            result_status = report.get("status") if report else None
            completed = exit_code == 0
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
            _write_json_atomic(self._state_path(run_dir), state)
            if self._log_handle is not None:
                self._log_handle.close()
            self._log_handle = None
            self._process = None

    def _current_report_locked(
        self,
        state: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        report = _read_json_object(self.settings.report_path)
        if report is None or state is None:
            return report
        before = state.get("report_mtime_before_ns")
        current = self.settings.report_path.stat().st_mtime_ns
        if before == current:
            return None
        return report

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

    def result_report(self) -> tuple[dict[str, Any] | None, bool, dict[str, Any]]:
        with self._lock:
            state = self._refresh_locked()
            report = self._current_report_locked(state)
            if report is None:
                return None, False, state or {"status": "idle"}
            if state is None:
                return report, True, {"status": "idle"}
            return report, state.get("status") not in ACTIVE_RUN_STATUSES, state


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


def _object_observations(
    connection: sqlite3.Connection,
    object_id: str,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            o.id,
            o.object_id,
            o.proposal_id,
            o.source_image_id,
            o.crop_path,
            o.mask_path,
            o.overlay_path,
            o.description,
            o.created_at,
            s.relative_path AS source_path,
            p.prompt AS sam_text_prompt,
            p.score AS sam_score,
            d.decision,
            d.matched_object_id,
            d.confidence AS decision_confidence,
            d.reason_code,
            d.short_reason,
            d.raw_response_path
        FROM observations AS o
        JOIN source_images AS s ON s.id = o.source_image_id
        JOIN proposals AS p ON p.id = o.proposal_id
        LEFT JOIN decisions AS d
          ON d.proposal_id = p.id
         AND d.attempt = (
             SELECT MAX(d2.attempt)
             FROM decisions AS d2
             WHERE d2.proposal_id = p.id
         )
        WHERE o.object_id = ?
        ORDER BY o.created_at, o.id
        """,
        (object_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def _read_objects(connection: sqlite3.Connection) -> list[dict[str, Any]]:
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
        item["material"] = _json_list(item.pop("material_json", "[]"))
        item["color"] = _json_list(item.pop("color_json", "[]"))
        item["observations"] = _object_observations(connection, str(item["id"]))
        objects.append(item)
    return objects


def _read_candidates(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            p.*,
            s.run_id,
            s.relative_path AS source_path,
            s.status AS source_status,
            d.decision,
            d.matched_object_id,
            d.confidence AS decision_confidence,
            d.reason_code,
            d.short_reason,
            d.raw_response_path,
            o.id AS observation_id,
            o.object_id AS observation_object_id
        FROM proposals AS p
        JOIN source_images AS s ON s.id = p.source_image_id
        LEFT JOIN decisions AS d
          ON d.proposal_id = p.id
         AND d.attempt = (
             SELECT MAX(d2.attempt)
             FROM decisions AS d2
             WHERE d2.proposal_id = p.id
         )
        LEFT JOIN observations AS o ON o.proposal_id = p.id
        ORDER BY p.created_at, p.id
        """
    ).fetchall()
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

    database = memory_root.expanduser().resolve() / database_filename
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
                "schema_version": int(
                    connection.execute("PRAGMA user_version").fetchone()[0]
                ),
                "counts": counts,
                "objects": _read_objects(connection),
                "candidates": _read_candidates(connection),
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
    return {
        "status": report.get("status"),
        "run_id": run.get("run_id"),
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

    @app.post("/api/runs", status_code=202)
    def start_run() -> dict[str, Any]:
        try:
            return manager.start()
        except (ExperimentBusyError, InputValidationError) as exc:
            raise _input_http_error(exc) from exc
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
    def get_results() -> dict[str, Any]:
        try:
            report, is_current, state = manager.result_report()
        except MemoryReadError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "available": report is not None,
            "is_current_run": is_current,
            "state": state,
            "report": report,
            "summary": (
                deterministic_result_summary(report, state)
                if report is not None
                else None
            ),
        }

    @app.get("/api/memory")
    def get_memory() -> dict[str, Any]:
        try:
            return read_memory_snapshot(
                resolved.memory_root,
                resolved.database_filename,
            )
        except MemoryReadError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/input-asset")
    def get_input_asset(path: str = Query(..., min_length=1)) -> FileResponse:
        try:
            return _asset_response(
                resolve_input_file(resolved.input_root, path),
                cache_control="no-store",
            )
        except SafePathError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="Input image not found",
            ) from exc

    @app.get("/api/memory-asset")
    def get_memory_asset(path: str = Query(..., min_length=1)) -> FileResponse:
        try:
            return _asset_response(resolve_memory_image(resolved.memory_root, path))
        except SafePathError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="Memory image not found",
            ) from exc

    @app.get("/api/audit-json")
    def get_audit_json(path: str = Query(..., min_length=1)) -> JSONResponse:
        try:
            artifact = resolve_audit_json(resolved.memory_root, path)
            if artifact.stat().st_size > resolved.max_audit_json_bytes:
                raise HTTPException(status_code=413, detail="Audit JSON is too large")
            payload = json.loads(artifact.read_text(encoding="utf-8"))
        except SafePathError as exc:
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
    "SafePathError",
    "WebSettings",
    "create_app",
    "default_web_settings",
    "delete_input_image",
    "deterministic_result_summary",
    "input_listing_payload",
    "list_input_images",
    "read_memory_snapshot",
    "resolve_audit_json",
    "resolve_input_file",
    "resolve_memory_image",
    "save_input_uploads",
]
