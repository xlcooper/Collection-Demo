"""Structured progress events for long-running object-memory workflows."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


STAGE_BOUNDARIES: dict[str, tuple[float, float]] = {
    "input_registration": (0.0, 10.0),
    "scene_guidance": (10.0, 35.0),
    "sam3": (35.0, 65.0),
    "candidate_reasoning": (65.0, 95.0),
    "report": (95.0, 100.0),
}


class ProgressWriteError(RuntimeError):
    """Raised when an explicitly requested progress event cannot be persisted."""


class ProgressSink(Protocol):
    """Destination for one already-structured progress record."""

    def write(self, record: dict[str, Any]) -> None: ...


class JsonlProgressWriter:
    """Write one flushed UTF-8 JSON object per line to a fresh event log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.flush()
        except Exception as exc:  # noqa: BLE001 - expose requested audit failure
            raise ProgressWriteError(
                f"Unable to initialize progress file {self.path}: {exc}"
            ) from exc

    def write(self, record: dict[str, Any]) -> None:
        try:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            with self._lock, self.path.open(
                "a",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                handle.write(line + "\n")
                handle.flush()
        except Exception as exc:  # noqa: BLE001 - event loss must be explicit
            raise ProgressWriteError(
                f"Unable to append progress event to {self.path}: {exc}"
            ) from exc


class ProgressReporter:
    """Sequence, timestamp, and persist progress records for one CLI invocation."""

    def __init__(self, sink: ProgressSink, *, run_id: str | None = None) -> None:
        self.sink = sink
        self.run_id = run_id
        self._started = time.perf_counter()
        self._sequence = 0
        self._last_overall_percent = 0.0
        self._lock = threading.Lock()

    @property
    def last_overall_percent(self) -> float:
        return self._last_overall_percent

    def set_run_id(self, run_id: str) -> None:
        if not run_id:
            raise ValueError("run_id must not be empty")
        if self.run_id is not None and self.run_id != run_id:
            raise ValueError(
                f"Progress reporter already belongs to run {self.run_id}"
            )
        self.run_id = run_id

    def emit(
        self,
        *,
        event: str,
        stage: str,
        status: str,
        current: int,
        total: int,
        message: str,
        data: dict[str, Any] | None = None,
        overall_percent: float | None = None,
    ) -> dict[str, Any]:
        if not event or not stage or not status or not message:
            raise ValueError("Progress event text fields must not be empty")
        if current < 0 or total < 0:
            raise ValueError("Progress current and total must not be negative")
        if total and current > total:
            raise ValueError("Progress current must not exceed total")

        with self._lock:
            computed_percent = (
                self._stage_percent(stage, current, total, status)
                if overall_percent is None
                else overall_percent
            )
            if not 0.0 <= computed_percent <= 100.0:
                raise ValueError("overall_percent must be between 0 and 100")
            computed_percent = max(
                self._last_overall_percent,
                computed_percent,
            )
            sequence = self._sequence + 1
            record = {
                "sequence": sequence,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.perf_counter() - self._started, 3),
                "run_id": self.run_id,
                "event": event,
                "stage": stage,
                "status": status,
                "current": current,
                "total": total,
                "overall_percent": round(computed_percent, 3),
                "message": message,
                "data": data or {},
            }
            try:
                self.sink.write(record)
            except ProgressWriteError:
                raise
            except Exception as exc:  # noqa: BLE001 - normalize sink failures
                raise ProgressWriteError(
                    f"Unable to persist progress event {event}: {exc}"
                ) from exc
            self._sequence = sequence
            self._last_overall_percent = computed_percent
            return record

    @staticmethod
    def _stage_percent(
        stage: str,
        current: int,
        total: int,
        status: str,
    ) -> float:
        boundary = STAGE_BOUNDARIES.get(stage)
        if boundary is None:
            return 0.0
        start, end = boundary
        if total == 0:
            if status in {"completed", "completed_with_errors", "failed", "skipped"}:
                return end
            return start
        fraction = min(max(current / total, 0.0), 1.0)
        return start + (end - start) * fraction
