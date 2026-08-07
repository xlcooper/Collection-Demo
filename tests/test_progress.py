"""Tests for durable progress records and current failure-report schema."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from object_memory.progress import JsonlProgressWriter, ProgressReporter, ProgressWriteError
from scripts.run_object_memory import failure_report


REQUIRED_EVENT_FIELDS = {
    "sequence",
    "timestamp_utc",
    "elapsed_seconds",
    "run_id",
    "event",
    "stage",
    "status",
    "current",
    "total",
    "overall_percent",
    "message",
    "data",
}


class FailingSink:
    def write(self, record: dict[str, object]) -> None:
        raise OSError("synthetic progress write failure")


class ProgressReporterTests(unittest.TestCase):
    def test_jsonl_records_have_required_fields_and_are_immediately_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "progress.jsonl"
            reporter = ProgressReporter(JsonlProgressWriter(path), run_id="run_1")
            reporter.emit(
                event="visual_identity_started",
                stage="candidate_reasoning",
                status="running",
                current=0,
                total=2,
                overall_percent=60.0,
                message="DINOv3 fingerprinting",
                data={"source_id": "src_1"},
            )
            record = json.loads(path.read_text(encoding="utf-8").strip())
        self.assertTrue(REQUIRED_EVENT_FIELDS.issubset(record))
        self.assertEqual(record["event"], "visual_identity_started")

    def test_sink_failure_is_explicit(self) -> None:
        reporter = ProgressReporter(FailingSink(), run_id="run_1")
        with self.assertRaises(ProgressWriteError):
            reporter.emit(
                event="run_started",
                stage="run",
                status="running",
                current=0,
                total=1,
                message="start",
                data={},
            )

    def test_failure_report_uses_single_pass_schema(self) -> None:
        report = failure_report(RuntimeError("boom"), run_id="run_1")
        self.assertEqual(report["schema_version"], 7)
        self.assertEqual(report["run"]["run_id"], "run_1")


if __name__ == "__main__":
    unittest.main()
