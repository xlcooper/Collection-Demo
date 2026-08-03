"""Deterministic progress-event tests that never load real models."""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch

import numpy as np
from PIL import Image

from object_memory.assets import MemoryPaths
from object_memory.config import DEFAULT_CONFIG_PATH, load_config
from object_memory.mllm_adapter import MllmPrediction
from object_memory.pipeline import ObjectMemoryPipeline
from object_memory.progress import (
    JsonlProgressWriter,
    ProgressReporter,
    ProgressWriteError,
)
from object_memory.sam3_adapter import RawSamCandidate, Sam3Prediction
from scripts.run_object_memory import main as run_object_memory_main


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


class RecordingSink:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write(self, record: dict[str, Any]) -> None:
        self.records.append(record)


class FailingSink:
    def write(self, record: dict[str, Any]) -> None:
        raise OSError("synthetic progress write failure")


class FailOnEventSink:
    def __init__(self, event: str) -> None:
        self.event = event
        self.records: list[dict[str, Any]] = []

    def write(self, record: dict[str, Any]) -> None:
        if record["event"] == self.event:
            raise OSError(f"synthetic failure at {self.event}")
        self.records.append(record)


class FakeSamRuntime:
    model_load_seconds = 0.01

    def load(self) -> None:
        return None

    def predict(
        self,
        image: Image.Image,
        prompts: Sequence[str],
    ) -> Sam3Prediction:
        mask = np.zeros((image.height, image.width), dtype=bool)
        mask[4:20, 4:12] = True
        prompt = prompts[0]
        candidates = (
            RawSamCandidate(
                raw_candidate_id="candidate-main",
                prompt=prompt,
                score=0.95,
                bbox_xyxy=(4.0, 4.0, 12.0, 20.0),
                mask=mask,
            ),
            RawSamCandidate(
                raw_candidate_id="candidate-duplicate",
                prompt=prompt,
                score=0.90,
                bbox_xyxy=(4.0, 4.0, 12.0, 20.0),
                mask=mask.copy(),
            ),
        )
        return Sam3Prediction(
            candidates=candidates,
            prompt_counts={prompt: 2},
            inference_seconds=0.02,
        )

    @property
    def peak_memory_mib(self) -> float:
        return 10.0

    def close(self) -> None:
        return None


class FakeQwenRuntime:
    model_load_seconds = 0.01
    model_placement = ["0"]
    resolved_snapshot = "fake-snapshot"

    def load(self) -> None:
        return None

    def predict(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> MllmPrediction:
        text = "\n".join(
            item["text"]
            for message in messages
            for item in message["content"]
            if item["type"] == "text"
        )
        if "SCENE_BATCH begins" in text:
            source_ids = list(
                dict.fromkeys(re.findall(r"source_id=(src_[A-Za-z0-9_-]+)", text))
            )
            payload = {
                "images": [
                    {
                        "source_id": source_id,
                        "scene_summary": "桌面上有一个杯子",
                        "targets": [
                            {
                                "target_id": "target_001",
                                "object_name_zh": "杯子",
                                "sam_text_prompt": "coffee cup",
                                "priority": "high",
                                "confidence": 0.95,
                                "selection_reason_code": "manipulable",
                                "selection_short_reason": "独立且可操作",
                            }
                        ],
                        "no_target_reason": None,
                    }
                    for source_id in source_ids
                ]
            }
        else:
            proposal_ids = list(
                dict.fromkeys(
                    re.findall(r"proposal_id=(prop_[A-Za-z0-9_-]+)", text)
                )
            )
            annotation = {
                "coarse_category": "cup",
                "fine_category": "coffee cup",
                "material": ["ceramic"],
                "color": ["white"],
                "shape": "round with handle",
                "description": "white ceramic cup with handle",
                "annotation_confidence": 0.96,
            }
            payload = {
                "candidates": [
                    {
                        "proposal_id": proposal_id,
                        "validity": "valid",
                        "validity_confidence": 0.96,
                        "validity_reason_code": "valid_candidate",
                        "validity_short_reason": "complete independent object",
                        "temporary_annotation": annotation,
                        "decision": "new",
                        "matched_object_id": None,
                        "confidence": 0.96,
                        "reason_code": "new_object",
                        "short_reason": "no matching object card",
                        "final_annotation": annotation,
                    }
                    for proposal_id in proposal_ids
                ]
            }
        return MllmPrediction(
            raw_text=json.dumps(payload, ensure_ascii=False),
            input_tokens=10,
            generated_tokens=10,
            inference_seconds=0.02,
        )

    @property
    def peak_memory_mib(self) -> float:
        return 20.0

    def close(self) -> None:
        return None


class FailingLoadQwenRuntime(FakeQwenRuntime):
    def load(self) -> None:
        raise RuntimeError("synthetic Qwen load failure")


class ProgressReporterTests(unittest.TestCase):
    def test_jsonl_records_have_required_fields_and_are_immediately_readable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "progress.jsonl"
            reporter = ProgressReporter(
                JsonlProgressWriter(path),
                run_id="run_progress_writer",
            )

            reporter.emit(
                event="input_registration_started",
                stage="input_registration",
                status="running",
                current=0,
                total=2,
                message="registration started",
            )
            first_lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(first_lines), 1)

            reporter.emit(
                event="input_registered",
                stage="input_registration",
                status="completed",
                current=1,
                total=2,
                message="one input registered",
                data={"filename": "one.jpg"},
            )
            records = [json.loads(line) for line in path.read_text().splitlines()]

            self.assertEqual([item["sequence"] for item in records], [1, 2])
            self.assertTrue(all(set(item) == REQUIRED_EVENT_FIELDS for item in records))
            self.assertEqual(records[1]["run_id"], "run_progress_writer")
            self.assertEqual(records[1]["overall_percent"], 5.0)
            self.assertEqual(records[1]["data"], {"filename": "one.jpg"})

    def test_sink_failure_is_explicit(self) -> None:
        reporter = ProgressReporter(FailingSink(), run_id="run_broken_sink")

        with self.assertRaises(ProgressWriteError):
            reporter.emit(
                event="run_started",
                stage="run",
                status="running",
                current=0,
                total=1,
                overall_percent=0.0,
                message="run started",
            )

    def test_pipeline_emits_stage_details_without_changing_report_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "scene.png"
            Image.new("RGB", (24, 24), (255, 255, 255)).save(image_path)
            sink = RecordingSink()
            pipeline = ObjectMemoryPipeline(
                config=load_config(DEFAULT_CONFIG_PATH),
                paths=MemoryPaths(root / "memory"),
                sam_runtime=FakeSamRuntime(),
                mllm_runtime=FakeQwenRuntime(),
                progress=ProgressReporter(sink),
            )

            report = pipeline.run([image_path], run_id="run_progress_pipeline")

            self.assertEqual(report["schema_version"], 6)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(
                [record["sequence"] for record in sink.records],
                list(range(1, len(sink.records) + 1)),
            )
            percentages = [record["overall_percent"] for record in sink.records]
            self.assertEqual(percentages, sorted(percentages))
            self.assertEqual(percentages[-1], 100.0)
            self.assertTrue(
                all(set(record) == REQUIRED_EVENT_FIELDS for record in sink.records)
            )

            by_event = {record["event"]: record for record in sink.records}
            self.assertIn("run_started", by_event)
            self.assertIn("input_registered", by_event)
            self.assertIn("scene_guidance_batch_completed", by_event)
            self.assertIn("sam3_image_completed", by_event)
            self.assertIn("candidate_reasoning_image_completed", by_event)
            self.assertIn("report_completed", by_event)
            self.assertEqual(
                by_event["input_registration_completed"]["overall_percent"],
                10.0,
            )
            self.assertEqual(
                by_event["scene_guidance_completed"]["overall_percent"],
                35.0,
            )
            self.assertEqual(by_event["sam3_completed"]["overall_percent"], 65.0)
            self.assertEqual(
                by_event["candidate_reasoning_completed"]["overall_percent"],
                95.0,
            )
            sam_data = by_event["sam3_image_completed"]["data"]
            self.assertEqual(len(sam_data["kept"]), 1)
            self.assertEqual(len(sam_data["filtered"]), 1)
            self.assertTrue(
                sam_data["filtered"][0]["filter_reason"].startswith(
                    "duplicate_mask:"
                )
            )
            decisions = by_event["candidate_reasoning_image_completed"]["data"][
                "decisions"
            ]
            self.assertEqual(decisions[0]["decision"], "new")
            self.assertEqual(by_event["report_completed"]["overall_percent"], 100.0)

    def test_scene_model_load_failure_remains_visible_per_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "scene.png"
            Image.new("RGB", (24, 24), (255, 255, 255)).save(image_path)
            sink = RecordingSink()
            pipeline = ObjectMemoryPipeline(
                config=load_config(DEFAULT_CONFIG_PATH),
                paths=MemoryPaths(root / "memory"),
                sam_runtime=FakeSamRuntime(),
                mllm_runtime=FailingLoadQwenRuntime(),
                progress=ProgressReporter(sink),
            )

            report = pipeline.run([image_path], run_id="run_qwen_load_failure")

        guidance = report["images"][0]["scene_guidance"]
        self.assertEqual(report["status"], "completed_with_errors")
        self.assertEqual(guidance["qwen_calls"], 0)
        self.assertIn("synthetic Qwen load failure", guidance["errors"][0])
        scene_finished = next(
            record
            for record in sink.records
            if record["event"] == "scene_guidance_completed"
        )
        self.assertEqual(scene_finished["status"], "failed")

    def test_pipeline_closes_run_and_source_after_progress_write_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "scene.png"
            Image.new("RGB", (24, 24), (255, 255, 255)).save(image_path)
            paths = MemoryPaths(root / "memory")
            pipeline = ObjectMemoryPipeline(
                config=load_config(DEFAULT_CONFIG_PATH),
                paths=paths,
                sam_runtime=FakeSamRuntime(),
                mllm_runtime=FakeQwenRuntime(),
                progress=ProgressReporter(FailOnEventSink("input_registered")),
            )

            with self.assertRaisesRegex(
                ProgressWriteError,
                "input_registered",
            ):
                pipeline.run([image_path], run_id="run_progress_interrupted")

            summary = pipeline.store.run_summary("run_progress_interrupted")
            self.assertEqual(summary.status.value, "completed_with_errors")
            self.assertEqual(summary.source_counts["failed"], 1)
            self.assertEqual(summary.source_counts["processing"], 0)
            with sqlite3.connect(paths.database) as connection:
                error_message = connection.execute(
                    "SELECT error_message FROM runs WHERE id = ?",
                    ("run_progress_interrupted",),
                ).fetchone()[0]
            self.assertIn("ProgressWriteError", error_message)
            self.assertIn("input_registered", error_message)

    def test_cleanup_failure_does_not_replace_progress_write_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "scene.png"
            Image.new("RGB", (24, 24), (255, 255, 255)).save(image_path)
            pipeline = ObjectMemoryPipeline(
                config=load_config(DEFAULT_CONFIG_PATH),
                paths=MemoryPaths(root / "memory"),
                sam_runtime=FakeSamRuntime(),
                mllm_runtime=FakeQwenRuntime(),
                progress=ProgressReporter(FailOnEventSink("input_registered")),
            )

            with (
                patch.object(
                    pipeline.loop,
                    "fail_source",
                    side_effect=RuntimeError("synthetic source cleanup failure"),
                ),
                self.assertRaises(ProgressWriteError) as raised,
            ):
                pipeline.run([image_path], run_id="run_cleanup_failure")

            self.assertTrue(
                any(
                    "synthetic source cleanup failure" in note
                    for note in getattr(raised.exception, "__notes__", [])
                )
            )
            self.assertEqual(
                pipeline.store.run_summary("run_cleanup_failure").status.value,
                "completed_with_errors",
            )

    def test_cli_startup_failure_is_written_as_a_terminal_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            progress_path = Path(temporary_directory) / "failure.jsonl"
            argv = [
                "run_object_memory.py",
                "--input-dir",
                temporary_directory,
                "--progress-file",
                str(progress_path),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch(
                    "scripts.run_object_memory.load_config",
                    side_effect=ValueError("synthetic startup failure"),
                ),
                redirect_stdout(StringIO()),
            ):
                exit_code = run_object_memory_main()

            records = [
                json.loads(line)
                for line in progress_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(exit_code, 1)
            self.assertEqual(records[-1]["event"], "cli_failed")
            self.assertEqual(records[-1]["status"], "failed")
            self.assertIsNone(records[-1]["run_id"])
            self.assertEqual(
                records[-1]["data"]["error"]["type"],
                "ValueError",
            )


if __name__ == "__main__":
    unittest.main()
