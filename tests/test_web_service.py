from __future__ import annotations

import base64
import io
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import BinaryIO

from PIL import Image

from object_memory.assets import MemoryPaths
from object_memory.memory_store import MemoryStore
from object_memory.web_service import (
    ExperimentBusyError,
    ExperimentManager,
    InputValidationError,
    SafePathError,
    WebSettings,
    _basic_authorized,
    create_app,
    deterministic_result_summary,
    input_listing_payload,
    list_input_images,
    read_memory_snapshot,
    resolve_audit_json,
    resolve_input_file,
    resolve_memory_image,
    save_input_uploads,
)
from scripts.run_object_memory_web import is_loopback_host


class FakeUpload:
    def __init__(self, filename: str, payload: bytes) -> None:
        self.filename = filename
        self.file: BinaryIO = io.BytesIO(payload)


def png_bytes(color: tuple[int, int, int] = (20, 40, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_settings(root: Path, *, password: str | None = None) -> WebSettings:
    project = root / "project"
    for relative in ("data/input", "environment", "temp", "web_static", "scripts"):
        (project / relative).mkdir(parents=True, exist_ok=True)
    (project / "scripts" / "run_object_memory.py").write_text(
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    return WebSettings(
        project_root=project,
        input_root=project / "data" / "input",
        memory_root=project / "data" / "memory",
        database_filename="memory.sqlite",
        report_path=project / "environment" / "run_report.json",
        run_state_root=project / "temp" / "web_runs",
        static_root=project / "web_static",
        python_executable=Path(os.sys.executable),
        basic_password=password,
    ).resolved()


class InputFileTests(unittest.TestCase):
    def test_upload_list_and_delete_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "input"
            uploaded = save_input_uploads(
                root,
                [FakeUpload("scene.png", png_bytes())],  # type: ignore[list-item]
                max_bytes=1024 * 1024,
                max_pixels=1000,
            )

            self.assertEqual(uploaded, ["scene.png"])
            items = list_input_images(root)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["path"], "scene.png")
            self.assertEqual(items[0]["width"], 8)
            self.assertEqual(len(items[0]["sha256"]), 64)
            self.assertEqual(resolve_input_file(root, "scene.png"), root / "scene.png")

    def test_input_payload_marks_content_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "input"
            root.mkdir()
            payload = png_bytes()
            (root / "first.png").write_bytes(payload)
            (root / "second.png").write_bytes(payload)

            listing = input_listing_payload(root, locked=False)

        self.assertEqual(listing["total"], 2)
        self.assertEqual(listing["unique"], 1)
        self.assertEqual(listing["duplicates"], 1)
        self.assertFalse(listing["items"][0]["is_duplicate"])
        self.assertEqual(listing["items"][1]["duplicate_of"], "first.png")

    def test_upload_rejects_invalid_content_and_rolls_back_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "input"

            with self.assertRaises(InputValidationError):
                save_input_uploads(
                    root,
                    [
                        FakeUpload("good.png", png_bytes()),
                        FakeUpload("bad.png", b"not an image"),
                    ],  # type: ignore[list-item]
                    max_bytes=1024 * 1024,
                    max_pixels=1000,
                )

            self.assertFalse((root / "good.png").exists())
            self.assertFalse((root / "bad.png").exists())
            self.assertEqual(list(root.glob(".upload-*.tmp")), [])

    def test_upload_rejects_unsafe_names_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "input"
            root.mkdir()
            (root / "exists.png").write_bytes(png_bytes())

            for filename in ("../escape.png", "folder/image.png", "bad.txt"):
                with self.subTest(filename=filename):
                    with self.assertRaises(InputValidationError):
                        save_input_uploads(
                            root,
                            [
                                FakeUpload(filename, png_bytes())
                            ],  # type: ignore[list-item]
                            max_bytes=1024 * 1024,
                            max_pixels=1000,
                        )
            with self.assertRaises(FileExistsError):
                save_input_uploads(
                    root,
                    [FakeUpload("exists.png", png_bytes())],  # type: ignore[list-item]
                    max_bytes=1024 * 1024,
                    max_pixels=1000,
                )


class AssetBoundaryTests(unittest.TestCase):
    def test_assets_are_limited_by_collection_and_type(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            memory = root / "memory"
            image = memory / "objects" / "obj_1" / "observations" / "obs_1" / "crop.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(png_bytes())
            audit = memory / "run_reports" / "run_1.json"
            audit.parent.mkdir(parents=True)
            audit.write_text("{}", encoding="utf-8")

            self.assertEqual(
                resolve_memory_image(
                    memory,
                    "objects/obj_1/observations/obs_1/crop.png",
                ),
                image,
            )
            self.assertEqual(
                resolve_audit_json(memory, "run_reports/run_1.json"),
                audit,
            )
            for path in (
                "../memory.sqlite",
                "memory.sqlite",
                "raw_responses/run_1/response.png",
                "run_reports/run_1.json",
            ):
                with self.subTest(path=path):
                    with self.assertRaises(SafePathError):
                        resolve_memory_image(memory, path)

    def test_input_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_root = root / "input"
            input_root.mkdir()
            outside = root / "outside.png"
            outside.write_bytes(png_bytes())
            link = input_root / "link.png"
            try:
                link.symlink_to(outside)
            except OSError:
                self.skipTest("Symbolic links are unavailable")

            with self.assertRaises(SafePathError):
                resolve_input_file(input_root, "link.png")


class MemoryReadTests(unittest.TestCase):
    def test_missing_database_returns_uninitialized_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            snapshot = read_memory_snapshot(Path(temporary_directory))

        self.assertFalse(snapshot["initialized"])
        self.assertEqual(snapshot["objects"], [])
        self.assertEqual(snapshot["candidates"], [])

    def test_object_timeline_and_candidate_lineage_are_joined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = MemoryPaths(Path(temporary_directory) / "memory")
            MemoryStore(paths).initialize()
            now = "2026-08-03T00:00:00+00:00"
            with sqlite3.connect(paths.database) as connection:
                connection.execute(
                    """
                    INSERT INTO runs (
                        id, status, started_at, completed_at, config_digest,
                        sam_model_id, qwen_model_id, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("run_1", "completed", now, now, "a" * 64, "sam", "qwen", None),
                )
                connection.execute(
                    """
                    INSERT INTO source_images (
                        id, run_id, sha256, relative_path, width, height, status,
                        error_message, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "src_1",
                        "run_1",
                        "b" * 64,
                        "sources/source.png",
                        8,
                        6,
                        "completed",
                        None,
                        now,
                        now,
                    ),
                )
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
                        "prop_1",
                        "src_1",
                        "raw_1",
                        "water bottle",
                        0.9,
                        0.0,
                        0.0,
                        7.0,
                        5.0,
                        30,
                        0.625,
                        "proposals/run_1/prop_1/mask.png",
                        "proposals/run_1/prop_1/crop.png",
                        "proposals/run_1/prop_1/overlay.jpg",
                        "decided",
                        None,
                        None,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO objects (
                        id, coarse_category, fine_category, material_json,
                        color_json, shape, description, annotation_confidence,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "obj_1",
                        "容器",
                        "水瓶",
                        '["塑料"]',
                        '["白色"]',
                        "圆柱形",
                        "白色水瓶",
                        0.95,
                        "active",
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO observations (
                        id, object_id, proposal_id, source_image_id, crop_path,
                        mask_path, overlay_path, description, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "obs_1",
                        "obj_1",
                        "prop_1",
                        "src_1",
                        "objects/obj_1/observations/obs_1/crop.png",
                        "objects/obj_1/observations/obs_1/mask.png",
                        "objects/obj_1/observations/obs_1/overlay.jpg",
                        "正面视角",
                        now,
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
                        "dec_1",
                        "prop_1",
                        "new",
                        None,
                        0.95,
                        "new_object",
                        "新物体",
                        "v1",
                        "raw_responses/run_1/src_1/response.json",
                        1,
                        now,
                    ),
                )
                connection.commit()

            snapshot = read_memory_snapshot(paths.root)

        self.assertTrue(snapshot["initialized"])
        self.assertEqual(snapshot["counts"]["objects"], 1)
        self.assertEqual(snapshot["objects"][0]["material"], ["塑料"])
        self.assertEqual(
            snapshot["objects"][0]["observations"][0]["source_path"],
            "sources/source.png",
        )
        self.assertEqual(snapshot["candidates"][0]["decision"], "new")
        self.assertEqual(snapshot["candidates"][0]["observation_object_id"], "obj_1")
        self.assertEqual(snapshot["runs"][0]["observation_count"], 1)


class ResultSummaryTests(unittest.TestCase):
    def test_summary_is_derived_only_from_report_counters(self) -> None:
        report = {
            "status": "passed",
            "run": {
                "run_id": "run_1",
                "source_counts": {
                    "pending": 0,
                    "processing": 0,
                    "completed": 2,
                    "failed": 0,
                },
                "proposal_counts": {"decided": 2, "filtered": 1},
                "decision_counts": {
                    "new": 1,
                    "existing": 1,
                    "ignored": 0,
                    "uncertain": 0,
                },
                "duplicate_sources_skipped": 1,
                "observations_added": 2,
                "active_objects_total": 1,
            },
            "images": [
                {
                    "scene_guidance": {"target_count": 2},
                    "sam": {
                        "prompt_detection_counts": {
                            "water bottle": 1,
                            "computer mouse": 0,
                        },
                        "zero_candidate_prompts": ["computer mouse"],
                        "above_confidence_threshold_candidates": 2,
                        "kept": 1,
                        "filtered": 1,
                    },
                    "error": None,
                }
            ],
            "external_errors": [],
        }

        summary = deterministic_result_summary(
            report,
            {"elapsed_seconds": 12.5},
        )

        self.assertEqual(summary["scene_targets"], 2)
        self.assertEqual(summary["sam_prompts"], 2)
        self.assertEqual(summary["sam_zero_candidate_prompts"], 1)
        self.assertEqual(summary["decision_counts"]["existing"], 1)
        self.assertEqual(summary["elapsed_seconds"], 12.5)
        self.assertTrue(summary["manual_review_required"])

    def test_summary_merges_all_top_level_report_errors(self) -> None:
        report = {
            "status": "failed",
            "error": {"type": "RuntimeError", "message": "pipeline failed"},
            "progress_error": {
                "type": "OSError",
                "message": "progress file failed",
            },
            "external_errors": ["scene guidance failed"],
            "images": [{"error": "image registration failed"}],
        }

        summary = deterministic_result_summary(report)

        self.assertEqual(summary["error_count"], 4)
        self.assertEqual(
            [error["source"] for error in summary["errors"]],
            [
                "report.error",
                "report.progress_error",
                "external_errors",
                "images[0].error",
            ],
        )
        self.assertEqual(summary["errors"][0]["type"], "RuntimeError")
        self.assertEqual(summary["errors"][1]["message"], "progress file failed")

    def test_summary_deduplicates_candidate_errors_repeated_by_decisions(self) -> None:
        repeated = "MllmOutputError: candidate response was invalid"
        report = {
            "status": "completed_with_errors",
            "images": [
                {
                    "candidate_reasoning": {"errors": [repeated]},
                    "decisions": [
                        {"errors": [repeated]},
                        {"errors": [repeated]},
                    ],
                }
            ],
            "external_errors": [],
        }

        summary = deterministic_result_summary(report)

        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["errors"][0]["message"], repeated)
        self.assertEqual(
            summary["errors"][0]["source"],
            "images[0].candidate_reasoning.errors",
        )


class RunStateTests(unittest.TestCase):
    def test_current_endpoint_payload_filters_events_by_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = test_settings(Path(temporary_directory))
            run_dir = settings.run_state_root / "web_run_20260803T000000000000Z"
            run_dir.mkdir(parents=True)
            state = {
                "web_run_id": run_dir.name,
                "run_id": "run_1",
                "status": "completed",
                "stage": "run",
                "stage_status": "completed",
                "started_at_utc": "2026-08-03T00:00:00+00:00",
                "updated_at_utc": "2026-08-03T00:00:01+00:00",
                "completed_at_utc": "2026-08-03T00:00:01+00:00",
                "elapsed_seconds": 1.0,
                "overall_percent": 100.0,
                "current": 1,
                "total": 1,
                "message": "done",
                "last_sequence": 2,
                "pid": None,
            }
            (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
            events = [
                {"sequence": 1, "event": "one"},
                {"sequence": 2, "event": "two"},
            ]
            (run_dir / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )

            payload = ExperimentManager(settings).current(after_sequence=1)

        self.assertFalse(payload["active"])
        self.assertEqual([event["sequence"] for event in payload["events"]], [2])

    def test_persisted_live_pid_locks_input_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = test_settings(Path(temporary_directory))
            run_dir = settings.run_state_root / "web_run_20260803T000000000000Z"
            run_dir.mkdir(parents=True)
            state = {
                "web_run_id": run_dir.name,
                "run_id": "run_1",
                "status": "running",
                "stage": "sam3",
                "started_at_utc": "2026-08-03T00:00:00+00:00",
                "last_sequence": 0,
                "elapsed_seconds": 0.0,
                "overall_percent": 35.0,
                "message": "running",
                "pid": os.getpid(),
            }
            (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
            (run_dir / "events.jsonl").touch()
            manager = ExperimentManager(settings)

            with self.assertRaises(ExperimentBusyError):
                with manager.input_mutation():
                    pass

    def test_recovery_does_not_treat_pipeline_completion_as_cli_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = test_settings(Path(temporary_directory))
            run_dir = settings.run_state_root / "web_run_20260803T000000000000Z"
            run_dir.mkdir(parents=True)
            state = {
                "web_run_id": run_dir.name,
                "status": "running",
                "stage": "run",
                "stage_status": "running",
                "started_at_utc": "2026-08-03T00:00:00+00:00",
                "last_sequence": 0,
                "elapsed_seconds": 0.0,
                "overall_percent": 95.0,
                "message": "running",
                "pid": -1,
            }
            (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
            event = {
                "sequence": 1,
                "event": "run_completed",
                "stage": "run",
                "status": "passed",
                "timestamp_utc": "2026-08-03T00:00:01+00:00",
                "elapsed_seconds": 1.0,
                "overall_percent": 100.0,
            }
            (run_dir / "events.jsonl").write_text(
                json.dumps(event) + "\n",
                encoding="utf-8",
            )

            recovered = ExperimentManager(settings).current()["state"]

        self.assertEqual(recovered["last_event"], "run_completed")
        self.assertEqual(recovered["status"], "failed")
        self.assertIsNone(recovered["result_status"])

    def test_recovery_accepts_cli_completion_as_official_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = test_settings(Path(temporary_directory))
            run_dir = settings.run_state_root / "web_run_20260803T000000000000Z"
            run_dir.mkdir(parents=True)
            state = {
                "web_run_id": run_dir.name,
                "status": "running",
                "stage": "cli",
                "stage_status": "running",
                "started_at_utc": "2026-08-03T00:00:00+00:00",
                "last_sequence": 0,
                "elapsed_seconds": 0.0,
                "overall_percent": 95.0,
                "message": "running",
                "pid": -1,
            }
            (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
            event = {
                "sequence": 1,
                "event": "cli_completed",
                "stage": "cli",
                "status": "passed",
                "timestamp_utc": "2026-08-03T00:00:01+00:00",
                "elapsed_seconds": 1.0,
                "overall_percent": 100.0,
            }
            (run_dir / "events.jsonl").write_text(
                json.dumps(event) + "\n",
                encoding="utf-8",
            )

            recovered = ExperimentManager(settings).current()["state"]

        self.assertEqual(recovered["last_event"], "cli_completed")
        self.assertEqual(recovered["status"], "completed")
        self.assertEqual(recovered["result_status"], "passed")

    def test_unexpected_nonzero_exit_captures_bounded_process_log_tail(self) -> None:
        class FinishedProcess:
            def poll(self) -> int:
                return 17

            def wait(self) -> int:
                return 17

        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = test_settings(Path(temporary_directory))
            run_dir = settings.run_state_root / "web_run_20260803T000000000000Z"
            run_dir.mkdir(parents=True)
            state = {
                "web_run_id": run_dir.name,
                "status": "running",
                "stage": "startup",
                "stage_status": "running",
                "started_at_utc": "2026-08-03T00:00:00+00:00",
                "last_sequence": 0,
                "elapsed_seconds": 0.0,
                "overall_percent": 0.0,
                "message": "running",
                "pid": 12345,
                "report_mtime_before_ns": None,
            }
            (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
            marker = "final subprocess failure"
            (run_dir / "process.log").write_bytes(
                (b"x" * (20 * 1024)) + b"\xff\n" + marker.encode("utf-8")
            )
            manager = ExperimentManager(settings)

            manager._watch_process(run_dir, FinishedProcess())  # type: ignore[arg-type]
            recovered = manager.current()["state"]

        self.assertEqual(recovered["status"], "failed")
        self.assertEqual(recovered["exit_code"], 17)
        self.assertEqual(recovered["process_error"]["exit_code"], 17)
        self.assertIn(marker, recovered["process_error"]["log_tail"])
        self.assertLessEqual(
            len(recovered["process_error"]["log_tail"]),
            16 * 1024,
        )

    def test_cli_failed_without_report_still_captures_process_log(self) -> None:
        class FinishedProcess:
            def poll(self) -> int:
                return 1

            def wait(self) -> int:
                return 1

        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = test_settings(Path(temporary_directory))
            run_dir = settings.run_state_root / "web_run_20260803T000000000000Z"
            run_dir.mkdir(parents=True)
            state = {
                "web_run_id": run_dir.name,
                "status": "running",
                "stage": "cli",
                "stage_status": "running",
                "started_at_utc": "2026-08-03T00:00:00+00:00",
                "last_sequence": 0,
                "elapsed_seconds": 0.0,
                "overall_percent": 40.0,
                "message": "running",
                "pid": 12345,
                "report_mtime_before_ns": None,
            }
            (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
            event = {
                "sequence": 1,
                "event": "cli_failed",
                "stage": "cli",
                "status": "failed",
                "timestamp_utc": "2026-08-03T00:00:01+00:00",
                "elapsed_seconds": 1.0,
                "overall_percent": 40.0,
            }
            (run_dir / "events.jsonl").write_text(
                json.dumps(event) + "\n",
                encoding="utf-8",
            )
            (run_dir / "process.log").write_text(
                "traceback from cli failure",
                encoding="utf-8",
            )
            manager = ExperimentManager(settings)

            manager._watch_process(run_dir, FinishedProcess())  # type: ignore[arg-type]
            recovered = manager.current()["state"]

        self.assertEqual(recovered["last_event"], "cli_failed")
        self.assertEqual(recovered["process_error"]["exit_code"], 1)
        self.assertIn("traceback", recovered["process_error"]["log_tail"])

    def test_restart_recovery_without_terminal_evidence_captures_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = test_settings(Path(temporary_directory))
            run_dir = settings.run_state_root / "web_run_20260803T000000000000Z"
            run_dir.mkdir(parents=True)
            state = {
                "web_run_id": run_dir.name,
                "status": "running",
                "stage": "sam3",
                "stage_status": "running",
                "started_at_utc": "2026-08-03T00:00:00+00:00",
                "last_sequence": 0,
                "elapsed_seconds": 0.0,
                "overall_percent": 35.0,
                "message": "running",
                "pid": -1,
                "report_mtime_before_ns": None,
            }
            (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
            (run_dir / "process.log").write_text(
                "process vanished after web service restart",
                encoding="utf-8",
            )

            recovered = ExperimentManager(settings).current()["state"]

        self.assertEqual(recovered["status"], "failed")
        self.assertIsNone(recovered["exit_code"])
        self.assertEqual(
            recovered["process_error"]["kind"],
            "unexpected_recovery_exit",
        )
        self.assertIsNone(recovered["process_error"]["exit_code"])
        self.assertIn("process vanished", recovered["process_error"]["log_tail"])

    def test_subprocess_command_is_fixed_and_uses_progress_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = test_settings(Path(temporary_directory))
            manager = ExperimentManager(settings)
            progress = settings.run_state_root / "web_run_x" / "events.jsonl"

            command = manager._command(progress)

        self.assertIn("--progress-file", command)
        self.assertIn("--validate-demo", command)
        self.assertIn(str(settings.input_root), command)
        self.assertIn(str(settings.memory_root), command)
        self.assertNotIn("--allow-network", command)


class AppContractTests(unittest.TestCase):
    def test_route_contract_and_optional_basic_auth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = test_settings(
                Path(temporary_directory),
                password="correct horse battery staple",
            )
            app = create_app(settings)

        route_paths = {route.path for route in app.routes}
        self.assertTrue(
            {
                "/",
                "/static",
                "/api/inputs",
                "/api/runs",
                "/api/runs/current",
                "/api/results",
                "/api/memory",
                "/api/input-asset",
                "/api/memory-asset",
                "/api/audit-json",
            }.issubset(route_paths)
        )
        token = base64.b64encode(
            b"object-memory:correct horse battery staple"
        ).decode("ascii")
        self.assertTrue(
            _basic_authorized(
                f"Basic {token}",
                expected_username="object-memory",
                expected_password="correct horse battery staple",
            )
        )
        self.assertFalse(
            _basic_authorized(
                "Basic invalid",
                expected_username="object-memory",
                expected_password="correct horse battery staple",
            )
        )

    def test_basic_auth_supports_non_ascii_credentials(self) -> None:
        username = "实验员"
        password = "安全密码-对象记忆"
        token = base64.b64encode(
            f"{username}:{password}".encode("utf-8")
        ).decode("ascii")

        self.assertTrue(
            _basic_authorized(
                f"Basic {token}",
                expected_username=username,
                expected_password=password,
            )
        )
        self.assertFalse(
            _basic_authorized(
                f"Basic {token}",
                expected_username=username,
                expected_password=f"{password}-wrong",
            )
        )

    def test_loopback_detection_is_conservative(self) -> None:
        self.assertTrue(is_loopback_host("127.0.0.1"))
        self.assertTrue(is_loopback_host("::1"))
        self.assertTrue(is_loopback_host("localhost"))
        self.assertFalse(is_loopback_host("0.0.0.0"))
        self.assertFalse(is_loopback_host("example.test"))


if __name__ == "__main__":
    unittest.main()
