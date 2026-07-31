"""Regression tests for configuration, schemas, paths, and storage."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from pydantic import ValidationError

from object_memory.assets import MemoryPaths
from object_memory.cli import main as cli_main
from object_memory.config import (
    AppConfig,
    DEFAULT_CONFIG_PATH,
    config_digest,
    load_config,
)
from object_memory.memory_store import CORE_TABLES, SCHEMA_VERSION, MemoryStore
from object_memory.schemas import (
    BoundingBox,
    Decision,
    DecisionType,
    MemoryObject,
    Observation,
    Proposal,
    Run,
    SourceImage,
)


class ConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        self.assertEqual(config.schema_version, 2)
        self.assertEqual(
            config.models.qwen_model_id,
            "Qwen/Qwen3-VL-8B-Instruct-FP8",
        )
        self.assertEqual(
            config.mllm_pipeline.prompt_version,
            "guided-image-batch-memory-reasoning-v2",
        )
        self.assertEqual(config.mllm_pipeline.scene_batch_size, 4)
        self.assertEqual(
            config.mllm_pipeline.max_new_tokens,
            4096,
        )
        self.assertEqual(
            config.sam3_pipeline.crop_background_color,
            (127, 127, 127),
        )
        self.assertEqual(len(config_digest(config)), 64)

    def test_scene_target_count_cannot_exceed_candidate_capacity(self) -> None:
        payload = load_config(DEFAULT_CONFIG_PATH).model_dump(mode="python")
        payload["mllm_pipeline"]["max_scene_targets_per_image"] = 13
        payload["sam3_pipeline"]["max_candidates_per_image"] = 12

        with self.assertRaises(ValidationError):
            AppConfig.model_validate(payload)


class SchemaTests(unittest.TestCase):
    def test_core_records_validate(self) -> None:
        run = Run(
            config_digest="a" * 64,
            sam_model_id="sam3",
            qwen_model_id="qwen3-vl",
        )
        source = SourceImage(
            run_id=run.id,
            sha256="b" * 64,
            relative_path="sources/example.jpg",
            width=640,
            height=480,
        )
        proposal = Proposal(
            source_image_id=source.id,
            raw_candidate_id="candidate-1",
            score=0.9,
            bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        )
        memory_object = MemoryObject(
            coarse_category="杯子",
            fine_category="马克杯",
            material=["陶瓷"],
            color=["红色"],
            shape="圆柱形",
            description="带白色把手的红色杯子",
            annotation_confidence=0.9,
        )
        observation = Observation(
            object_id=memory_object.id,
            proposal_id=proposal.id,
            source_image_id=source.id,
            crop_path="objects/example/observations/one/crop.png",
            mask_path="objects/example/observations/one/mask.png",
            overlay_path="objects/example/observations/one/overlay.jpg",
            description="正面可见",
        )
        decision = Decision(
            proposal_id=proposal.id,
            decision=DecisionType.EXISTING,
            matched_object_id=memory_object.id,
            confidence=0.88,
            reason_code="visual_instance_match",
            short_reason="外观特征一致",
            prompt_version="identity-v1",
        )
        self.assertEqual(observation.object_id, decision.matched_object_id)

    def test_existing_decision_requires_match(self) -> None:
        with self.assertRaises(ValidationError):
            Decision(
                proposal_id="prop_example",
                decision=DecisionType.EXISTING,
                confidence=0.8,
                reason_code="match",
                short_reason="matched",
                prompt_version="identity-v1",
            )


class MemoryPathTests(unittest.TestCase):
    def test_relative_asset_round_trip_and_traversal_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = MemoryPaths(Path(temporary_directory))
            asset = paths.observation_dir("obj_1", "obs_1") / "crop.png"
            relative = paths.relative_asset(asset)
            self.assertEqual(
                paths.resolve_asset(relative),
                asset.resolve(),
            )
            with self.assertRaises(ValueError):
                paths.resolve_asset("../outside.png")


class MemoryStoreTests(unittest.TestCase):
    def test_initialize_is_repeatable_and_empty_status_is_queryable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = MemoryPaths(Path(temporary_directory) / "memory")
            store = MemoryStore(paths)
            first_status = store.initialize()
            second_status = store.initialize()

            self.assertEqual(first_status.schema_version, SCHEMA_VERSION)
            self.assertEqual(second_status.schema_version, SCHEMA_VERSION)
            self.assertEqual(set(second_status.counts), set(CORE_TABLES))
            self.assertTrue(all(count == 0 for count in second_status.counts.values()))
            self.assertTrue(paths.database.is_file())

    def test_cli_initializes_and_reports_empty_store(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            memory_root = str(Path(temporary_directory) / "memory")
            output = StringIO()
            with redirect_stdout(output):
                init_code = cli_main(
                    ["init", "--memory-root", memory_root, "--json"]
                )
                status_code = cli_main(
                    ["status", "--memory-root", memory_root, "--json"]
                )
            self.assertEqual(init_code, 0)
            self.assertEqual(status_code, 0)
            self.assertIn('"status": "ready"', output.getvalue())
            self.assertIn('"objects": 0', output.getvalue())

    def test_schema_version_one_migrates_to_current(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = MemoryPaths(Path(temporary_directory) / "memory")
            paths.ensure_layout()
            with sqlite3.connect(paths.database) as connection:
                connection.executescript(
                    """
                    CREATE TABLE proposals (
                        id TEXT PRIMARY KEY,
                        source_image_id TEXT NOT NULL,
                        raw_candidate_id TEXT NOT NULL,
                        score REAL NOT NULL,
                        bbox_x_min REAL NOT NULL,
                        bbox_y_min REAL NOT NULL,
                        bbox_x_max REAL NOT NULL,
                        bbox_y_max REAL NOT NULL,
                        mask_path TEXT,
                        crop_path TEXT,
                        overlay_path TEXT,
                        status TEXT NOT NULL,
                        filter_reason TEXT,
                        error_message TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    PRAGMA user_version = 1;
                    """
                )

            status = MemoryStore(paths).initialize()
            with sqlite3.connect(paths.database) as connection:
                proposal_columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(proposals)")
                }
            self.assertEqual(status.schema_version, SCHEMA_VERSION)
            self.assertTrue(
                {"prompt", "mask_area_pixels", "mask_area_ratio"}.issubset(
                    proposal_columns
                )
            )


if __name__ == "__main__":
    unittest.main()
