"""Regression tests for configuration, schemas, paths, and SQLite v3."""

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
from object_memory.config import AppConfig, DEFAULT_CONFIG_PATH, config_digest, load_config
from object_memory.memory_store import (
    CORE_TABLES,
    SCHEMA_VERSION,
    MemoryStore,
    MemoryStoreError,
)
from object_memory.schemas import (
    ClusterReview,
    ClusterVerdict,
    IdentityHypothesis,
    ObjectSummary,
    PartAppearance,
)


class ConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        self.assertEqual(config.schema_version, 5)
        self.assertEqual(config.sam3_pipeline.points_per_side, 16)
        self.assertEqual(config.mllm_pipeline.max_clusters_per_batch, 8)
        self.assertEqual(
            config.mllm_pipeline.prompt_version,
            "object-memory-cluster-review-v1",
        )
        self.assertEqual(
            config.models.dinov3_model_id,
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
        )
        self.assertEqual(config.visual_fingerprint.input_size, 512)
        self.assertAlmostEqual(
            config.visual_fingerprint.global_weight
            + config.visual_fingerprint.local_weight,
            1.0,
        )
        self.assertEqual(config.sam3_pipeline.overlay_alpha, 0.0)
        self.assertEqual(len(config_digest(config)), 64)

    def test_cluster_representative_count_has_a_small_upper_bound(self) -> None:
        payload = load_config(DEFAULT_CONFIG_PATH).model_dump(mode="python")
        payload["visual_fingerprint"]["max_cluster_representatives"] = 9
        with self.assertRaises(ValidationError):
            AppConfig.model_validate(payload)

    def test_visual_weights_must_sum_to_one(self) -> None:
        payload = load_config(DEFAULT_CONFIG_PATH).model_dump(mode="python")
        payload["visual_fingerprint"]["global_weight"] = 0.8
        with self.assertRaises(ValidationError):
            AppConfig.model_validate(payload)


class SchemaTests(unittest.TestCase):
    def test_part_aware_summary_and_existing_cluster_validate(self) -> None:
        summary = ObjectSummary(
            object_name_zh="人体工学鼠标",
            coarse_category="电子设备",
            fine_category="鼠标",
            stable_description="银灰色非对称鼠标",
            stable_identity_features=["非左右对称"],
            brand_or_markings=[],
            part_appearance=[],
            summary_confidence=0.9,
        )
        review = ClusterReview(
            cluster_id="clu_mouse",
            verdict=ClusterVerdict.OBJECT,
            identity_hypothesis=IdentityHypothesis.EXISTING,
            matched_object_id="obj_1",
            short_reason="轮廓一致",
            object_summary=summary,
        )
        self.assertEqual(review.matched_object_id, "obj_1")

    def test_existing_hypothesis_requires_object_id(self) -> None:
        summary = ObjectSummary(
            object_name_zh="鼠标",
            coarse_category="设备",
            fine_category="鼠标",
            stable_description="鼠标",
            summary_confidence=0.5,
        )
        with self.assertRaises(ValidationError):
            ClusterReview(
                cluster_id="clu_mouse",
                verdict=ClusterVerdict.OBJECT,
                identity_hypothesis=IdentityHypothesis.EXISTING,
                short_reason="不完整",
                object_summary=summary,
            )

    def test_object_summary_rejects_duplicate_part_labels(self) -> None:
        with self.assertRaises(ValidationError):
            ObjectSummary(
                object_name_zh="鼠标",
                coarse_category="设备",
                fine_category="鼠标",
                stable_description="银灰色鼠标",
                part_appearance=[
                    PartAppearance(part="外壳", color=["银灰色"]),
                    PartAppearance(part="外壳", color=["黑色"]),
                ],
                summary_confidence=0.8,
            )


class MemoryPathTests(unittest.TestCase):
    def test_relative_asset_round_trip_and_traversal_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = MemoryPaths(Path(temporary_directory))
            asset = paths.proposal_dir("run_1", "prop_1") / "fingerprint.npz"
            self.assertEqual(paths.resolve_asset(paths.relative_asset(asset)), asset.resolve())
            with self.assertRaises(ValueError):
                paths.resolve_asset("../outside.npz")


class MemoryStoreTests(unittest.TestCase):
    def test_initialize_is_repeatable_and_empty_status_is_queryable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = MemoryPaths(Path(temporary_directory) / "memory")
            store = MemoryStore(paths)
            first = store.initialize()
            second = store.initialize()
            self.assertEqual(first.schema_version, SCHEMA_VERSION)
            self.assertEqual(second.schema_version, SCHEMA_VERSION)
            self.assertEqual(set(second.counts), set(CORE_TABLES))
            self.assertTrue(all(count == 0 for count in second.counts.values()))

    def test_cli_initializes_and_reports_empty_store(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            memory_root = str(Path(temporary_directory) / "memory")
            output = StringIO()
            with redirect_stdout(output):
                init_code = cli_main(["init", "--memory-root", memory_root, "--json"])
                status_code = cli_main(["status", "--memory-root", memory_root, "--json"])
            self.assertEqual(init_code, 0)
            self.assertEqual(status_code, 0)
            self.assertIn('"status": "ready"', output.getvalue())

    def test_legacy_schema_is_not_migrated_or_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = MemoryPaths(Path(temporary_directory) / "memory")
            paths.root.mkdir(parents=True)
            with sqlite3.connect(paths.database) as connection:
                connection.execute("CREATE TABLE objects (id TEXT PRIMARY KEY)")
                connection.execute("PRAGMA user_version = 2")
            with self.assertRaises(MemoryStoreError):
                MemoryStore(paths).initialize()
            with sqlite3.connect(paths.database) as connection:
                self.assertEqual(
                    int(connection.execute("PRAGMA user_version").fetchone()[0]),
                    2,
                )
            self.assertFalse(paths.sources.exists())


if __name__ == "__main__":
    unittest.main()
