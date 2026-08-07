"""Deterministic tests for the SAM3 -> DINOv3 -> Qwen cluster loop."""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from object_memory.assets import MemoryPaths
from object_memory.config import DEFAULT_CONFIG_PATH, load_config
from object_memory.dinov3_adapter import FingerprintData
from object_memory.memory_store import MemoryStore
from object_memory.mllm_adapter import MllmPrediction
from object_memory.pipeline import ObjectMemoryPipeline
from object_memory.sam3_adapter import (
    AUTOMATIC_CANDIDATE_SOURCE,
    RawSamCandidate,
    Sam3Prediction,
)


def summary_payload(name: str = "银灰色人体工学鼠标") -> dict[str, Any]:
    return {
        "object_name_zh": name,
        "coarse_category": "电子设备",
        "fine_category": "鼠标",
        "stable_description": "银灰色非对称鼠标，右侧轮廓明显隆起。",
        "stable_identity_features": ["整体非左右对称", "右侧轮廓隆起"],
        "brand_or_markings": [],
        "part_appearance": [
            {"part": "外壳", "color": ["银灰色"], "material": ["塑料"]}
        ],
        "summary_confidence": 0.95,
    }


class FakeQwenRuntime:
    model_load_seconds = 0.1
    model_placement = ["cuda:0"]
    resolved_snapshot = "fake-qwen"

    def __init__(
        self,
        events: list[str],
        *,
        invalid: bool = False,
        ignore: bool = False,
    ) -> None:
        self.events = events
        self.invalid = invalid
        self.ignore = ignore
        self.calls = 0

    def load(self) -> None:
        self.events.append("qwen.load")

    def predict(self, messages: Sequence[dict[str, Any]]) -> MllmPrediction:
        self.events.append("qwen.predict")
        self.calls += 1
        text = "\n".join(
            item["text"]
            for message in messages
            for item in message["content"]
            if item["type"] == "text"
        )
        if self.invalid:
            raw = "{}"
        else:
            cluster_ids = list(dict.fromkeys(re.findall(r"CLUSTER (clu_[a-z0-9]+)", text)))
            reviews = []
            for index, cluster_id in enumerate(cluster_ids, start=1):
                reviews.append(
                    {
                        "cluster_id": cluster_id,
                        "verdict": "ignore" if self.ignore else "object",
                        "identity_hypothesis": (
                            "uncertain" if self.ignore else "new"
                        ),
                        "matched_object_id": None,
                        "short_reason": (
                            "背景区域" if self.ignore else "多视角轮廓一致且为完整物体"
                        ),
                        "object_summary": (
                            None
                            if self.ignore
                            else summary_payload(f"银灰色人体工学鼠标{index}")
                        ),
                    }
                )
            raw = json.dumps({"reviews": reviews}, ensure_ascii=False)
        return MllmPrediction(raw, 100, 80, 0.2)

    @property
    def peak_memory_mib(self) -> float:
        return 1000.0

    def close(self) -> None:
        self.events.append("qwen.close")


class FakeSamRuntime:
    model_load_seconds = 0.1

    def __init__(self, events: list[str], *, two_candidates: bool = False) -> None:
        self.events = events
        self.two_candidates = two_candidates
        self.calls = 0

    def load(self) -> None:
        self.events.append("sam.load")

    def predict(self, image: Image.Image) -> Sam3Prediction:
        self.events.append("sam.predict")
        self.calls += 1
        boxes = [(4, 4, 16, 28)]
        if self.two_candidates:
            boxes.append((18, 4, 29, 28))
        candidates = []
        for index, (x_min, y_min, x_max, y_max) in enumerate(boxes):
            mask = np.zeros((image.height, image.width), dtype=bool)
            mask[y_min:y_max, x_min:x_max] = True
            candidates.append(
                RawSamCandidate(
                    raw_candidate_id=f"grid_point_{index:06d}",
                    prompt=AUTOMATIC_CANDIDATE_SOURCE,
                    score=0.95 - index * 0.01,
                    bbox_xyxy=(x_min, y_min, x_max, y_max),
                    mask=mask,
                )
            )
        return Sam3Prediction(
            candidates=tuple(candidates),
            inference_seconds=0.1,
        )

    @property
    def peak_memory_mib(self) -> float:
        return 1500.0

    def close(self) -> None:
        self.events.append("sam.close")


class FakeDinoRuntime:
    model_load_seconds = 0.1
    model_placement = ["cuda:0"]
    resolved_snapshot = "a" * 40
    feature_layer = "last_hidden_state"
    last_inference_seconds = 0.05

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.extract_count = 0

    def load(self) -> None:
        self.events.append("dino.load")

    def extract(self, **_: Any) -> FingerprintData:
        self.events.append("dino.extract")
        self.extract_count += 1
        return FingerprintData(
            global_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            local_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            local_patch_indices=np.asarray([[0, 0]], dtype=np.int32),
        )

    @property
    def peak_memory_mib(self) -> float:
        return 2000.0

    def close(self) -> None:
        self.events.append("dino.close")


def make_pipeline(
    root: Path,
    *,
    qwen: FakeQwenRuntime,
    sam: FakeSamRuntime,
    dino: FakeDinoRuntime,
) -> ObjectMemoryPipeline:
    return ObjectMemoryPipeline(
        config=load_config(DEFAULT_CONFIG_PATH),
        paths=MemoryPaths(root / "memory"),
        sam_runtime=sam,
        mllm_runtime=qwen,
        dino_runtime=dino,
    )


class PipelineTests(unittest.TestCase):
    def test_cross_image_cluster_creates_one_object_with_two_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.png"
            second = root / "second.png"
            duplicate = root / "duplicate.png"
            Image.new("RGB", (32, 32), (255, 0, 0)).save(first)
            Image.new("RGB", (32, 32), (0, 0, 255)).save(second)
            shutil.copy2(first, duplicate)
            events: list[str] = []
            qwen = FakeQwenRuntime(events)
            sam = FakeSamRuntime(events)
            dino = FakeDinoRuntime(events)
            pipeline = make_pipeline(root, qwen=qwen, sam=sam, dino=dino)
            report = pipeline.run([first, second, duplicate], run_id="run_demo")

            self.assertEqual(report["status"], "passed")
            self.assertEqual(sam.calls, 2)
            self.assertEqual(qwen.calls, 1)
            self.assertEqual(len(report["clusters"]), 1)
            self.assertEqual(report["run"]["decision_counts"]["new"], 1)
            self.assertEqual(report["run"]["decision_counts"]["existing"], 1)
            self.assertEqual(report["run"]["active_objects_total"], 1)
            self.assertEqual(report["run"]["observations_added"], 2)
            self.assertLess(events.index("sam.close"), events.index("dino.load"))
            self.assertLess(events.index("dino.close"), events.index("qwen.load"))
            with sqlite3.connect(pipeline.paths.database) as connection:
                fingerprints = connection.execute(
                    "SELECT fingerprint_json FROM proposals ORDER BY created_at"
                ).fetchall()
            self.assertTrue(all(row[0] for row in fingerprints))
            cards = MemoryStore(pipeline.paths).list_object_cards()
            self.assertIn("轮廓明显隆起", cards[0].summary.stable_description)

    def test_same_source_candidates_never_merge_into_one_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "two.png"
            Image.new("RGB", (32, 32), (100, 100, 100)).save(image)
            events: list[str] = []
            qwen = FakeQwenRuntime(events)
            sam = FakeSamRuntime(events, two_candidates=True)
            dino = FakeDinoRuntime(events)
            report = make_pipeline(root, qwen=qwen, sam=sam, dino=dino).run(
                [image], run_id="run_two"
            )
            self.assertEqual(qwen.calls, 1)
            self.assertEqual(len(report["clusters"]), 2)
            self.assertEqual(report["run"]["decision_counts"]["new"], 2)
            self.assertEqual(report["run"]["active_objects_total"], 2)

    def test_invalid_cluster_review_fails_after_sam_and_dino(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "one.png"
            Image.new("RGB", (32, 32), (10, 20, 30)).save(image)
            events: list[str] = []
            qwen = FakeQwenRuntime(events, invalid=True)
            sam = FakeSamRuntime(events)
            dino = FakeDinoRuntime(events)
            pipeline = make_pipeline(root, qwen=qwen, sam=sam, dino=dino)
            report = pipeline.run([image], run_id="run_invalid")
            self.assertEqual(qwen.calls, 1)
            self.assertIn("sam.predict", events)
            self.assertIn("dino.extract", events)
            self.assertEqual(report["status"], "completed_with_errors")
            raw_response = report["clusters"][0]["raw_response"]
            self.assertTrue(raw_response)
            self.assertTrue(pipeline.paths.resolve_asset(raw_response).is_file())

    def test_qwen_ignore_filters_cluster_without_creating_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "background.png"
            Image.new("RGB", (32, 32), (10, 20, 30)).save(image)
            events: list[str] = []
            qwen = FakeQwenRuntime(events, ignore=True)
            sam = FakeSamRuntime(events)
            dino = FakeDinoRuntime(events)
            report = make_pipeline(root, qwen=qwen, sam=sam, dino=dino).run(
                [image], run_id="run_ignore"
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["cluster_counts"], {"ignored": 1})
            self.assertEqual(report["run"]["active_objects_total"], 0)
            self.assertEqual(report["run"]["proposal_counts"]["filtered"], 1)


if __name__ == "__main__":
    unittest.main()
