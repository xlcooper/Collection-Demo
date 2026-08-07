"""Deterministic tests for the per-image Qwen/SAM3/DINOv3 loop."""

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
from object_memory.sam3_adapter import RawSamCandidate, Sam3Prediction


def summary_payload(description: str) -> dict[str, Any]:
    return {
        "object_name_zh": "银灰色人体工学鼠标",
        "coarse_category": "电子设备",
        "fine_category": "鼠标",
        "stable_description": description,
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

    def __init__(self, events: list[str], *, invalid: bool = False, two_targets: bool = False, no_targets: bool = False) -> None:
        self.events = events
        self.invalid = invalid
        self.two_targets = two_targets
        self.no_targets = no_targets
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
            source_id = re.search(r"source_id=(src_[A-Za-z0-9_-]+)", text).group(1)
            object_ids = re.findall(r'"object_id": "(obj_[^"]+)"', text)
            if self.no_targets:
                targets: list[dict[str, Any]] = []
            else:
                targets = []
                count = 2 if self.two_targets else 1
                for index in range(count):
                    existing = bool(object_ids)
                    x_min, x_max = ((0.125, 0.5) if index == 0 else (0.55, 0.9))
                    targets.append(
                        {
                            "target_id": f"target_{index + 1:03d}",
                            "object_name_zh": "银灰色人体工学鼠标",
                            "sam_text_prompt": "computer mouse",
                            "current_view_facts": {
                                "category": "鼠标",
                                "visible_identity_features": ["右侧轮廓隆起"],
                                "brand_or_markings": [],
                                "part_appearance": [],
                            },
                            "identity_hypothesis": "existing" if existing else "new",
                            "matched_object_id": object_ids[0] if existing else None,
                            "identity_short_reason": "轮廓证据",
                            "proposed_object_summary": summary_payload(
                                "银灰色非对称鼠标，右侧轮廓明显隆起。"
                                if existing
                                else "银灰色非对称鼠标。"
                            ),
                            "temporary_target_anchor": {
                                "x_min": x_min,
                                "y_min": 0.125,
                                "x_max": x_max,
                                "y_max": 0.875,
                            },
                        }
                    )
            raw = json.dumps(
                {
                    "image": {
                        "source_id": source_id,
                        "scene_summary": "桌面上的鼠标",
                        "targets": targets,
                        "no_target_reason": "没有目标" if not targets else None,
                    }
                },
                ensure_ascii=False,
            )
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
        self.received_prompts: list[tuple[str, ...]] = []

    def load(self) -> None:
        self.events.append("sam.load")

    def predict(self, image: Image.Image, prompts: Sequence[str]) -> Sam3Prediction:
        self.events.append("sam.predict")
        self.received_prompts.append(tuple(prompts))
        candidates = []
        boxes = [(4, 4, 16, 28)]
        if self.two_candidates:
            boxes.append((18, 4, 29, 28))
        for index, (x_min, y_min, x_max, y_max) in enumerate(boxes):
            mask = np.zeros((image.height, image.width), dtype=bool)
            mask[y_min:y_max, x_min:x_max] = True
            candidates.append(
                RawSamCandidate(
                    raw_candidate_id=f"candidate_{index}",
                    prompt=prompts[0],
                    score=0.95 - index * 0.01,
                    bbox_xyxy=(x_min, y_min, x_max, y_max),
                    mask=mask,
                )
            )
        return Sam3Prediction(
            candidates=tuple(candidates),
            prompt_counts={prompts[0]: len(candidates)},
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
    def test_each_unique_image_calls_qwen_once_and_iterates_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first, second, duplicate = root / "first.png", root / "second.png", root / "duplicate.png"
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
            self.assertEqual(qwen.calls, 2)
            self.assertEqual(report["models"]["qwen"]["calls"], 2)
            self.assertEqual(
                report["models"]["dinov3"]["result_counts"],
                {"match": 1, "no_match": 1, "ambiguous": 0},
            )
            self.assertFalse(report["strategy"]["second_qwen_stage"])
            self.assertEqual(report["run"]["decision_counts"]["new"], 1)
            self.assertEqual(report["run"]["decision_counts"]["existing"], 1)
            self.assertEqual(report["run"]["active_objects_total"], 1)
            self.assertEqual(report["run"]["observations_added"], 2)
            self.assertTrue(
                all(
                    proposal["status"] == "decided"
                    for image_report in report["images"]
                    for proposal in image_report["kept_proposals"]
                )
            )
            self.assertLess(events.index("dino.load"), events.index("qwen.predict"))
            cards = MemoryStore(pipeline.paths).list_object_cards()
            self.assertIn("明显隆起", cards[0].summary.stable_description)
            with sqlite3.connect(pipeline.paths.database) as connection:
                fingerprints = connection.execute(
                    "SELECT fingerprint_json FROM proposals ORDER BY created_at"
                ).fetchall()
            self.assertTrue(all(row[0] for row in fingerprints))

    def test_duplicate_prompt_targets_share_query_and_pre_image_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "two.png"
            Image.new("RGB", (32, 32), (100, 100, 100)).save(image)
            events: list[str] = []
            qwen = FakeQwenRuntime(events, two_targets=True)
            sam = FakeSamRuntime(events, two_candidates=True)
            dino = FakeDinoRuntime(events)
            report = make_pipeline(root, qwen=qwen, sam=sam, dino=dino).run(
                [image], run_id="run_two"
            )
            self.assertEqual(qwen.calls, 1)
            self.assertEqual(sam.received_prompts, [("computer mouse",)])
            self.assertEqual(len(report["images"][0]["decisions"]), 2)
            self.assertEqual(report["run"]["decision_counts"]["new"], 2)
            self.assertEqual(report["run"]["decision_counts"]["uncertain"], 0)
            self.assertEqual(report["run"]["active_objects_total"], 2)

    def test_invalid_qwen_output_fails_source_after_one_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "one.png"
            Image.new("RGB", (32, 32), (10, 20, 30)).save(image)
            events: list[str] = []
            qwen = FakeQwenRuntime(events, invalid=True)
            sam = FakeSamRuntime(events)
            dino = FakeDinoRuntime(events)
            report = make_pipeline(root, qwen=qwen, sam=sam, dino=dino).run(
                [image], run_id="run_invalid"
            )
            self.assertEqual(qwen.calls, 1)
            self.assertNotIn("sam.predict", events)
            self.assertEqual(report["status"], "completed_with_errors")

    def test_empty_target_response_skips_sam_and_dino(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "empty.png"
            Image.new("RGB", (32, 32), (10, 20, 30)).save(image)
            events: list[str] = []
            qwen = FakeQwenRuntime(events, no_targets=True)
            sam = FakeSamRuntime(events)
            dino = FakeDinoRuntime(events)
            report = make_pipeline(root, qwen=qwen, sam=sam, dino=dino).run(
                [image], run_id="run_empty"
            )
            self.assertEqual(report["status"], "passed")
            self.assertNotIn("sam.predict", events)
            self.assertEqual(dino.extract_count, 0)


if __name__ == "__main__":
    unittest.main()
