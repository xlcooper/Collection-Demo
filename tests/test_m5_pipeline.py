"""Deterministic end-to-end orchestration tests for M5."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from object_memory.assets import MemoryPaths
from object_memory.config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from object_memory.memory_store import MemoryStore
from object_memory.mllm_adapter import MllmPrediction
from object_memory.pipeline import ObjectMemoryPipeline
from object_memory.sam3_adapter import RawSamCandidate, Sam3Prediction


def test_config(*, max_error_attempts: int = 2) -> AppConfig:
    payload = load_config(DEFAULT_CONFIG_PATH).model_dump(mode="python")
    payload["mllm_pipeline"]["max_error_attempts"] = max_error_attempts
    return AppConfig.model_validate(payload)


def annotation_payload(*, existing: bool = False) -> dict[str, Any]:
    return {
        "coarse_category": "cup",
        "fine_category": "coffee cup",
        "material": ["ceramic"],
        "color": ["white"],
        "shape": "round with handle",
        "description": (
            "updated cumulative white ceramic cup annotation"
            if existing
            else "white ceramic cup with handle"
        ),
        "annotation_confidence": 0.96,
    }


def batch_candidate(
    proposal_id: str,
    *,
    decision: str,
    object_id: str | None = None,
) -> dict[str, Any]:
    if decision == "ignored":
        return {
            "proposal_id": proposal_id,
            "validity": "ignored",
            "validity_confidence": 0.95,
            "validity_reason_code": "invalid_candidate",
            "validity_short_reason": "not an independent physical object",
            "temporary_annotation": None,
            "decision": "ignored",
            "matched_object_id": None,
            "confidence": 0.95,
            "reason_code": "invalid_candidate",
            "short_reason": "shadow or fragment",
            "final_annotation": None,
        }
    reasons = {
        "new": "new_object",
        "existing": "visual_instance_match",
        "uncertain": "insufficient_evidence",
    }
    return {
        "proposal_id": proposal_id,
        "validity": "valid",
        "validity_confidence": 0.95,
        "validity_reason_code": "valid_candidate",
        "validity_short_reason": "complete independent object",
        "temporary_annotation": annotation_payload(),
        "decision": decision,
        "matched_object_id": object_id,
        "confidence": 0.95,
        "reason_code": reasons[decision],
        "short_reason": "deterministic batch decision",
        "final_annotation": annotation_payload(existing=decision == "existing"),
    }


class FakeSamRuntime:
    def __init__(
        self,
        events: list[str],
        *,
        duplicate_candidate: bool = False,
        second_candidate: bool = False,
    ) -> None:
        self.events = events
        self.duplicate_candidate = duplicate_candidate
        self.second_candidate = second_candidate
        self.model_load_seconds = 0.1
        self._peak_memory_mib = 100.0

    def load(self) -> None:
        self.events.append("sam.load")

    def predict(self, image: Image.Image) -> Sam3Prediction:
        self.events.append("sam.predict")
        y_min = image.height // 4
        y_max = image.height - y_min
        first_x_min = image.width // 8
        first_x_max = image.width // 2 - 1
        first_mask = np.zeros((image.height, image.width), dtype=bool)
        first_mask[y_min:y_max, first_x_min:first_x_max] = True
        candidates = [
            RawSamCandidate(
                raw_candidate_id="candidate-main",
                prompt="automatic_point_grid",
                score=0.95,
                bbox_xyxy=(first_x_min, y_min, first_x_max, y_max),
                mask=first_mask,
            )
        ]
        if self.duplicate_candidate:
            candidates.append(
                RawSamCandidate(
                    raw_candidate_id="candidate-duplicate",
                    prompt="automatic_point_grid",
                    score=0.90,
                    bbox_xyxy=(first_x_min, y_min, first_x_max, y_max),
                    mask=first_mask.copy(),
                )
            )
        if self.second_candidate:
            second_x_min = image.width // 2 + 1
            second_x_max = image.width - image.width // 8
            second_mask = np.zeros((image.height, image.width), dtype=bool)
            second_mask[y_min:y_max, second_x_min:second_x_max] = True
            candidates.append(
                RawSamCandidate(
                    raw_candidate_id="candidate-second",
                    prompt="automatic_point_grid",
                    score=0.94,
                    bbox_xyxy=(second_x_min, y_min, second_x_max, y_max),
                    mask=second_mask,
                )
            )
        return Sam3Prediction(
            candidates=tuple(candidates),
            prompt_counts={"automatic_point_grid": len(candidates)},
            inference_seconds=0.2,
        )

    @property
    def peak_memory_mib(self) -> float:
        return self._peak_memory_mib

    def close(self) -> None:
        self.events.append("sam.close")


class FakeQwenRuntime:
    def __init__(
        self,
        events: list[str],
        *,
        responses: list[str] | None = None,
        existing_decision: str = "existing",
    ) -> None:
        self.events = events
        self.responses = list(responses or [])
        self.existing_decision = existing_decision
        self.model_load_seconds = 0.3
        self.model_placement = ["0"]
        self.resolved_snapshot = "fake-snapshot"
        self._peak_memory_mib = 200.0
        self.call_count = 0

    def load(self) -> None:
        self.events.append("qwen.load")

    def predict(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> MllmPrediction:
        self.events.append("qwen.predict")
        self.call_count += 1
        if self.responses:
            raw_text = self.responses.pop(0)
        else:
            all_text = "\n".join(
                item["text"]
                for message in messages
                for item in message["content"]
                if item["type"] == "text"
            )
            proposal_ids = re.findall(r"proposal_id=(prop_[A-Za-z0-9_-]+)", all_text)
            object_ids = re.findall(r'"object_id": "(obj_[^"]+)"', all_text)
            results: list[dict[str, Any]] = []
            for index, proposal_id in enumerate(proposal_ids):
                if index > 0:
                    results.append(
                        batch_candidate(proposal_id, decision="ignored")
                    )
                elif object_ids:
                    results.append(
                        batch_candidate(
                            proposal_id,
                            decision=self.existing_decision,
                            object_id=(
                                object_ids[0]
                                if self.existing_decision == "existing"
                                else None
                            ),
                        )
                    )
                else:
                    results.append(batch_candidate(proposal_id, decision="new"))
            raw_text = json.dumps({"candidates": results})
        return MllmPrediction(
            raw_text=raw_text,
            input_tokens=100,
            generated_tokens=80,
            inference_seconds=0.4,
        )

    @property
    def peak_memory_mib(self) -> float:
        return self._peak_memory_mib

    def close(self) -> None:
        self.events.append("qwen.close")


class M5PipelineTests(unittest.TestCase):
    def test_batch_runs_models_sequentially_and_updates_memory_card(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.png"
            second = root / "second.png"
            duplicate = root / "duplicate.png"
            Image.new("RGB", (24, 24), (255, 0, 0)).save(first)
            Image.new("RGB", (24, 24), (0, 0, 255)).save(second)
            shutil.copy2(first, duplicate)
            events: list[str] = []
            paths = MemoryPaths(root / "memory")
            qwen = FakeQwenRuntime(events)
            pipeline = ObjectMemoryPipeline(
                config=test_config(),
                paths=paths,
                sam_runtime=FakeSamRuntime(events, duplicate_candidate=True),
                mllm_runtime=qwen,
            )

            report = pipeline.run(
                [first, second, duplicate],
                run_id="run_m5_batch",
            )

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["schema_version"], 2)
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(
                report["core_counts"],
                {
                    "runs": 1,
                    "source_images": 2,
                    "proposals": 4,
                    "objects": 1,
                    "observations": 2,
                    "decisions": 2,
                },
            )
            self.assertEqual(report["run"]["duplicate_sources_skipped"], 1)
            self.assertEqual(report["run"]["proposal_counts"]["filtered"], 2)
            self.assertEqual(report["run"]["decision_counts"]["new"], 1)
            self.assertEqual(report["run"]["decision_counts"]["existing"], 1)
            self.assertEqual(report["models"]["qwen"]["image_batch_calls"], 2)
            self.assertEqual(report["images"][0]["qwen_batch"]["candidate_count"], 1)
            self.assertEqual(report["images"][0]["qwen_batch"]["object_card_count"], 0)
            self.assertEqual(report["images"][1]["qwen_batch"]["object_card_count"], 1)
            self.assertEqual(
                len(report["images"][1]["qwen_batch"]["object_card_ids"]),
                1,
            )
            self.assertEqual(
                report["images"][1]["decisions"][0]["temporary_annotation"][
                    "fine_category"
                ],
                "coffee cup",
            )
            self.assertEqual(
                report["images"][1]["decisions"][0]["final_annotation"][
                    "description"
                ],
                "updated cumulative white ceramic cup annotation",
            )
            cards = MemoryStore(paths).list_object_cards(max_reference_views=2)
            self.assertEqual(
                cards[0].description,
                "updated cumulative white ceramic cup annotation",
            )
            self.assertLess(events.index("sam.close"), events.index("qwen.load"))
            self.assertEqual(qwen.call_count, 2)

    def test_all_candidates_from_one_image_share_one_qwen_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "scene.png"
            Image.new("RGB", (24, 24), (255, 255, 255)).save(image)
            events: list[str] = []
            qwen = FakeQwenRuntime(events)
            pipeline = ObjectMemoryPipeline(
                config=test_config(),
                paths=MemoryPaths(root / "memory"),
                sam_runtime=FakeSamRuntime(events, second_candidate=True),
                mllm_runtime=qwen,
            )

            report = pipeline.run([image], run_id="run_m5_one_image_batch")

            self.assertEqual(report["status"], "passed")
            self.assertEqual(qwen.call_count, 1)
            self.assertEqual(report["images"][0]["qwen_batch"]["candidate_count"], 2)
            self.assertEqual(len(report["images"][0]["decisions"]), 2)
            self.assertEqual(report["run"]["decision_counts"]["new"], 1)
            self.assertEqual(report["run"]["decision_counts"]["ignored"], 1)

    def test_invalid_batch_output_retries_without_duplicate_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "image.png"
            Image.new("RGB", (24, 24), (255, 255, 255)).save(image)
            events: list[str] = []
            qwen = FakeQwenRuntime(events, responses=["not json"])
            paths = MemoryPaths(root / "memory")
            pipeline = ObjectMemoryPipeline(
                config=test_config(max_error_attempts=2),
                paths=paths,
                sam_runtime=FakeSamRuntime(events),
                mllm_runtime=qwen,
            )

            report = pipeline.run([image], run_id="run_m5_retry")

            self.assertEqual(report["status"], "passed")
            self.assertEqual(qwen.call_count, 2)
            self.assertEqual(report["core_counts"]["decisions"], 1)
            self.assertEqual(report["images"][0]["qwen_batch"]["pipeline_attempts"], 2)
            self.assertEqual(report["images"][0]["qwen_batch"]["qwen_calls"], 2)

    def test_uncertain_is_persisted_without_immediate_second_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.png"
            second = root / "second.png"
            Image.new("RGB", (24, 24), (128, 128, 128)).save(first)
            Image.new("RGB", (24, 24), (64, 64, 64)).save(second)
            events: list[str] = []
            qwen = FakeQwenRuntime(events, existing_decision="uncertain")
            pipeline = ObjectMemoryPipeline(
                config=test_config(),
                paths=MemoryPaths(root / "memory"),
                sam_runtime=FakeSamRuntime(events),
                mllm_runtime=qwen,
            )

            report = pipeline.run(
                [first, second],
                run_id="run_m5_uncertain",
            )

            self.assertEqual(report["status"], "completed_with_errors")
            self.assertEqual(qwen.call_count, 2)
            self.assertEqual(report["run"]["proposal_counts"]["pending"], 1)
            self.assertEqual(report["run"]["decision_counts"]["uncertain"], 1)


if __name__ == "__main__":
    unittest.main()
