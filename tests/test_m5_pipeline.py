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
from object_memory.mllm_adapter import MllmPrediction
from object_memory.pipeline import ObjectMemoryPipeline
from object_memory.sam3_adapter import RawSamCandidate, Sam3Prediction


def test_config(*, max_error_attempts: int = 2) -> AppConfig:
    payload = load_config(DEFAULT_CONFIG_PATH).model_dump(mode="python")
    payload["mllm_pipeline"]["max_error_attempts"] = max_error_attempts
    return AppConfig.model_validate(payload)


def response_text(decision: str, object_id: str | None = None) -> str:
    reasons = {
        "new": "new_object",
        "existing": "visual_instance_match",
        "ignored": "invalid_candidate",
        "uncertain": "insufficient_evidence",
    }
    return json.dumps(
        {
            "decision": decision,
            "matched_object_id": object_id,
            "confidence": 0.95,
            "reason_code": reasons[decision],
            "short_reason": "deterministic M5 response",
            "annotation": (
                {
                    "coarse_category": "cup",
                    "fine_category": "coffee cup",
                    "material": ["ceramic"],
                    "color": ["white"],
                    "shape": "round with handle",
                    "description": "white ceramic cup with handle",
                    "annotation_confidence": 0.96,
                }
                if decision in {"new", "existing"}
                else None
            ),
        }
    )


class FakeSamRuntime:
    def __init__(
        self,
        events: list[str],
        *,
        duplicate_candidate: bool = False,
    ) -> None:
        self.events = events
        self.duplicate_candidate = duplicate_candidate
        self.model_load_seconds = 0.1
        self._peak_memory_mib = 100.0

    def load(self) -> None:
        self.events.append("sam.load")

    def predict(
        self,
        image: Image.Image,
    ) -> Sam3Prediction:
        self.events.append("sam.predict")
        x_min = image.width // 4
        y_min = image.height // 4
        x_max = image.width - x_min
        y_max = image.height - y_min
        mask = np.zeros((image.height, image.width), dtype=bool)
        mask[y_min:y_max, x_min:x_max] = True
        candidates = [
            RawSamCandidate(
                raw_candidate_id="candidate-main",
                prompt="automatic_point_grid",
                score=0.95,
                bbox_xyxy=(x_min, y_min, x_max, y_max),
                mask=mask,
            )
        ]
        if self.duplicate_candidate:
            candidates.append(
                RawSamCandidate(
                    raw_candidate_id="candidate-duplicate",
                    prompt="automatic_point_grid",
                    score=0.90,
                    bbox_xyxy=(x_min, y_min, x_max, y_max),
                    mask=mask,
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
        responses: list[str] | None = None,
    ) -> None:
        self.events = events
        self.responses = list(responses or [])
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
            object_ids = re.findall(r'"object_id": "(obj_[^"]+)"', all_text)
            raw_text = (
                response_text("existing", object_ids[0])
                if object_ids
                else response_text("new")
            )
        return MllmPrediction(
            raw_text=raw_text,
            input_tokens=100,
            generated_tokens=50,
            inference_seconds=0.4,
        )

    @property
    def peak_memory_mib(self) -> float:
        return self._peak_memory_mib

    def close(self) -> None:
        self.events.append("qwen.close")


class M5PipelineTests(unittest.TestCase):
    def test_batch_runs_models_sequentially_and_builds_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.png"
            second = root / "second.png"
            duplicate = root / "duplicate.png"
            Image.new("RGB", (20, 20), (255, 0, 0)).save(first)
            Image.new("RGB", (20, 20), (0, 0, 255)).save(second)
            shutil.copy2(first, duplicate)
            events: list[str] = []
            paths = MemoryPaths(root / "memory")
            pipeline = ObjectMemoryPipeline(
                config=test_config(),
                paths=paths,
                sam_runtime=FakeSamRuntime(events, duplicate_candidate=True),
                mllm_runtime=FakeQwenRuntime(events),
            )

            report = pipeline.run(
                [first, second, duplicate],
                run_id="run_m5_batch",
            )

            self.assertEqual(report["status"], "passed")
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
            self.assertLess(events.index("sam.close"), events.index("qwen.load"))
            self.assertTrue(paths.resolve_asset(report["run_report"]).is_file())

    def test_invalid_qwen_output_retries_without_duplicate_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "image.png"
            Image.new("RGB", (20, 20), (255, 255, 255)).save(image)
            events: list[str] = []
            qwen = FakeQwenRuntime(
                events,
                responses=["not json", response_text("new")],
            )
            paths = MemoryPaths(root / "memory")
            pipeline = ObjectMemoryPipeline(
                config=test_config(max_error_attempts=2),
                paths=paths,
                sam_runtime=FakeSamRuntime(events),
                mllm_runtime=qwen,
            )

            report = pipeline.run(
                [image],
                run_id="run_m5_retry",
            )

            self.assertEqual(report["status"], "passed")
            self.assertEqual(qwen.call_count, 2)
            self.assertEqual(report["core_counts"]["decisions"], 1)
            self.assertEqual(report["images"][0]["decisions"][0]["decision"], "new")
            self.assertEqual(
                report["images"][0]["decisions"][0]["qwen_call_attempts"],
                2,
            )

    def test_uncertain_is_persisted_without_immediate_second_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "image.png"
            Image.new("RGB", (20, 20), (128, 128, 128)).save(image)
            events: list[str] = []
            qwen = FakeQwenRuntime(
                events,
                responses=[response_text("uncertain")],
            )
            paths = MemoryPaths(root / "memory")
            pipeline = ObjectMemoryPipeline(
                config=test_config(max_error_attempts=2),
                paths=paths,
                sam_runtime=FakeSamRuntime(events),
                mllm_runtime=qwen,
            )

            report = pipeline.run(
                [image],
                run_id="run_m5_uncertain",
            )

            self.assertEqual(report["status"], "completed_with_errors")
            self.assertEqual(qwen.call_count, 1)
            self.assertEqual(report["run"]["proposal_counts"]["pending"], 1)
            self.assertEqual(report["run"]["decision_counts"]["uncertain"], 1)


if __name__ == "__main__":
    unittest.main()
