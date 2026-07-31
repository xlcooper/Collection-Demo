"""Deterministic end-to-end orchestration tests for the Demo pipeline."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch

import numpy as np
from PIL import Image

import object_memory.pipeline as pipeline_module
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
        self.received_prompts: list[tuple[str, ...]] = []
        self.received_pixels: list[tuple[int, int, int]] = []

    def load(self) -> None:
        self.events.append("sam.load")

    def predict(
        self,
        image: Image.Image,
        prompts: Sequence[str],
    ) -> Sam3Prediction:
        self.events.append("sam.predict")
        self.received_pixels.append(image.convert("RGB").getpixel((0, 0)))
        normalized_prompts = tuple(prompts)
        self.received_prompts.append(normalized_prompts)
        prompt = normalized_prompts[0]
        y_min = image.height // 4
        y_max = image.height - y_min
        first_x_min = image.width // 8
        first_x_max = image.width // 2 - 1
        first_mask = np.zeros((image.height, image.width), dtype=bool)
        first_mask[y_min:y_max, first_x_min:first_x_max] = True
        candidates = [
            RawSamCandidate(
                raw_candidate_id="candidate-main",
                prompt=prompt,
                score=0.95,
                bbox_xyxy=(first_x_min, y_min, first_x_max, y_max),
                mask=first_mask,
            )
        ]
        if self.duplicate_candidate:
            candidates.append(
                RawSamCandidate(
                    raw_candidate_id="candidate-duplicate",
                    prompt=prompt,
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
                    prompt=prompt,
                    score=0.94,
                    bbox_xyxy=(second_x_min, y_min, second_x_max, y_max),
                    mask=second_mask,
                )
            )
        for prompt_index, extra_prompt in enumerate(
            normalized_prompts[1:],
            start=1,
        ):
            extra_x_min = image.width // 2 + 1
            extra_x_max = image.width - image.width // 8
            extra_mask = np.zeros((image.height, image.width), dtype=bool)
            extra_mask[y_min:y_max, extra_x_min:extra_x_max] = True
            candidates.append(
                RawSamCandidate(
                    raw_candidate_id=f"candidate-prompt-{prompt_index}",
                    prompt=extra_prompt,
                    score=0.93 - prompt_index * 0.01,
                    bbox_xyxy=(extra_x_min, y_min, extra_x_max, y_max),
                    mask=extra_mask,
                )
            )
        prompt_counts = {
            current_prompt: sum(
                candidate.prompt == current_prompt for candidate in candidates
            )
            for current_prompt in normalized_prompts
        }
        return Sam3Prediction(
            candidates=tuple(candidates),
            prompt_counts=prompt_counts,
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
        scene_responses: list[str] | None = None,
        candidate_responses: list[str] | None = None,
        existing_decision: str = "existing",
        no_scene_targets: bool = False,
        scene_prompts: tuple[str, ...] = ("coffee cup",),
        scene_failures: int = 0,
        mutate_path_after_first_close: Path | None = None,
    ) -> None:
        self.events = events
        self.scene_responses = list(scene_responses or [])
        self.candidate_responses = list(candidate_responses or [])
        self.existing_decision = existing_decision
        self.no_scene_targets = no_scene_targets
        self.scene_prompts = scene_prompts
        self.scene_failures = scene_failures
        self.mutate_path_after_first_close = mutate_path_after_first_close
        self.model_load_seconds = 0.3
        self.model_placement = ["0"]
        self.resolved_snapshot = "fake-snapshot"
        self._peak_memory_mib = 200.0
        self.call_count = 0
        self.scene_call_count = 0
        self.candidate_call_count = 0
        self.close_count = 0

    def load(self) -> None:
        self.events.append("qwen.load")

    def predict(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> MllmPrediction:
        self.events.append("qwen.predict")
        self.call_count += 1
        all_text = "\n".join(
            item["text"]
            for message in messages
            for item in message["content"]
            if item["type"] == "text"
        )
        if "SCENE_BATCH begins" in all_text:
            self.scene_call_count += 1
            if self.scene_failures > 0:
                self.scene_failures -= 1
                raise RuntimeError("synthetic scene inference failure")
            if self.scene_responses:
                raw_text = self.scene_responses.pop(0)
            else:
                source_ids = re.findall(
                    r"source_id=(src_[A-Za-z0-9_-]+)",
                    all_text,
                )
                images = []
                for source_id in source_ids:
                    targets = [] if self.no_scene_targets else [
                        {
                            "target_id": f"target_{index:03d}",
                            "object_name_zh": "测试物体",
                            "sam_text_prompt": prompt,
                            "priority": "high",
                            "confidence": 0.95,
                            "selection_reason_code": "manipulable",
                            "selection_short_reason": "独立且可操作",
                        }
                        for index, prompt in enumerate(
                            self.scene_prompts,
                            start=1,
                        )
                    ]
                    images.append(
                        {
                            "source_id": source_id,
                            "scene_summary": "测试工作台",
                            "targets": targets,
                            "no_target_reason": (
                                "没有值得观察的独立物体"
                                if self.no_scene_targets
                                else None
                            ),
                        }
                    )
                raw_text = json.dumps({"images": images}, ensure_ascii=False)
        else:
            self.candidate_call_count += 1
            if self.candidate_responses:
                raw_text = self.candidate_responses.pop(0)
            else:
                proposal_ids = re.findall(
                    r"proposal_id=(prop_[A-Za-z0-9_-]+)",
                    all_text,
                )
                object_ids = re.findall(
                    r'"object_id": "(obj_[^"]+)"',
                    all_text,
                )
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
                        results.append(
                            batch_candidate(proposal_id, decision="new")
                        )
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
        self.close_count += 1
        if self.close_count == 1 and self.mutate_path_after_first_close:
            Image.new("RGB", (24, 24), (0, 255, 0)).save(
                self.mutate_path_after_first_close
            )


class PipelineTests(unittest.TestCase):
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
            sam = FakeSamRuntime(events, duplicate_candidate=True)
            pipeline = ObjectMemoryPipeline(
                config=test_config(),
                paths=paths,
                sam_runtime=sam,
                mllm_runtime=qwen,
            )

            report = pipeline.run(
                [first, second, duplicate],
                run_id="run_demo_batch",
            )

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["schema_version"], 3)
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
            self.assertEqual(report["models"]["qwen"]["scene_batch_calls"], 1)
            self.assertEqual(
                report["models"]["qwen"]["candidate_reasoning_calls"],
                2,
            )
            self.assertEqual(report["models"]["qwen"]["total_calls"], 3)
            self.assertEqual(report["models"]["qwen"]["load_count"], 2)
            self.assertEqual(
                report["images"][0]["candidate_reasoning"]["candidate_count"],
                1,
            )
            self.assertEqual(
                report["images"][0]["candidate_reasoning"][
                    "object_card_count"
                ],
                0,
            )
            self.assertEqual(
                report["images"][1]["candidate_reasoning"][
                    "object_card_count"
                ],
                1,
            )
            self.assertEqual(
                len(
                    report["images"][1]["candidate_reasoning"][
                        "object_card_ids"
                    ]
                ),
                1,
            )
            self.assertEqual(
                report["images"][0]["scene_guidance"]["target_count"],
                1,
            )
            self.assertEqual(sam.received_prompts, [("coffee cup",)] * 2)
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
            qwen_loads = [
                index
                for index, event in enumerate(events)
                if event == "qwen.load"
            ]
            self.assertLess(qwen_loads[0], events.index("sam.load"))
            self.assertLess(events.index("qwen.close"), events.index("sam.load"))
            self.assertLess(events.index("sam.close"), qwen_loads[1])
            self.assertEqual(qwen.call_count, 3)

    def test_all_candidates_from_one_image_share_one_detailed_qwen_call(self) -> None:
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

            report = pipeline.run([image], run_id="run_demo_one_image_batch")

            self.assertEqual(report["status"], "passed")
            self.assertEqual(qwen.call_count, 2)
            self.assertEqual(qwen.scene_call_count, 1)
            self.assertEqual(qwen.candidate_call_count, 1)
            self.assertEqual(
                report["images"][0]["candidate_reasoning"]["candidate_count"],
                2,
            )
            self.assertEqual(len(report["images"][0]["decisions"]), 2)
            self.assertEqual(report["run"]["decision_counts"]["new"], 1)
            self.assertEqual(report["run"]["decision_counts"]["ignored"], 1)

    def test_multiple_scene_concepts_reach_sam_and_candidate_reasoning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "scene.png"
            Image.new("RGB", (24, 24), (255, 255, 255)).save(image)
            events: list[str] = []
            qwen = FakeQwenRuntime(
                events,
                scene_prompts=("coffee cup", "computer mouse"),
            )
            sam = FakeSamRuntime(events)
            pipeline = ObjectMemoryPipeline(
                config=test_config(),
                paths=MemoryPaths(root / "memory"),
                sam_runtime=sam,
                mllm_runtime=qwen,
            )

            report = pipeline.run([image], run_id="run_demo_two_concepts")

            self.assertEqual(report["status"], "passed")
            self.assertEqual(
                sam.received_prompts,
                [("coffee cup", "computer mouse")],
            )
            self.assertEqual(
                report["images"][0]["sam"]["prompt_detection_counts"],
                {"coffee cup": 1, "computer mouse": 1},
            )
            self.assertEqual(
                report["images"][0]["candidate_reasoning"]["candidate_count"],
                2,
            )

    def test_invalid_batch_output_retries_without_duplicate_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "image.png"
            Image.new("RGB", (24, 24), (255, 255, 255)).save(image)
            events: list[str] = []
            qwen = FakeQwenRuntime(
                events,
                candidate_responses=["not json"],
            )
            paths = MemoryPaths(root / "memory")
            pipeline = ObjectMemoryPipeline(
                config=test_config(max_error_attempts=2),
                paths=paths,
                sam_runtime=FakeSamRuntime(events),
                mllm_runtime=qwen,
            )

            report = pipeline.run([image], run_id="run_demo_retry")

            self.assertEqual(report["status"], "passed")
            self.assertEqual(qwen.call_count, 3)
            self.assertEqual(report["core_counts"]["decisions"], 1)
            self.assertEqual(
                report["images"][0]["candidate_reasoning"][
                    "pipeline_attempts"
                ],
                2,
            )
            self.assertEqual(
                report["images"][0]["candidate_reasoning"]["qwen_calls"],
                2,
            )
            self.assertEqual(
                len(
                    report["images"][0]["candidate_reasoning"][
                        "attempt_raw_responses"
                    ]
                ),
                2,
            )
            first_attempt_path = report["images"][0]["candidate_reasoning"][
                "attempt_raw_responses"
            ][0]
            first_attempt = json.loads(
                paths.resolve_asset(first_attempt_path).read_text(encoding="utf-8")
            )
            self.assertEqual(len(first_attempt["expected_proposal_ids"]), 1)
            self.assertEqual(first_attempt["memory_context"]["object_card_count"], 0)

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
                run_id="run_demo_uncertain",
            )

            self.assertEqual(report["status"], "completed_with_errors")
            self.assertEqual(qwen.call_count, 3)
            self.assertEqual(report["run"]["proposal_counts"]["pending"], 1)
            self.assertEqual(report["run"]["decision_counts"]["uncertain"], 1)

    def test_empty_scene_guidance_skips_sam_and_candidate_reasoning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "empty.png"
            Image.new("RGB", (24, 24), (255, 255, 255)).save(image)
            events: list[str] = []
            qwen = FakeQwenRuntime(events, no_scene_targets=True)
            pipeline = ObjectMemoryPipeline(
                config=test_config(),
                paths=MemoryPaths(root / "memory"),
                sam_runtime=FakeSamRuntime(events),
                mllm_runtime=qwen,
            )

            report = pipeline.run([image], run_id="run_demo_no_targets")

            self.assertEqual(report["status"], "passed")
            self.assertEqual(qwen.call_count, 1)
            self.assertEqual(qwen.candidate_call_count, 0)
            self.assertNotIn("sam.predict", events)
            self.assertEqual(report["core_counts"]["proposals"], 0)
            self.assertEqual(
                report["images"][0]["scene_guidance"]["target_count"],
                0,
            )

    def test_scene_guidance_uses_configured_four_image_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images: list[Path] = []
            for index in range(5):
                image = root / f"scene_{index}.png"
                Image.new("RGB", (24, 24), (index * 20, 0, 0)).save(image)
                images.append(image)
            events: list[str] = []
            qwen = FakeQwenRuntime(events)
            pipeline = ObjectMemoryPipeline(
                config=test_config(),
                paths=MemoryPaths(root / "memory"),
                sam_runtime=FakeSamRuntime(events),
                mllm_runtime=qwen,
            )

            report = pipeline.run(images, run_id="run_demo_scene_batches")

            self.assertEqual(report["status"], "passed")
            self.assertEqual(qwen.scene_call_count, 2)
            self.assertEqual(qwen.candidate_call_count, 5)
            self.assertEqual(report["models"]["qwen"]["total_calls"], 7)
            self.assertEqual(
                [
                    image_report["scene_guidance"]["batch_index"]
                    for image_report in report["images"]
                ],
                [1, 1, 1, 1, 2],
            )

    def test_invalid_multi_image_scene_batch_is_rescued_per_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images: list[Path] = []
            for index in range(2):
                image = root / f"scene_{index}.png"
                Image.new("RGB", (24, 24), (index * 40, 0, 0)).save(image)
                images.append(image)
            events: list[str] = []
            qwen = FakeQwenRuntime(
                events,
                scene_responses=["not json", "still not json"],
            )
            pipeline = ObjectMemoryPipeline(
                config=test_config(max_error_attempts=2),
                paths=MemoryPaths(root / "memory"),
                sam_runtime=FakeSamRuntime(events),
                mllm_runtime=qwen,
            )

            report = pipeline.run(images, run_id="run_demo_scene_rescue")

            self.assertEqual(report["status"], "passed")
            self.assertEqual(qwen.scene_call_count, 4)
            self.assertEqual(
                report["models"]["qwen"]["phases"]["scene_guidance"][
                    "rescue_sources"
                ],
                2,
            )
            for image_report in report["images"]:
                guidance = image_report["scene_guidance"]
                self.assertEqual(
                    guidance["rescued_from_scope"],
                    "scene_batch_0001",
                )
                self.assertEqual(len(guidance["attempt_raw_responses"]), 3)
                self.assertIsNotNone(guidance["accepted_raw_response"])

    def test_runtime_exception_counts_as_a_qwen_call_and_keeps_raw_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "scene.png"
            Image.new("RGB", (24, 24), (255, 0, 0)).save(image)
            events: list[str] = []
            paths = MemoryPaths(root / "memory")
            qwen = FakeQwenRuntime(events, scene_failures=1)
            pipeline = ObjectMemoryPipeline(
                config=test_config(max_error_attempts=2),
                paths=paths,
                sam_runtime=FakeSamRuntime(events),
                mllm_runtime=qwen,
            )

            report = pipeline.run([image], run_id="run_demo_scene_exception")

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["models"]["qwen"]["scene_batch_calls"], 2)
            guidance = report["images"][0]["scene_guidance"]
            self.assertEqual(guidance["scope_calls"], 2)
            self.assertEqual(len(guidance["attempt_raw_responses"]), 2)
            for relative_path in guidance["attempt_raw_responses"]:
                self.assertTrue(paths.resolve_asset(relative_path).is_file())
            first_attempt = json.loads(
                paths.resolve_asset(guidance["attempt_raw_responses"][0]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(first_attempt["expected_source_ids"]), 1)
            self.assertEqual(first_attempt["predictions"], [])

    def test_sam_reads_the_canonical_source_after_input_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "scene.png"
            Image.new("RGB", (24, 24), (255, 0, 0)).save(image)
            events: list[str] = []
            qwen = FakeQwenRuntime(
                events,
                mutate_path_after_first_close=image,
            )
            sam = FakeSamRuntime(events)
            pipeline = ObjectMemoryPipeline(
                config=test_config(),
                paths=MemoryPaths(root / "memory"),
                sam_runtime=sam,
                mllm_runtime=qwen,
            )

            report = pipeline.run([image], run_id="run_demo_canonical_source")

            self.assertEqual(report["status"], "passed")
            self.assertEqual(sam.received_pixels, [(255, 0, 0)])

    def test_scene_raw_write_failure_stops_without_model_or_rescue_retries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "scene.png"
            Image.new("RGB", (24, 24), (255, 0, 0)).save(image)
            events: list[str] = []
            qwen = FakeQwenRuntime(events)
            original_write = pipeline_module.write_json_atomic

            def fail_scene_raw(path: Path, payload: dict[str, Any]) -> None:
                if "raw_responses" in path.parts:
                    raise OSError("synthetic raw storage failure")
                original_write(path, payload)

            pipeline = ObjectMemoryPipeline(
                config=test_config(max_error_attempts=2),
                paths=MemoryPaths(root / "memory"),
                sam_runtime=FakeSamRuntime(events),
                mllm_runtime=qwen,
            )
            with patch.object(
                pipeline_module,
                "write_json_atomic",
                side_effect=fail_scene_raw,
            ):
                report = pipeline.run([image], run_id="run_demo_scene_raw_failure")

            self.assertNotEqual(report["status"], "passed")
            self.assertEqual(qwen.scene_call_count, 1)
            self.assertEqual(qwen.candidate_call_count, 0)
            self.assertNotIn("sam.predict", events)
            self.assertEqual(
                report["models"]["qwen"]["phases"]["scene_guidance"][
                    "rescue_sources"
                ],
                0,
            )

    def test_candidate_raw_write_failure_never_persists_a_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "scene.png"
            Image.new("RGB", (24, 24), (255, 0, 0)).save(image)
            events: list[str] = []
            qwen = FakeQwenRuntime(events)
            original_write = pipeline_module.write_json_atomic

            def fail_candidate_raw(path: Path, payload: dict[str, Any]) -> None:
                if (
                    "raw_responses" in path.parts
                    and path.parent.name.startswith("src_")
                ):
                    raise OSError("synthetic candidate raw storage failure")
                original_write(path, payload)

            pipeline = ObjectMemoryPipeline(
                config=test_config(max_error_attempts=2),
                paths=MemoryPaths(root / "memory"),
                sam_runtime=FakeSamRuntime(events),
                mllm_runtime=qwen,
            )
            with patch.object(
                pipeline_module,
                "write_json_atomic",
                side_effect=fail_candidate_raw,
            ):
                report = pipeline.run(
                    [image],
                    run_id="run_demo_candidate_raw_failure",
                )

            self.assertNotEqual(report["status"], "passed")
            self.assertEqual(qwen.scene_call_count, 1)
            self.assertEqual(qwen.candidate_call_count, 1)
            self.assertEqual(report["core_counts"]["decisions"], 0)
            self.assertEqual(report["run"]["proposal_counts"]["failed"], 1)


if __name__ == "__main__":
    unittest.main()
