"""Deterministic tests for first-pass robot scene guidance."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from object_memory.config import MllmPipelineConfig
from object_memory.mllm_adapter import MllmPrediction
from object_memory.scene_guidance import (
    MllmOutputError,
    SceneImageInput,
    build_scene_guidance_messages,
    evaluate_scene_guidance_batch,
    parse_scene_guidance_response,
)


def target_payload(
    target_id: str,
    *,
    sam_text_prompt: str = "water bottle",
) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "object_name_zh": "水瓶",
        "sam_text_prompt": sam_text_prompt,
        "priority": "high",
        "confidence": 0.93,
        "selection_reason_code": "manipulable",
        "selection_short_reason": "独立、可移动且值得多视角观察",
    }


def guidance_payload(
    source_id: str,
    *,
    targets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selected = [target_payload("target_001")] if targets is None else targets
    return {
        "source_id": source_id,
        "scene_summary": "机械臂工作台的一角",
        "targets": selected,
        "no_target_reason": None if selected else "未看到独立任务物体",
    }


class QueuePredictor:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def predict(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> MllmPrediction:
        self.calls += 1
        return MllmPrediction(
            raw_text=self.response,
            input_tokens=120,
            generated_tokens=90,
            inference_seconds=0.2,
        )


class SceneGuidanceResponseTests(unittest.TestCase):
    def test_response_covers_each_source_and_allows_empty_targets(self) -> None:
        raw = json.dumps(
            {
                "images": [
                    guidance_payload("src_one"),
                    guidance_payload("src_two", targets=[]),
                ]
            },
            ensure_ascii=False,
        )

        response = parse_scene_guidance_response(
            raw,
            expected_source_ids=["src_one", "src_two"],
            max_targets_per_image=12,
        )

        self.assertEqual(len(response.images), 2)
        self.assertEqual(response.images[0].targets[0].sam_text_prompt, "water bottle")
        self.assertEqual(response.images[1].targets, [])

    def test_response_rejects_missing_source(self) -> None:
        raw = json.dumps({"images": [guidance_payload("src_one")]})

        with self.assertRaises(MllmOutputError):
            parse_scene_guidance_response(
                raw,
                expected_source_ids=["src_one", "src_two"],
                max_targets_per_image=12,
            )

    def test_response_rejects_duplicate_or_generic_sam_prompts(self) -> None:
        duplicate = guidance_payload(
            "src_one",
            targets=[
                target_payload("target_001"),
                target_payload("target_002"),
            ],
        )
        with self.assertRaises(MllmOutputError):
            parse_scene_guidance_response(
                json.dumps({"images": [duplicate]}),
                expected_source_ids=["src_one"],
                max_targets_per_image=12,
            )

        generic = guidance_payload(
            "src_one",
            targets=[target_payload("target_001", sam_text_prompt="object")],
        )
        with self.assertRaises(MllmOutputError):
            parse_scene_guidance_response(
                json.dumps({"images": [generic]}),
                expected_source_ids=["src_one"],
                max_targets_per_image=12,
            )

        for invalid_prompt in (
            "bottle and mouse",
            "left bottle",
            "cup. mouse",
            "cup-or-bottle",
            "cup on table",
            "small object",
            "123",
        ):
            with self.subTest(invalid_prompt=invalid_prompt):
                invalid = guidance_payload(
                    "src_one",
                    targets=[
                        target_payload(
                            "target_001",
                            sam_text_prompt=invalid_prompt,
                        )
                    ],
                )
                with self.assertRaises(MllmOutputError):
                    parse_scene_guidance_response(
                        json.dumps({"images": [invalid]}),
                        expected_source_ids=["src_one"],
                        max_targets_per_image=12,
                    )


class SceneGuidancePromptTests(unittest.TestCase):
    def test_four_image_batch_builds_one_call_with_exact_source_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images: list[SceneImageInput] = []
            for index in range(4):
                path = root / f"scene_{index}.png"
                Image.new("RGB", (8, 8)).save(path)
                images.append(
                    SceneImageInput(
                        source_id=f"src_{index}",
                        image_path=path,
                    )
                )

            messages = build_scene_guidance_messages(
                images=images,
                settings=MllmPipelineConfig(),
            )
            all_text = "\n".join(
                item["text"]
                for message in messages
                for item in message["content"]
                if item["type"] == "text"
            )
            image_count = sum(
                item["type"] == "image"
                for message in messages
                for item in message["content"]
            )

            for index in range(4):
                self.assertIn(f"src_{index}", all_text)
            self.assertIn("robot-oriented observation plan", all_text)
            self.assertIn("only from its own visible pixels", all_text)
            self.assertEqual(image_count, 4)

    def test_evaluation_uses_one_call_for_multiple_source_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scene_one = root / "one.png"
            scene_two = root / "two.png"
            Image.new("RGB", (8, 8)).save(scene_one)
            Image.new("RGB", (8, 8)).save(scene_two)
            predictor = QueuePredictor(
                json.dumps(
                    {
                        "images": [
                            guidance_payload("src_one"),
                            guidance_payload("src_two"),
                        ]
                    }
                )
            )

            evaluation = evaluate_scene_guidance_batch(
                predictor,
                images=[
                    SceneImageInput("src_one", scene_one),
                    SceneImageInput("src_two", scene_two),
                ],
                settings=MllmPipelineConfig(),
            )

            self.assertEqual(predictor.calls, 1)
            self.assertEqual(len(evaluation.response.images), 2)


if __name__ == "__main__":
    unittest.main()
