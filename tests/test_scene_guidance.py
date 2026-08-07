"""Tests for the one-call Qwen discovery and text-memory contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from object_memory.config import DEFAULT_CONFIG_PATH, load_config
from object_memory.identity import MllmOutputError
from object_memory.mllm_adapter import MllmPrediction
from object_memory.scene_guidance import (
    SceneImageInput,
    build_scene_guidance_messages,
    evaluate_scene_guidance,
    parse_scene_guidance_response,
)
from object_memory.schemas import ObjectCard, ObjectSummary


def summary(name: str = "人体工学鼠标") -> dict[str, Any]:
    return {
        "object_name_zh": name,
        "coarse_category": "电子设备",
        "fine_category": "鼠标",
        "stable_description": "银灰色非对称鼠标，右侧轮廓隆起。",
        "stable_identity_features": ["非左右对称"],
        "brand_or_markings": [],
        "part_appearance": [
            {"part": "外壳", "color": ["银灰色"], "material": ["塑料"]}
        ],
        "summary_confidence": 0.9,
    }


def target_payload(
    target_id: str,
    *,
    hypothesis: str = "new",
    matched_object_id: str | None = None,
    anchor: list[float] | None = None,
) -> dict[str, Any]:
    x_min, y_min, x_max, y_max = anchor or [0.1, 0.1, 0.5, 0.8]
    return {
        "target_id": target_id,
        "object_name_zh": "银灰色人体工学鼠标",
        "sam_text_prompt": "computer mouse",
        "current_view_facts": {
            "category": "鼠标",
            "visible_identity_features": ["右侧轮廓隆起"],
            "brand_or_markings": [],
            "part_appearance": [],
        },
        "identity_hypothesis": hypothesis,
        "matched_object_id": matched_object_id,
        "identity_short_reason": "可见轮廓证据",
        "proposed_object_summary": summary(),
        "temporary_target_anchor": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        },
    }


def response_payload(*targets: dict[str, Any]) -> str:
    return json.dumps(
        {
            "image": {
                "source_id": "src_1",
                "scene_summary": "桌面上有两个鼠标",
                "targets": list(targets),
                "no_target_reason": None if targets else "没有目标",
            }
        },
        ensure_ascii=False,
    )


class FakePredictor:
    def __init__(self, raw_text: str) -> None:
        self.raw_text = raw_text
        self.calls = 0

    def predict(self, messages: Sequence[dict[str, Any]]) -> MllmPrediction:
        self.calls += 1
        return MllmPrediction(self.raw_text, 10, 10, 0.1)


class SceneGuidanceTests(unittest.TestCase):
    def test_duplicate_sam_prompts_are_allowed_for_distinct_instances(self) -> None:
        response = parse_scene_guidance_response(
            response_payload(
                target_payload("target_001"),
                target_payload("target_002", anchor=[0.5, 0.1, 0.9, 0.8]),
            ),
            expected_source_id="src_1",
            allowed_object_ids=set(),
            max_targets_per_image=12,
        )
        self.assertEqual(len(response.targets), 2)

    def test_existing_id_must_come_from_text_cards(self) -> None:
        with self.assertRaises(MllmOutputError):
            parse_scene_guidance_response(
                response_payload(
                    target_payload(
                        "target_001",
                        hypothesis="existing",
                        matched_object_id="obj_missing",
                    )
                ),
                expected_source_id="src_1",
                allowed_object_ids={"obj_known"},
                max_targets_per_image=12,
            )

    def test_empty_memory_requires_new_hypothesis(self) -> None:
        with self.assertRaises(MllmOutputError):
            parse_scene_guidance_response(
                response_payload(target_payload("target_001", hypothesis="uncertain")),
                expected_source_id="src_1",
                allowed_object_ids=set(),
                max_targets_per_image=12,
            )

    def test_message_contains_one_image_and_text_summaries_without_old_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "source.png"
            Image.new("RGB", (16, 16), (20, 30, 40)).save(image_path)
            card = ObjectCard(
                object_id="obj_mouse",
                summary=ObjectSummary.model_validate(summary()),
            )
            messages = build_scene_guidance_messages(
                image=SceneImageInput("src_1", image_path),
                cards=[card],
                settings=load_config(DEFAULT_CONFIG_PATH).mllm_pipeline,
            )
        content = messages[1]["content"]
        self.assertEqual(sum(item["type"] == "image" for item in content), 1)
        text = "\n".join(item["text"] for item in content if item["type"] == "text")
        self.assertIn("obj_mouse", text)
        self.assertIn("stable_identity_features", text)
        self.assertNotIn("representative_view", text)

    def test_evaluation_performs_exactly_one_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "source.png"
            Image.new("RGB", (16, 16), (20, 30, 40)).save(image_path)
            predictor = FakePredictor(response_payload(target_payload("target_001")))
            result = evaluate_scene_guidance(
                predictor,
                image=SceneImageInput("src_1", image_path),
                cards=[],
                settings=load_config(DEFAULT_CONFIG_PATH).mllm_pipeline,
            )
        self.assertEqual(predictor.calls, 1)
        self.assertEqual(result.response.targets[0].target_id, "target_001")


if __name__ == "__main__":
    unittest.main()
