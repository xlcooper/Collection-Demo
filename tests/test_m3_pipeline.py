"""Deterministic tests for M3 prompting, validation, and card batching."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from object_memory.assets import MemoryPaths
from object_memory.config import MllmPipelineConfig
from object_memory.identity import (
    MllmOutputError,
    build_identity_messages,
    evaluate_candidate,
    parse_mllm_response,
)
from object_memory.mllm_adapter import MllmPrediction
from object_memory.schemas import DecisionType, ObjectCard


def annotation_payload() -> dict[str, Any]:
    return {
        "coarse_category": "容器",
        "fine_category": "马克杯",
        "material": ["陶瓷"],
        "color": ["白色"],
        "shape": "带把手的圆柱形",
        "description": "白色陶瓷马克杯",
        "annotation_confidence": 0.92,
    }


def response_payload(
    decision: str,
    *,
    matched_object_id: str | None = None,
    confidence: float = 0.9,
) -> str:
    reason_codes = {
        "new": "new_object",
        "existing": "visual_instance_match",
        "ignored": "invalid_candidate",
        "uncertain": "insufficient_evidence",
    }
    return json.dumps(
        {
            "decision": decision,
            "matched_object_id": matched_object_id,
            "confidence": confidence,
            "reason_code": reason_codes[decision],
            "short_reason": "确定性测试响应",
            "annotation": (
                annotation_payload() if decision in {"new", "existing"} else None
            ),
        },
        ensure_ascii=False,
    )


class QueuePredictor:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.messages: list[Sequence[dict[str, Any]]] = []

    def predict(self, messages: Sequence[dict[str, Any]]) -> MllmPrediction:
        self.messages.append(messages)
        return MllmPrediction(
            raw_text=self.responses.pop(0),
            input_tokens=100,
            generated_tokens=50,
            inference_seconds=0.1,
        )


def object_card(object_id: str, view_path: str | None = None) -> ObjectCard:
    return ObjectCard(
        object_id=object_id,
        coarse_category="容器",
        fine_category="马克杯",
        material=["陶瓷"],
        color=["白色"],
        shape="带把手的圆柱形",
        description="白色陶瓷马克杯",
        representative_view_paths=[view_path] if view_path else [],
    )


class M3ResponseTests(unittest.TestCase):
    def test_fenced_json_validates_annotation(self) -> None:
        raw = f"```json\n{response_payload('new')}\n```"
        response = parse_mllm_response(raw, allowed_object_ids=set())
        self.assertEqual(response.decision, DecisionType.NEW)
        self.assertEqual(response.annotation.fine_category, "马克杯")

    def test_existing_match_must_come_from_current_card_batch(self) -> None:
        with self.assertRaises(MllmOutputError):
            parse_mllm_response(
                response_payload("existing", matched_object_id="obj_not_shown"),
                allowed_object_ids={"obj_shown"},
            )

    def test_reason_code_must_agree_with_decision(self) -> None:
        payload = json.loads(response_payload("new"))
        payload["reason_code"] = "visual_instance_match"
        with self.assertRaises(MllmOutputError):
            parse_mllm_response(json.dumps(payload), allowed_object_ids=set())


class M3IdentityPipelineTests(unittest.TestCase):
    def test_messages_map_reference_image_to_object_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = MemoryPaths(Path(temporary_directory) / "assets")
            paths.ensure_layout()
            crop = paths.proposals / "crop.png"
            overlay = paths.proposals / "overlay.jpg"
            reference = paths.objects / "reference.png"
            for path in (crop, overlay, reference):
                Image.new("RGB", (8, 8), (200, 200, 200)).save(path)
            card = object_card("obj_reference", paths.relative_asset(reference))

            messages = build_identity_messages(
                candidate_crop=crop,
                candidate_overlay=overlay,
                sam_prompt="cup",
                cards=[card],
                card_assets=paths,
                max_reference_views_per_object=2,
                max_pixels=1024,
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
            self.assertIn("obj_reference", all_text)
            self.assertIn("physical object", all_text)
            self.assertEqual(image_count, 3)

    def test_all_card_batches_are_checked_before_new_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            crop = root / "crop.png"
            overlay = root / "overlay.jpg"
            Image.new("RGB", (8, 8), (255, 255, 255)).save(crop)
            Image.new("RGB", (8, 8), (255, 255, 255)).save(overlay)
            cards = [object_card(f"obj_{index}") for index in range(3)]
            predictor = QueuePredictor(
                [
                    response_payload("new", confidence=0.88),
                    response_payload(
                        "existing",
                        matched_object_id="obj_2",
                        confidence=0.94,
                    ),
                ]
            )
            settings = MllmPipelineConfig(
                object_card_batch_size=2,
                existing_min_confidence=0.8,
            )

            evaluation = evaluate_candidate(
                predictor,
                candidate_crop=crop,
                candidate_overlay=overlay,
                sam_prompt="cup",
                cards=cards,
                card_assets=None,
                settings=settings,
            )

            self.assertEqual(len(evaluation.batches), 2)
            self.assertEqual(
                evaluation.final_response.decision,
                DecisionType.EXISTING,
            )
            self.assertEqual(
                evaluation.final_response.matched_object_id,
                "obj_2",
            )

    def test_low_confidence_match_becomes_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            crop = root / "crop.png"
            overlay = root / "overlay.jpg"
            Image.new("RGB", (8, 8)).save(crop)
            Image.new("RGB", (8, 8)).save(overlay)
            predictor = QueuePredictor(
                [
                    response_payload(
                        "existing",
                        matched_object_id="obj_1",
                        confidence=0.6,
                    )
                ]
            )

            evaluation = evaluate_candidate(
                predictor,
                candidate_crop=crop,
                candidate_overlay=overlay,
                sam_prompt="cup",
                cards=[object_card("obj_1")],
                card_assets=None,
                settings=MllmPipelineConfig(existing_min_confidence=0.8),
            )

            self.assertEqual(
                evaluation.final_response.decision,
                DecisionType.UNCERTAIN,
            )

    def test_strong_match_with_conflicting_batch_becomes_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            crop = root / "crop.png"
            overlay = root / "overlay.jpg"
            Image.new("RGB", (8, 8)).save(crop)
            Image.new("RGB", (8, 8)).save(overlay)
            predictor = QueuePredictor(
                [
                    response_payload(
                        "existing",
                        matched_object_id="obj_0",
                        confidence=0.95,
                    ),
                    response_payload("uncertain", confidence=0.7),
                ]
            )

            evaluation = evaluate_candidate(
                predictor,
                candidate_crop=crop,
                candidate_overlay=overlay,
                sam_prompt="cup",
                cards=[object_card("obj_0"), object_card("obj_1")],
                card_assets=None,
                settings=MllmPipelineConfig(object_card_batch_size=1),
            )

            self.assertEqual(
                evaluation.final_response.decision,
                DecisionType.UNCERTAIN,
            )


if __name__ == "__main__":
    unittest.main()
