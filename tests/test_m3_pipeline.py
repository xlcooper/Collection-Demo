"""Deterministic tests for image-level Qwen candidate and memory reasoning."""

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
    BatchCandidateInput,
    MllmOutputError,
    build_image_batch_messages,
    evaluate_image_batch,
    parse_image_batch_response,
)
from object_memory.mllm_adapter import MllmPrediction
from object_memory.schemas import CandidateValidity, DecisionType, ObjectCard


def annotation_payload(
    *,
    description: str = "白色陶瓷马克杯，带弧形把手",
) -> dict[str, Any]:
    return {
        "coarse_category": "容器",
        "fine_category": "马克杯",
        "material": ["陶瓷"],
        "color": ["白色"],
        "shape": "带把手的圆柱形",
        "description": description,
        "annotation_confidence": 0.94,
    }


def candidate_payload(
    proposal_id: str,
    *,
    validity: str = "valid",
    decision: str = "new",
    matched_object_id: str | None = None,
    confidence: float = 0.93,
    final_description: str = "白色陶瓷马克杯，带弧形把手",
) -> dict[str, Any]:
    if validity == "ignored":
        return {
            "proposal_id": proposal_id,
            "validity": "ignored",
            "validity_confidence": 0.96,
            "validity_reason_code": "invalid_candidate",
            "validity_short_reason": "这是物体投下的阴影",
            "temporary_annotation": None,
            "decision": "ignored",
            "matched_object_id": None,
            "confidence": 0.96,
            "reason_code": "invalid_candidate",
            "short_reason": "不是独立物体",
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
        "validity_short_reason": "轮廓构成完整独立物体",
        "temporary_annotation": annotation_payload(),
        "decision": decision,
        "matched_object_id": matched_object_id,
        "confidence": confidence,
        "reason_code": reasons[decision],
        "short_reason": "完成全部卡片和参考图比较",
        "final_annotation": annotation_payload(
            description=final_description,
        ),
    }


def batch_payload(*items: dict[str, Any]) -> str:
    return json.dumps({"candidates": list(items)}, ensure_ascii=False)


def object_card(
    object_id: str,
    *,
    view_path: str | None = None,
) -> ObjectCard:
    return ObjectCard(
        object_id=object_id,
        coarse_category="容器",
        fine_category="马克杯",
        material=["陶瓷"],
        color=["白色"],
        shape="带把手的圆柱形",
        description="白色陶瓷马克杯，杯口有细窄蓝边",
        representative_view_paths=[view_path] if view_path else [],
    )


class QueuePredictor:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def predict(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> MllmPrediction:
        self.calls += 1
        return MllmPrediction(
            raw_text=self.responses.pop(0),
            input_tokens=100,
            generated_tokens=80,
            inference_seconds=0.2,
        )


class BatchResponseTests(unittest.TestCase):
    def test_fenced_response_covers_valid_and_ignored_candidates(self) -> None:
        raw = batch_payload(
            candidate_payload("prop_cup"),
            candidate_payload("prop_shadow", validity="ignored"),
        )
        response = parse_image_batch_response(
            f"```json\n{raw}\n```",
            expected_proposal_ids=["prop_cup", "prop_shadow"],
            allowed_object_ids=set(),
            existing_min_confidence=0.8,
        )

        self.assertEqual(len(response.candidates), 2)
        self.assertEqual(
            response.candidates[0].validity,
            CandidateValidity.VALID,
        )
        self.assertEqual(
            response.candidates[1].decision,
            DecisionType.IGNORED,
        )

    def test_response_must_cover_every_requested_candidate_once(self) -> None:
        raw = batch_payload(candidate_payload("prop_one"))
        with self.assertRaises(MllmOutputError):
            parse_image_batch_response(
                raw,
                expected_proposal_ids=["prop_one", "prop_two"],
                allowed_object_ids=set(),
                existing_min_confidence=0.8,
            )

    def test_existing_object_id_must_come_from_supplied_cards(self) -> None:
        raw = batch_payload(
            candidate_payload(
                "prop_cup",
                decision="existing",
                matched_object_id="obj_not_supplied",
            )
        )
        with self.assertRaises(MllmOutputError):
            parse_image_batch_response(
                raw,
                expected_proposal_ids=["prop_cup"],
                allowed_object_ids={"obj_supplied"},
                existing_min_confidence=0.8,
            )

    def test_valid_candidate_requires_temporary_and_final_annotations(self) -> None:
        payload = candidate_payload("prop_cup")
        payload["temporary_annotation"] = None
        with self.assertRaises(MllmOutputError):
            parse_image_batch_response(
                batch_payload(payload),
                expected_proposal_ids=["prop_cup"],
                allowed_object_ids=set(),
                existing_min_confidence=0.8,
            )

    def test_low_confidence_existing_is_rejected(self) -> None:
        raw = batch_payload(
            candidate_payload(
                "prop_cup",
                decision="existing",
                matched_object_id="obj_cup",
                confidence=0.6,
            )
        )
        with self.assertRaises(MllmOutputError):
            parse_image_batch_response(
                raw,
                expected_proposal_ids=["prop_cup"],
                allowed_object_ids={"obj_cup"},
                existing_min_confidence=0.8,
            )

    def test_valid_candidate_is_new_when_memory_is_empty(self) -> None:
        raw = batch_payload(
            candidate_payload("prop_cup", decision="uncertain")
        )
        with self.assertRaises(MllmOutputError):
            parse_image_batch_response(
                raw,
                expected_proposal_ids=["prop_cup"],
                allowed_object_ids=set(),
                existing_min_confidence=0.8,
            )


class BatchPromptTests(unittest.TestCase):
    def test_one_message_contains_all_candidates_cards_and_references(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = MemoryPaths(Path(temporary_directory) / "assets")
            paths.ensure_layout()
            candidates: list[BatchCandidateInput] = []
            for index in range(2):
                crop = paths.proposals / f"crop_{index}.png"
                overlay = paths.proposals / f"overlay_{index}.jpg"
                Image.new("RGB", (8, 8)).save(crop)
                Image.new("RGB", (8, 8)).save(overlay)
                candidates.append(
                    BatchCandidateInput(
                        proposal_id=f"prop_{index}",
                        crop_path=crop,
                        overlay_path=overlay,
                        sam_prompt="automatic_point_grid",
                    )
                )
            references: list[str] = []
            for index in range(2):
                reference = paths.objects / f"reference_{index}.png"
                Image.new("RGB", (8, 8)).save(reference)
                references.append(paths.relative_asset(reference))
            cards = [
                object_card("obj_one", view_path=references[0]),
                object_card("obj_two", view_path=references[1]),
            ]

            messages, reference_count = build_image_batch_messages(
                candidates=candidates,
                cards=cards,
                card_assets=paths,
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

            self.assertIn("prop_0", all_text)
            self.assertIn("prop_1", all_text)
            self.assertIn("obj_one", all_text)
            self.assertIn("obj_two", all_text)
            self.assertIn("No script-side similarity ranking", all_text)
            self.assertIn("temporary_annotation", all_text)
            self.assertIn("final_annotation", all_text)
            self.assertEqual(reference_count, 2)
            self.assertEqual(image_count, 6)

    def test_evaluation_uses_one_call_for_multiple_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidates: list[BatchCandidateInput] = []
            for index in range(2):
                crop = root / f"crop_{index}.png"
                overlay = root / f"overlay_{index}.jpg"
                Image.new("RGB", (8, 8)).save(crop)
                Image.new("RGB", (8, 8)).save(overlay)
                candidates.append(
                    BatchCandidateInput(
                        proposal_id=f"prop_{index}",
                        crop_path=crop,
                        overlay_path=overlay,
                        sam_prompt="automatic_point_grid",
                    )
                )
            predictor = QueuePredictor(
                [
                    batch_payload(
                        candidate_payload("prop_0"),
                        candidate_payload("prop_1", validity="ignored"),
                    )
                ]
            )

            evaluation = evaluate_image_batch(
                predictor,
                candidates=candidates,
                cards=[],
                card_assets=None,
                settings=MllmPipelineConfig(),
            )

            self.assertEqual(predictor.calls, 1)
            self.assertEqual(len(evaluation.response.candidates), 2)
            self.assertEqual(evaluation.object_card_count, 0)


if __name__ == "__main__":
    unittest.main()
