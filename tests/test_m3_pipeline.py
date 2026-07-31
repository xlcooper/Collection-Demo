"""Deterministic tests for two-stage Qwen analysis and identity retrieval."""

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
    build_candidate_analysis_messages,
    build_identity_confirmation_messages,
    evaluate_candidate,
    parse_candidate_analysis,
    parse_identity_response,
    retrieve_object_cards,
)
from object_memory.mllm_adapter import MllmPrediction
from object_memory.schemas import (
    CandidateValidity,
    DecisionType,
    ObjectAnnotation,
    ObjectCard,
)


def annotation_payload(
    *,
    coarse_category: str = "容器",
    fine_category: str = "马克杯",
    description: str = "白色陶瓷马克杯，带弧形把手",
) -> dict[str, Any]:
    return {
        "coarse_category": coarse_category,
        "fine_category": fine_category,
        "material": ["陶瓷"],
        "color": ["白色"],
        "shape": "带把手的圆柱形",
        "description": description,
        "annotation_confidence": 0.92,
    }


def analysis_payload(*, valid: bool = True) -> str:
    return json.dumps(
        {
            "validity": "valid" if valid else "ignored",
            "confidence": 0.94,
            "reason_code": "valid_candidate" if valid else "invalid_candidate",
            "short_reason": "完整物体" if valid else "阴影",
            "annotation": annotation_payload() if valid else None,
        },
        ensure_ascii=False,
    )


def identity_payload(
    decision: str,
    *,
    matched_object_id: str | None = None,
    confidence: float = 0.9,
) -> str:
    reason_codes = {
        "new": "new_object",
        "existing": "visual_instance_match",
        "uncertain": "insufficient_evidence",
    }
    return json.dumps(
        {
            "decision": decision,
            "matched_object_id": matched_object_id,
            "confidence": confidence,
            "reason_code": reason_codes[decision],
            "short_reason": "确定性身份测试响应",
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


def object_card(
    object_id: str,
    *,
    coarse_category: str = "容器",
    fine_category: str = "马克杯",
    description: str = "白色陶瓷马克杯，带弧形把手",
    view_path: str | None = None,
) -> ObjectCard:
    return ObjectCard(
        object_id=object_id,
        coarse_category=coarse_category,
        fine_category=fine_category,
        material=["陶瓷"],
        color=["白色"],
        shape="带把手的圆柱形",
        description=description,
        representative_view_paths=[view_path] if view_path else [],
    )


class ResponseValidationTests(unittest.TestCase):
    def test_fenced_candidate_analysis_validates_annotation(self) -> None:
        response = parse_candidate_analysis(f"```json\n{analysis_payload()}\n```")
        self.assertEqual(response.validity, CandidateValidity.VALID)
        self.assertEqual(response.annotation.fine_category, "马克杯")

    def test_ignored_candidate_must_not_include_annotation(self) -> None:
        payload = json.loads(analysis_payload(valid=False))
        payload["annotation"] = annotation_payload()
        with self.assertRaises(MllmOutputError):
            parse_candidate_analysis(json.dumps(payload, ensure_ascii=False))

    def test_existing_match_must_come_from_shortlist(self) -> None:
        with self.assertRaises(MllmOutputError):
            parse_identity_response(
                identity_payload(
                    "existing",
                    matched_object_id="obj_not_shown",
                ),
                allowed_object_ids={"obj_shown"},
            )

    def test_identity_stage_cannot_ignore_a_valid_candidate(self) -> None:
        payload = {
            "decision": "ignored",
            "matched_object_id": None,
            "confidence": 0.9,
            "reason_code": "invalid_candidate",
            "short_reason": "不匹配已有卡片",
        }
        with self.assertRaises(MllmOutputError):
            parse_identity_response(
                json.dumps(payload, ensure_ascii=False),
                allowed_object_ids={"obj_shown"},
            )


class PromptAndRetrievalTests(unittest.TestCase):
    def test_analysis_call_has_candidate_and_overlay_but_no_memory_cards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            crop = root / "crop.png"
            overlay = root / "overlay.jpg"
            Image.new("RGB", (8, 8)).save(crop)
            Image.new("RGB", (8, 8)).save(overlay)

            messages = build_candidate_analysis_messages(
                candidate_crop=crop,
                candidate_overlay=overlay,
                sam_prompt="automatic_point_grid",
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
            self.assertIn("without seeing or considering any memory", all_text)
            self.assertIn("mask-isolated", all_text)
            self.assertIn("No category hint was supplied", all_text)
            self.assertNotIn("object_id", all_text)
            self.assertEqual(image_count, 2)

    def test_identity_call_uses_temporary_annotation_and_reference_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = MemoryPaths(Path(temporary_directory) / "assets")
            paths.ensure_layout()
            crop = paths.proposals / "crop.png"
            reference = paths.objects / "reference.png"
            Image.new("RGB", (8, 8)).save(crop)
            Image.new("RGB", (8, 8)).save(reference)
            annotation = ObjectAnnotation.model_validate(annotation_payload())
            card = object_card(
                "obj_reference",
                view_path=paths.relative_asset(reference),
            )

            messages = build_identity_confirmation_messages(
                candidate_crop=crop,
                temporary_annotation=annotation,
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
            image_uris = [
                item["image"]
                for message in messages
                for item in message["content"]
                if item["type"] == "image"
            ]
            self.assertIn("temporary annotation", all_text)
            self.assertIn("obj_reference", all_text)
            self.assertIn("REFERENCE_IMAGE_CARD_1_VIEW_1", all_text)
            self.assertIn("It can never be ignored", all_text)
            self.assertEqual(
                image_uris,
                [crop.resolve().as_uri(), reference.resolve().as_uri()],
            )

    def test_semantic_retrieval_prefers_matching_card_and_honors_limit(self) -> None:
        annotation = ObjectAnnotation.model_validate(annotation_payload())
        cards = [
            object_card(
                "obj_mouse",
                coarse_category="电子设备",
                fine_category="鼠标",
                description="灰色无线鼠标",
            ),
            object_card("obj_cup"),
            object_card(
                "obj_bottle",
                fine_category="水瓶",
                description="透明塑料水瓶",
            ),
        ]

        retrieved = retrieve_object_cards(annotation, cards, limit=2)

        self.assertEqual(len(retrieved), 2)
        self.assertEqual(retrieved[0].card.object_id, "obj_cup")
        self.assertGreater(retrieved[0].score, retrieved[1].score)
        self.assertIn("fine_category", retrieved[0].matched_fields)


class TwoStageEvaluationTests(unittest.TestCase):
    def _assets(self, root: Path) -> tuple[Path, Path]:
        crop = root / "crop.png"
        overlay = root / "overlay.jpg"
        Image.new("RGB", (8, 8)).save(crop)
        Image.new("RGB", (8, 8)).save(overlay)
        return crop, overlay

    def test_invalid_candidate_stops_before_memory_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            crop, overlay = self._assets(Path(temporary_directory))
            predictor = QueuePredictor([analysis_payload(valid=False)])

            def unexpected_cards() -> list[ObjectCard]:
                raise AssertionError("ignored candidates must not query memory cards")

            def unexpected_references(
                object_ids: Sequence[str],
            ) -> list[ObjectCard]:
                raise AssertionError("ignored candidates must not query references")

            evaluation = evaluate_candidate(
                predictor,
                candidate_crop=crop,
                candidate_overlay=overlay,
                sam_prompt="automatic_point_grid",
                get_card_texts=unexpected_cards,
                get_reference_cards=unexpected_references,
                card_assets=None,
                settings=MllmPipelineConfig(),
            )

            self.assertEqual(
                evaluation.final_response.decision,
                DecisionType.IGNORED,
            )
            self.assertEqual(len(evaluation.predictions), 1)

    def test_valid_candidate_with_empty_memory_becomes_new_in_one_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            crop, overlay = self._assets(Path(temporary_directory))
            predictor = QueuePredictor([analysis_payload()])

            evaluation = evaluate_candidate(
                predictor,
                candidate_crop=crop,
                candidate_overlay=overlay,
                sam_prompt="automatic_point_grid",
                get_card_texts=lambda: [],
                get_reference_cards=lambda object_ids: [],
                card_assets=None,
                settings=MllmPipelineConfig(),
            )

            self.assertEqual(evaluation.final_response.decision, DecisionType.NEW)
            self.assertEqual(len(evaluation.predictions), 1)

    def test_valid_candidate_uses_shortlist_then_visual_identity_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            crop, overlay = self._assets(Path(temporary_directory))
            predictor = QueuePredictor(
                [
                    analysis_payload(),
                    identity_payload(
                        "existing",
                        matched_object_id="obj_cup",
                        confidence=0.94,
                    ),
                ]
            )
            cards = [
                object_card(
                    "obj_mouse",
                    coarse_category="电子设备",
                    fine_category="鼠标",
                    description="灰色鼠标",
                ),
                object_card("obj_cup"),
            ]
            requested_reference_ids: list[str] = []

            def reference_cards(object_ids: Sequence[str]) -> list[ObjectCard]:
                requested_reference_ids.extend(object_ids)
                return [
                    card for card in cards if card.object_id in object_ids
                ]

            evaluation = evaluate_candidate(
                predictor,
                candidate_crop=crop,
                candidate_overlay=overlay,
                sam_prompt="automatic_point_grid",
                get_card_texts=lambda: cards,
                get_reference_cards=reference_cards,
                card_assets=None,
                settings=MllmPipelineConfig(object_card_shortlist_size=1),
            )

            self.assertEqual(len(evaluation.predictions), 2)
            self.assertEqual(len(evaluation.retrieved_cards), 1)
            self.assertEqual(
                evaluation.retrieved_cards[0].card.object_id,
                "obj_cup",
            )
            self.assertEqual(requested_reference_ids, ["obj_cup"])
            self.assertEqual(
                evaluation.final_response.decision,
                DecisionType.EXISTING,
            )
            self.assertEqual(
                evaluation.final_response.matched_object_id,
                "obj_cup",
            )

    def test_low_confidence_visual_match_becomes_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            crop, overlay = self._assets(Path(temporary_directory))
            predictor = QueuePredictor(
                [
                    analysis_payload(),
                    identity_payload(
                        "existing",
                        matched_object_id="obj_cup",
                        confidence=0.6,
                    ),
                ]
            )

            evaluation = evaluate_candidate(
                predictor,
                candidate_crop=crop,
                candidate_overlay=overlay,
                sam_prompt="automatic_point_grid",
                get_card_texts=lambda: [object_card("obj_cup")],
                get_reference_cards=lambda object_ids: [
                    object_card("obj_cup")
                ],
                card_assets=None,
                settings=MllmPipelineConfig(existing_min_confidence=0.8),
            )

            self.assertEqual(
                evaluation.final_response.decision,
                DecisionType.UNCERTAIN,
            )
            self.assertIsNotNone(evaluation.final_response.annotation)


if __name__ == "__main__":
    unittest.main()
