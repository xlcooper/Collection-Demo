"""Tests for target association and final two-evidence identity decisions."""

from __future__ import annotations

import unittest

from object_memory.identity import MllmOutputError, extract_json_object
from object_memory.identity_decision import associate_targets, decide_identity
from object_memory.schemas import (
    BoundingBox,
    CurrentViewFacts,
    DecisionType,
    IdentityHypothesis,
    NormalizedBoundingBox,
    ObjectSummary,
    Proposal,
    SceneTarget,
    VisualEvidence,
    VisualMatchType,
)


def summary() -> ObjectSummary:
    return ObjectSummary(
        object_name_zh="银灰色人体工学鼠标",
        coarse_category="电子设备",
        fine_category="鼠标",
        stable_description="银灰色非对称人体工学鼠标，右侧轮廓隆起。",
        stable_identity_features=["整体非左右对称", "右侧轮廓隆起"],
        brand_or_markings=[],
        part_appearance=[],
        summary_confidence=0.92,
    )


def target(
    hypothesis: IdentityHypothesis,
    *,
    matched_object_id: str | None = None,
    target_id: str = "target_001",
    anchor: tuple[float, float, float, float] = (0.0, 0.0, 0.5, 1.0),
) -> SceneTarget:
    return SceneTarget(
        target_id=target_id,
        object_name_zh="银灰色人体工学鼠标",
        sam_text_prompt="computer mouse",
        current_view_facts=CurrentViewFacts(
            category="鼠标",
            visible_identity_features=["右侧轮廓隆起"],
        ),
        identity_hypothesis=hypothesis,
        matched_object_id=matched_object_id,
        identity_short_reason="轮廓证据",
        proposed_object_summary=summary(),
        temporary_target_anchor=NormalizedBoundingBox(
            x_min=anchor[0], y_min=anchor[1], x_max=anchor[2], y_max=anchor[3]
        ),
    )


class JsonExtractionTests(unittest.TestCase):
    def test_fenced_object_is_extracted(self) -> None:
        self.assertEqual(extract_json_object('```json\n{"a": 1}\n```'), {"a": 1})

    def test_non_object_is_rejected(self) -> None:
        with self.assertRaises(MllmOutputError):
            extract_json_object("[]")


class TargetAssociationTests(unittest.TestCase):
    def test_duplicate_prompt_instances_are_paired_one_to_one_by_anchor(self) -> None:
        proposals = [
            Proposal(
                id="prop_left",
                source_image_id="src_1",
                raw_candidate_id="raw_left",
                prompt="computer mouse",
                score=0.9,
                bbox=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=100),
            ),
            Proposal(
                id="prop_right",
                source_image_id="src_1",
                raw_candidate_id="raw_right",
                prompt="computer mouse",
                score=0.9,
                bbox=BoundingBox(x_min=50, y_min=0, x_max=100, y_max=100),
            ),
        ]
        targets = [
            target(IdentityHypothesis.NEW),
            target(
                IdentityHypothesis.NEW,
                target_id="target_002",
                anchor=(0.5, 0.0, 1.0, 1.0),
            ),
        ]
        result = associate_targets(
            proposals,
            targets,
            image_width=100,
            image_height=100,
            minimum_iou=0.1,
        )
        self.assertEqual(result["prop_left"].target_id, "target_001")
        self.assertEqual(result["prop_right"].target_id, "target_002")


class FinalDecisionTests(unittest.TestCase):
    def test_new_requires_visual_no_match(self) -> None:
        result = decide_identity(
            target(IdentityHypothesis.NEW),
            VisualEvidence(result=VisualMatchType.NO_MATCH),
        )
        self.assertEqual(result.decision, DecisionType.NEW)
        self.assertIsNotNone(result.object_summary)

    def test_existing_requires_same_visual_object(self) -> None:
        visual = VisualEvidence(
            result=VisualMatchType.MATCH,
            matched_object_id="obj_mouse",
            matched_observation_id="obs_1",
            global_similarity=0.9,
            local_match_ratio=0.8,
            visual_score=0.85,
        )
        result = decide_identity(
            target(IdentityHypothesis.EXISTING, matched_object_id="obj_mouse"),
            visual,
        )
        self.assertEqual(result.decision, DecisionType.EXISTING)
        self.assertEqual(result.matched_object_id, "obj_mouse")

    def test_text_visual_conflict_is_uncertain(self) -> None:
        visual = VisualEvidence(
            result=VisualMatchType.MATCH,
            matched_object_id="obj_other",
            matched_observation_id="obs_2",
            visual_score=0.9,
            global_similarity=0.9,
            local_match_ratio=0.9,
        )
        result = decide_identity(
            target(IdentityHypothesis.EXISTING, matched_object_id="obj_mouse"),
            visual,
        )
        self.assertEqual(result.decision, DecisionType.UNCERTAIN)
        self.assertIsNone(result.object_summary)

    def test_ambiguous_visual_result_is_terminal_uncertain(self) -> None:
        result = decide_identity(
            target(IdentityHypothesis.NEW),
            VisualEvidence(
                result=VisualMatchType.AMBIGUOUS,
                visual_score=0.9,
                second_best_score=0.88,
                score_margin=0.02,
            ),
        )
        self.assertEqual(result.decision, DecisionType.UNCERTAIN)


if __name__ == "__main__":
    unittest.main()
