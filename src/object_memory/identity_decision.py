"""Deterministic target association and Qwen/DINOv3 decision agreement."""

from __future__ import annotations

from typing import Sequence

from .schemas import (
    DecisionReasonCode,
    DecisionType,
    FinalIdentityDecision,
    IdentityHypothesis,
    NormalizedBoundingBox,
    Proposal,
    SceneTarget,
    VisualEvidence,
    VisualMatchType,
)


def _normalized_proposal_box(
    proposal: Proposal,
    *,
    image_width: int,
    image_height: int,
) -> NormalizedBoundingBox:
    return NormalizedBoundingBox(
        x_min=proposal.bbox.x_min / image_width,
        y_min=proposal.bbox.y_min / image_height,
        x_max=proposal.bbox.x_max / image_width,
        y_max=proposal.bbox.y_max / image_height,
    )


def box_iou(first: NormalizedBoundingBox, second: NormalizedBoundingBox) -> float:
    intersection_width = max(
        0.0,
        min(first.x_max, second.x_max) - max(first.x_min, second.x_min),
    )
    intersection_height = max(
        0.0,
        min(first.y_max, second.y_max) - max(first.y_min, second.y_min),
    )
    intersection = intersection_width * intersection_height
    first_area = (first.x_max - first.x_min) * (first.y_max - first.y_min)
    second_area = (second.x_max - second.x_min) * (second.y_max - second.y_min)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def associate_targets(
    proposals: Sequence[Proposal],
    targets: Sequence[SceneTarget],
    *,
    image_width: int,
    image_height: int,
    minimum_iou: float,
) -> dict[str, SceneTarget]:
    """Greedily form deterministic one-to-one pairs within each SAM prompt."""

    pairs: list[tuple[float, str, str, Proposal, SceneTarget]] = []
    for proposal in proposals:
        proposal_box = _normalized_proposal_box(
            proposal,
            image_width=image_width,
            image_height=image_height,
        )
        for target in targets:
            if proposal.prompt != target.sam_text_prompt:
                continue
            overlap = box_iou(proposal_box, target.temporary_target_anchor)
            if overlap >= minimum_iou:
                pairs.append(
                    (-overlap, proposal.id, target.target_id, proposal, target)
                )
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    assigned_proposals: set[str] = set()
    assigned_targets: set[str] = set()
    result: dict[str, SceneTarget] = {}
    for _, proposal_id, target_id, proposal, target in pairs:
        if proposal_id in assigned_proposals or target_id in assigned_targets:
            continue
        assigned_proposals.add(proposal_id)
        assigned_targets.add(target_id)
        proposal.target_id = target.target_id
        proposal.target_object_name_zh = target.object_name_zh
        proposal.target_anchor = target.temporary_target_anchor
        result[proposal.id] = target
    return result


def decide_identity(
    target: SceneTarget,
    visual: VisualEvidence,
) -> FinalIdentityDecision:
    """Commit new/existing only when the text hypothesis and visual result agree."""

    if target.identity_hypothesis is IdentityHypothesis.UNCERTAIN:
        return FinalIdentityDecision(
            decision=DecisionType.UNCERTAIN,
            confidence=0.0,
            reason_code=(
                DecisionReasonCode.AMBIGUOUS_MATCH
                if visual.result is VisualMatchType.AMBIGUOUS
                else DecisionReasonCode.INSUFFICIENT_EVIDENCE
            ),
            short_reason="Qwen文本身份证据不足，保持待定。",
            qwen_hypothesis=target.identity_hypothesis,
            visual_evidence=visual,
        )

    if visual.result is VisualMatchType.AMBIGUOUS:
        return FinalIdentityDecision(
            decision=DecisionType.UNCERTAIN,
            confidence=0.0,
            reason_code=DecisionReasonCode.AMBIGUOUS_MATCH,
            short_reason="第一与第二视觉候选差距不足，保持待定。",
            qwen_hypothesis=target.identity_hypothesis,
            qwen_matched_object_id=target.matched_object_id,
            visual_evidence=visual,
        )

    if target.identity_hypothesis is IdentityHypothesis.NEW:
        if visual.result is VisualMatchType.NO_MATCH:
            return FinalIdentityDecision(
                decision=DecisionType.NEW,
                confidence=target.proposed_object_summary.summary_confidence,
                reason_code=DecisionReasonCode.NEW_OBJECT,
                short_reason="Qwen判断为新对象，且历史视觉指纹均未达到匹配阈值。",
                qwen_hypothesis=target.identity_hypothesis,
                visual_evidence=visual,
                object_summary=target.proposed_object_summary,
            )
        return FinalIdentityDecision(
            decision=DecisionType.UNCERTAIN,
            confidence=0.0,
            reason_code=DecisionReasonCode.INSUFFICIENT_EVIDENCE,
            short_reason="Qwen判断为新对象，但视觉指纹明确匹配已有对象。",
            qwen_hypothesis=target.identity_hypothesis,
            visual_evidence=visual,
        )

    assert target.identity_hypothesis is IdentityHypothesis.EXISTING
    if (
        visual.result is VisualMatchType.MATCH
        and visual.matched_object_id == target.matched_object_id
    ):
        return FinalIdentityDecision(
            decision=DecisionType.EXISTING,
            matched_object_id=target.matched_object_id,
            confidence=target.proposed_object_summary.summary_confidence,
            reason_code=DecisionReasonCode.VISUAL_INSTANCE_MATCH,
            short_reason="Qwen对象假设与DINOv3最佳历史视角一致。",
            qwen_hypothesis=target.identity_hypothesis,
            qwen_matched_object_id=target.matched_object_id,
            visual_evidence=visual,
            object_summary=target.proposed_object_summary,
        )
    return FinalIdentityDecision(
        decision=DecisionType.UNCERTAIN,
        confidence=0.0,
        reason_code=DecisionReasonCode.INSUFFICIENT_EVIDENCE,
        short_reason="Qwen已有对象假设未得到同一DINOv3对象匹配支持。",
        qwen_hypothesis=target.identity_hypothesis,
        qwen_matched_object_id=target.matched_object_id,
        visual_evidence=visual,
    )


def unmatched_proposal_decision(visual: VisualEvidence) -> FinalIdentityDecision:
    """Persist an unassociated SAM proposal as uncertain without another model call."""

    return FinalIdentityDecision(
        decision=DecisionType.UNCERTAIN,
        confidence=0.0,
        reason_code=DecisionReasonCode.INSUFFICIENT_EVIDENCE,
        short_reason="SAM3候选无法与Qwen临时目标框形成唯一对应。",
        qwen_hypothesis=IdentityHypothesis.UNCERTAIN,
        visual_evidence=visual,
    )
