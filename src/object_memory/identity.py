"""Object-card prompting, constrained response parsing, and batch aggregation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from pydantic import ValidationError

from .assets import MemoryPaths
from .config import MllmPipelineConfig
from .mllm_adapter import MllmPrediction
from .schemas import (
    DecisionReasonCode,
    DecisionType,
    MllmResponse,
    ObjectCard,
)


SYSTEM_PROMPT = """You are the visual identity component of an object-memory system.
Judge the physical object isolated by SAM, not its contents, nearby objects, scene, brand, or product model.
Two items with the same category are not automatically the same physical instance. Match only when visible instance-level evidence supports it. When evidence is weak or conflicting, answer uncertain.
Return exactly one JSON object with no Markdown or extra text. Use concise Chinese annotation values.
"""


OUTPUT_RULES = """Required JSON structure:
{
  "decision": "new | existing | ignored | uncertain",
  "matched_object_id": "known object ID when existing, otherwise null",
  "confidence": 0.0,
  "reason_code": "new_object | visual_instance_match | invalid_candidate | ambiguous_match | insufficient_evidence",
  "short_reason": "brief reason",
  "annotation": {
    "coarse_category": "physical object category",
    "fine_category": "more specific category or unknown",
    "material": ["visible material"],
    "color": ["visible color"],
    "shape": "visible shape",
    "description": "visible facts about the isolated object",
    "annotation_confidence": 0.0
  }
}
annotation is required for new and existing; use null for ignored and when no reliable annotation is possible.
Decision rules: valid object with no matching card -> new; same physical instance -> existing; invalid/non-object candidate -> ignored; insufficient or ambiguous evidence -> uncertain.
Reason codes must agree with the decision: new_object for new, visual_instance_match for existing, invalid_candidate for ignored, and ambiguous_match or insufficient_evidence for uncertain.
"""


class MllmOutputError(ValueError):
    """Raised when a model response is not safe to use."""


class MllmPredictor(Protocol):
    def predict(self, messages: Sequence[dict[str, Any]]) -> MllmPrediction: ...


@dataclass(frozen=True, slots=True)
class BatchEvaluation:
    object_ids: tuple[str, ...]
    response: MllmResponse
    prediction: MllmPrediction


@dataclass(frozen=True, slots=True)
class IdentityEvaluation:
    final_response: MllmResponse
    batches: tuple[BatchEvaluation, ...]


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Extract one JSON object while tolerating a surrounding code fence."""

    stripped = raw_text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as original_error:
        start = stripped.find("{")
        if start < 0:
            raise MllmOutputError("Qwen response contains no JSON object") from original_error
        try:
            payload, _ = json.JSONDecoder().raw_decode(stripped[start:])
        except json.JSONDecodeError as exc:
            raise MllmOutputError("Qwen response contains invalid JSON") from exc

    if not isinstance(payload, dict):
        raise MllmOutputError("Qwen response must be one JSON object")
    return payload


def parse_mllm_response(
    raw_text: str,
    *,
    allowed_object_ids: set[str],
) -> MllmResponse:
    """Validate JSON fields and prevent references to cards not in the batch."""

    try:
        response = MllmResponse.model_validate(extract_json_object(raw_text))
    except (ValidationError, ValueError) as exc:
        raise MllmOutputError(f"Qwen response failed schema validation: {exc}") from exc

    if response.decision is DecisionType.EXISTING:
        if response.matched_object_id not in allowed_object_ids:
            raise MllmOutputError(
                "Qwen matched an object ID that was not present in its card batch"
            )
    return response


def partition_object_cards(
    cards: Sequence[ObjectCard], batch_size: int
) -> list[tuple[ObjectCard, ...]]:
    if batch_size <= 0:
        raise ValueError("object card batch size must be positive")
    if not cards:
        return [tuple()]
    return [
        tuple(cards[index : index + batch_size])
        for index in range(0, len(cards), batch_size)
    ]


def _card_summary(card: ObjectCard) -> dict[str, Any]:
    return {
        "object_id": card.object_id,
        "coarse_category": card.coarse_category,
        "fine_category": card.fine_category,
        "material": card.material,
        "color": card.color,
        "shape": card.shape,
        "description": card.description,
    }


def build_identity_messages(
    *,
    candidate_crop: Path,
    candidate_overlay: Path,
    sam_prompt: str,
    cards: Sequence[ObjectCard],
    card_assets: MemoryPaths | None,
    max_reference_views_per_object: int,
    max_pixels: int,
) -> list[dict[str, Any]]:
    """Build an interleaved image/text request with explicit image ownership."""

    crop = candidate_crop.expanduser().resolve()
    overlay = candidate_overlay.expanduser().resolve()
    if not crop.is_file() or not overlay.is_file():
        raise FileNotFoundError("Candidate crop and overlay must both exist")
    if max_reference_views_per_object <= 0 or max_pixels <= 0:
        raise ValueError("MLLM image limits must be positive")

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"SAM category hint: {sam_prompt.strip() or 'unknown'}. "
                "Candidate crop follows. Judge the isolated physical object."
            ),
        },
        {"type": "image", "image": crop.as_uri(), "max_pixels": max_pixels},
        {
            "type": "text",
            "text": "Source overlay follows; the colored mask marks the candidate.",
        },
        {"type": "image", "image": overlay.as_uri(), "max_pixels": max_pixels},
    ]

    if not cards:
        content.append(
            {
                "type": "text",
                "text": "Known-object card batch is empty. Do not invent an object ID.",
            }
        )
    for card in cards:
        content.append(
            {
                "type": "text",
                "text": (
                    "Known-object card:\n"
                    + json.dumps(_card_summary(card), ensure_ascii=False, sort_keys=True)
                ),
            }
        )
        view_paths = card.representative_view_paths[
            :max_reference_views_per_object
        ]
        if view_paths and card_assets is None:
            raise ValueError("card_assets is required for representative views")
        for view_index, relative_path in enumerate(view_paths, start=1):
            assert card_assets is not None
            view = card_assets.resolve_asset(relative_path)
            if not view.is_file():
                raise FileNotFoundError(f"Object-card view not found: {view}")
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"Reference view {view_index} for object_id "
                            f"{card.object_id} follows."
                        ),
                    },
                    {
                        "type": "image",
                        "image": view.as_uri(),
                        "max_pixels": max_pixels,
                    },
                ]
            )

    content.append({"type": "text", "text": OUTPUT_RULES})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def aggregate_batch_responses(
    batches: Sequence[BatchEvaluation],
    *,
    existing_min_confidence: float,
) -> MllmResponse:
    """Conservatively combine fixed-size card comparisons."""

    if not batches:
        raise ValueError("At least one MLLM batch response is required")
    if not 0.0 <= existing_min_confidence <= 1.0:
        raise ValueError("existing_min_confidence must be between 0 and 1")

    strong_existing = [
        batch.response
        for batch in batches
        if batch.response.decision is DecisionType.EXISTING
        and batch.response.confidence >= existing_min_confidence
    ]
    matched_ids = {response.matched_object_id for response in strong_existing}
    if len(matched_ids) == 1:
        matched_id = next(iter(matched_ids))
        has_conflict = any(
            response.decision in {DecisionType.IGNORED, DecisionType.UNCERTAIN}
            or (
                response.decision is DecisionType.EXISTING
                and response.matched_object_id != matched_id
            )
            for response in (batch.response for batch in batches)
        )
        if not has_conflict:
            return max(strong_existing, key=lambda response: response.confidence)
    if matched_ids:
        return MllmResponse(
            decision=DecisionType.UNCERTAIN,
            matched_object_id=None,
            confidence=max(response.confidence for response in strong_existing),
            reason_code=DecisionReasonCode.AMBIGUOUS_MATCH,
            short_reason="对象卡片批次给出了冲突或不确定匹配",
            annotation=strong_existing[0].annotation,
        )

    responses = [batch.response for batch in batches]
    if all(response.decision is DecisionType.NEW for response in responses):
        selected = max(responses, key=lambda response: response.confidence)
        return selected.model_copy(
            update={"confidence": min(response.confidence for response in responses)}
        )
    if all(response.decision is DecisionType.IGNORED for response in responses):
        return max(responses, key=lambda response: response.confidence)

    annotation = next(
        (response.annotation for response in responses if response.annotation is not None),
        None,
    )
    confidence = max(response.confidence for response in responses)
    return MllmResponse(
        decision=DecisionType.UNCERTAIN,
        matched_object_id=None,
        confidence=confidence,
        reason_code=DecisionReasonCode.INSUFFICIENT_EVIDENCE,
        short_reason="对象卡片批次的判断不一致或匹配置信度不足",
        annotation=annotation,
    )


def evaluate_candidate(
    predictor: MllmPredictor,
    *,
    candidate_crop: Path,
    candidate_overlay: Path,
    sam_prompt: str,
    cards: Sequence[ObjectCard],
    card_assets: MemoryPaths | None,
    settings: MllmPipelineConfig,
) -> IdentityEvaluation:
    """Compare one candidate against every object card in fixed-size batches."""

    evaluations: list[BatchEvaluation] = []
    for card_batch in partition_object_cards(cards, settings.object_card_batch_size):
        messages = build_identity_messages(
            candidate_crop=candidate_crop,
            candidate_overlay=candidate_overlay,
            sam_prompt=sam_prompt,
            cards=card_batch,
            card_assets=card_assets,
            max_reference_views_per_object=settings.max_reference_views_per_object,
            max_pixels=settings.max_pixels,
        )
        prediction = predictor.predict(messages)
        object_ids = tuple(card.object_id for card in card_batch)
        response = parse_mllm_response(
            prediction.raw_text,
            allowed_object_ids=set(object_ids),
        )
        evaluations.append(
            BatchEvaluation(
                object_ids=object_ids,
                response=response,
                prediction=prediction,
            )
        )

    final_response = aggregate_batch_responses(
        evaluations,
        existing_min_confidence=settings.existing_min_confidence,
    )
    return IdentityEvaluation(
        final_response=final_response,
        batches=tuple(evaluations),
    )
