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


SYSTEM_PROMPT = """You are the single-call identity and annotation component of an object-memory system.
Follow the ordered stages below, then return exactly one JSON object with no Markdown or extra text.
Stage 1 - validity: decide whether IMAGE_A_CANDIDATE is one valid physical object. Judge the isolated object, not its contents, nearby objects, scene, brand, or product model.
Stage 2 - identity: compare IMAGE_A_CANDIDATE only with images explicitly labelled REFERENCE_IMAGE for known object cards. Never compare the colored context overlay with a reference image. Card text is prior context, not evidence that an unlabeled candidate is different.
Stage 3 - annotation: describe IMAGE_A_CANDIDATE only, after deciding identity. Use concise Chinese values and visible evidence only.
Two objects of the same category are not automatically the same instance. However, pixel-identical images or the same distinctive visible details are strong instance evidence and must not be rejected by inventing color, material, or shape differences. If evidence is weak or conflicting, answer uncertain rather than new.
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
Decision rules, in order:
1. invalid/non-object IMAGE_A_CANDIDATE -> ignored.
2. IMAGE_A_CANDIDATE matches a REFERENCE_IMAGE for one shown object card -> existing with that exact object_id.
3. evidence is insufficient, contradictory, or close to more than one card -> uncertain.
4. valid candidate and no shown card matches -> new. For one card batch, new means no match in this batch; the program accepts final new only after every batch returns new.
Do not claim a color, material, or shape mismatch unless that difference is directly visible between IMAGE_A_CANDIDATE and the relevant REFERENCE_IMAGE.
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
        "reference_view_count": len(card.representative_view_paths),
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
                "IMAGE_A_CANDIDATE follows. This crop is the current proposal, "
                "the only image to annotate, and the left-hand side of every "
                "identity comparison. "
                f"SAM category hint: {sam_prompt.strip() or 'unknown'}."
            ),
        },
        {"type": "image", "image": crop.as_uri(), "max_pixels": max_pixels},
    ]

    if not cards:
        content.append(
            {
                "type": "text",
                "text": (
                    "KNOWN_OBJECT_CARD_BATCH is empty. If IMAGE_A_CANDIDATE is "
                    "valid, decision must be new. Do not invent an object ID."
                ),
            }
        )
    for card_index, card in enumerate(cards, start=1):
        content.append(
            {
                "type": "text",
                "text": (
                    f"OBJECT_CARD_{card_index} begins. Its exact object_id and "
                    "stored attributes are:\n"
                    + json.dumps(_card_summary(card), ensure_ascii=False, sort_keys=True)
                    + "\nCompare IMAGE_A_CANDIDATE with only the REFERENCE_IMAGE "
                    "items explicitly assigned to this object_id."
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
                            f"REFERENCE_IMAGE_CARD_{card_index}_VIEW_{view_index} "
                            f"for object_id={card.object_id} follows immediately. "
                            "This is an identity reference, not the image to annotate."
                        ),
                    },
                    {
                        "type": "image",
                        "image": view.as_uri(),
                        "max_pixels": max_pixels,
                    },
                ]
            )

    content.extend(
        [
            {
                "type": "text",
                "text": (
                    "IMAGE_Z_CONTEXT_OVERLAY follows. It only shows where the SAM "
                    "mask lies in the source scene. Its colored mask changes visible "
                    "colors. Never compare IMAGE_Z_CONTEXT_OVERLAY with any "
                    "REFERENCE_IMAGE and never annotate it."
                ),
            },
            {"type": "image", "image": overlay.as_uri(), "max_pixels": max_pixels},
            {
                "type": "text",
                "text": (
                    "Now follow Stage 1 validity, Stage 2 identity using only "
                    "IMAGE_A_CANDIDATE versus REFERENCE_IMAGE items, and Stage 3 "
                    "annotation of IMAGE_A_CANDIDATE.\n" + OUTPUT_RULES
                ),
            },
        ]
    )
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
