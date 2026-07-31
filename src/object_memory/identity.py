"""Image-level Qwen reasoning over all candidates and all active memory cards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from pydantic import ValidationError

from .assets import MemoryPaths
from .config import MllmPipelineConfig
from .mllm_adapter import MllmPrediction
from .schemas import ImageBatchResponse, ObjectCard


BATCH_SYSTEM_PROMPT = """You are the detailed visual reasoning component of a robot-oriented object-memory system.
You receive every retained SAM candidate from one source image, followed by all active memory object cards and their available reference images.

Each candidate was retrieved from a fallible first-pass scene-survey concept. Its SAM text prompt is retrieval metadata, not ground truth. Never accept, label, or match a candidate merely because the upstream concept names that object.

For every candidate, follow this order:
1. Judge candidate validity from its mask-isolated crop and context overlay. You may compare candidates from this source image to reject redundant masks or non-independent parts, but do not use memory-card similarity to decide validity.
2. If valid, create a structured temporary annotation from the candidate's visible evidence.
3. Compare that temporary annotation and candidate image with every supplied object card and its reference images.
4. Decide new, existing, or uncertain, then produce a final annotation updated with the useful evidence from comparison.

Reject shadows, reflections, background regions, support surfaces, fixed structural components or built-in controls, textures, merged groups, meaningless fragments, redundant masks of another complete candidate, and non-independent object parts. A correctly segmented region can still be outside the robot-arm object-memory scope. Same category, color, material, upstream concept, or spatial proximity alone never proves that two observations are the same physical instance. Reference images and distinctive instance details are stronger evidence.
Use Chinese for annotation values and brief reason text, while keeping JSON keys and enum values exactly as specified.
Return one JSON object covering every supplied candidate exactly once. Do not return Markdown, extra text, or hidden chain-of-thought; use only brief reason fields.
"""


def _output_rules(existing_min_confidence: float) -> str:
    return f"""Required JSON structure:
{{
  "candidates": [
    {{
      "proposal_id": "exact supplied proposal ID",
      "validity": "valid | ignored",
      "validity_confidence": 0.0,
      "validity_reason_code": "valid_candidate | invalid_candidate",
      "validity_short_reason": "brief visible-evidence reason",
      "temporary_annotation": {{
        "coarse_category": "physical object category",
        "fine_category": "more specific category or unknown",
        "material": ["visible material"],
        "color": ["visible color"],
        "shape": "visible shape",
        "description": "visible facts and distinctive instance cues",
        "annotation_confidence": 0.0
      }},
      "decision": "new | existing | ignored | uncertain",
      "matched_object_id": "supplied object ID only when existing, otherwise null",
      "confidence": 0.0,
      "reason_code": "new_object | visual_instance_match | invalid_candidate | ambiguous_match | insufficient_evidence",
      "short_reason": "brief final reason",
      "final_annotation": {{
        "coarse_category": "final object category",
        "fine_category": "final specific category",
        "material": ["final material knowledge"],
        "color": ["final color knowledge"],
        "shape": "final shape knowledge",
        "description": "concise accumulated object description",
        "annotation_confidence": 0.0
      }}
    }}
  ]
}}

Rules:
1. Output every supplied proposal_id exactly once and do not invent IDs.
2. ignored requires invalid_candidate; temporary_annotation, matched_object_id, and final_annotation must be null.
3. valid requires both temporary_annotation and final_annotation, and decision must be new, existing, or uncertain.
4. existing requires one exact supplied object_id, visual_instance_match, and confidence >= {existing_min_confidence:.2f}.
5. new requires no matching supplied card, matched_object_id null, and new_object.
6. uncertain requires matched_object_id null and ambiguous_match or insufficient_evidence.
7. For new, final_annotation refines the temporary annotation using only visible evidence.
8. For existing, final_annotation is the updated cumulative object-card annotation: preserve supported stable facts from the matched card and add useful current-view evidence without inventing facts.
9. For uncertain, final_annotation describes the current valid candidate but must not borrow uncertain identity facts.
10. When no memory cards are supplied, every valid candidate must be new.
"""


class MllmOutputError(ValueError):
    """Raised when a model response is not safe to persist."""


class MllmPredictor(Protocol):
    def predict(self, messages: Sequence[dict[str, Any]]) -> MllmPrediction: ...


@dataclass(frozen=True, slots=True)
class BatchCandidateInput:
    """One retained SAM candidate included in an image-level Qwen call."""

    proposal_id: str
    crop_path: Path
    overlay_path: Path
    sam_prompt: str


@dataclass(frozen=True, slots=True)
class ImageBatchEvaluation:
    """Validated result and context summary for one source-image call."""

    response: ImageBatchResponse
    prediction: MllmPrediction
    object_card_count: int
    object_card_ids: tuple[str, ...]
    reference_image_count: int


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


def parse_image_batch_response(
    raw_text: str,
    *,
    expected_proposal_ids: Sequence[str],
    allowed_object_ids: set[str],
    existing_min_confidence: float,
) -> ImageBatchResponse:
    """Validate complete candidate coverage and memory-object references."""

    expected = list(expected_proposal_ids)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected proposal IDs must be non-empty and unique")
    try:
        response = ImageBatchResponse.model_validate(extract_json_object(raw_text))
    except (ValidationError, ValueError) as exc:
        raise MllmOutputError(
            f"Qwen image-batch response failed schema validation: {exc}"
        ) from exc

    received = [item.proposal_id for item in response.candidates]
    if len(received) != len(set(received)):
        raise MllmOutputError("Qwen returned a proposal ID more than once")
    if set(received) != set(expected):
        missing = sorted(set(expected) - set(received))
        unexpected = sorted(set(received) - set(expected))
        raise MllmOutputError(
            "Qwen candidate coverage does not match the request; "
            f"missing={missing}, unexpected={unexpected}"
        )

    for item in response.candidates:
        if (
            not allowed_object_ids
            and item.validity.value == "valid"
            and item.decision.value != "new"
        ):
            raise MllmOutputError(
                "A valid candidate must be new when no memory cards were supplied: "
                f"{item.proposal_id}"
            )
        if item.matched_object_id is not None:
            if item.matched_object_id not in allowed_object_ids:
                raise MllmOutputError(
                    "Qwen matched an object ID absent from the supplied cards: "
                    f"{item.matched_object_id}"
                )
            if item.confidence < existing_min_confidence:
                raise MllmOutputError(
                    "Qwen existing confidence is below the configured threshold: "
                    f"{item.proposal_id}"
                )
    return response


def _candidate_origin(sam_prompt: str) -> str:
    normalized_source = sam_prompt.strip()
    return (
        "First-pass scene-survey SAM text prompt="
        f"{normalized_source or 'unknown'!r}. Treat it only as a retrieval hypothesis "
        "and verify the candidate independently from visible pixels."
    )


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


def build_image_batch_messages(
    *,
    candidates: Sequence[BatchCandidateInput],
    cards: Sequence[ObjectCard],
    card_assets: MemoryPaths | None,
    settings: MllmPipelineConfig,
) -> tuple[list[dict[str, Any]], int]:
    """Build one call containing all source-image candidates and memory cards."""

    candidate_list = list(candidates)
    card_list = list(cards)
    if not candidate_list:
        raise ValueError("Image-batch reasoning requires at least one candidate")
    proposal_ids = [candidate.proposal_id for candidate in candidate_list]
    if len(proposal_ids) != len(set(proposal_ids)):
        raise ValueError("Candidate proposal IDs must be unique")
    object_ids = [card.object_id for card in card_list]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("Object-card IDs must be unique")
    if settings.max_reference_views_per_object <= 0 or settings.max_pixels <= 0:
        raise ValueError("MLLM image limits must be positive")

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"CANDIDATE_SET begins. It contains {len(candidate_list)} retained "
                "SAM candidates from the same source image. Analyze each candidate "
                "independently for validity before using memory evidence."
            ),
        }
    ]
    for index, candidate in enumerate(candidate_list, start=1):
        crop = candidate.crop_path.expanduser().resolve()
        overlay = candidate.overlay_path.expanduser().resolve()
        if not crop.is_file() or not overlay.is_file():
            raise FileNotFoundError(
                f"Candidate crop and overlay must exist: {candidate.proposal_id}"
            )
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        f"CANDIDATE_{index} proposal_id={candidate.proposal_id}. "
                        "MASK_ISOLATED_CROP follows. It is the annotation target; "
                        "pixels outside the SAM mask are neutral gray. Judge its "
                        "visible pixels before reading any retrieval metadata."
                    ),
                },
                {
                    "type": "image",
                    "image": crop.as_uri(),
                    "max_pixels": settings.max_pixels,
                },
                {
                    "type": "text",
                    "text": (
                        f"CANDIDATE_{index}_CONTEXT_OVERLAY follows. Use it only "
                        "for boundaries and scene position; its colored mask is not "
                        "the target and must not determine object color."
                    ),
                },
                {
                    "type": "image",
                    "image": overlay.as_uri(),
                    "max_pixels": settings.max_pixels,
                },
                {
                    "type": "text",
                    "text": (
                        f"CANDIDATE_{index}_RETRIEVAL_METADATA: "
                        + _candidate_origin(candidate.sam_prompt)
                    ),
                },
            ]
        )

    content.append(
        {
            "type": "text",
            "text": (
                f"MEMORY_OBJECT_CARDS begins. Exactly {len(card_list)} active cards "
                "are supplied. No script-side similarity ranking or semantic "
                "shortlist was applied. Compare every valid candidate with every card."
            ),
        }
    )
    reference_image_count = 0
    for card_index, card in enumerate(card_list, start=1):
        content.append(
            {
                "type": "text",
                "text": (
                    f"OBJECT_CARD_{card_index}:\n"
                    + json.dumps(
                        _card_summary(card),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                ),
            }
        )
        view_paths = card.representative_view_paths[
            : settings.max_reference_views_per_object
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
                            f"for object_id={card.object_id} follows."
                        ),
                    },
                    {
                        "type": "image",
                        "image": view.as_uri(),
                        "max_pixels": settings.max_pixels,
                    },
                ]
            )
            reference_image_count += 1

    if not card_list:
        content.append(
            {
                "type": "text",
                "text": (
                    "The memory is empty. Valid candidates must be classified as new."
                ),
            }
        )
    content.append(
        {
            "type": "text",
            "text": (
                "Now return the complete structured result. The presence or absence "
                "of a matching card must never change whether a candidate is a real, "
                "independent physical object.\n"
                + _output_rules(settings.existing_min_confidence)
            ),
        }
    )
    return (
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": BATCH_SYSTEM_PROMPT}],
            },
            {"role": "user", "content": content},
        ],
        reference_image_count,
    )


def evaluate_image_batch(
    predictor: MllmPredictor,
    *,
    candidates: Sequence[BatchCandidateInput],
    cards: Sequence[ObjectCard],
    card_assets: MemoryPaths | None,
    settings: MllmPipelineConfig,
) -> ImageBatchEvaluation:
    """Run and validate one source-image candidate/memory reasoning call."""

    candidate_list = list(candidates)
    card_list = list(cards)
    messages, reference_image_count = build_image_batch_messages(
        candidates=candidate_list,
        cards=card_list,
        card_assets=card_assets,
        settings=settings,
    )
    prediction = predictor.predict(messages)
    response = parse_image_batch_response(
        prediction.raw_text,
        expected_proposal_ids=[
            candidate.proposal_id for candidate in candidate_list
        ],
        allowed_object_ids={card.object_id for card in card_list},
        existing_min_confidence=settings.existing_min_confidence,
    )
    return ImageBatchEvaluation(
        response=response,
        prediction=prediction,
        object_card_count=len(card_list),
        object_card_ids=tuple(card.object_id for card in card_list),
        reference_image_count=reference_image_count,
    )
