"""Two-stage candidate analysis, semantic retrieval, and identity confirmation."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from pydantic import ValidationError

from .assets import MemoryPaths
from .config import MllmPipelineConfig
from .mllm_adapter import MllmPrediction
from .schemas import (
    CandidateAnalysis,
    CandidateValidity,
    DecisionReasonCode,
    DecisionType,
    IdentityMatchResponse,
    MllmResponse,
    ObjectAnnotation,
    ObjectCard,
)


ANALYSIS_SYSTEM_PROMPT = """You are the candidate-analysis component of an object-memory system.
Analyze exactly one mask-isolated candidate without seeing or considering any memory object cards.
First decide whether it is one complete, independent physical object. Reject shadows, reflections, background regions, textures, merged groups, meaningless fragments, and object parts that are not independent objects.
If valid, create a concise temporary annotation using only visible evidence. Use Chinese annotation values. Do not decide whether the object is new or existing; memory comparison happens later.
Return exactly one JSON object with no Markdown or extra text.
"""


ANALYSIS_OUTPUT_RULES = """Required JSON structure:
{
  "validity": "valid | ignored",
  "confidence": 0.0,
  "reason_code": "valid_candidate | invalid_candidate",
  "short_reason": "brief reason",
  "annotation": {
    "coarse_category": "physical object category",
    "fine_category": "more specific category or unknown",
    "material": ["visible material"],
    "color": ["visible color"],
    "shape": "visible shape",
    "description": "visible facts and distinctive instance cues",
    "annotation_confidence": 0.0
  }
}
For valid, reason_code must be valid_candidate and annotation is required.
For ignored, reason_code must be invalid_candidate and annotation must be null.
Do not use new, existing, or uncertain in this stage.
"""


IDENTITY_SYSTEM_PROMPT = """You are the identity-confirmation component of an object-memory system.
The candidate has already been confirmed as a complete physical object and has a temporary annotation. It can never be ignored in this stage.
Compare the candidate with only the shortlisted memory object cards shown in this request. Use card text for semantic comparison and REFERENCE_IMAGE items for concrete instance evidence.
Same category, color, or material alone does not prove that two observations are the same physical instance. Distinctive visible details or matching reference views are stronger evidence.
Return existing only for one supported instance match, new when no shown card matches, and uncertain when evidence is weak, conflicting, or ambiguous.
Return exactly one JSON object with no Markdown or extra text.
"""


IDENTITY_OUTPUT_RULES = """Required JSON structure:
{
  "decision": "new | existing | uncertain",
  "matched_object_id": "known object ID when existing, otherwise null",
  "confidence": 0.0,
  "reason_code": "new_object | visual_instance_match | ambiguous_match | insufficient_evidence",
  "short_reason": "brief reason"
}
Decision rules:
1. one supported instance match -> existing with that exact object_id.
2. weak, conflicting, or ambiguous identity evidence -> uncertain.
3. no shown card matches -> new.
ignored and invalid_candidate are forbidden because validity was resolved before memory lookup.
Reason codes must agree with the decision.
"""


class MllmOutputError(ValueError):
    """Raised when a model response is not safe to use."""


class MllmPredictor(Protocol):
    def predict(self, messages: Sequence[dict[str, Any]]) -> MllmPrediction: ...


@dataclass(frozen=True, slots=True)
class RetrievedObjectCard:
    card: ObjectCard
    score: float
    matched_fields: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.card.object_id,
            "score": self.score,
            "matched_fields": list(self.matched_fields),
        }


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    analysis: CandidateAnalysis
    final_response: MllmResponse
    analysis_prediction: MllmPrediction
    memory_lookup_performed: bool
    available_object_cards: int
    retrieved_cards: tuple[RetrievedObjectCard, ...] = ()
    identity_response: IdentityMatchResponse | None = None
    identity_prediction: MllmPrediction | None = None

    @property
    def predictions(self) -> tuple[MllmPrediction, ...]:
        if self.identity_prediction is None:
            return (self.analysis_prediction,)
        return (self.analysis_prediction, self.identity_prediction)


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


def parse_candidate_analysis(raw_text: str) -> CandidateAnalysis:
    """Validate the memory-independent candidate analysis."""

    try:
        return CandidateAnalysis.model_validate(extract_json_object(raw_text))
    except (ValidationError, ValueError) as exc:
        raise MllmOutputError(
            f"Qwen candidate analysis failed schema validation: {exc}"
        ) from exc


def parse_identity_response(
    raw_text: str,
    *,
    allowed_object_ids: set[str],
) -> IdentityMatchResponse:
    """Validate identity output and reject object IDs outside the shortlist."""

    try:
        response = IdentityMatchResponse.model_validate(
            extract_json_object(raw_text)
        )
    except (ValidationError, ValueError) as exc:
        raise MllmOutputError(
            f"Qwen identity response failed schema validation: {exc}"
        ) from exc

    if (
        response.decision is DecisionType.EXISTING
        and response.matched_object_id not in allowed_object_ids
    ):
        raise MllmOutputError(
            "Qwen matched an object ID that was not present in the shortlist"
        )
    return response


def _candidate_origin(sam_prompt: str) -> str:
    normalized_source = sam_prompt.strip()
    if normalized_source == "automatic_point_grid":
        return (
            "No category hint was supplied. The project generated this candidate "
            "automatically from a point grid; infer its category only from pixels."
        )
    return f"Historical SAM category hint: {normalized_source or 'unknown'}."


def build_candidate_analysis_messages(
    *,
    candidate_crop: Path,
    candidate_overlay: Path,
    sam_prompt: str,
    max_pixels: int,
) -> list[dict[str, Any]]:
    """Build the first call: validity and a temporary visible annotation."""

    crop = candidate_crop.expanduser().resolve()
    overlay = candidate_overlay.expanduser().resolve()
    if not crop.is_file() or not overlay.is_file():
        raise FileNotFoundError("Candidate crop and overlay must both exist")
    if max_pixels <= 0:
        raise ValueError("MLLM image limits must be positive")

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "IMAGE_A_CANDIDATE follows. This mask-isolated crop is the only "
                "annotation target. Pixels outside the SAM mask were replaced with "
                "uniform neutral gray; judge only the preserved mask pixels. "
                + _candidate_origin(sam_prompt)
            ),
        },
        {"type": "image", "image": crop.as_uri(), "max_pixels": max_pixels},
        {
            "type": "text",
            "text": (
                "IMAGE_Z_CONTEXT_OVERLAY follows. It only shows the candidate's "
                "location in the source scene. Its colored mask changes visible "
                "colors; use it only to understand boundaries and context, never "
                "as the annotation target."
            ),
        },
        {"type": "image", "image": overlay.as_uri(), "max_pixels": max_pixels},
        {
            "type": "text",
            "text": (
                "Decide candidate validity first. If valid, produce a temporary "
                "annotation with visible category, attributes, and distinctive "
                "instance cues. Do not compare with memory in this call.\n"
                + ANALYSIS_OUTPUT_RULES
            ),
        },
    ]
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": ANALYSIS_SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
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


def build_identity_confirmation_messages(
    *,
    candidate_crop: Path,
    temporary_annotation: ObjectAnnotation,
    cards: Sequence[ObjectCard],
    card_assets: MemoryPaths | None,
    max_reference_views_per_object: int,
    max_pixels: int,
) -> list[dict[str, Any]]:
    """Build the second call for a semantically retrieved object-card shortlist."""

    crop = candidate_crop.expanduser().resolve()
    if not crop.is_file():
        raise FileNotFoundError(f"Candidate crop not found: {crop}")
    if not cards:
        raise ValueError("Identity confirmation requires at least one object card")
    if max_reference_views_per_object <= 0 or max_pixels <= 0:
        raise ValueError("MLLM image limits must be positive")

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "VALID_IMAGE_A_CANDIDATE follows. It already passed independent "
                "validity analysis. Its temporary annotation is:\n"
                + json.dumps(
                    temporary_annotation.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\nCompare this valid candidate with the shortlisted cards below."
            ),
        },
        {"type": "image", "image": crop.as_uri(), "max_pixels": max_pixels},
    ]

    for card_index, card in enumerate(cards, start=1):
        content.append(
            {
                "type": "text",
                "text": (
                    f"SHORTLISTED_OBJECT_CARD_{card_index} begins:\n"
                    + json.dumps(
                        _card_summary(card),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\nIts REFERENCE_IMAGE items, when present, are concrete "
                    "identity evidence for this exact object_id."
                ),
            }
        )
        view_paths = card.representative_view_paths[
            :max_reference_views_per_object
        ]
        if view_paths and card_assets is None:
            raise ValueError("card_assets is required for representative views")
        if not view_paths:
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"OBJECT_CARD_{card_index} has no reference image. Its text "
                        "alone is not sufficient for a confident exact-instance match."
                    ),
                }
            )
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
                        "max_pixels": max_pixels,
                    },
                ]
            )

    content.append(
        {
            "type": "text",
            "text": (
                "Compare temporary text semantics first, then use reference images "
                "to confirm or reject exact instance identity.\n"
                + IDENTITY_OUTPUT_RULES
            ),
        }
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": IDENTITY_SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
    ]


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+", normalized))


def _text_features(value: str) -> set[str]:
    normalized = _normalize_text(value)
    features = set(re.findall(r"[a-z0-9]+", normalized))
    for run in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", normalized):
        features.add(run)
        features.update(run)
        features.update(run[index : index + 2] for index in range(len(run) - 1))
    return {feature for feature in features if feature}


def _text_similarity(left: str, right: str) -> float:
    left_normalized = _normalize_text(left)
    right_normalized = _normalize_text(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0
    if left_normalized in right_normalized or right_normalized in left_normalized:
        return 0.8
    left_features = _text_features(left)
    right_features = _text_features(right)
    union = left_features | right_features
    if not union:
        return 0.0
    return len(left_features & right_features) / len(union)


def _list_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    return _text_similarity(" ".join(left), " ".join(right))


def _card_relevance(
    annotation: ObjectAnnotation,
    card: ObjectCard,
) -> tuple[float, tuple[str, ...]]:
    weighted_fields = (
        (
            "coarse_category",
            3.0,
            _text_similarity(annotation.coarse_category, card.coarse_category),
        ),
        (
            "fine_category",
            4.0,
            _text_similarity(annotation.fine_category, card.fine_category),
        ),
        (
            "material",
            1.0,
            _list_similarity(annotation.material, card.material),
        ),
        (
            "color",
            1.0,
            _list_similarity(annotation.color, card.color),
        ),
        (
            "shape",
            1.5,
            _text_similarity(annotation.shape, card.shape),
        ),
        (
            "description",
            2.5,
            _text_similarity(annotation.description, card.description),
        ),
    )
    total_weight = sum(weight for _, weight, _ in weighted_fields)
    score = sum(weight * similarity for _, weight, similarity in weighted_fields)
    matched_fields = tuple(
        field_name
        for field_name, _, similarity in weighted_fields
        if similarity > 0.0
    )
    return round(score / total_weight, 6), matched_fields


def retrieve_object_cards(
    annotation: ObjectAnnotation,
    cards: Sequence[ObjectCard],
    *,
    limit: int,
) -> tuple[RetrievedObjectCard, ...]:
    """Rank all card text locally and retain only a bounded visual shortlist."""

    if limit <= 0:
        raise ValueError("object-card shortlist limit must be positive")
    object_ids = [card.object_id for card in cards]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("object-card IDs must be unique")

    ranked: list[RetrievedObjectCard] = []
    for card in cards:
        score, matched_fields = _card_relevance(annotation, card)
        ranked.append(
            RetrievedObjectCard(
                card=card,
                score=score,
                matched_fields=matched_fields,
            )
        )
    ranked.sort(key=lambda item: (-item.score, item.card.object_id))
    return tuple(ranked[:limit])


def _final_from_analysis(
    analysis: CandidateAnalysis,
    identity: IdentityMatchResponse | None,
) -> MllmResponse:
    if analysis.validity is CandidateValidity.IGNORED:
        return MllmResponse(
            decision=DecisionType.IGNORED,
            matched_object_id=None,
            confidence=analysis.confidence,
            reason_code=DecisionReasonCode.INVALID_CANDIDATE,
            short_reason=analysis.short_reason,
            annotation=None,
        )

    assert analysis.annotation is not None
    if identity is None:
        return MllmResponse(
            decision=DecisionType.NEW,
            matched_object_id=None,
            confidence=analysis.confidence,
            reason_code=DecisionReasonCode.NEW_OBJECT,
            short_reason="候选是有效物体，且当前记忆库为空",
            annotation=analysis.annotation,
        )
    return MllmResponse(
        decision=identity.decision,
        matched_object_id=identity.matched_object_id,
        confidence=min(analysis.confidence, identity.confidence),
        reason_code=identity.reason_code,
        short_reason=identity.short_reason,
        annotation=analysis.annotation,
    )


def evaluate_candidate(
    predictor: MllmPredictor,
    *,
    candidate_crop: Path,
    candidate_overlay: Path,
    sam_prompt: str,
    get_card_texts: Callable[[], Sequence[ObjectCard]],
    get_reference_cards: Callable[[Sequence[str]], Sequence[ObjectCard]],
    card_assets: MemoryPaths | None,
    settings: MllmPipelineConfig,
) -> CandidateEvaluation:
    """Analyze one candidate, retrieve card text, then confirm identity visually."""

    analysis_prediction = predictor.predict(
        build_candidate_analysis_messages(
            candidate_crop=candidate_crop,
            candidate_overlay=candidate_overlay,
            sam_prompt=sam_prompt,
            max_pixels=settings.max_pixels,
        )
    )
    analysis = parse_candidate_analysis(analysis_prediction.raw_text)
    if analysis.validity is CandidateValidity.IGNORED:
        return CandidateEvaluation(
            analysis=analysis,
            final_response=_final_from_analysis(analysis, None),
            analysis_prediction=analysis_prediction,
            memory_lookup_performed=False,
            available_object_cards=0,
        )

    card_texts = list(get_card_texts())
    if not card_texts:
        return CandidateEvaluation(
            analysis=analysis,
            final_response=_final_from_analysis(analysis, None),
            analysis_prediction=analysis_prediction,
            memory_lookup_performed=True,
            available_object_cards=0,
        )

    assert analysis.annotation is not None
    retrieved = retrieve_object_cards(
        analysis.annotation,
        card_texts,
        limit=settings.object_card_shortlist_size,
    )
    selected_ids = [item.card.object_id for item in retrieved]
    hydrated_cards = list(get_reference_cards(selected_ids))
    hydrated_by_id = {card.object_id: card for card in hydrated_cards}
    if set(hydrated_by_id) != set(selected_ids):
        raise ValueError("reference-card query did not return the complete shortlist")
    shortlisted_cards = [hydrated_by_id[object_id] for object_id in selected_ids]
    retrieved = tuple(
        RetrievedObjectCard(
            card=hydrated_by_id[item.card.object_id],
            score=item.score,
            matched_fields=item.matched_fields,
        )
        for item in retrieved
    )
    identity_prediction = predictor.predict(
        build_identity_confirmation_messages(
            candidate_crop=candidate_crop,
            temporary_annotation=analysis.annotation,
            cards=shortlisted_cards,
            card_assets=card_assets,
            max_reference_views_per_object=settings.max_reference_views_per_object,
            max_pixels=settings.max_pixels,
        )
    )
    identity = parse_identity_response(
        identity_prediction.raw_text,
        allowed_object_ids={card.object_id for card in shortlisted_cards},
    )
    if (
        identity.decision is DecisionType.EXISTING
        and identity.confidence < settings.existing_min_confidence
    ):
        identity = IdentityMatchResponse(
            decision=DecisionType.UNCERTAIN,
            matched_object_id=None,
            confidence=identity.confidence,
            reason_code=DecisionReasonCode.INSUFFICIENT_EVIDENCE,
            short_reason="候选可能匹配已有对象，但身份置信度低于阈值",
        )

    return CandidateEvaluation(
        analysis=analysis,
        final_response=_final_from_analysis(analysis, identity),
        analysis_prediction=analysis_prediction,
        memory_lookup_performed=True,
        available_object_cards=len(card_texts),
        retrieved_cards=retrieved,
        identity_response=identity,
        identity_prediction=identity_prediction,
    )
