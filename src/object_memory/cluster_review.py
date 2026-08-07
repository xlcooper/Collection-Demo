"""Batched Qwen semantic review for DINOv3 candidate clusters."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

from pydantic import ValidationError

from .candidate_clustering import CandidateCluster
from .config import MllmPipelineConfig
from .identity import MllmOutputError, extract_json_object
from .mllm_adapter import MllmPrediction
from .schemas import (
    ClusterReview,
    ClusterReviewResponse,
    ClusterVerdict,
    IdentityHypothesis,
    ObjectCard,
    VisualEvidence,
)


SYSTEM_PROMPT = """You are the semantic reviewer at the end of an object-memory pipeline.

SAM3 has already proposed class-agnostic masks and DINOv3 has already grouped visually similar proposals across source images. Each supplied board is one DINOv3 cluster hypothesis. The left image in every row is a mask-isolated appearance crop; the right image is the original scene with a thin location box.

For every cluster you must decide exactly one of:
- object: a coherent set of views of one complete independent physical object worth remembering;
- ignore: background, support, shadow, reflection, texture, fixed scene structure, redundant fragment, attached part when the complete object is visible, or another clearly invalid memory target;
- uncertain: the board is mixed, incomplete, ambiguous, or lacks enough evidence for a safe decision.

For an accepted object, name it in Chinese and create one accumulated object summary from all visible views. Describe intra-class identity features such as asymmetry, silhouette, proportions, component layout, texture, legible markings, and distinctive wear. Attach colors and materials to named parts. Do not infer brands or models that are not legible. For an existing object, return the complete updated summary: preserve compatible stable facts from its supplied card, add new visible stable facts, and remove facts contradicted by stronger evidence.

DINOv3 historical evidence is a retrieval hypothesis, not a semantic fact. Use existing only when the supplied historical evidence clearly matches the same active object and the visual/text evidence does not conflict. Same category, color, or material alone never proves instance identity. If memory is empty, accepted objects must be new.

Return one JSON object only. Use Chinese for descriptive values, keep keys and enum values exact, and do not return Markdown or hidden reasoning.
"""


class MllmPredictor(Protocol):
    def predict(self, messages: Sequence[dict[str, Any]]) -> MllmPrediction: ...


@dataclass(frozen=True, slots=True)
class ClusterReviewInput:
    cluster: CandidateCluster
    contact_sheet_path: Path
    historical_evidence: VisualEvidence


@dataclass(frozen=True, slots=True)
class ClusterReviewEvaluation:
    reviews: tuple[ClusterReview, ...]
    prediction: MllmPrediction
    object_card_count: int
    object_card_ids: tuple[str, ...]


def _output_rules(cluster_ids: Sequence[str]) -> str:
    return f"""Required JSON structure:
{{
  "reviews": [
    {{
      "cluster_id": "exact supplied cluster ID",
      "verdict": "object | ignore | uncertain",
      "identity_hypothesis": "new | existing | uncertain",
      "matched_object_id": "active object ID only for existing, otherwise null",
      "short_reason": "brief evidence-based Chinese reason",
      "object_summary": {{
        "object_name_zh": "instance-aware Chinese name",
        "coarse_category": "broad category",
        "fine_category": "specific category",
        "stable_description": "concise multi-view description",
        "stable_identity_features": ["stable intra-class feature"],
        "brand_or_markings": ["only clearly legible marking"],
        "part_appearance": [
          {{"part": "part name", "color": ["color"], "material": ["material"]}}
        ],
        "summary_confidence": 0.8
      }}
    }}
  ]
}}

Rules:
1. Cover every supplied cluster exactly once and no others: {json.dumps(list(cluster_ids))}.
2. object requires new or existing plus object_summary.
3. existing requires one exact supplied active object ID.
4. ignore and uncertain require identity_hypothesis=uncertain, null matched_object_id, and null object_summary.
5. If a cluster contains conflicting objects or uncertain whole/part boundaries, choose uncertain rather than forcing a memory object.
6. summary_confidence must reflect the visible evidence; do not copy the example value mechanically.
"""


def _card_payload(card: ObjectCard) -> dict[str, Any]:
    return {
        "object_id": card.object_id,
        "summary": card.summary.model_dump(mode="json"),
    }


def parse_cluster_review_response(
    raw_text: str,
    *,
    expected_cluster_ids: set[str],
    allowed_object_ids: set[str],
) -> tuple[ClusterReview, ...]:
    try:
        response = ClusterReviewResponse.model_validate(extract_json_object(raw_text))
    except (ValidationError, ValueError) as exc:
        raise MllmOutputError(
            f"Qwen cluster-review response failed schema validation: {exc}"
        ) from exc
    received = {review.cluster_id for review in response.reviews}
    if received != expected_cluster_ids or len(response.reviews) != len(
        expected_cluster_ids
    ):
        raise MllmOutputError(
            "Qwen cluster coverage does not match the requested batch; "
            f"expected={sorted(expected_cluster_ids)}, received={sorted(received)}"
        )
    for review in response.reviews:
        if (
            review.matched_object_id is not None
            and review.matched_object_id not in allowed_object_ids
        ):
            raise MllmOutputError(
                "Qwen referenced an object absent from the supplied cards: "
                f"{review.matched_object_id}"
            )
        if (
            not allowed_object_ids
            and review.verdict is ClusterVerdict.OBJECT
            and review.identity_hypothesis is not IdentityHypothesis.NEW
        ):
            raise MllmOutputError(
                "Accepted clusters must be new when no object cards were supplied"
            )
    return tuple(sorted(response.reviews, key=lambda review: review.cluster_id))


def build_cluster_review_messages(
    *,
    inputs: Sequence[ClusterReviewInput],
    cards: Sequence[ObjectCard],
    settings: MllmPipelineConfig,
) -> list[dict[str, Any]]:
    input_list = list(inputs)
    if not input_list:
        raise ValueError("At least one candidate cluster is required")
    if len(input_list) > settings.max_clusters_per_batch:
        raise ValueError("Cluster-review batch exceeds max_clusters_per_batch")
    cluster_ids = [item.cluster.id for item in input_list]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("Cluster IDs must be unique within one Qwen batch")
    card_list = list(cards)
    object_ids = [card.object_id for card in card_list]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("Object-card IDs must be unique")

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"ACTIVE_OBJECT_CARDS count={len(card_list)}:\n"
                + json.dumps(
                    [_card_payload(card) for card in card_list],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            ),
        }
    ]
    for item in input_list:
        board = item.contact_sheet_path.expanduser().resolve()
        if not board.is_file():
            raise FileNotFoundError(f"Cluster contact sheet not found: {board}")
        cluster = item.cluster
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        f"CLUSTER {cluster.id}:\n"
                        + json.dumps(
                            {
                                **cluster.report(),
                                "historical_visual_evidence": (
                                    item.historical_evidence.model_dump(mode="json")
                                ),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    ),
                },
                {
                    "type": "image",
                    "image": board.as_uri(),
                    "max_pixels": settings.max_pixels,
                },
            ]
        )
    content.append({"type": "text", "text": _output_rules(cluster_ids)})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def evaluate_cluster_reviews(
    predictor: MllmPredictor,
    *,
    inputs: Sequence[ClusterReviewInput],
    cards: Sequence[ObjectCard],
    settings: MllmPipelineConfig,
) -> ClusterReviewEvaluation:
    input_list = list(inputs)
    card_list = list(cards)
    prediction = predictor.predict(
        build_cluster_review_messages(
            inputs=input_list,
            cards=card_list,
            settings=settings,
        )
    )
    reviews = parse_cluster_review_response(
        prediction.raw_text,
        expected_cluster_ids={item.cluster.id for item in input_list},
        allowed_object_ids={card.object_id for card in card_list},
    )
    return ClusterReviewEvaluation(
        reviews=reviews,
        prediction=prediction,
        object_card_count=len(card_list),
        object_card_ids=tuple(card.object_id for card in card_list),
    )
