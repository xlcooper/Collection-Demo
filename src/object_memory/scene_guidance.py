"""Single Qwen call for discovery, SAM guidance, and text-memory iteration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

from pydantic import ValidationError

from .config import MllmPipelineConfig
from .identity import MllmOutputError, extract_json_object
from .mllm_adapter import MllmPrediction
from .schemas import (
    IdentityHypothesis,
    ObjectCard,
    SceneGuidanceResponse,
    SceneImageGuidance,
)


SYSTEM_PROMPT = """You are the only vision-language call for one image in a robot-oriented object-memory system.

You receive one current source image and the compact text summaries of all active memory objects. In this one response you must:
1. discover every visible complete independent physical object worth remembering;
2. emit a concise English whole-object category for SAM3;
3. describe current visible identity evidence in Chinese;
4. make a text-based new/existing/uncertain identity hypothesis against the supplied summaries;
5. propose the single updated object summary that may be committed only if later DINOv3 visual evidence agrees.

Discovery and SAM3 rules:
- Prefer complete movable, graspable, detachable, task-relevant, or identity-worthy objects.
- Exclude supports, walls, fixed structures, shadows, reflections, textures, people, animals, merged groups, fragments, and attached parts when the complete object is visible.
- Give every visible instance a separate target, even when multiple instances share the same sam_text_prompt.
- sam_text_prompt is only retrieval metadata. Use a lowercase concrete base category or one stable useful modifier plus that category. Do not use positions, alternatives, feature clauses, or lists of attributes.
- A bottle-shaped vessel remains `water bottle`; a wide-rim cup or tumbler is `drink cup`. Name the same complete object in Chinese and English.

Text-memory rules:
- Category-level facts are not enough for instance identity. Examine intra-class differences: asymmetry, silhouette, proportions, component layout, texture, markings, visible brand/model text, and distinctive wear or damage.
- Record a brand, model, or marking only when it is visibly legible; never infer it from shape.
- Attach color and material to a named part in part_appearance. Do not output flat object color/material lists.
- current_view_facts contains only facts visible now. A feature not visible in this view is not evidence that it disappeared.
- proposed_object_summary is the one complete accumulated card after accepting this view. For existing, preserve supported old facts and add only non-conflicting visible evidence. Never add object IDs, match claims, tables, nearby objects, positions, reflections, or annotation-box colors.
- Same category, color, or material alone never proves the same physical instance. If the text evidence cannot distinguish plausible objects, use uncertain.
- When memory is empty, every selected target must use new. Otherwise existing must name exactly one supplied object_id.

Return one JSON object only. Use Chinese for descriptive values, keep keys and enum values exact, and do not return Markdown or hidden reasoning.
"""


def _output_rules(max_targets: int) -> str:
    return f"""Required JSON structure:
{{
  "image": {{
    "source_id": "exact supplied source ID",
    "scene_summary": "brief Chinese scene summary",
    "targets": [
      {{
        "target_id": "target_001",
        "object_name_zh": "instance-aware Chinese object name",
        "sam_text_prompt": "short lowercase English noun phrase",
        "current_view_facts": {{
          "category": "visible category",
          "visible_identity_features": ["visible intra-class feature"],
          "brand_or_markings": ["only clearly legible marking"],
          "part_appearance": [
            {{"part": "part name", "color": ["color"], "material": ["material"]}}
          ]
        }},
        "identity_hypothesis": "new | existing | uncertain",
        "matched_object_id": "supplied object ID only for existing, otherwise null",
        "identity_short_reason": "brief visible-evidence reason",
        "proposed_object_summary": {{
          "object_name_zh": "current accumulated object name",
          "coarse_category": "broad category",
          "fine_category": "specific category",
          "stable_description": "concise accumulated description with intra-class cues",
          "stable_identity_features": ["stable distinguishing feature"],
          "brand_or_markings": ["confirmed marking"],
          "part_appearance": [
            {{"part": "part name", "color": ["color"], "material": ["material"]}}
          ],
          "summary_confidence": 0.0
        }},
        "temporary_target_anchor": {{
          "x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0
        }}
      }}
    ],
    "no_target_reason": null
  }}
}}

Rules:
1. Return the supplied source_id exactly and at most {max_targets} targets.
2. target_id values are unique. Duplicate sam_text_prompt values are allowed for different visible instances.
3. temporary_target_anchor is a tight normalized [0,1] box around this instance and is used only for target-to-mask association.
4. existing requires one exact supplied object_id. new and uncertain require matched_object_id null.
5. When no memory cards are supplied, every target is new.
6. If no qualifying target is visible, targets is empty and no_target_reason is non-empty; otherwise no_target_reason is null.
7. proposed_object_summary is required for every target but will be committed only after the final identity decision.
"""


class MllmPredictor(Protocol):
    def predict(self, messages: Sequence[dict[str, Any]]) -> MllmPrediction: ...


@dataclass(frozen=True, slots=True)
class SceneImageInput:
    source_id: str
    image_path: Path


@dataclass(frozen=True, slots=True)
class SceneGuidanceEvaluation:
    response: SceneImageGuidance
    prediction: MllmPrediction
    object_card_count: int
    object_card_ids: tuple[str, ...]


def _card_payload(card: ObjectCard) -> dict[str, Any]:
    return {
        "object_id": card.object_id,
        "summary": card.summary.model_dump(mode="json"),
    }


def parse_scene_guidance_response(
    raw_text: str,
    *,
    expected_source_id: str,
    allowed_object_ids: set[str],
    max_targets_per_image: int,
) -> SceneImageGuidance:
    """Validate exact image coverage and all text-memory references."""

    try:
        response = SceneGuidanceResponse.model_validate(extract_json_object(raw_text))
    except (ValidationError, ValueError) as exc:
        raise MllmOutputError(
            f"Qwen single-pass response failed schema validation: {exc}"
        ) from exc
    image = response.image
    if image.source_id != expected_source_id:
        raise MllmOutputError(
            "Qwen source coverage does not match the request; "
            f"expected={expected_source_id}, received={image.source_id}"
        )
    if len(image.targets) > max_targets_per_image:
        raise MllmOutputError(
            f"Qwen returned too many targets for {expected_source_id}"
        )
    for target in image.targets:
        if not allowed_object_ids and target.identity_hypothesis is not IdentityHypothesis.NEW:
            raise MllmOutputError(
                "Every target must be new when no object cards were supplied: "
                f"{target.target_id}"
            )
        if (
            target.matched_object_id is not None
            and target.matched_object_id not in allowed_object_ids
        ):
            raise MllmOutputError(
                "Qwen referenced an object absent from the supplied cards: "
                f"{target.matched_object_id}"
            )
    return image


def build_scene_guidance_messages(
    *,
    image: SceneImageInput,
    cards: Sequence[ObjectCard],
    settings: MllmPipelineConfig,
) -> list[dict[str, Any]]:
    """Build the one permitted Qwen request for a source image."""

    image_path = image.image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Scene source image not found: {image_path}")
    card_list = list(cards)
    object_ids = [card.object_id for card in card_list]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("Object-card IDs must be unique")
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"CURRENT_SOURCE source_id={image.source_id}. "
                "The following image is the only current visual input."
            ),
        },
        {
            "type": "image",
            "image": image_path.as_uri(),
            "max_pixels": settings.max_pixels,
        },
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
        },
        {
            "type": "text",
            "text": _output_rules(settings.max_scene_targets_per_image),
        },
    ]
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def evaluate_scene_guidance(
    predictor: MllmPredictor,
    *,
    image: SceneImageInput,
    cards: Sequence[ObjectCard],
    settings: MllmPipelineConfig,
) -> SceneGuidanceEvaluation:
    """Run the one Qwen call and validate its object-memory contract."""

    card_list = list(cards)
    prediction = predictor.predict(
        build_scene_guidance_messages(image=image, cards=card_list, settings=settings)
    )
    response = parse_scene_guidance_response(
        prediction.raw_text,
        expected_source_id=image.source_id,
        allowed_object_ids={card.object_id for card in card_list},
        max_targets_per_image=settings.max_scene_targets_per_image,
    )
    return SceneGuidanceEvaluation(
        response=response,
        prediction=prediction,
        object_card_count=len(card_list),
        object_card_ids=tuple(card.object_id for card in card_list),
    )
