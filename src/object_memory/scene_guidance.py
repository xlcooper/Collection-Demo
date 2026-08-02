"""First-pass Qwen scene survey for robot-oriented SAM3 text guidance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from pydantic import ValidationError

from .config import MllmPipelineConfig
from .identity import MllmOutputError, extract_json_object
from .mllm_adapter import MllmPrediction
from .schemas import SceneGuidanceBatchResponse


SCENE_GUIDANCE_SYSTEM_PROMPT = """You are the scene-survey planner for autonomous object-memory acquisition by a robot arm.

Each supplied image is one viewpoint of a specific task workspace or an unfamiliar operational scene. This first pass does not segment pixels, compare persistent memory, identify cross-image instances, or decide new/existing. Its only job is to decide which visible physical object concepts deserve targeted SAM3 segmentation and later detailed inspection.

Select independent physical objects whose appearance, shape, state, identity, or affordance could support later recognition, multi-view memory, grasping, manipulation, task planning, or change detection.

Prioritize:
- free-standing, movable, graspable, detachable, or otherwise task-interactable objects;
- tools, containers, consumables, portable control devices, and loose components in the working area;
- objects with distinctive identity or state that benefits from closer or multi-view observation;
- unfamiliar, transparent, partially occluded, or category-uncertain regions when they still plausibly form one independent physical object.

Prefer the complete object. Do not separately select an attached cap, handle, label, button, straw, or other part. Select such a part only when it is visibly detached and independently manipulable.
When an attached part is the most visible region of a partially visible object, name the whole object concept. A lid and straw attached to a cup must be returned as a `drink cup`, not as a lid or straw. The Chinese object name and the SAM3 text must refer to that same complete object.

Exclude:
- floors, walls, desktops, shelves, support boards, structural partitions, and other scene-support surfaces;
- decoration, fixed connectors, installed fasteners, permanently routed cables, and robot/camera hardware;
- shadows, reflections, lighting, textures, printed patterns, screen contents, stains, and empty regions;
- merged multi-object regions, meaningless fragments, and duplicate descriptions of the same visible object;
- people and animals.

Recall policy:
This stage gates downstream discovery. Do not restrict selection to familiar categories, previously known objects, or only the most obvious items. When a bounded foreground region plausibly represents an independent task-relevant object, include it with a concrete but not overly narrow category rather than silently omitting it.

SAM3 prompt policy:
SAM3 retrieval text is not an image caption. For each target, return exactly one concise lowercase English concrete noun phrase within 64 characters that names one complete physical object. Prefer the shortest stable category that can match across viewpoints, such as `computer mouse`, `water bottle`, or `drink cup`. Do not add color, material, transparency, wireless status, contents, or attached parts unless the base category is genuinely ambiguous and the attribute is clearly visible in this exact image. A partially occluded object still uses its whole-object category.
`with` and `and` may describe features belonging to that same object, but never use them to join separate objects. Never use `or` or punctuation-separated alternatives. Never use scene-relative position or vague words such as object, item, thing, stuff, region, foreground, or background. If multiple visible instances share the same concept, one prompt is enough because SAM3 returns matching instances.

Use Chinese for scene summaries, object names, and brief reasons. Keep JSON keys and enum values exactly as specified. Return one JSON object covering every supplied source_id exactly once. Do not return Markdown, extra text, or hidden chain-of-thought.
"""


def _output_rules(max_targets_per_image: int) -> str:
    return f"""Required JSON structure:
{{
  "images": [
    {{
      "source_id": "exact supplied source ID",
      "scene_summary": "brief Chinese scene summary",
      "targets": [
        {{
          "target_id": "target_001",
          "object_name_zh": "concise Chinese object name",
          "sam_text_prompt": "short lowercase English noun phrase",
          "priority": "high | medium",
          "confidence": 0.0,
          "selection_reason_code": "manipulable | task_relevant | identity_worthy | uncertain_standalone",
          "selection_short_reason": "brief Chinese reason"
        }}
      ],
      "no_target_reason": null
    }}
  ]
}}

Rules:
1. Output every supplied source_id exactly once and do not invent source IDs. Judge each image only from its own visible pixels; never copy or infer a target for one source merely because it appears in another source.
2. Return at most {max_targets_per_image} targets per image, ordered by priority and usefulness.
3. target_id values must be unique within an image.
4. object_name_zh and sam_text_prompt must name the same complete independent object; never name an attached part when the larger object is visible.
5. sam_text_prompt values must be unique within an image and follow the SAM3 prompt policy. Use stable category-level retrieval text rather than a detailed caption.
6. If no qualifying target is visible, return an empty targets list and a brief non-empty no_target_reason.
7. If targets is non-empty, no_target_reason must be null.
8. Confidence expresses confidence that the region is an independent worthwhile object, not confidence in its exact category name.
"""


class MllmPredictor(Protocol):
    def predict(self, messages: Sequence[dict[str, Any]]) -> MllmPrediction: ...


@dataclass(frozen=True, slots=True)
class SceneImageInput:
    source_id: str
    image_path: Path


@dataclass(frozen=True, slots=True)
class SceneGuidanceEvaluation:
    response: SceneGuidanceBatchResponse
    prediction: MllmPrediction


def parse_scene_guidance_response(
    raw_text: str,
    *,
    expected_source_ids: Sequence[str],
    max_targets_per_image: int,
) -> SceneGuidanceBatchResponse:
    """Validate source coverage and the Qwen-to-SAM3 text contract."""

    expected = list(expected_source_ids)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected source IDs must be non-empty and unique")
    try:
        response = SceneGuidanceBatchResponse.model_validate(
            extract_json_object(raw_text)
        )
    except (ValidationError, ValueError) as exc:
        raise MllmOutputError(
            f"Qwen scene-guidance response failed schema validation: {exc}"
        ) from exc

    received = [item.source_id for item in response.images]
    if len(received) != len(set(received)):
        raise MllmOutputError("Qwen returned a source ID more than once")
    if set(received) != set(expected):
        missing = sorted(set(expected) - set(received))
        unexpected = sorted(set(received) - set(expected))
        raise MllmOutputError(
            "Qwen scene coverage does not match the request; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for image in response.images:
        if len(image.targets) > max_targets_per_image:
            raise MllmOutputError(
                f"Qwen returned too many scene targets for {image.source_id}"
            )
    return response


def build_scene_guidance_messages(
    *,
    images: Sequence[SceneImageInput],
    settings: MllmPipelineConfig,
) -> list[dict[str, Any]]:
    """Build one first-pass request for up to the configured number of scenes."""

    image_list = list(images)
    if not image_list:
        raise ValueError("Scene guidance requires at least one source image")
    if len(image_list) > settings.scene_batch_size:
        raise ValueError("Scene guidance batch exceeds configured batch size")
    source_ids = [item.source_id for item in image_list]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Scene-guidance source IDs must be unique")

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"SCENE_BATCH begins. It contains {len(image_list)} independent "
                "source viewpoints. Analyze and report each source separately."
            ),
        }
    ]
    for index, item in enumerate(image_list, start=1):
        image_path = item.image_path.expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Scene source image not found: {image_path}")
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        f"SCENE_{index} source_id={item.source_id}. "
                        "The following image is one viewpoint of this exact scene."
                    ),
                },
                {
                    "type": "image",
                    "image": image_path.as_uri(),
                    "max_pixels": settings.max_pixels,
                },
            ]
        )
    content.append(
        {
            "type": "text",
            "text": (
                "Create the robot-oriented observation plan now.\n"
                + _output_rules(settings.max_scene_targets_per_image)
            ),
        }
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SCENE_GUIDANCE_SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
    ]


def evaluate_scene_guidance_batch(
    predictor: MllmPredictor,
    *,
    images: Sequence[SceneImageInput],
    settings: MllmPipelineConfig,
) -> SceneGuidanceEvaluation:
    """Run and validate one batched first-pass scene survey."""

    image_list = list(images)
    prediction = predictor.predict(
        build_scene_guidance_messages(images=image_list, settings=settings)
    )
    response = parse_scene_guidance_response(
        prediction.raw_text,
        expected_source_ids=[item.source_id for item in image_list],
        max_targets_per_image=settings.max_scene_targets_per_image,
    )
    return SceneGuidanceEvaluation(response=response, prediction=prediction)
