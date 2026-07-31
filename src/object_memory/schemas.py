"""Validated data exchanged between pipeline stages."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
import re
from typing import Annotated
from uuid import uuid4

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _relative_asset_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Asset path must be a non-empty string")
    if "\\" in value:
        raise ValueError("Asset paths must use forward slashes")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("Asset paths must remain relative to the memory root")
    return path.as_posix()


Identifier = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$"),
]
RelativeAssetPath = Annotated[str, BeforeValidator(_relative_asset_path)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
NonEmptyText = Annotated[str, Field(min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"


class SourceImageStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProposalStatus(str, Enum):
    PENDING = "pending"
    FILTERED = "filtered"
    DECIDED = "decided"
    FAILED = "failed"


class ObjectStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class DecisionType(str, Enum):
    NEW = "new"
    EXISTING = "existing"
    IGNORED = "ignored"
    UNCERTAIN = "uncertain"


class DecisionReasonCode(str, Enum):
    VALID_CANDIDATE = "valid_candidate"
    NEW_OBJECT = "new_object"
    VISUAL_INSTANCE_MATCH = "visual_instance_match"
    INVALID_CANDIDATE = "invalid_candidate"
    AMBIGUOUS_MATCH = "ambiguous_match"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class CandidateValidity(str, Enum):
    VALID = "valid"
    IGNORED = "ignored"


class SceneTargetPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"


class SceneTargetReasonCode(str, Enum):
    MANIPULABLE = "manipulable"
    TASK_RELEVANT = "task_relevant"
    IDENTITY_WORTHY = "identity_worthy"
    UNCERTAIN_STANDALONE = "uncertain_standalone"


class SceneTarget(StrictModel):
    """One first-pass object concept that may be sent to SAM3."""

    target_id: Identifier
    object_name_zh: NonEmptyText
    sam_text_prompt: NonEmptyText
    priority: SceneTargetPriority
    confidence: Confidence
    selection_reason_code: SceneTargetReasonCode
    selection_short_reason: NonEmptyText

    @field_validator("sam_text_prompt")
    @classmethod
    def validate_sam_text_prompt(cls, value: str) -> str:
        normalized = " ".join(value.strip().lower().split())
        if not 2 <= len(normalized) <= 64:
            raise ValueError("sam_text_prompt must contain 2 to 64 characters")
        words = normalized.split()
        if len(words) > 6:
            raise ValueError("sam_text_prompt must contain at most 6 words")
        if not re.fullmatch(r"[a-z][a-z0-9]*(?:[ '][a-z0-9]+)*", normalized):
            raise ValueError(
                "sam_text_prompt must be a short lowercase English noun phrase"
            )
        alternative_words = {"and", "or"}
        if alternative_words.intersection(words):
            raise ValueError(
                "sam_text_prompt must describe one object concept without alternatives"
            )
        position_words = {
            "left",
            "right",
            "upper",
            "lower",
            "top",
            "bottom",
            "front",
            "rear",
            "near",
            "on",
            "in",
            "under",
            "above",
            "below",
            "behind",
            "beside",
            "between",
            "next",
            "to",
            "by",
            "from",
        }
        if position_words.intersection(words):
            raise ValueError(
                "sam_text_prompt must not use scene-relative position words"
            )
        vague_words = {
            "object",
            "objects",
            "item",
            "items",
            "thing",
            "things",
            "stuff",
            "scene",
            "region",
            "area",
            "background",
            "foreground",
        }
        if vague_words.intersection(words):
            raise ValueError("sam_text_prompt must use a concrete object category")
        return normalized


class SceneImageGuidance(StrictModel):
    """First-pass scene survey for one exact source image."""

    source_id: Identifier
    scene_summary: NonEmptyText
    targets: list[SceneTarget] = Field(default_factory=list)
    no_target_reason: str | None = None

    @model_validator(mode="after")
    def validate_scene_targets(self) -> "SceneImageGuidance":
        target_ids = [target.target_id for target in self.targets]
        prompts = [target.sam_text_prompt for target in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("scene target IDs must be unique within one image")
        if len(prompts) != len(set(prompts)):
            raise ValueError("SAM3 text prompts must be unique within one image")
        if self.targets and self.no_target_reason is not None:
            raise ValueError("no_target_reason is only valid when targets is empty")
        if not self.targets and not (self.no_target_reason or "").strip():
            raise ValueError("an empty target list requires no_target_reason")
        return self


class SceneGuidanceBatchResponse(StrictModel):
    """One Qwen response covering a batch of source-scene images."""

    images: list[SceneImageGuidance] = Field(min_length=1)


class BoundingBox(StrictModel):
    x_min: Annotated[float, Field(ge=0.0)]
    y_min: Annotated[float, Field(ge=0.0)]
    x_max: Annotated[float, Field(gt=0.0)]
    y_max: Annotated[float, Field(gt=0.0)]

    @model_validator(mode="after")
    def validate_corners(self) -> "BoundingBox":
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Bounding box maximums must exceed minimums")
        return self


class Run(StrictModel):
    id: Identifier = Field(default_factory=lambda: new_id("run"))
    status: RunStatus = RunStatus.RUNNING
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    config_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    sam_model_id: NonEmptyText
    qwen_model_id: NonEmptyText
    error_message: str | None = None


class SourceImage(StrictModel):
    id: Identifier = Field(default_factory=lambda: new_id("src"))
    run_id: Identifier
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    relative_path: RelativeAssetPath
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    status: SourceImageStatus = SourceImageStatus.PENDING
    error_message: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Proposal(StrictModel):
    id: Identifier = Field(default_factory=lambda: new_id("prop"))
    source_image_id: Identifier
    raw_candidate_id: NonEmptyText
    prompt: NonEmptyText = "unknown"
    score: Confidence
    bbox: BoundingBox
    mask_area_pixels: int = Field(default=0, ge=0)
    mask_area_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    mask_path: RelativeAssetPath | None = None
    crop_path: RelativeAssetPath | None = None
    overlay_path: RelativeAssetPath | None = None
    status: ProposalStatus = ProposalStatus.PENDING
    filter_reason: str | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class MemoryObject(StrictModel):
    id: Identifier = Field(default_factory=lambda: new_id("obj"))
    coarse_category: NonEmptyText
    fine_category: NonEmptyText
    material: list[str] = Field(default_factory=list)
    color: list[str] = Field(default_factory=list)
    shape: NonEmptyText
    description: NonEmptyText
    annotation_confidence: Confidence
    status: ObjectStatus = ObjectStatus.ACTIVE
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ObjectAnnotation(StrictModel):
    """Visible facts produced by the MLLM for one candidate observation."""

    coarse_category: NonEmptyText
    fine_category: NonEmptyText
    material: list[str] = Field(default_factory=list)
    color: list[str] = Field(default_factory=list)
    shape: NonEmptyText
    description: NonEmptyText
    annotation_confidence: Confidence


class ObjectCard(StrictModel):
    """Compact known-object context sent to the MLLM."""

    object_id: Identifier
    coarse_category: NonEmptyText
    fine_category: NonEmptyText
    material: list[str] = Field(default_factory=list)
    color: list[str] = Field(default_factory=list)
    shape: NonEmptyText
    description: NonEmptyText
    representative_view_paths: list[RelativeAssetPath] = Field(default_factory=list)


class BatchCandidateDecision(StrictModel):
    """One candidate's ordered validity, annotation, and memory decision."""

    proposal_id: Identifier
    validity: CandidateValidity
    validity_confidence: Confidence
    validity_reason_code: DecisionReasonCode
    validity_short_reason: NonEmptyText
    temporary_annotation: ObjectAnnotation | None
    decision: DecisionType
    matched_object_id: Identifier | None
    confidence: Confidence
    reason_code: DecisionReasonCode
    short_reason: NonEmptyText
    final_annotation: ObjectAnnotation | None

    @model_validator(mode="after")
    def validate_batch_candidate(self) -> "BatchCandidateDecision":
        if self.validity is CandidateValidity.IGNORED:
            if self.validity_reason_code is not DecisionReasonCode.INVALID_CANDIDATE:
                raise ValueError("ignored validity requires invalid_candidate")
            if self.decision is not DecisionType.IGNORED:
                raise ValueError("ignored validity requires ignored decision")
            if self.reason_code is not DecisionReasonCode.INVALID_CANDIDATE:
                raise ValueError("ignored decision requires invalid_candidate")
            if (
                self.temporary_annotation is not None
                or self.final_annotation is not None
                or self.matched_object_id is not None
            ):
                raise ValueError(
                    "ignored candidates cannot carry annotations or object IDs"
                )
            return self

        if self.validity_reason_code is not DecisionReasonCode.VALID_CANDIDATE:
            raise ValueError("valid candidates require valid_candidate")
        if self.temporary_annotation is None or self.final_annotation is None:
            raise ValueError(
                "valid candidates require temporary and final annotations"
            )
        if self.decision is DecisionType.IGNORED:
            raise ValueError("valid candidates cannot be ignored")
        if self.decision is DecisionType.EXISTING:
            if self.matched_object_id is None:
                raise ValueError("existing requires matched_object_id")
        elif self.matched_object_id is not None:
            raise ValueError("only existing may carry matched_object_id")
        allowed_reasons = {
            DecisionType.NEW: {DecisionReasonCode.NEW_OBJECT},
            DecisionType.EXISTING: {DecisionReasonCode.VISUAL_INSTANCE_MATCH},
            DecisionType.UNCERTAIN: {
                DecisionReasonCode.AMBIGUOUS_MATCH,
                DecisionReasonCode.INSUFFICIENT_EVIDENCE,
            },
        }
        if self.reason_code not in allowed_reasons[self.decision]:
            raise ValueError("reason_code does not agree with decision")
        return self

    def to_mllm_response(self) -> "MllmResponse":
        return MllmResponse(
            decision=self.decision,
            matched_object_id=self.matched_object_id,
            confidence=self.confidence,
            reason_code=self.reason_code,
            short_reason=self.short_reason,
            annotation=self.final_annotation,
        )


class ImageBatchResponse(StrictModel):
    """One Qwen response covering all retained candidates from one source image."""

    candidates: list[BatchCandidateDecision] = Field(min_length=1)


class MllmResponse(StrictModel):
    """Validated combined identity decision and visible-object annotation."""

    decision: DecisionType
    matched_object_id: Identifier | None = None
    confidence: Confidence
    reason_code: DecisionReasonCode
    short_reason: NonEmptyText
    annotation: ObjectAnnotation | None = None

    @model_validator(mode="after")
    def validate_decision_payload(self) -> "MllmResponse":
        if self.decision is DecisionType.EXISTING and self.matched_object_id is None:
            raise ValueError("existing responses require matched_object_id")
        if self.decision is not DecisionType.EXISTING and self.matched_object_id:
            raise ValueError(
                "matched_object_id is only valid for existing responses"
            )
        if self.decision in {DecisionType.NEW, DecisionType.EXISTING}:
            if self.annotation is None:
                raise ValueError("new and existing responses require annotation")
        allowed_reasons = {
            DecisionType.NEW: {DecisionReasonCode.NEW_OBJECT},
            DecisionType.EXISTING: {DecisionReasonCode.VISUAL_INSTANCE_MATCH},
            DecisionType.IGNORED: {DecisionReasonCode.INVALID_CANDIDATE},
            DecisionType.UNCERTAIN: {
                DecisionReasonCode.AMBIGUOUS_MATCH,
                DecisionReasonCode.INSUFFICIENT_EVIDENCE,
            },
        }
        if self.reason_code not in allowed_reasons[self.decision]:
            raise ValueError("reason_code does not agree with decision")
        return self


class Observation(StrictModel):
    id: Identifier = Field(default_factory=lambda: new_id("obs"))
    object_id: Identifier
    proposal_id: Identifier
    source_image_id: Identifier
    crop_path: RelativeAssetPath
    mask_path: RelativeAssetPath
    overlay_path: RelativeAssetPath
    description: NonEmptyText
    created_at: datetime = Field(default_factory=utc_now)


class Decision(StrictModel):
    id: Identifier = Field(default_factory=lambda: new_id("dec"))
    proposal_id: Identifier
    decision: DecisionType
    matched_object_id: Identifier | None = None
    confidence: Confidence
    reason_code: NonEmptyText
    short_reason: NonEmptyText
    prompt_version: NonEmptyText
    raw_response_path: RelativeAssetPath | None = None
    attempt: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_match(self) -> "Decision":
        if self.decision is DecisionType.EXISTING and self.matched_object_id is None:
            raise ValueError("existing decisions require matched_object_id")
        if self.decision is not DecisionType.EXISTING and self.matched_object_id:
            raise ValueError("matched_object_id is only valid for existing decisions")
        return self
