"""Validated data exchanged between pipeline stages."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Annotated
from uuid import uuid4

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
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
    NEW_OBJECT = "new_object"
    VISUAL_INSTANCE_MATCH = "visual_instance_match"
    INVALID_CANDIDATE = "invalid_candidate"
    AMBIGUOUS_MATCH = "ambiguous_match"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


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
