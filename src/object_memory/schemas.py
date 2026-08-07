"""Validated data exchanged between the single-pass object-memory stages."""

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
CosineScore = Annotated[float, Field(ge=-1.0, le=1.0)]
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
    UNCERTAIN = "uncertain"


class DecisionReasonCode(str, Enum):
    NEW_OBJECT = "new_object"
    VISUAL_INSTANCE_MATCH = "visual_instance_match"
    AMBIGUOUS_MATCH = "ambiguous_match"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class IdentityHypothesis(str, Enum):
    NEW = "new"
    EXISTING = "existing"
    UNCERTAIN = "uncertain"


class VisualMatchType(str, Enum):
    MATCH = "match"
    NO_MATCH = "no_match"
    AMBIGUOUS = "ambiguous"


class PartAppearance(StrictModel):
    part: NonEmptyText
    color: list[NonEmptyText] = Field(default_factory=list)
    material: list[NonEmptyText] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_values(self) -> "PartAppearance":
        if len(self.color) != len(set(self.color)):
            raise ValueError("part color values must be unique")
        if len(self.material) != len(set(self.material)):
            raise ValueError("part material values must be unique")
        return self


class ObjectSummary(StrictModel):
    """The only long-lived text description for one physical object."""

    object_name_zh: NonEmptyText
    coarse_category: NonEmptyText
    fine_category: NonEmptyText
    stable_description: NonEmptyText
    stable_identity_features: list[NonEmptyText] = Field(default_factory=list)
    brand_or_markings: list[NonEmptyText] = Field(default_factory=list)
    part_appearance: list[PartAppearance] = Field(default_factory=list)
    summary_confidence: Confidence

    @model_validator(mode="after")
    def validate_unique_summary_entries(self) -> "ObjectSummary":
        part_names = [item.part for item in self.part_appearance]
        if len(part_names) != len(set(part_names)):
            raise ValueError("object summary part names must be unique")
        if len(self.stable_identity_features) != len(
            set(self.stable_identity_features)
        ):
            raise ValueError("stable identity features must be unique")
        if len(self.brand_or_markings) != len(set(self.brand_or_markings)):
            raise ValueError("brand or marking values must be unique")
        return self


class CurrentViewFacts(StrictModel):
    """Visible facts retained in the audited Qwen response, not the object card."""

    category: NonEmptyText
    visible_identity_features: list[NonEmptyText] = Field(default_factory=list)
    brand_or_markings: list[NonEmptyText] = Field(default_factory=list)
    part_appearance: list[PartAppearance] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_view_entries(self) -> "CurrentViewFacts":
        part_names = [item.part for item in self.part_appearance]
        if len(part_names) != len(set(part_names)):
            raise ValueError("current-view part names must be unique")
        return self


class NormalizedBoundingBox(StrictModel):
    x_min: Annotated[float, Field(ge=0.0, le=1.0)]
    y_min: Annotated[float, Field(ge=0.0, le=1.0)]
    x_max: Annotated[float, Field(ge=0.0, le=1.0)]
    y_max: Annotated[float, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def validate_corners(self) -> "NormalizedBoundingBox":
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Normalized bounding-box maximums must exceed minimums")
        return self


class SceneTarget(StrictModel):
    """One Qwen-discovered instance, its SAM query, and its text identity guess."""

    target_id: Identifier
    object_name_zh: NonEmptyText
    sam_text_prompt: NonEmptyText
    current_view_facts: CurrentViewFacts
    identity_hypothesis: IdentityHypothesis
    matched_object_id: Identifier | None = None
    identity_short_reason: NonEmptyText
    proposed_object_summary: ObjectSummary
    temporary_target_anchor: NormalizedBoundingBox

    @field_validator("sam_text_prompt")
    @classmethod
    def validate_sam_text_prompt(cls, value: str) -> str:
        if not 2 <= len(value) <= 64:
            raise ValueError("sam_text_prompt must contain 2 to 64 characters")
        words = value.split()
        if not re.fullmatch(r"[a-z][a-z0-9]*(?:[ '][a-z0-9]+)*", value):
            raise ValueError(
                "sam_text_prompt must be a short lowercase English noun phrase"
            )
        if "or" in words:
            raise ValueError(
                "sam_text_prompt must describe one object concept without alternatives"
            )
        forbidden = {
            "left", "right", "upper", "lower", "top", "bottom", "front",
            "rear", "near", "on", "in", "under", "above", "below",
            "behind", "beside", "between", "next", "to", "by", "from",
            "object", "objects", "item", "items", "thing", "things",
            "stuff", "scene", "region", "area", "background", "foreground",
        }
        if forbidden.intersection(words):
            raise ValueError(
                "sam_text_prompt must use a concrete category without scene position"
            )
        return value

    @model_validator(mode="after")
    def validate_identity_hypothesis(self) -> "SceneTarget":
        if self.identity_hypothesis is IdentityHypothesis.EXISTING:
            if self.matched_object_id is None:
                raise ValueError("existing hypothesis requires matched_object_id")
        elif self.matched_object_id is not None:
            raise ValueError("only existing hypothesis may carry matched_object_id")
        return self


class SceneImageGuidance(StrictModel):
    """The only Qwen response for one exact source image."""

    source_id: Identifier
    scene_summary: NonEmptyText
    targets: list[SceneTarget] = Field(default_factory=list)
    no_target_reason: str | None = None

    @model_validator(mode="after")
    def validate_scene_targets(self) -> "SceneImageGuidance":
        target_ids = [target.target_id for target in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("scene target IDs must be unique within one image")
        if self.targets and self.no_target_reason is not None:
            raise ValueError("no_target_reason is only valid when targets is empty")
        if not self.targets and not (self.no_target_reason or "").strip():
            raise ValueError("an empty target list requires no_target_reason")
        return self


class SceneGuidanceResponse(StrictModel):
    image: SceneImageGuidance


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
    dinov3_model_id: NonEmptyText
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
    target_id: Identifier | None = None
    target_object_name_zh: str | None = None
    target_anchor: NormalizedBoundingBox | None = None
    score: Confidence
    bbox: BoundingBox
    mask_area_pixels: int = Field(default=0, ge=0)
    mask_area_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    mask_path: RelativeAssetPath | None = None
    crop_path: RelativeAssetPath | None = None
    overlay_path: RelativeAssetPath | None = None
    fingerprint: VisualFingerprint | None = None
    status: ProposalStatus = ProposalStatus.PENDING
    filter_reason: str | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class MemoryObject(StrictModel):
    id: Identifier = Field(default_factory=lambda: new_id("obj"))
    summary: ObjectSummary
    status: ObjectStatus = ObjectStatus.ACTIVE
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ObjectCard(StrictModel):
    object_id: Identifier
    summary: ObjectSummary


class VisualFingerprint(StrictModel):
    path: RelativeAssetPath
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_id: NonEmptyText
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    feature_layer: NonEmptyText
    input_size: int = Field(gt=0)
    storage_dtype: NonEmptyText
    global_dimension: int = Field(gt=0)
    local_count: int = Field(ge=0)
    l2_normalized: bool


class VisualObjectScore(StrictModel):
    object_id: Identifier
    observation_id: Identifier
    global_similarity: CosineScore
    local_match_ratio: Confidence
    visual_score: CosineScore


class VisualEvidence(StrictModel):
    result: VisualMatchType
    matched_object_id: Identifier | None = None
    matched_observation_id: Identifier | None = None
    global_similarity: CosineScore | None = None
    local_match_ratio: Confidence | None = None
    visual_score: CosineScore | None = None
    second_best_score: CosineScore | None = None
    score_margin: Annotated[float, Field(ge=0.0, le=2.0)] | None = None
    object_scores: list[VisualObjectScore] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_match(self) -> "VisualEvidence":
        if self.result is VisualMatchType.MATCH:
            if self.matched_object_id is None or self.matched_observation_id is None:
                raise ValueError("visual match requires object and observation IDs")
        elif self.matched_object_id is not None or self.matched_observation_id is not None:
            raise ValueError("only a visual match may carry matched IDs")
        return self


class FinalIdentityDecision(StrictModel):
    decision: DecisionType
    matched_object_id: Identifier | None = None
    confidence: Confidence
    reason_code: DecisionReasonCode
    short_reason: NonEmptyText
    qwen_hypothesis: IdentityHypothesis
    qwen_matched_object_id: Identifier | None = None
    visual_evidence: VisualEvidence
    object_summary: ObjectSummary | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "FinalIdentityDecision":
        if self.decision is DecisionType.EXISTING:
            if self.matched_object_id is None:
                raise ValueError("existing decision requires matched_object_id")
        elif self.matched_object_id is not None:
            raise ValueError("only existing decision may carry matched_object_id")
        if self.decision in {DecisionType.NEW, DecisionType.EXISTING}:
            if self.object_summary is None:
                raise ValueError("new and existing decisions require object_summary")
        elif self.object_summary is not None:
            raise ValueError("uncertain decisions cannot update an object summary")
        if self.qwen_hypothesis is IdentityHypothesis.EXISTING:
            if self.qwen_matched_object_id is None:
                raise ValueError("existing Qwen hypothesis requires an object ID")
        elif self.qwen_matched_object_id is not None:
            raise ValueError("only existing Qwen hypothesis may carry an object ID")
        return self


class Observation(StrictModel):
    id: Identifier = Field(default_factory=lambda: new_id("obs"))
    object_id: Identifier
    proposal_id: Identifier
    source_image_id: Identifier
    fingerprint: VisualFingerprint
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
    qwen_hypothesis: IdentityHypothesis
    qwen_matched_object_id: Identifier | None = None
    visual_evidence: VisualEvidence
    raw_response_path: RelativeAssetPath | None = None
    attempt: int = Field(default=1, ge=1, le=1)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_match(self) -> "Decision":
        if self.decision is DecisionType.EXISTING and self.matched_object_id is None:
            raise ValueError("existing decisions require matched_object_id")
        if self.decision is not DecisionType.EXISTING and self.matched_object_id:
            raise ValueError("matched_object_id is only valid for existing decisions")
        return self


Proposal.model_rebuild()
