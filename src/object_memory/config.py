"""Project configuration loading and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class StorageConfig(BaseModel):
    """Filesystem and SQLite locations."""

    model_config = ConfigDict(extra="forbid")

    memory_root: Path = Path("data/memory")
    database_filename: str = "memory.sqlite"

    @field_validator("database_filename")
    @classmethod
    def validate_database_filename(cls, value: str) -> str:
        candidate = Path(value)
        if candidate.name != value or candidate.suffix != ".sqlite":
            raise ValueError(
                "database_filename must be one relative filename ending in .sqlite"
            )
        return value


class ModelConfig(BaseModel):
    """Model identifiers shared by later pipeline stages."""

    model_config = ConfigDict(extra="forbid")

    sam3_checkpoint: Path = Path("weights/sam3/sam3.pt")
    qwen_model_id: str = Field(
        default="Qwen/Qwen3-VL-8B-Instruct-FP8",
        min_length=1,
    )
    dinov3_model_id: str = Field(
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        min_length=1,
    )
    dinov3_model_path: Path = Path(
        "weights/dinov3/dinov3-vitb16-pretrain-lvd1689m"
    )
    dinov3_revision: str = Field(
        default="5931719e67bbdb9737e363e781fb0c67687896bc",
        pattern=r"^[a-f0-9]{40}$",
    )


class Sam3PipelineConfig(BaseModel):
    """Deterministic settings for class-agnostic point-grid candidates."""

    model_config = ConfigDict(extra="forbid")

    points_per_side: int = Field(default=16, ge=2, le=64)
    points_per_batch: int = Field(default=32, ge=1, le=256)
    confidence_threshold: float = Field(default=0.88, ge=0.0, le=1.0)
    min_mask_area_ratio: float = Field(default=0.0005, ge=0.0, le=1.0)
    max_mask_area_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    duplicate_mask_iou_threshold: float = Field(default=0.9, gt=0.0, le=1.0)
    contained_mask_overlap_threshold: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
    )
    max_candidates_per_image: int = Field(default=24, ge=1, le=256)
    crop_padding_pixels: int = Field(default=8, ge=0)
    crop_background_color: tuple[int, int, int] = (127, 127, 127)
    overlay_alpha: float = Field(default=0.0, ge=0.0, le=0.0)
    overlay_color: tuple[int, int, int] = (255, 64, 64)

    @field_validator("crop_background_color", "overlay_color")
    @classmethod
    def validate_rgb_color(
        cls, value: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        if any(channel < 0 or channel > 255 for channel in value):
            raise ValueError("RGB color channels must be between 0 and 255")
        return value

    @model_validator(mode="after")
    def validate_mask_area_range(self) -> "Sam3PipelineConfig":
        if self.min_mask_area_ratio >= self.max_mask_area_ratio:
            raise ValueError(
                "min_mask_area_ratio must be smaller than max_mask_area_ratio"
            )
        return self


class MllmPipelineConfig(BaseModel):
    """Settings for batched Qwen review after visual clustering."""

    model_config = ConfigDict(extra="forbid")

    prompt_version: Literal[
        "object-memory-cluster-review-v1"
    ] = "object-memory-cluster-review-v1"
    max_clusters_per_batch: int = Field(default=8, ge=1, le=16)
    max_pixels: int = Field(default=1024 * 1024, gt=0)
    max_new_tokens: int = Field(default=4096, gt=0)


class VisualFingerprintConfig(BaseModel):
    """Fixed DINOv3 extraction and interpretable matching settings."""

    model_config = ConfigDict(extra="forbid")

    input_size: int = Field(default=512, ge=224)
    storage_dtype: Literal["float16"] = "float16"
    similarity_metric: Literal["cosine"] = "cosine"
    global_feature: Literal["cls_token"] = "cls_token"
    local_feature: Literal["patch_tokens"] = "patch_tokens"
    min_patch_mask_coverage: float = Field(default=0.5, gt=0.0, le=1.0)
    local_patch_similarity_threshold: float = Field(
        default=0.7,
        ge=-1.0,
        le=1.0,
    )
    global_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    local_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    match_threshold: float = Field(default=0.75, ge=-1.0, le=1.0)
    ambiguity_margin: float = Field(default=0.05, ge=0.0, le=2.0)
    local_top_k: int = Field(default=3, ge=1, le=64)
    cluster_global_similarity_threshold: float = Field(
        default=0.75,
        ge=-1.0,
        le=1.0,
    )
    max_cluster_representatives: int = Field(default=4, ge=1, le=8)
    contact_sheet_cell_size: int = Field(default=320, ge=160, le=640)

    @model_validator(mode="after")
    def validate_matching_weights(self) -> "VisualFingerprintConfig":
        if abs((self.global_weight + self.local_weight) - 1.0) > 1e-9:
            raise ValueError("global_weight and local_weight must sum to 1.0")
        if self.input_size % 16:
            raise ValueError("input_size must be divisible by the ViT-B/16 patch size")
        return self


class AppConfig(BaseModel):
    """Validated top-level project configuration."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[5] = 5
    storage: StorageConfig = Field(default_factory=StorageConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    sam3_pipeline: Sam3PipelineConfig = Field(default_factory=Sam3PipelineConfig)
    mllm_pipeline: MllmPipelineConfig = Field(default_factory=MllmPipelineConfig)
    visual_fingerprint: VisualFingerprintConfig = Field(
        default_factory=VisualFingerprintConfig
    )


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load one YAML configuration file and reject unknown fields."""

    config_path = Path(path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("The configuration root must be a mapping.")
    return AppConfig.model_validate(raw)


def resolve_memory_root(
    config: AppConfig,
    override: str | Path | None = None,
    *,
    base_dir: str | Path | None = None,
) -> Path:
    """Resolve the memory root; relative paths are based on the working directory."""

    root = Path(override) if override is not None else config.storage.memory_root
    root = root.expanduser()
    if not root.is_absolute():
        root = Path(base_dir or Path.cwd()) / root
    return root.resolve()


def config_digest(config: AppConfig) -> str:
    """Return a stable digest for recording the exact settings of one run."""

    canonical = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
