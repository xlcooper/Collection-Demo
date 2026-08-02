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


class Sam3PipelineConfig(BaseModel):
    """Deterministic settings for text-guided SAM3 candidate generation."""

    model_config = ConfigDict(extra="forbid")

    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
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
    overlay_alpha: float = Field(default=0.45, gt=0.0, le=1.0)
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
    """Settings for scene guidance and constrained object-memory reasoning."""

    model_config = ConfigDict(extra="forbid")

    scene_prompt_version: Literal[
        "robot-scene-guidance-v3"
    ] = "robot-scene-guidance-v3"
    prompt_version: Literal[
        "guided-image-batch-memory-reasoning-v2"
    ] = "guided-image-batch-memory-reasoning-v2"
    scene_batch_size: int = Field(default=4, ge=1, le=8)
    max_scene_targets_per_image: int = Field(default=12, ge=1, le=24)
    max_pixels: int = Field(default=1024 * 1024, gt=0)
    max_new_tokens: int = Field(default=4096, gt=0)
    max_reference_views_per_object: int = Field(default=2, gt=0)
    existing_min_confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class AppConfig(BaseModel):
    """Validated top-level project configuration."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[3] = 3
    storage: StorageConfig = Field(default_factory=StorageConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    sam3_pipeline: Sam3PipelineConfig = Field(default_factory=Sam3PipelineConfig)
    mllm_pipeline: MllmPipelineConfig = Field(default_factory=MllmPipelineConfig)

    @model_validator(mode="after")
    def validate_scene_target_capacity(self) -> "AppConfig":
        if (
            self.mllm_pipeline.max_scene_targets_per_image
            > self.sam3_pipeline.max_candidates_per_image
        ):
            raise ValueError(
                "max_scene_targets_per_image must not exceed "
                "max_candidates_per_image"
            )
        return self


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
