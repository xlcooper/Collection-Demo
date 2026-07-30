"""Project configuration loading and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    """Deterministic settings for explicit-prompt SAM3 candidate generation."""

    model_config = ConfigDict(extra="forbid")

    prompt_strategy: Literal["explicit_category_list"] = "explicit_category_list"
    prompts: list[str] = Field(default_factory=list)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_mask_area_ratio: float = Field(default=0.0005, ge=0.0, le=1.0)
    duplicate_mask_iou_threshold: float = Field(default=0.9, gt=0.0, le=1.0)
    crop_padding_pixels: int = Field(default=8, ge=0)
    overlay_alpha: float = Field(default=0.45, gt=0.0, le=1.0)
    overlay_color: tuple[int, int, int] = (255, 64, 64)

    @field_validator("prompts")
    @classmethod
    def normalize_prompts(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("SAM3 prompts must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("SAM3 prompts must be unique")
        return normalized

    @field_validator("overlay_color")
    @classmethod
    def validate_overlay_color(
        cls, value: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        if any(channel < 0 or channel > 255 for channel in value):
            raise ValueError("overlay_color channels must be between 0 and 255")
        return value


class MllmPipelineConfig(BaseModel):
    """Settings for constrained Qwen object annotation and identity checks."""

    model_config = ConfigDict(extra="forbid")

    prompt_version: Literal["m3-object-identity-v2"] = "m3-object-identity-v2"
    max_pixels: int = Field(default=1024 * 1024, gt=0)
    max_new_tokens: int = Field(default=512, gt=0)
    object_card_batch_size: int = Field(default=8, gt=0)
    max_reference_views_per_object: int = Field(default=2, gt=0)
    existing_min_confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class AppConfig(BaseModel):
    """Validated top-level project configuration."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    storage: StorageConfig = Field(default_factory=StorageConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    sam3_pipeline: Sam3PipelineConfig = Field(default_factory=Sam3PipelineConfig)
    mllm_pipeline: MllmPipelineConfig = Field(default_factory=MllmPipelineConfig)


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
