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


class AppConfig(BaseModel):
    """Validated top-level project configuration."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    storage: StorageConfig = Field(default_factory=StorageConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)


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
