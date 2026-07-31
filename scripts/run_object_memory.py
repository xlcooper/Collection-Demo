#!/usr/bin/env python3
"""Run the batch object-memory Demo workflow."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from object_memory.assets import MemoryPaths  # noqa: E402
from object_memory.config import (  # noqa: E402
    AppConfig,
    DEFAULT_CONFIG_PATH,
    load_config,
    resolve_memory_root,
)
from object_memory.mllm_adapter import QwenMllmAdapter  # noqa: E402
from object_memory.pipeline import (  # noqa: E402
    ObjectMemoryPipeline,
    discover_images,
    write_json_atomic,
)
from object_memory.sam3_adapter import Sam3Adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process an image directory into persistent object memory."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--memory-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--qwen-model")
    parser.add_argument("--revision")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument(
        "--validate-demo",
        action="store_true",
        help=(
            "Require coverage for at least two new objects, one existing match, "
            "one duplicate image, one filtered/ignored candidate, and no pending "
            "or failed proposals."
        ),
    )
    parser.add_argument(
        "--report",
        help="Optional second copy of the run report, such as an environment report.",
    )
    return parser.parse_args()


def runtime_config(base: AppConfig, args: argparse.Namespace) -> AppConfig:
    payload = base.model_dump(mode="python")
    if args.checkpoint:
        payload["models"]["sam3_checkpoint"] = args.checkpoint
    if args.qwen_model:
        payload["models"]["qwen_model_id"] = args.qwen_model
    return AppConfig.model_validate(payload)


def resolve_checkpoint(config: AppConfig) -> Path:
    checkpoint = config.models.sam3_checkpoint.expanduser()
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    return checkpoint.resolve()


def validate_directory_separation(input_root: Path, memory_root: Path) -> None:
    if input_root == memory_root:
        raise ValueError("Input directory and memory root must be different")
    if input_root.is_relative_to(memory_root) or memory_root.is_relative_to(input_root):
        raise ValueError(
            "Input directory and memory root must not contain one another"
        )


def failure_report(exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test": "object_memory_demo_batch",
        "status": "failed",
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }


def add_demo_coverage(report: dict[str, Any]) -> None:
    run = report["run"]
    decisions = run["decision_counts"]
    proposals = run["proposal_counts"]
    coverage = {
        "at_least_two_new_object_archives": (
            decisions["new"] >= 2 and run["active_objects_total"] >= 2
        ),
        "same_object_observation_was_merged": decisions["existing"] >= 1,
        "exact_duplicate_image_was_skipped": (
            run["duplicate_sources_skipped"] >= 1
        ),
        "invalid_or_duplicate_mask_was_filtered": (
            proposals["filtered"] + decisions["ignored"] >= 1
        ),
        "no_pending_or_failed_proposals": (
            proposals["pending"] == 0 and proposals["failed"] == 0
        ),
    }
    report["demo_coverage"] = coverage
    if report["status"] != "passed" or not all(coverage.values()):
        report["pipeline_status"] = report["status"]
        report["status"] = "failed"


def main() -> int:
    args = parse_args()
    try:
        config = runtime_config(load_config(args.config), args)
        input_root = Path(args.input_dir).expanduser().resolve()
        memory_root = resolve_memory_root(
            config,
            args.memory_root,
            base_dir=PROJECT_ROOT,
        )
        validate_directory_separation(input_root, memory_root)
        image_paths = discover_images(input_root)
        paths = MemoryPaths(memory_root, config.storage.database_filename)
        checkpoint = resolve_checkpoint(config)
        sam_runtime = Sam3Adapter(
            checkpoint,
            config.sam3_pipeline.confidence_threshold,
            points_per_side=config.sam3_pipeline.points_per_side,
            points_per_batch=config.sam3_pipeline.points_per_batch,
        )
        mllm_runtime = QwenMllmAdapter(
            config.models.qwen_model_id,
            revision=args.revision,
            allow_network=args.allow_network,
            max_pixels=config.mllm_pipeline.max_pixels,
            max_new_tokens=config.mllm_pipeline.max_new_tokens,
        )
        pipeline = ObjectMemoryPipeline(
            config=config,
            paths=paths,
            sam_runtime=sam_runtime,
            mllm_runtime=mllm_runtime,
        )
        report = pipeline.run(image_paths)
        if args.validate_demo:
            add_demo_coverage(report)
            write_json_atomic(paths.resolve_asset(report["run_report"]), report)
        if args.report:
            write_json_atomic(Path(args.report).expanduser().resolve(), report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "passed" else 4
    except Exception as exc:  # noqa: BLE001 - CLI must preserve failure evidence
        report = failure_report(exc)
        if args.report:
            write_json_atomic(Path(args.report).expanduser().resolve(), report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
