#!/usr/bin/env python3
"""Run the per-image Qwen, SAM3, and DINOv3 object-memory workflow."""

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
from object_memory.dinov3_adapter import DinoV3Adapter  # noqa: E402
from object_memory.mllm_adapter import QwenMllmAdapter  # noqa: E402
from object_memory.memory_store import MemoryStore, MemoryStoreError  # noqa: E402
from object_memory.pipeline import (  # noqa: E402
    ObjectMemoryPipeline,
    discover_images,
    write_json_atomic,
)
from object_memory.progress import (  # noqa: E402
    JsonlProgressWriter,
    ProgressReporter,
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
    parser.add_argument("--dinov3-model-path")
    parser.add_argument("--revision")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument(
        "--validate-demo",
        action="store_true",
        help=(
            "Require coverage for at least two new objects, one existing match, "
            "one duplicate image, and no pending or failed proposals."
        ),
    )
    parser.add_argument(
        "--report",
        help="Optional second copy of the run report, such as an environment report.",
    )
    parser.add_argument(
        "--progress-file",
        help=(
            "Optional JSONL progress event file. Each event is flushed "
            "immediately for live monitoring."
        ),
    )
    return parser.parse_args()


def runtime_config(base: AppConfig, args: argparse.Namespace) -> AppConfig:
    payload = base.model_dump(mode="python")
    if args.checkpoint:
        payload["models"]["sam3_checkpoint"] = args.checkpoint
    if args.qwen_model:
        payload["models"]["qwen_model_id"] = args.qwen_model
    if args.dinov3_model_path:
        payload["models"]["dinov3_model_path"] = args.dinov3_model_path
    return AppConfig.model_validate(payload)


def resolve_checkpoint(config: AppConfig) -> Path:
    checkpoint = config.models.sam3_checkpoint.expanduser()
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    return checkpoint.resolve()


def resolve_dinov3_model_path(config: AppConfig) -> Path:
    model_path = config.models.dinov3_model_path.expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    return model_path.resolve()


def validate_directory_separation(input_root: Path, memory_root: Path) -> None:
    if input_root == memory_root:
        raise ValueError("Input directory and memory root must be different")
    if input_root.is_relative_to(memory_root) or memory_root.is_relative_to(input_root):
        raise ValueError(
            "Input directory and memory root must not contain one another"
        )


def failure_report(
    exc: Exception,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 7,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test": "object_memory_demo_single_pass_dinov3",
        "status": "failed",
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }
    if run_id:
        report["run"] = {"run_id": run_id}
        report["run_report"] = f"run_reports/{run_id}.json"
    return report


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
        "no_pending_or_failed_proposals": (
            proposals["pending"] == 0 and proposals["failed"] == 0
        ),
    }
    report["demo_coverage"] = coverage
    report["demo_observations"] = {
        "filtered_candidate_observed": proposals["filtered"] >= 1
    }
    if report["status"] != "passed" or not all(coverage.values()):
        report["pipeline_status"] = report["status"]
        report["status"] = "failed"


def main() -> int:
    args = parse_args()
    progress: ProgressReporter | None = None
    paths: MemoryPaths | None = None
    try:
        if args.progress_file:
            progress_path = Path(args.progress_file).expanduser().resolve()
            progress = ProgressReporter(JsonlProgressWriter(progress_path))
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
        )
        mllm_runtime = QwenMllmAdapter(
            config.models.qwen_model_id,
            revision=args.revision,
            allow_network=args.allow_network,
            max_pixels=config.mllm_pipeline.max_pixels,
            max_new_tokens=config.mllm_pipeline.max_new_tokens,
        )
        dino_runtime = DinoV3Adapter(
            resolve_dinov3_model_path(config),
            model_id=config.models.dinov3_model_id,
            revision=config.models.dinov3_revision,
            settings=config.visual_fingerprint,
        )
        pipeline = ObjectMemoryPipeline(
            config=config,
            paths=paths,
            sam_runtime=sam_runtime,
            mllm_runtime=mllm_runtime,
            dino_runtime=dino_runtime,
            progress=progress,
        )
        report = pipeline.run(image_paths)
        if args.validate_demo:
            add_demo_coverage(report)
            write_json_atomic(paths.resolve_asset(report["run_report"]), report)
        if args.report:
            write_json_atomic(Path(args.report).expanduser().resolve(), report)
        if progress is not None:
            progress.emit(
                event="cli_completed",
                stage="cli",
                status=report["status"],
                current=1,
                total=1,
                overall_percent=100.0,
                message=(
                    "Command-line run completed with "
                    f"report status={report['status']}"
                ),
                data={
                    "report_status": report["status"],
                    "external_report": (
                        str(Path(args.report).expanduser().resolve())
                        if args.report
                        else None
                    ),
                },
            )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "passed" else 4
    except Exception as exc:  # noqa: BLE001 - CLI must preserve failure evidence
        run_id = progress.run_id if progress is not None else None
        report = failure_report(exc, run_id=run_id)
        if progress is not None:
            try:
                progress.emit(
                    event="cli_failed",
                    stage="cli",
                    status="failed",
                    current=0,
                    total=0,
                    overall_percent=progress.last_overall_percent,
                    message=f"Command-line run failed: {type(exc).__name__}: {exc}",
                    data={
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    },
                )
            except Exception as progress_exc:  # noqa: BLE001 - expose both failures
                report["progress_error"] = {
                    "type": type(progress_exc).__name__,
                    "message": str(progress_exc),
                }
        writable_memory = False
        if paths is not None:
            try:
                MemoryStore(paths).status()
                writable_memory = True
            except (FileNotFoundError, MemoryStoreError):
                pass
        if writable_memory and paths is not None and run_id:
            try:
                write_json_atomic(paths.run_reports / f"{run_id}.json", report)
            except Exception as internal_report_exc:  # noqa: BLE001
                report["internal_report_error"] = {
                    "type": type(internal_report_exc).__name__,
                    "message": str(internal_report_exc),
                }
        if args.report:
            write_json_atomic(Path(args.report).expanduser().resolve(), report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
