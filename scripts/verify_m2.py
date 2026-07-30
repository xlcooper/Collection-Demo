#!/usr/bin/env python3
"""Verify SAM3 candidate generation without loading Qwen."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from object_memory.assets import MemoryPaths
from object_memory.config import (
    DEFAULT_CONFIG_PATH,
    Sam3PipelineConfig,
    load_config,
)
from object_memory.sam3_adapter import Sam3Adapter
from object_memory.sam3_postprocess import PostprocessResult, process_candidates
from object_memory.schemas import Proposal, SourceImage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify real SAM3 candidates.")
    parser.add_argument("--image", required=True, help="Source scene image path.")
    parser.add_argument(
        "--prompt",
        action="append",
        help=(
            "Optional historical category prompt. Omit it to verify the current "
            "automatic point-grid strategy."
        ),
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint", help="Override the configured checkpoint.")
    parser.add_argument("--output-dir", default="runs/m2")
    parser.add_argument(
        "--report",
        default="environment/m2_sam3_pipeline_report.json",
        help="Compact JSON report path intended for Git.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def proposal_summary(proposal: Proposal) -> dict[str, Any]:
    return {
        "id": proposal.id,
        "raw_candidate_id": proposal.raw_candidate_id,
        "prompt": proposal.prompt,
        "score": round(proposal.score, 6),
        "bbox": proposal.bbox.model_dump(),
        "mask_area_pixels": proposal.mask_area_pixels,
        "mask_area_ratio": round(proposal.mask_area_ratio, 8),
        "status": proposal.status.value,
        "filter_reason": proposal.filter_reason,
        "assets": {
            "crop": proposal.crop_path,
            "mask": proposal.mask_path,
            "overlay": proposal.overlay_path,
        },
    }


def inspect_assets(
    result: PostprocessResult,
    paths: MemoryPaths,
    source_size: tuple[int, int],
) -> dict[str, bool]:
    complete = True
    crops_nonempty = True
    masks_binary = True
    masks_match_source = True
    overlays_match_source = True
    for proposal in result.kept:
        if not proposal.crop_path or not proposal.mask_path or not proposal.overlay_path:
            complete = False
            continue
        crop_path = paths.resolve_asset(proposal.crop_path)
        mask_path = paths.resolve_asset(proposal.mask_path)
        overlay_path = paths.resolve_asset(proposal.overlay_path)
        if not all(path.is_file() for path in (crop_path, mask_path, overlay_path)):
            complete = False
            continue
        with Image.open(crop_path) as crop:
            crops_nonempty = crops_nonempty and crop.width > 0 and crop.height > 0
        with Image.open(mask_path) as mask:
            masks_match_source = masks_match_source and mask.size == source_size
            colors = mask.convert("L").getcolors(maxcolors=256) or []
            masks_binary = masks_binary and all(
                value in {0, 255} for _, value in colors
            )
        with Image.open(overlay_path) as overlay:
            overlays_match_source = overlays_match_source and overlay.size == source_size
    return {
        "kept_assets_complete": complete,
        "crop_assets_nonempty": crops_nonempty,
        "mask_assets_binary": masks_binary,
        "mask_dimensions_match_source": masks_match_source,
        "overlay_dimensions_match_source": overlays_match_source,
    }


def run_verification(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    settings_payload = config.sam3_pipeline.model_dump()
    if args.prompt:
        settings_payload["prompt_strategy"] = "explicit_category_list"
        settings_payload["prompts"] = args.prompt
    else:
        settings_payload["prompt_strategy"] = "automatic_point_grid"
        settings_payload["prompts"] = []
    settings = Sam3PipelineConfig.model_validate(settings_payload)

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Source image not found: {image_path}")
    checkpoint = Path(args.checkpoint or config.models.sam3_checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    checkpoint = checkpoint.resolve()

    output_root = Path(args.output_dir).expanduser().resolve()
    paths = MemoryPaths(output_root)
    paths.ensure_layout()
    with Image.open(image_path) as opened_image:
        image = opened_image.convert("RGB")
    image_sha256 = sha256_file(image_path)
    source_suffix = image_path.suffix.lower()
    if source_suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        source_suffix = ".png"
    source_asset = paths.sources / f"{image_sha256}{source_suffix}"
    if not source_asset.exists():
        if image_path.suffix.lower() == source_suffix:
            shutil.copy2(image_path, source_asset)
        else:
            image.save(source_asset, format="PNG")

    generated_at = datetime.now(timezone.utc)
    run_prefix = "m2" if settings.prompts else "m5_auto"
    run_id = generated_at.strftime(f"{run_prefix}_%Y%m%dT%H%M%S%fZ")
    source = SourceImage(
        id=f"src_{image_sha256[:32]}",
        run_id=run_id,
        sha256=image_sha256,
        relative_path=paths.relative_asset(source_asset),
        width=image.width,
        height=image.height,
    )

    adapter = Sam3Adapter(
        checkpoint,
        settings.confidence_threshold,
        points_per_side=settings.points_per_side,
        points_per_batch=settings.points_per_batch,
    )
    try:
        adapter.load()
        prediction = (
            adapter.predict(image, settings.prompts)
            if settings.prompt_strategy == "explicit_category_list"
            else adapter.predict(image)
        )
        peak_memory_mib = adapter.peak_memory_mib
        model_load_seconds = adapter.model_load_seconds
    finally:
        adapter.close()

    result = process_candidates(
        prediction.candidates,
        image=image,
        source_image_id=source.id,
        run_id=run_id,
        paths=paths,
        settings=settings,
    )
    asset_checks = inspect_assets(result, paths, image.size)
    checks = {
        "candidate_strategy_valid": settings.prompt_strategy
        in {"automatic_point_grid", "explicit_category_list"},
        "raw_candidates_nonempty": len(prediction.candidates) > 0,
        "kept_candidates_nonempty": len(result.kept) > 0,
        "source_association_valid": all(
            proposal.source_image_id == source.id for proposal in result.proposals
        ),
        "filtered_candidates_explainable": all(
            proposal.filter_reason for proposal in result.filtered
        ),
        **asset_checks,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "strategy": {
            "name": settings.prompt_strategy,
            "prompts": settings.prompts,
            "external_category_prompts": bool(settings.prompts),
            "scope": (
                "automatic class-agnostic point-grid candidates"
                if settings.prompt_strategy == "automatic_point_grid"
                else "configured categories only"
            ),
        },
        "input": {
            "filename": image_path.name,
            "sha256": image_sha256,
            "width": image.width,
            "height": image.height,
            "stored_source": source.relative_path,
            "source_image_id": source.id,
        },
        "checkpoint": {
            "filename": checkpoint.name,
            "size_gib": round(checkpoint.stat().st_size / (1024**3), 3),
        },
        "candidate_source_counts": prediction.prompt_counts,
        "counts": {
            "raw": len(prediction.candidates),
            "kept": len(result.kept),
            "filtered": len(result.filtered),
        },
        "filter_counts": result.filter_counts,
        "settings": settings.model_dump(),
        "timing_seconds": {
            "model_load": round(model_load_seconds, 3),
            "candidate_inference": round(prediction.inference_seconds, 3),
        },
        "cuda": {"peak_memory_mib": round(peak_memory_mib, 2)},
        "proposals": [proposal_summary(item) for item in result.proposals],
    }


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test": (
            "m2_sam3_pipeline"
            if args.prompt
            else "sam3_automatic_candidate_generation"
        ),
        "status": "failed",
    }
    try:
        report.update(run_verification(args))
        return_code = 0 if report["status"] == "passed" else 4
    except Exception as exc:  # noqa: BLE001 - the report must survive failures
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        return_code = 1

    report_path = Path(args.report).expanduser().resolve()
    write_json(report_path, report)
    print(f"M2 verification report: {report_path}")
    print(f"Status: {report['status']}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
