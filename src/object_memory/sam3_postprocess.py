"""Deterministic SAM3 filtering, deduplication, and asset generation."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from .assets import MemoryPaths
from .config import Sam3PipelineConfig
from .sam3_adapter import RawSamCandidate
from .schemas import BoundingBox, Proposal, ProposalStatus, utc_now


@dataclass(frozen=True, slots=True)
class PostprocessResult:
    proposals: tuple[Proposal, ...]
    kept: tuple[Proposal, ...]
    filtered: tuple[Proposal, ...]
    filter_counts: dict[str, int]


@dataclass(slots=True)
class _PreparedCandidate:
    raw: RawSamCandidate
    proposal: Proposal


def process_candidates(
    candidates: list[RawSamCandidate] | tuple[RawSamCandidate, ...],
    *,
    image: Image.Image,
    source_image_id: str,
    run_id: str,
    paths: MemoryPaths,
    settings: Sam3PipelineConfig,
) -> PostprocessResult:
    """Filter candidates, deduplicate masks, and save assets for kept proposals."""

    rgb_image = image.convert("RGB")
    image_width, image_height = rgb_image.size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Source image dimensions must be positive")
    image_area = image_width * image_height
    paths.ensure_layout()

    prepared: list[_PreparedCandidate] = []
    filtered: list[Proposal] = []
    filter_counts: Counter[str] = Counter()

    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: (-candidate.score, candidate.raw_candidate_id),
    )
    for raw in ranked_candidates:
        mask_area = int(np.count_nonzero(raw.mask))
        mask_area_ratio = mask_area / image_area
        bbox = _clamp_bbox(raw.bbox_xyxy, image_width, image_height)
        safe_bbox = bbox or BoundingBox(
            x_min=0.0,
            y_min=0.0,
            x_max=float(min(image_width, 1)),
            y_max=float(min(image_height, 1)),
        )
        proposal = Proposal(
            source_image_id=source_image_id,
            raw_candidate_id=raw.raw_candidate_id,
            prompt=raw.prompt,
            score=raw.score,
            bbox=safe_bbox,
            mask_area_pixels=mask_area,
            mask_area_ratio=mask_area_ratio,
        )

        reason: str | None = None
        if raw.score < settings.confidence_threshold:
            reason = "low_confidence"
        elif raw.mask.shape != (image_height, image_width):
            reason = "mask_shape_mismatch"
        elif mask_area == 0:
            reason = "empty_mask"
        elif mask_area_ratio < settings.min_mask_area_ratio:
            reason = "mask_too_small"
        elif bbox is None:
            reason = "invalid_bbox"

        if reason is not None:
            _mark_filtered(proposal, reason)
            filtered.append(proposal)
            filter_counts[reason] += 1
            continue
        prepared.append(_PreparedCandidate(raw=raw, proposal=proposal))

    kept_prepared: list[_PreparedCandidate] = []
    for candidate in prepared:
        duplicate_of: Proposal | None = None
        for kept_candidate in kept_prepared:
            if (
                mask_iou(candidate.raw.mask, kept_candidate.raw.mask)
                >= settings.duplicate_mask_iou_threshold
            ):
                duplicate_of = kept_candidate.proposal
                break
        if duplicate_of is not None:
            reason = f"duplicate_mask:{duplicate_of.id}"
            _mark_filtered(candidate.proposal, reason)
            filtered.append(candidate.proposal)
            filter_counts["duplicate_mask"] += 1
            continue
        kept_prepared.append(candidate)

    kept: list[Proposal] = []
    for candidate in kept_prepared:
        _save_candidate_assets(
            candidate,
            image=rgb_image,
            run_id=run_id,
            paths=paths,
            settings=settings,
        )
        kept.append(candidate.proposal)

    proposals = [candidate.proposal for candidate in kept_prepared] + filtered
    return PostprocessResult(
        proposals=tuple(proposals),
        kept=tuple(kept),
        filtered=tuple(filtered),
        filter_counts=dict(sorted(filter_counts.items())),
    )


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Return binary mask intersection-over-union."""

    if first.shape != second.shape:
        return 0.0
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return intersection / union if union else 0.0


def _clamp_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> BoundingBox | None:
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0.0, min(float(width), x_min))
    y_min = max(0.0, min(float(height), y_min))
    x_max = max(0.0, min(float(width), x_max))
    y_max = max(0.0, min(float(height), y_max))
    if x_max <= x_min or y_max <= y_min:
        return None
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def _mark_filtered(proposal: Proposal, reason: str) -> None:
    proposal.status = ProposalStatus.FILTERED
    proposal.filter_reason = reason
    proposal.updated_at = utc_now()


def _save_candidate_assets(
    candidate: _PreparedCandidate,
    *,
    image: Image.Image,
    run_id: str,
    paths: MemoryPaths,
    settings: Sam3PipelineConfig,
) -> None:
    proposal = candidate.proposal
    output_dir = paths.proposal_dir(run_id, proposal.id)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_path = output_dir / "mask.png"
    crop_path = output_dir / "crop.png"
    overlay_path = output_dir / "overlay.jpg"

    mask_image = Image.fromarray(candidate.raw.mask.astype(np.uint8) * 255, mode="L")
    _save_image_atomic(mask_image, mask_path, "PNG")

    crop_box = _crop_box(
        proposal.bbox,
        image.width,
        image.height,
        settings.crop_padding_pixels,
    )
    _save_image_atomic(image.crop(crop_box), crop_path, "PNG")

    overlay = _make_overlay(
        image,
        candidate.raw.mask,
        proposal.bbox,
        settings.overlay_color,
        settings.overlay_alpha,
    )
    _save_image_atomic(overlay, overlay_path, "JPEG")

    proposal.mask_path = paths.relative_asset(mask_path)
    proposal.crop_path = paths.relative_asset(crop_path)
    proposal.overlay_path = paths.relative_asset(overlay_path)
    proposal.updated_at = utc_now()


def _crop_box(
    bbox: BoundingBox,
    width: int,
    height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    left = max(0, math.floor(bbox.x_min) - padding)
    top = max(0, math.floor(bbox.y_min) - padding)
    right = min(width, math.ceil(bbox.x_max) + padding)
    bottom = min(height, math.ceil(bbox.y_max) + padding)
    return left, top, right, bottom


def _make_overlay(
    image: Image.Image,
    mask: np.ndarray,
    bbox: BoundingBox,
    color: tuple[int, int, int],
    alpha: float,
) -> Image.Image:
    image_array = np.asarray(image, dtype=np.uint8).copy()
    color_array = np.asarray(color, dtype=np.float32)
    selected = image_array[mask].astype(np.float32)
    image_array[mask] = np.clip(
        selected * (1.0 - alpha) + color_array * alpha,
        0,
        255,
    ).astype(np.uint8)
    overlay = Image.fromarray(image_array, mode="RGB")
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(
        (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max),
        outline=color,
        width=3,
    )
    return overlay


def _save_image_atomic(image: Image.Image, path: Path, image_format: str) -> None:
    temporary_path = path.with_name(
        f".{path.stem}.{uuid4().hex}.tmp{path.suffix}"
    )
    try:
        save_kwargs = {"quality": 92} if image_format == "JPEG" else {}
        image.save(temporary_path, format=image_format, **save_kwargs)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)

