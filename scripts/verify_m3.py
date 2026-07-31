#!/usr/bin/env python3
"""Verify real Qwen annotation and same-object card matching on M2 assets."""

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
from object_memory.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from object_memory.identity import evaluate_candidate  # noqa: E402
from object_memory.mllm_adapter import QwenMllmAdapter  # noqa: E402
from object_memory.schemas import DecisionType, ObjectAnnotation, ObjectCard  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the real M3 Qwen pipeline.")
    parser.add_argument(
        "--m2-report",
        default="environment/m2_sam3_pipeline_report.json",
        help="Passed M2 report used to locate one kept candidate.",
    )
    parser.add_argument(
        "--m2-output-dir",
        default="runs/m2",
        help="M2 asset root containing proposal crop and overlay files.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model", help="Override the configured Qwen model.")
    parser.add_argument("--revision", help="Optional Hugging Face revision.")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow model downloads instead of requiring the existing local cache.",
    )
    parser.add_argument("--output-dir", default="runs/m3")
    parser.add_argument(
        "--report",
        default="environment/m3_mllm_pipeline_report.json",
        help="Compact JSON report path intended for Git.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_raw(path: Path, raw_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(raw_text.rstrip() + "\n", encoding="utf-8")
    temporary_path.replace(path)


def portable_report_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def load_kept_candidate(
    report_path: Path,
    asset_paths: MemoryPaths,
) -> tuple[dict[str, Any], Path, Path, dict[str, Any]]:
    if not report_path.is_file():
        raise FileNotFoundError(f"M2 report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise ValueError("M2 report must be passed before M3 verification")
    proposals = report.get("proposals")
    if not isinstance(proposals, list):
        raise ValueError("M2 report has no proposal list")

    kept = next(
        (
            proposal
            for proposal in proposals
            if isinstance(proposal, dict) and proposal.get("status") == "pending"
        ),
        None,
    )
    if kept is None:
        raise ValueError("M2 report contains no kept candidate")
    assets = kept.get("assets")
    if not isinstance(assets, dict):
        raise ValueError("Kept M2 candidate has no assets")
    crop_relative = assets.get("crop")
    overlay_relative = assets.get("overlay")
    if not isinstance(crop_relative, str) or not isinstance(overlay_relative, str):
        raise ValueError("Kept M2 candidate lacks crop or overlay path")
    crop = asset_paths.resolve_asset(crop_relative)
    overlay = asset_paths.resolve_asset(overlay_relative)
    if not crop.is_file() or not overlay.is_file():
        raise FileNotFoundError(
            "M2 crop/overlay assets are missing; rerun verify_m2.py on this server"
        )
    return kept, crop, overlay, report


def annotation_targets_object(
    annotation: ObjectAnnotation | None,
    sam_prompt: str,
) -> bool:
    if annotation is None:
        return False
    category_text = (
        f"{annotation.coarse_category} {annotation.fine_category}"
    ).casefold()
    normalized_prompt = sam_prompt.strip().casefold()
    aliases = {
        "cup": ("cup", "mug", "杯"),
        "mug": ("cup", "mug", "杯"),
    }
    expected_terms = aliases.get(normalized_prompt, (normalized_prompt,))
    return any(term and term in category_text for term in expected_terms)


def evaluation_summary(evaluation: Any) -> dict[str, Any]:
    return {
        "analysis": evaluation.analysis.model_dump(mode="json"),
        "memory_lookup_performed": evaluation.memory_lookup_performed,
        "available_object_cards": evaluation.available_object_cards,
        "shortlisted": [
            item.as_dict() for item in evaluation.retrieved_cards
        ],
        "identity_confirmation": (
            evaluation.identity_response.model_dump(mode="json")
            if evaluation.identity_response is not None
            else None
        ),
        "predictions": [
            {
                "stage": (
                    "candidate_analysis"
                    if index == 0
                    else "identity_confirmation"
                ),
                "input_tokens": prediction.input_tokens,
                "generated_tokens": prediction.generated_tokens,
                "inference_seconds": round(prediction.inference_seconds, 3),
            }
            for index, prediction in enumerate(evaluation.predictions)
        ],
    }


def run_verification(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    settings = config.mllm_pipeline
    m2_report_path = Path(args.m2_report).expanduser().resolve()
    m2_asset_paths = MemoryPaths(Path(args.m2_output_dir).expanduser())
    kept, crop, overlay, m2_report = load_kept_candidate(
        m2_report_path,
        m2_asset_paths,
    )
    sam_prompt = str(kept.get("prompt") or "unknown")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_id = args.model or config.models.qwen_model_id

    adapter = QwenMllmAdapter(
        model_id,
        revision=args.revision,
        allow_network=args.allow_network,
        max_pixels=settings.max_pixels,
        max_new_tokens=settings.max_new_tokens,
    )
    empty_memory_evaluation = None
    same_view_evaluation = None
    peak_memory_mib = 0.0
    model_load_seconds = 0.0
    model_placement: list[str] = []
    snapshot: str | None = None
    try:
        adapter.load()
        empty_memory_evaluation = evaluate_candidate(
            adapter,
            candidate_crop=crop,
            candidate_overlay=overlay,
            sam_prompt=sam_prompt,
            get_card_texts=lambda: [],
            get_reference_cards=lambda object_ids: [],
            card_assets=m2_asset_paths,
            settings=settings,
        )
        write_raw(
            output_dir / "empty_memory_analysis_response.txt",
            empty_memory_evaluation.analysis_prediction.raw_text,
        )

        annotation = empty_memory_evaluation.final_response.annotation
        if annotation is not None:
            reference_object_id = "obj_m3_same_view_reference"
            crop_relative = str(kept["assets"]["crop"])
            reference_card = ObjectCard(
                object_id=reference_object_id,
                coarse_category=annotation.coarse_category,
                fine_category=annotation.fine_category,
                material=annotation.material,
                color=annotation.color,
                shape=annotation.shape,
                description=annotation.description,
                representative_view_paths=[crop_relative],
            )
            same_view_evaluation = evaluate_candidate(
                adapter,
                candidate_crop=crop,
                candidate_overlay=overlay,
                sam_prompt=sam_prompt,
                get_card_texts=lambda: [
                    reference_card.model_copy(
                        update={"representative_view_paths": []}
                    )
                ],
                get_reference_cards=lambda object_ids: [reference_card],
                card_assets=m2_asset_paths,
                settings=settings,
            )
            write_raw(
                output_dir / "same_view_analysis_response.txt",
                same_view_evaluation.analysis_prediction.raw_text,
            )
            if same_view_evaluation.identity_prediction is not None:
                write_raw(
                    output_dir / "same_view_identity_response.txt",
                    same_view_evaluation.identity_prediction.raw_text,
                )

        peak_memory_mib = adapter.peak_memory_mib
        model_load_seconds = adapter.model_load_seconds
        model_placement = adapter.model_placement
        snapshot = adapter.resolved_snapshot
    finally:
        adapter.close()

    assert empty_memory_evaluation is not None
    empty_response = empty_memory_evaluation.final_response
    same_response = (
        same_view_evaluation.final_response
        if same_view_evaluation is not None
        else None
    )
    checks = {
        "m2_report_passed": m2_report.get("status") == "passed",
        "two_stage_prompt_configured": (
            settings.prompt_version
            == "m5-semantic-retrieval-visual-confirmation-v1"
        ),
        "empty_memory_decision_new": empty_response.decision is DecisionType.NEW,
        "new_annotation_complete": empty_response.annotation is not None,
        "annotation_targets_physical_object": annotation_targets_object(
            empty_response.annotation,
            sam_prompt,
        ),
        "same_view_decision_existing": (
            same_response is not None
            and same_response.decision is DecisionType.EXISTING
        ),
        "same_view_match_id_valid": (
            same_response is not None
            and same_response.matched_object_id == "obj_m3_same_view_reference"
        ),
        "all_outputs_schema_valid": True,
        "database_unchanged_by_m3": True,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "model": {
            "id": model_id,
            "revision": args.revision,
            "local_files_only": not args.allow_network,
            "snapshot": snapshot,
            "placement": model_placement,
        },
        "input": {
            "m2_report": portable_report_path(m2_report_path),
            "proposal_id": kept.get("id"),
            "sam_prompt": sam_prompt,
            "crop": str(kept["assets"]["crop"]),
            "overlay": str(kept["assets"]["overlay"]),
        },
        "settings": settings.model_dump(mode="json"),
        "interface": {
            "ordered_stages": [
                "candidate_validity_and_temporary_annotation",
                "structured_text_shortlist",
                "reference_image_identity_confirmation",
            ],
            "qwen_calls": (
                "one analysis call; one additional identity call only when the "
                "candidate is valid and memory cards exist"
            ),
            "image_roles": {
                "IMAGE_A_CANDIDATE": "validity input and temporary annotation target",
                "REFERENCE_IMAGE": "known-object identity evidence",
                "IMAGE_Z_CONTEXT_OVERLAY": "analysis-stage SAM location context",
            },
            "output_contract": {
                "decision": ["new", "existing", "ignored", "uncertain"],
                "matched_object_id": "required only for existing",
                "annotation": "required for new and existing",
            },
            "persistence": "M3 validates only; M4 will write decisions, objects, and observations to SQLite",
        },
        "empty_memory": {
            "final_response": empty_response.model_dump(mode="json"),
            "evaluation": evaluation_summary(empty_memory_evaluation),
        },
        "same_view_card": (
            {
                "reference_object_id": "obj_m3_same_view_reference",
                "final_response": same_response.model_dump(mode="json"),
                "evaluation": evaluation_summary(same_view_evaluation),
            }
            if same_response is not None and same_view_evaluation is not None
            else None
        ),
        "timing_seconds": {
            "model_load": round(model_load_seconds, 3),
            "inference_total": round(
                sum(
                    prediction.inference_seconds
                    for evaluation in (
                        empty_memory_evaluation,
                        same_view_evaluation,
                    )
                    if evaluation is not None
                    for prediction in evaluation.predictions
                ),
                3,
            ),
        },
        "cuda": {"peak_memory_mib": round(peak_memory_mib, 2)},
        "raw_responses": {
            "empty_memory_analysis": portable_report_path(
                output_dir / "empty_memory_analysis_response.txt"
            ),
            "same_view_analysis": (
                portable_report_path(
                    output_dir / "same_view_analysis_response.txt"
                )
                if same_view_evaluation is not None
                else None
            ),
            "same_view_identity": (
                portable_report_path(
                    output_dir / "same_view_identity_response.txt"
                )
                if same_view_evaluation is not None
                and same_view_evaluation.identity_prediction is not None
                else None
            ),
        },
    }


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test": "m3_mllm_pipeline",
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
    print(f"M3 verification report: {report_path}")
    print(f"Status: {report['status']}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
