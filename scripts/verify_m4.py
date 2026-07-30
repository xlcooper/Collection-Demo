#!/usr/bin/env python3
"""Verify the deterministic M4 SQLite memory loop in a temporary store."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from object_memory.assets import MemoryPaths  # noqa: E402
from object_memory.memory_loop import MemoryLoop  # noqa: E402
from object_memory.memory_store import MemoryStore, MemoryStoreError  # noqa: E402
from object_memory.schemas import (  # noqa: E402
    BoundingBox,
    DecisionReasonCode,
    DecisionType,
    MllmResponse,
    ObjectAnnotation,
    Proposal,
    ProposalStatus,
    Run,
    RunStatus,
    SourceImage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the M4 memory loop.")
    parser.add_argument(
        "--report",
        default="environment/m4_memory_loop_report.json",
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


def make_run(name: str) -> Run:
    return Run(
        id=f"run_m4_{name}",
        config_digest="a" * 64,
        sam_model_id="sam3-m4-fixture",
        qwen_model_id="qwen-m4-fixture",
    )


def make_source(run: Run, name: str, digest_character: str) -> SourceImage:
    return SourceImage(
        id=f"src_m4_{name}",
        run_id=run.id,
        sha256=digest_character * 64,
        relative_path=f"sources/{name}.jpg",
        width=64,
        height=48,
    )


def make_proposal(
    paths: MemoryPaths,
    run: Run,
    source_id: str,
    name: str,
) -> Proposal:
    proposal_id = f"prop_m4_{name}"
    directory = paths.proposal_dir(run.id, proposal_id)
    directory.mkdir(parents=True)
    assets = {
        "crop": directory / "crop.png",
        "mask": directory / "mask.png",
        "overlay": directory / "overlay.jpg",
    }
    for role, path in assets.items():
        path.write_bytes(f"m4-{name}-{role}".encode("utf-8"))
    return Proposal(
        id=proposal_id,
        source_image_id=source_id,
        raw_candidate_id=f"candidate-{name}",
        prompt="cup",
        score=0.91,
        bbox=BoundingBox(x_min=2, y_min=3, x_max=40, y_max=42),
        mask_area_pixels=400,
        mask_area_ratio=0.13,
        crop_path=paths.relative_asset(assets["crop"]),
        mask_path=paths.relative_asset(assets["mask"]),
        overlay_path=paths.relative_asset(assets["overlay"]),
    )


def annotation() -> ObjectAnnotation:
    return ObjectAnnotation(
        coarse_category="cup",
        fine_category="coffee cup",
        material=["ceramic"],
        color=["white"],
        shape="round with handle",
        description="white ceramic cup with a round handle",
        annotation_confidence=0.98,
    )


def run_verification() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary_directory:
        paths = MemoryPaths(Path(temporary_directory) / "memory")
        store = MemoryStore(paths)
        store.initialize()
        loop = MemoryLoop(store)

        new_run = make_run("new")
        loop.begin_run(new_run)
        first_source = make_source(new_run, "first", "b")
        first_registration = loop.register_source(first_source)
        new_proposal = make_proposal(
            paths,
            new_run,
            first_registration.source_id,
            "new",
        )
        new_write = loop.apply_response(
            proposal=new_proposal,
            response=MllmResponse(
                decision=DecisionType.NEW,
                confidence=0.95,
                reason_code=DecisionReasonCode.NEW_OBJECT,
                short_reason="no matching object card",
                annotation=annotation(),
            ),
            prompt_version="m3-object-identity-v2",
        )
        loop.complete_source(first_registration.source_id)
        new_summary = loop.complete_run(new_run.id)

        existing_run = make_run("existing")
        loop.begin_run(existing_run)
        second_source = make_source(existing_run, "second", "c")
        second_registration = loop.register_source(second_source)
        existing_proposal = make_proposal(
            paths,
            existing_run,
            second_registration.source_id,
            "existing",
        )
        existing_write = loop.apply_response(
            proposal=existing_proposal,
            response=MllmResponse(
                decision=DecisionType.EXISTING,
                matched_object_id=new_write.object_id,
                confidence=0.99,
                reason_code=DecisionReasonCode.VISUAL_INSTANCE_MATCH,
                short_reason="same physical cup",
                annotation=annotation(),
            ),
            prompt_version="m3-object-identity-v2",
        )
        loop.complete_source(second_registration.source_id)
        existing_summary = loop.complete_run(existing_run.id)

        outcome_run = make_run("outcomes")
        loop.begin_run(outcome_run)
        outcome_source = make_source(outcome_run, "outcomes", "d")
        outcome_registration = loop.register_source(outcome_source)
        filtered = Proposal(
            id="prop_m4_filtered",
            source_image_id=outcome_registration.source_id,
            raw_candidate_id="candidate-filtered",
            prompt="cup",
            score=0.2,
            bbox=BoundingBox(x_min=1, y_min=1, x_max=5, y_max=5),
            status=ProposalStatus.FILTERED,
            filter_reason="below_confidence_threshold",
        )
        loop.record_filtered_proposal(filtered)

        uncertain = make_proposal(
            paths,
            outcome_run,
            outcome_registration.source_id,
            "uncertain",
        )
        uncertain_write = loop.apply_response(
            proposal=uncertain,
            response=MllmResponse(
                decision=DecisionType.UNCERTAIN,
                confidence=0.5,
                reason_code=DecisionReasonCode.INSUFFICIENT_EVIDENCE,
                short_reason="insufficient visible evidence",
            ),
            prompt_version="m3-object-identity-v2",
        )
        pending_summary = store.run_summary(outcome_run.id)
        resolved_write = loop.apply_response(
            proposal=uncertain,
            response=MllmResponse(
                decision=DecisionType.EXISTING,
                matched_object_id=new_write.object_id,
                confidence=0.9,
                reason_code=DecisionReasonCode.VISUAL_INSTANCE_MATCH,
                short_reason="retry resolved the pending identity",
                annotation=annotation(),
            ),
            prompt_version="m3-object-identity-v2",
            attempt=2,
        )

        ignored = make_proposal(
            paths,
            outcome_run,
            outcome_registration.source_id,
            "ignored",
        )
        ignored_write = loop.apply_response(
            proposal=ignored,
            response=MllmResponse(
                decision=DecisionType.IGNORED,
                confidence=0.9,
                reason_code=DecisionReasonCode.INVALID_CANDIDATE,
                short_reason="not a valid independent object",
            ),
            prompt_version="m3-object-identity-v2",
        )

        failed = make_proposal(
            paths,
            outcome_run,
            outcome_registration.source_id,
            "failed",
        )
        invalid_match_rolled_back = False
        try:
            loop.apply_response(
                proposal=failed,
                response=MllmResponse(
                    decision=DecisionType.EXISTING,
                    matched_object_id="obj_missing",
                    confidence=0.99,
                    reason_code=DecisionReasonCode.VISUAL_INSTANCE_MATCH,
                    short_reason="invalid fixture match",
                    annotation=annotation(),
                ),
                prompt_version="m3-object-identity-v2",
            )
        except MemoryStoreError:
            counts_after_rollback = store.status().counts
            invalid_match_rolled_back = (
                counts_after_rollback["proposals"] == 5
                and counts_after_rollback["objects"] == 1
                and counts_after_rollback["observations"] == 3
                and counts_after_rollback["decisions"] == 5
                and not (paths.objects / "obj_missing").exists()
            )
        loop.record_proposal_failure(failed, "matched object is missing")
        loop.complete_source(outcome_registration.source_id)
        outcome_summary = loop.complete_run(outcome_run.id)

        duplicate_run = make_run("duplicate")
        loop.begin_run(duplicate_run)
        duplicate_source = make_source(duplicate_run, "duplicate", "b")
        duplicate_registration = loop.register_source(duplicate_source)
        duplicate_summary = loop.complete_run(duplicate_run.id)

        cards = loop.object_cards(max_reference_views=2)
        final_counts = store.status().counts
        promoted_views_exist = bool(cards) and all(
            paths.resolve_asset(relative_path).is_file()
            for relative_path in cards[0].representative_view_paths
        )
        checks = {
            "new_creates_object_and_observation": (
                new_write.decision is DecisionType.NEW
                and new_write.object_id is not None
                and new_write.observation_id is not None
            ),
            "existing_reuses_object_and_adds_observation": (
                existing_write.decision is DecisionType.EXISTING
                and existing_write.object_id == new_write.object_id
                and final_counts["objects"] == 1
                and final_counts["observations"] == 3
            ),
            "completed_source_hash_is_skipped": (
                duplicate_registration.duplicate
                and duplicate_registration.source_id == first_registration.source_id
                and duplicate_summary.duplicate_sources_skipped == 1
                and final_counts["source_images"] == 3
            ),
            "filtered_candidate_is_recorded_without_qwen_decision": (
                outcome_summary.proposal_counts["filtered"] == 1
            ),
            "ignored_is_terminal_without_object_write": (
                ignored_write.decision is DecisionType.IGNORED
                and ignored_write.object_id is None
                and outcome_summary.decision_counts["ignored"] == 1
            ),
            "uncertain_is_pending_then_retryable": (
                uncertain_write.decision is DecisionType.UNCERTAIN
                and uncertain_write.proposal_status is ProposalStatus.PENDING
                and pending_summary.proposal_counts["pending"] == 1
                and resolved_write.decision is DecisionType.EXISTING
                and resolved_write.object_id == new_write.object_id
                and outcome_summary.proposal_counts["pending"] == 0
            ),
            "invalid_existing_rolls_back_before_failed_record": (
                invalid_match_rolled_back
                and outcome_summary.proposal_counts["failed"] == 1
            ),
            "object_cards_read_structured_labels_and_views": (
                len(cards) == 1
                and cards[0].object_id == new_write.object_id
                and cards[0].fine_category == "coffee cup"
                and len(cards[0].representative_view_paths) == 2
                and promoted_views_exist
            ),
            "run_status_reflects_pending_or_failed_work": (
                new_summary.status is RunStatus.COMPLETED
                and existing_summary.status is RunStatus.COMPLETED
                and outcome_summary.status is RunStatus.COMPLETED_WITH_ERRORS
                and duplicate_summary.status is RunStatus.COMPLETED
            ),
            "final_core_counts_expected": final_counts
            == {
                "runs": 4,
                "source_images": 3,
                "proposals": 6,
                "objects": 1,
                "observations": 3,
                "decisions": 5,
            },
        }
        return {
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "core_counts": final_counts,
            "runs": {
                "new": new_summary.as_dict(),
                "existing": existing_summary.as_dict(),
                "outcomes": outcome_summary.as_dict(),
                "duplicate": duplicate_summary.as_dict(),
            },
            "object_card": cards[0].model_dump(mode="json") if cards else None,
            "scope": (
                "deterministic SQLite and asset-write verification; real SAM3 and "
                "Qwen orchestration remains M5"
            ),
        }


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test": "m4_memory_loop",
        "status": "failed",
    }
    try:
        report.update(run_verification())
        return_code = 0 if report["status"] == "passed" else 4
    except Exception as exc:  # noqa: BLE001 - the report must survive failures
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        return_code = 1

    report_path = Path(args.report).expanduser().resolve()
    write_json(report_path, report)
    print(f"M4 verification report: {report_path}")
    print(f"Status: {report['status']}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
