"""Deterministic tests for M4 SQLite writes and object-card reads."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from object_memory.assets import MemoryPaths
from object_memory.memory_loop import MemoryLoop
from object_memory.memory_store import MemoryStore, MemoryStoreError
from object_memory.schemas import (
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


def make_run(name: str) -> Run:
    return Run(
        id=f"run_{name}",
        config_digest="a" * 64,
        sam_model_id="sam3-test",
        qwen_model_id="qwen-test",
    )


def make_source(run: Run, name: str, digest_character: str) -> SourceImage:
    return SourceImage(
        id=f"src_{name}",
        run_id=run.id,
        sha256=digest_character * 64,
        relative_path=f"sources/{name}.jpg",
        width=32,
        height=24,
    )


def make_proposal(
    paths: MemoryPaths,
    run: Run,
    source_id: str,
    name: str,
) -> Proposal:
    proposal_id = f"prop_{name}"
    directory = paths.proposal_dir(run.id, proposal_id)
    directory.mkdir(parents=True)
    asset_paths: dict[str, Path] = {
        "crop": directory / "crop.png",
        "mask": directory / "mask.png",
        "overlay": directory / "overlay.jpg",
    }
    for role, path in asset_paths.items():
        path.write_bytes(f"{name}-{role}".encode("utf-8"))
    return Proposal(
        id=proposal_id,
        source_image_id=source_id,
        raw_candidate_id=f"candidate-{name}",
        prompt="cup",
        score=0.9,
        bbox=BoundingBox(x_min=1, y_min=2, x_max=20, y_max=22),
        mask_area_pixels=100,
        mask_area_ratio=0.1,
        crop_path=paths.relative_asset(asset_paths["crop"]),
        mask_path=paths.relative_asset(asset_paths["mask"]),
        overlay_path=paths.relative_asset(asset_paths["overlay"]),
    )


def annotation() -> ObjectAnnotation:
    return ObjectAnnotation(
        coarse_category="容器",
        fine_category="马克杯",
        material=["陶瓷"],
        color=["白色"],
        shape="带把手的圆柱形",
        description="白色陶瓷马克杯",
        annotation_confidence=0.95,
    )


class M4MemoryLoopTests(unittest.TestCase):
    def test_new_existing_duplicate_and_card_query(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = MemoryPaths(Path(temporary_directory) / "memory")
            store = MemoryStore(paths)
            store.initialize()
            loop = MemoryLoop(store)

            new_run = make_run("new")
            loop.begin_run(new_run)
            first_source = make_source(new_run, "first", "b")
            first_registration = loop.register_source(first_source)
            first_proposal = make_proposal(
                paths,
                new_run,
                first_registration.source_id,
                "new",
            )
            new_result = loop.apply_response(
                proposal=first_proposal,
                response=MllmResponse(
                    decision=DecisionType.NEW,
                    confidence=0.96,
                    reason_code=DecisionReasonCode.NEW_OBJECT,
                    short_reason="没有匹配的已有对象",
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
            second_proposal = make_proposal(
                paths,
                existing_run,
                second_registration.source_id,
                "existing",
            )
            existing_result = loop.apply_response(
                proposal=second_proposal,
                response=MllmResponse(
                    decision=DecisionType.EXISTING,
                    matched_object_id=new_result.object_id,
                    confidence=0.98,
                    reason_code=DecisionReasonCode.VISUAL_INSTANCE_MATCH,
                    short_reason="视觉实例一致",
                    annotation=annotation(),
                ),
                prompt_version="m3-object-identity-v2",
            )
            loop.complete_source(second_registration.source_id)
            existing_summary = loop.complete_run(existing_run.id)

            duplicate_run = make_run("duplicate")
            loop.begin_run(duplicate_run)
            duplicate_source = make_source(duplicate_run, "duplicate", "b")
            duplicate_registration = loop.register_source(duplicate_source)
            duplicate_summary = loop.complete_run(duplicate_run.id)

            cards = loop.object_cards(max_reference_views=2)
            text_cards = loop.object_card_texts()
            hydrated_cards = loop.object_cards_by_ids(
                [new_result.object_id],
                max_reference_views=1,
            )
            counts = store.status().counts
            self.assertFalse(first_registration.duplicate)
            self.assertTrue(duplicate_registration.duplicate)
            self.assertEqual(duplicate_registration.source_id, first_source.id)
            self.assertEqual(new_result.object_id, existing_result.object_id)
            self.assertEqual(new_summary.status, RunStatus.COMPLETED)
            self.assertEqual(existing_summary.status, RunStatus.COMPLETED)
            self.assertEqual(duplicate_summary.status, RunStatus.COMPLETED)
            self.assertEqual(duplicate_summary.duplicate_sources_skipped, 1)
            self.assertEqual(counts["source_images"], 2)
            self.assertEqual(counts["objects"], 1)
            self.assertEqual(counts["observations"], 2)
            self.assertEqual(counts["decisions"], 2)
            self.assertEqual(len(cards), 1)
            self.assertEqual(cards[0].object_id, new_result.object_id)
            self.assertEqual(len(cards[0].representative_view_paths), 2)
            self.assertEqual(text_cards[0].representative_view_paths, [])
            self.assertEqual(
                len(hydrated_cards[0].representative_view_paths),
                1,
            )
            for relative_path in cards[0].representative_view_paths:
                self.assertTrue(paths.resolve_asset(relative_path).is_file())

    def test_filtered_ignored_uncertain_and_failure_are_traceable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = MemoryPaths(Path(temporary_directory) / "memory")
            store = MemoryStore(paths)
            store.initialize()
            loop = MemoryLoop(store)
            run = make_run("outcomes")
            loop.begin_run(run)
            source = make_source(run, "outcomes", "d")
            registration = loop.register_source(source)

            filtered = Proposal(
                id="prop_filtered",
                source_image_id=registration.source_id,
                raw_candidate_id="candidate-filtered",
                prompt="cup",
                score=0.2,
                bbox=BoundingBox(x_min=1, y_min=1, x_max=4, y_max=4),
                status=ProposalStatus.FILTERED,
                filter_reason="below_confidence_threshold",
            )
            loop.record_filtered_proposal(filtered)

            uncertain = make_proposal(
                paths,
                run,
                registration.source_id,
                "uncertain",
            )
            loop.apply_response(
                proposal=uncertain,
                response=MllmResponse(
                    decision=DecisionType.UNCERTAIN,
                    confidence=0.55,
                    reason_code=DecisionReasonCode.INSUFFICIENT_EVIDENCE,
                    short_reason="可见证据不足",
                ),
                prompt_version="m3-object-identity-v2",
            )
            pending_summary = store.run_summary(run.id)
            loop.apply_response(
                proposal=uncertain,
                response=MllmResponse(
                    decision=DecisionType.IGNORED,
                    confidence=0.85,
                    reason_code=DecisionReasonCode.INVALID_CANDIDATE,
                    short_reason="重试后确认不是独立物体",
                ),
                prompt_version="m3-object-identity-v2",
                attempt=2,
            )

            ignored = make_proposal(
                paths,
                run,
                registration.source_id,
                "ignored",
            )
            loop.apply_response(
                proposal=ignored,
                response=MllmResponse(
                    decision=DecisionType.IGNORED,
                    confidence=0.9,
                    reason_code=DecisionReasonCode.INVALID_CANDIDATE,
                    short_reason="不是有效独立物体",
                ),
                prompt_version="m3-object-identity-v2",
            )

            failed = make_proposal(
                paths,
                run,
                registration.source_id,
                "failed",
            )
            with self.assertRaises(MemoryStoreError):
                loop.apply_response(
                    proposal=failed,
                    response=MllmResponse(
                        decision=DecisionType.EXISTING,
                        matched_object_id="obj_missing",
                        confidence=0.99,
                        reason_code=DecisionReasonCode.VISUAL_INSTANCE_MATCH,
                        short_reason="引用了不存在的对象",
                        annotation=annotation(),
                    ),
                    prompt_version="m3-object-identity-v2",
                )
            loop.record_proposal_failure(failed, "matched object is missing")
            loop.complete_source(registration.source_id)
            summary = loop.complete_run(run.id)

            self.assertEqual(summary.status, RunStatus.COMPLETED_WITH_ERRORS)
            self.assertEqual(pending_summary.proposal_counts["pending"], 1)
            self.assertEqual(summary.proposal_counts["filtered"], 1)
            self.assertEqual(summary.proposal_counts["pending"], 0)
            self.assertEqual(summary.proposal_counts["decided"], 2)
            self.assertEqual(summary.proposal_counts["failed"], 1)
            self.assertEqual(summary.decision_counts["uncertain"], 1)
            self.assertEqual(summary.decision_counts["ignored"], 2)
            self.assertEqual(summary.observations_added, 0)
            self.assertEqual(summary.active_objects_total, 0)
            self.assertFalse((paths.objects / "obj_missing").exists())


if __name__ == "__main__":
    unittest.main()
