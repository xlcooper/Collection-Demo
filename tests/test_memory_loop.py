"""Transactional tests for summary iteration and fingerprint references."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from object_memory.assets import MemoryPaths
from object_memory.memory_loop import MemoryLoop
from object_memory.memory_store import MemoryStore
from object_memory.schemas import (
    BoundingBox,
    DecisionReasonCode,
    DecisionType,
    FinalIdentityDecision,
    IdentityHypothesis,
    ObjectSummary,
    Observation,
    Proposal,
    Run,
    SourceImage,
    VisualEvidence,
    VisualFingerprint,
    VisualMatchType,
)


def object_summary(description: str = "银灰色非对称鼠标") -> ObjectSummary:
    return ObjectSummary(
        object_name_zh="人体工学鼠标",
        coarse_category="电子设备",
        fine_category="鼠标",
        stable_description=description,
        stable_identity_features=["整体非左右对称"],
        brand_or_markings=[],
        part_appearance=[],
        summary_confidence=0.9,
    )


def fingerprint(path: str) -> VisualFingerprint:
    return VisualFingerprint(
        path=path,
        sha256="f" * 64,
        model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        revision="a" * 40,
        feature_layer="last_hidden_state",
        input_size=512,
        storage_dtype="float16",
        global_dimension=768,
        local_count=10,
        l2_normalized=True,
    )


def proposal(source_id: str, suffix: str) -> Proposal:
    return Proposal(
        id=f"prop_{suffix}",
        source_image_id=source_id,
        raw_candidate_id=f"raw_{suffix}",
        prompt="computer mouse",
        score=0.9,
        bbox=BoundingBox(x_min=1, y_min=1, x_max=10, y_max=10),
        crop_path=f"proposals/run_1/prop_{suffix}/crop.png",
        mask_path=f"proposals/run_1/prop_{suffix}/mask.png",
        overlay_path=f"proposals/run_1/prop_{suffix}/overlay.jpg",
    )


class MemoryLoopTests(unittest.TestCase):
    def test_new_and_existing_update_one_summary_without_copying_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = MemoryPaths(Path(temporary_directory) / "memory")
            store = MemoryStore(paths)
            store.initialize()
            loop = MemoryLoop(store)
            run = Run(
                id="run_1",
                config_digest="a" * 64,
                sam_model_id="sam",
                qwen_model_id="qwen",
                dinov3_model_id="dino",
            )
            loop.begin_run(run)
            first_source = SourceImage(
                id="src_1",
                run_id=run.id,
                sha256="b" * 64,
                relative_path="sources/one.png",
                width=20,
                height=20,
            )
            loop.register_source(first_source)
            first = loop.apply_decision(
                proposal=proposal(first_source.id, "one"),
                result=FinalIdentityDecision(
                    decision=DecisionType.NEW,
                    confidence=0.9,
                    reason_code=DecisionReasonCode.NEW_OBJECT,
                    short_reason="无历史匹配",
                    qwen_hypothesis=IdentityHypothesis.NEW,
                    visual_evidence=VisualEvidence(result=VisualMatchType.NO_MATCH),
                    object_summary=object_summary(),
                ),
                fingerprint=fingerprint("proposals/run_1/prop_one/fingerprint.npz"),
                prompt_version="object-memory-single-pass-v1",
                raw_response_path="raw_responses/run_1/src_1/response.json",
            )
            loop.complete_source(first_source.id)
            self.assertIsNotNone(first.object_id)

            second_source = SourceImage(
                id="src_2",
                run_id=run.id,
                sha256="c" * 64,
                relative_path="sources/two.png",
                width=20,
                height=20,
            )
            loop.register_source(second_source)
            updated = object_summary("银灰色非对称鼠标，右侧轮廓明显隆起。")
            second = loop.apply_decision(
                proposal=proposal(second_source.id, "two"),
                result=FinalIdentityDecision(
                    decision=DecisionType.EXISTING,
                    matched_object_id=first.object_id,
                    confidence=0.92,
                    reason_code=DecisionReasonCode.VISUAL_INSTANCE_MATCH,
                    short_reason="双证据一致",
                    qwen_hypothesis=IdentityHypothesis.EXISTING,
                    qwen_matched_object_id=first.object_id,
                    visual_evidence=VisualEvidence(
                        result=VisualMatchType.MATCH,
                        matched_object_id=first.object_id,
                        matched_observation_id=first.observation_id,
                        visual_score=0.9,
                    ),
                    object_summary=updated,
                ),
                fingerprint=fingerprint("proposals/run_1/prop_two/fingerprint.npz"),
                prompt_version="object-memory-single-pass-v1",
                raw_response_path="raw_responses/run_1/src_2/response.json",
            )
            loop.complete_source(second_source.id)

            cards = loop.object_cards()
            records = loop.fingerprint_records()
            self.assertEqual(len(cards), 1)
            self.assertEqual(cards[0].summary.stable_description, updated.stable_description)
            self.assertEqual(len(records), 2)
            self.assertFalse(any(paths.objects.rglob("crop.png")))
            with sqlite3.connect(paths.database) as connection:
                observations = connection.execute(
                    "SELECT COUNT(*) FROM observations"
                ).fetchone()[0]
            self.assertEqual(observations, 2)
            self.assertEqual(second.object_id, first.object_id)

    def test_uncertain_is_terminal_and_does_not_update_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = MemoryPaths(Path(temporary_directory) / "memory")
            store = MemoryStore(paths)
            store.initialize()
            loop = MemoryLoop(store)
            run = Run(
                id="run_1",
                config_digest="a" * 64,
                sam_model_id="sam",
                qwen_model_id="qwen",
                dinov3_model_id="dino",
            )
            loop.begin_run(run)
            source = SourceImage(
                id="src_1",
                run_id=run.id,
                sha256="b" * 64,
                relative_path="sources/one.png",
                width=20,
                height=20,
            )
            loop.register_source(source)
            result = loop.apply_decision(
                proposal=proposal(source.id, "one"),
                result=FinalIdentityDecision(
                    decision=DecisionType.UNCERTAIN,
                    confidence=0.2,
                    reason_code=DecisionReasonCode.INSUFFICIENT_EVIDENCE,
                    short_reason="证据冲突",
                    qwen_hypothesis=IdentityHypothesis.UNCERTAIN,
                    visual_evidence=VisualEvidence(result=VisualMatchType.NO_MATCH),
                ),
                fingerprint=fingerprint("proposals/run_1/prop_one/fingerprint.npz"),
                prompt_version="object-memory-single-pass-v1",
                raw_response_path="raw_responses/run_1/src_1/response.json",
            )
            loop.complete_source(source.id)
            summary = loop.complete_run(run.id)
            self.assertEqual(result.proposal_status.value, "decided")
            self.assertEqual(summary.status.value, "completed")
            self.assertEqual(summary.active_objects_total, 0)
            self.assertEqual(summary.observations_added, 0)


if __name__ == "__main__":
    unittest.main()
