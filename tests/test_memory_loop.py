"""Transactional tests for cluster writes and visual fingerprint references."""

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
    ClusterReview,
    ClusterVerdict,
    DecisionReasonCode,
    DecisionType,
    IdentityHypothesis,
    ObjectSummary,
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
        raw_candidate_id=f"grid_{suffix}",
        prompt="automatic_point_grid",
        score=0.9,
        bbox=BoundingBox(x_min=1, y_min=1, x_max=10, y_max=10),
        crop_path=f"proposals/run_1/prop_{suffix}/crop.png",
        mask_path=f"proposals/run_1/prop_{suffix}/mask.png",
        overlay_path=f"proposals/run_1/prop_{suffix}/overlay.jpg",
        fingerprint=fingerprint(f"proposals/run_1/prop_{suffix}/fingerprint.npz"),
    )


def review(
    cluster_id: str,
    *,
    hypothesis: IdentityHypothesis,
    matched_object_id: str | None = None,
    summary: ObjectSummary | None = None,
) -> ClusterReview:
    return ClusterReview(
        cluster_id=cluster_id,
        verdict=ClusterVerdict.OBJECT,
        identity_hypothesis=hypothesis,
        matched_object_id=matched_object_id,
        short_reason="聚类成员轮廓一致",
        object_summary=summary or object_summary(),
    )


def register_source(loop: MemoryLoop, run: Run, source_id: str, digest: str) -> SourceImage:
    source = SourceImage(
        id=source_id,
        run_id=run.id,
        sha256=digest * 64,
        relative_path=f"sources/{source_id}.png",
        width=20,
        height=20,
    )
    loop.register_source(source)
    return source


class MemoryLoopTests(unittest.TestCase):
    def test_new_cluster_creates_one_object_and_observation_per_view(self) -> None:
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
            first_source = register_source(loop, run, "src_1", "b")
            second_source = register_source(loop, run, "src_2", "c")
            proposals = [
                proposal(first_source.id, "one"),
                proposal(second_source.id, "two"),
            ]
            result = loop.apply_cluster_decision(
                proposals=proposals,
                review=review("clu_mouse", hypothesis=IdentityHypothesis.NEW),
                decision_type=DecisionType.NEW,
                visual_evidence=VisualEvidence(result=VisualMatchType.NO_MATCH),
                prompt_version="object-memory-cluster-review-v1",
                raw_response_path="raw_responses/run_1/cluster_batch_0001/response.json",
                reason_code=DecisionReasonCode.NEW_OBJECT,
                short_reason="未匹配历史对象",
            )
            loop.complete_source(first_source.id)
            loop.complete_source(second_source.id)

            self.assertIsNotNone(result.object_id)
            self.assertEqual(
                [item.decision for item in result.proposal_results],
                [DecisionType.NEW, DecisionType.EXISTING],
            )
            self.assertTrue(
                all(item.object_id == result.object_id for item in result.proposal_results)
            )
            self.assertEqual(len(loop.object_cards()), 1)
            self.assertEqual(len(loop.fingerprint_records()), 2)
            with sqlite3.connect(paths.database) as connection:
                observations = connection.execute(
                    "SELECT COUNT(*) FROM observations"
                ).fetchone()[0]
            self.assertEqual(observations, 2)

    def test_uncertain_cluster_is_terminal_without_object_or_observation(self) -> None:
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
            source = register_source(loop, run, "src_1", "b")
            result = loop.apply_cluster_decision(
                proposals=[proposal(source.id, "one")],
                review=review("clu_mouse", hypothesis=IdentityHypothesis.NEW),
                decision_type=DecisionType.UNCERTAIN,
                visual_evidence=VisualEvidence(result=VisualMatchType.AMBIGUOUS),
                prompt_version="object-memory-cluster-review-v1",
                raw_response_path="raw_responses/run_1/cluster_batch_0001/response.json",
                reason_code=DecisionReasonCode.AMBIGUOUS_MATCH,
                short_reason="视觉证据冲突",
            )
            loop.complete_source(source.id)
            summary = loop.complete_run(run.id)

            self.assertIsNone(result.object_id)
            self.assertEqual(result.proposal_results[0].decision, DecisionType.UNCERTAIN)
            self.assertEqual(summary.active_objects_total, 0)
            self.assertEqual(summary.observations_added, 0)


if __name__ == "__main__":
    unittest.main()
