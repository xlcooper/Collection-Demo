"""Tests for batched Qwen review of DINOv3 candidate clusters."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from object_memory.candidate_clustering import (
    CandidateCluster,
    FingerprintedCandidate,
)
from object_memory.cluster_review import (
    ClusterReviewInput,
    build_cluster_review_messages,
    evaluate_cluster_reviews,
    parse_cluster_review_response,
)
from object_memory.config import DEFAULT_CONFIG_PATH, load_config
from object_memory.dinov3_adapter import FingerprintData
from object_memory.identity import MllmOutputError
from object_memory.mllm_adapter import MllmPrediction
from object_memory.schemas import (
    BoundingBox,
    ObjectCard,
    ObjectSummary,
    Proposal,
    VisualEvidence,
    VisualMatchType,
)


def summary_payload(name: str = "人体工学鼠标") -> dict[str, Any]:
    return {
        "object_name_zh": name,
        "coarse_category": "电子设备",
        "fine_category": "鼠标",
        "stable_description": "银灰色非对称鼠标，右侧轮廓隆起。",
        "stable_identity_features": ["非左右对称"],
        "brand_or_markings": [],
        "part_appearance": [
            {"part": "外壳", "color": ["银灰色"], "material": ["塑料"]}
        ],
        "summary_confidence": 0.9,
    }


def cluster(cluster_id: str = "clu_test") -> CandidateCluster:
    proposal = Proposal(
        id="prop_1",
        source_image_id="src_1",
        raw_candidate_id="grid_point_000001",
        prompt="automatic_point_grid",
        score=0.95,
        bbox=BoundingBox(x_min=1, y_min=1, x_max=8, y_max=8),
    )
    member = FingerprintedCandidate(
        proposal=proposal,
        data=FingerprintData(
            global_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            local_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            local_patch_indices=np.asarray([[0, 0]], dtype=np.int32),
        ),
    )
    return CandidateCluster(
        id=cluster_id,
        members=(member,),
        representative_proposal_ids=(proposal.id,),
        global_similarity_min=1.0,
        global_similarity_mean=1.0,
        global_similarity_max=1.0,
    )


def review_payload(
    cluster_id: str,
    *,
    hypothesis: str = "new",
    matched_object_id: str | None = None,
) -> dict[str, Any]:
    return {
        "cluster_id": cluster_id,
        "verdict": "object",
        "identity_hypothesis": hypothesis,
        "matched_object_id": matched_object_id,
        "short_reason": "轮廓与部件布局一致",
        "object_summary": summary_payload(),
    }


class FakePredictor:
    def __init__(self, raw_text: str) -> None:
        self.raw_text = raw_text
        self.calls = 0

    def predict(self, messages: Sequence[dict[str, Any]]) -> MllmPrediction:
        self.calls += 1
        return MllmPrediction(self.raw_text, 10, 10, 0.1)


class ClusterReviewTests(unittest.TestCase):
    def test_response_must_cover_every_cluster_exactly_once(self) -> None:
        with self.assertRaises(MllmOutputError):
            parse_cluster_review_response(
                json.dumps({"reviews": [review_payload("clu_a")]}),
                expected_cluster_ids={"clu_a", "clu_b"},
                allowed_object_ids=set(),
            )

    def test_existing_id_must_come_from_supplied_cards(self) -> None:
        with self.assertRaises(MllmOutputError):
            parse_cluster_review_response(
                json.dumps(
                    {
                        "reviews": [
                            review_payload(
                                "clu_a",
                                hypothesis="existing",
                                matched_object_id="obj_missing",
                            )
                        ]
                    }
                ),
                expected_cluster_ids={"clu_a"},
                allowed_object_ids={"obj_known"},
            )

    def test_empty_memory_requires_new_for_accepted_object(self) -> None:
        with self.assertRaises(MllmOutputError):
            parse_cluster_review_response(
                json.dumps(
                    {
                        "reviews": [
                            review_payload(
                                "clu_a",
                                hypothesis="existing",
                                matched_object_id="obj_known",
                            )
                        ]
                    }
                ),
                expected_cluster_ids={"clu_a"},
                allowed_object_ids=set(),
            )

    def test_message_contains_one_board_per_cluster_and_text_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            board = Path(temporary_directory) / "cluster.jpg"
            Image.new("RGB", (16, 16), (20, 30, 40)).save(board)
            card = ObjectCard(
                object_id="obj_mouse",
                summary=ObjectSummary.model_validate(summary_payload()),
            )
            messages = build_cluster_review_messages(
                inputs=[
                    ClusterReviewInput(
                        cluster=cluster(),
                        contact_sheet_path=board,
                        historical_evidence=VisualEvidence(
                            result=VisualMatchType.NO_MATCH
                        ),
                    )
                ],
                cards=[card],
                settings=load_config(DEFAULT_CONFIG_PATH).mllm_pipeline,
            )
        content = messages[1]["content"]
        self.assertEqual(sum(item["type"] == "image" for item in content), 1)
        text = "\n".join(item["text"] for item in content if item["type"] == "text")
        self.assertIn("obj_mouse", text)
        self.assertIn("historical_visual_evidence", text)

    def test_evaluation_performs_one_call_for_a_cluster_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            board = Path(temporary_directory) / "cluster.jpg"
            Image.new("RGB", (16, 16), (20, 30, 40)).save(board)
            predictor = FakePredictor(
                json.dumps({"reviews": [review_payload("clu_test")]})
            )
            result = evaluate_cluster_reviews(
                predictor,
                inputs=[
                    ClusterReviewInput(
                        cluster=cluster(),
                        contact_sheet_path=board,
                        historical_evidence=VisualEvidence(
                            result=VisualMatchType.NO_MATCH
                        ),
                    )
                ],
                cards=[],
                settings=load_config(DEFAULT_CONFIG_PATH).mllm_pipeline,
            )
        self.assertEqual(predictor.calls, 1)
        self.assertEqual(result.reviews[0].cluster_id, "clu_test")


if __name__ == "__main__":
    unittest.main()
