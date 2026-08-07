"""Deterministic tests for cross-image DINOv3 candidate clustering."""

from __future__ import annotations

import unittest

import numpy as np

from object_memory.candidate_clustering import (
    FingerprintedCandidate,
    cluster_candidates,
)
from object_memory.config import VisualFingerprintConfig
from object_memory.dinov3_adapter import FingerprintData
from object_memory.schemas import BoundingBox, Proposal


def candidate(
    proposal_id: str,
    source_id: str,
    global_embedding: tuple[float, float],
) -> FingerprintedCandidate:
    vector = np.asarray(global_embedding, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    return FingerprintedCandidate(
        proposal=Proposal(
            id=proposal_id,
            source_image_id=source_id,
            raw_candidate_id=f"grid_{proposal_id}",
            prompt="automatic_point_grid",
            score=0.95,
            bbox=BoundingBox(x_min=1, y_min=1, x_max=10, y_max=10),
        ),
        data=FingerprintData(
            global_embedding=vector,
            local_embeddings=vector[None, :],
            local_patch_indices=np.asarray([[0, 0]], dtype=np.int32),
        ),
    )


class CandidateClusteringTests(unittest.TestCase):
    def test_similar_candidates_from_different_sources_form_one_cluster(self) -> None:
        clusters = cluster_candidates(
            [
                candidate("prop_a", "src_1", (1.0, 0.0)),
                candidate("prop_b", "src_2", (0.99, 0.1)),
            ],
            VisualFingerprintConfig(cluster_global_similarity_threshold=0.9),
        )

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].proposal_ids, ("prop_a", "prop_b"))
        self.assertGreater(clusters[0].global_similarity_min, 0.9)

    def test_same_source_candidates_are_never_merged(self) -> None:
        clusters = cluster_candidates(
            [
                candidate("prop_a", "src_1", (1.0, 0.0)),
                candidate("prop_b", "src_2", (1.0, 0.0)),
                candidate("prop_c", "src_1", (1.0, 0.0)),
            ],
            VisualFingerprintConfig(cluster_global_similarity_threshold=0.9),
        )

        self.assertEqual(len(clusters), 2)
        self.assertTrue(
            all(
                len(set(cluster.source_ids)) == len(cluster.source_ids)
                for cluster in clusters
            )
        )
        self.assertEqual(sorted(len(cluster.members) for cluster in clusters), [1, 2])

    def test_dissimilar_candidates_remain_separate_review_units(self) -> None:
        clusters = cluster_candidates(
            [
                candidate("prop_a", "src_1", (1.0, 0.0)),
                candidate("prop_b", "src_2", (0.0, 1.0)),
            ],
            VisualFingerprintConfig(cluster_global_similarity_threshold=0.75),
        )

        self.assertEqual(len(clusters), 2)
        self.assertTrue(all(len(cluster.members) == 1 for cluster in clusters))


if __name__ == "__main__":
    unittest.main()
