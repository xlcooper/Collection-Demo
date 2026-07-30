"""Deterministic tests for prompt-free SAM3 point-grid candidate handling."""

from __future__ import annotations

import unittest

import numpy as np

from object_memory.sam3_adapter import (
    AUTOMATIC_CANDIDATE_SOURCE,
    Sam3Adapter,
)


class AutomaticCandidateTests(unittest.TestCase):
    def test_point_grid_covers_cell_centers_without_category_text(self) -> None:
        adapter = Sam3Adapter(
            "unused-checkpoint.pt",
            0.88,
            points_per_side=2,
            points_per_batch=2,
        )

        points = adapter._point_grid(100, 50)

        np.testing.assert_allclose(
            points,
            np.array(
                [[25.0, 12.5], [75.0, 12.5], [25.0, 37.5], [75.0, 37.5]],
                dtype=np.float32,
            ),
        )

    def test_best_mask_per_point_becomes_an_unlabelled_candidate(self) -> None:
        adapter = Sam3Adapter("unused-checkpoint.pt", 0.88)
        masks = np.zeros((2, 3, 6, 8), dtype=bool)
        masks[0, 1, 1:4, 2:6] = True
        masks[1, 2, 3:5, 4:7] = True
        scores = np.array([[0.4, 0.95, 0.7], [0.2, 0.3, 0.9]])

        candidates = adapter._extract_point_candidates(
            masks,
            scores,
            batch_start=10,
            expected_count=2,
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].raw_candidate_id, "grid_point_000010")
        self.assertEqual(candidates[0].prompt, AUTOMATIC_CANDIDATE_SOURCE)
        self.assertEqual(candidates[0].bbox_xyxy, (2.0, 1.0, 6.0, 4.0))
        self.assertAlmostEqual(candidates[0].score, 0.95)
        self.assertEqual(candidates[1].bbox_xyxy, (4.0, 3.0, 7.0, 5.0))


if __name__ == "__main__":
    unittest.main()
