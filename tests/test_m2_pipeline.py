"""Server-side deterministic tests for M2 post-processing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from object_memory.assets import MemoryPaths
from object_memory.config import Sam3PipelineConfig
from object_memory.sam3_adapter import RawSamCandidate
from object_memory.sam3_postprocess import mask_iou, process_candidates
from object_memory.schemas import ProposalStatus


def candidate(
    candidate_id: str,
    prompt: str,
    score: float,
    mask: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> RawSamCandidate:
    return RawSamCandidate(
        raw_candidate_id=candidate_id,
        prompt=prompt,
        score=score,
        bbox_xyxy=bbox,
        mask=mask,
    )


class M2PostprocessTests(unittest.TestCase):
    def test_filters_deduplicates_and_generates_assets(self) -> None:
        image = Image.new("RGB", (100, 100), (220, 220, 220))
        main_mask = np.zeros((100, 100), dtype=bool)
        main_mask[20:50, 20:50] = True
        tiny_mask = np.zeros((100, 100), dtype=bool)
        tiny_mask[5:7, 5:7] = True
        other_mask = np.zeros((100, 100), dtype=bool)
        other_mask[60:80, 60:80] = True

        candidates = [
            candidate("main", "cup", 0.90, main_mask, (20, 20, 50, 50)),
            candidate("duplicate", "mug", 0.80, main_mask, (20, 20, 50, 50)),
            candidate("tiny", "cap", 0.95, tiny_mask, (5, 5, 7, 7)),
            candidate("weak", "bottle", 0.40, other_mask, (60, 60, 80, 80)),
        ]
        settings = Sam3PipelineConfig(
            prompts=["cup", "mug", "cap", "bottle"],
            confidence_threshold=0.5,
            min_mask_area_ratio=0.01,
            duplicate_mask_iou_threshold=0.9,
            crop_padding_pixels=2,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = MemoryPaths(Path(temporary_directory) / "assets")
            result = process_candidates(
                candidates,
                image=image,
                source_image_id="src_example",
                run_id="run_example",
                paths=paths,
                settings=settings,
            )

            self.assertEqual(len(result.kept), 1)
            self.assertEqual(len(result.filtered), 3)
            self.assertEqual(result.filter_counts["mask_too_small"], 1)
            self.assertEqual(result.filter_counts["low_confidence"], 1)
            self.assertEqual(result.filter_counts["duplicate_mask"], 1)

            kept = result.kept[0]
            self.assertEqual(kept.status, ProposalStatus.PENDING)
            self.assertEqual(kept.prompt, "cup")
            for relative_path in (
                kept.crop_path,
                kept.mask_path,
                kept.overlay_path,
            ):
                self.assertIsNotNone(relative_path)
                self.assertTrue(paths.resolve_asset(relative_path).is_file())

            reasons = [proposal.filter_reason or "" for proposal in result.filtered]
            self.assertTrue(any(reason.startswith("duplicate_mask:") for reason in reasons))

    def test_mask_iou_is_deterministic(self) -> None:
        first = np.array([[True, True], [False, False]])
        second = np.array([[True, False], [True, False]])
        self.assertAlmostEqual(mask_iou(first, second), 1 / 3)
        self.assertEqual(mask_iou(first, np.zeros((1, 1), dtype=bool)), 0.0)


if __name__ == "__main__":
    unittest.main()

