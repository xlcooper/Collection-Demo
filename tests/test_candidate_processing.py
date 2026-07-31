"""Deterministic tests for candidate filtering and asset generation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from object_memory.assets import MemoryPaths
from object_memory.config import Sam3PipelineConfig
from object_memory.sam3_adapter import RawSamCandidate
from object_memory.sam3_postprocess import (
    mask_containment,
    mask_iou,
    process_candidates,
)
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


class CandidatePostprocessTests(unittest.TestCase):
    def test_filters_deduplicates_and_generates_assets(self) -> None:
        image = Image.new("RGB", (100, 100), (220, 220, 220))
        main_mask = np.zeros((100, 100), dtype=bool)
        main_mask[20:50, 20:50] = True
        tiny_mask = np.zeros((100, 100), dtype=bool)
        tiny_mask[5:7, 5:7] = True
        other_mask = np.zeros((100, 100), dtype=bool)
        other_mask[60:80, 60:80] = True

        candidates = [
            candidate(
                "main",
                "coffee cup",
                0.90,
                main_mask,
                (20, 20, 50, 50),
            ),
            candidate(
                "duplicate",
                "coffee cup",
                0.80,
                main_mask,
                (20, 20, 50, 50),
            ),
            candidate(
                "tiny",
                "coffee cup",
                0.95,
                tiny_mask,
                (5, 5, 7, 7),
            ),
            candidate(
                "weak",
                "coffee cup",
                0.40,
                other_mask,
                (60, 60, 80, 80),
            ),
        ]
        settings = Sam3PipelineConfig(
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
            self.assertEqual(kept.prompt, "coffee cup")
            for relative_path in (
                kept.crop_path,
                kept.mask_path,
                kept.overlay_path,
            ):
                self.assertIsNotNone(relative_path)
                self.assertTrue(paths.resolve_asset(relative_path).is_file())

            reasons = [proposal.filter_reason or "" for proposal in result.filtered]
            self.assertTrue(any(reason.startswith("duplicate_mask:") for reason in reasons))

            assert kept.crop_path is not None
            with Image.open(paths.resolve_asset(kept.crop_path)) as crop:
                crop_pixels = np.asarray(crop.convert("RGB"))
            self.assertEqual(tuple(crop_pixels[0, 0]), (127, 127, 127))
            self.assertEqual(tuple(crop_pixels[5, 5]), (220, 220, 220))

    def test_mask_iou_is_deterministic(self) -> None:
        first = np.array([[True, True], [False, False]])
        second = np.array([[True, False], [True, False]])
        self.assertAlmostEqual(mask_iou(first, second), 1 / 3)
        self.assertEqual(mask_iou(first, np.zeros((1, 1), dtype=bool)), 0.0)

    def test_filters_lower_scored_mask_contained_by_complete_candidate(self) -> None:
        image = Image.new("RGB", (40, 40), (220, 220, 220))
        complete_mask = np.zeros((40, 40), dtype=bool)
        complete_mask[5:25, 5:25] = True
        part_mask = np.zeros((40, 40), dtype=bool)
        part_mask[10:15, 10:15] = True
        separate_mask = np.zeros((40, 40), dtype=bool)
        separate_mask[30:35, 30:35] = True
        candidates = [
            candidate(
                "complete",
                "coffee cup",
                0.98,
                complete_mask,
                (5, 5, 25, 25),
            ),
            candidate(
                "part",
                "coffee cup",
                0.95,
                part_mask,
                (10, 10, 15, 15),
            ),
            candidate(
                "separate",
                "coffee cup",
                0.94,
                separate_mask,
                (30, 30, 35, 35),
            ),
        ]
        settings = Sam3PipelineConfig(
            confidence_threshold=0.5,
            min_mask_area_ratio=0.001,
            contained_mask_overlap_threshold=0.9,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = process_candidates(
                candidates,
                image=image,
                source_image_id="src_containment",
                run_id="run_containment",
                paths=MemoryPaths(Path(temporary_directory) / "assets"),
                settings=settings,
            )

        self.assertEqual(
            [proposal.raw_candidate_id for proposal in result.kept],
            ["complete", "separate"],
        )
        self.assertEqual(result.filter_counts["contained_mask"], 1)
        contained = next(
            proposal
            for proposal in result.filtered
            if proposal.raw_candidate_id == "part"
        )
        self.assertTrue((contained.filter_reason or "").startswith("contained_mask:"))

    def test_mask_containment_is_directional(self) -> None:
        outer = np.ones((4, 4), dtype=bool)
        inner = np.zeros((4, 4), dtype=bool)
        inner[1:3, 1:3] = True
        self.assertEqual(mask_containment(inner, outer), 1.0)
        self.assertEqual(mask_containment(outer, inner), 0.25)
        self.assertEqual(
            mask_containment(inner, np.zeros((1, 1), dtype=bool)),
            0.0,
        )

    def test_candidate_limit_reserves_one_mask_per_text_prompt(self) -> None:
        image = Image.new("RGB", (60, 60), (220, 220, 220))
        masks: list[np.ndarray] = []
        for left, top in ((2, 2), (22, 2), (42, 42)):
            mask = np.zeros((60, 60), dtype=bool)
            mask[top : top + 10, left : left + 10] = True
            masks.append(mask)
        candidates = [
            candidate("cup_one", "coffee cup", 0.99, masks[0], (2, 2, 12, 12)),
            candidate("cup_two", "coffee cup", 0.98, masks[1], (22, 2, 32, 12)),
            candidate("mouse", "computer mouse", 0.60, masks[2], (42, 42, 52, 52)),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = process_candidates(
                candidates,
                image=image,
                source_image_id="src_prompt_fairness",
                run_id="run_prompt_fairness",
                paths=MemoryPaths(Path(temporary_directory) / "assets"),
                settings=Sam3PipelineConfig(max_candidates_per_image=2),
            )

        self.assertEqual(
            {proposal.prompt for proposal in result.kept},
            {"coffee cup", "computer mouse"},
        )
        self.assertEqual(result.filter_counts["candidate_limit"], 1)

    def test_containment_does_not_remove_a_different_text_concept(self) -> None:
        image = Image.new("RGB", (40, 40), (220, 220, 220))
        outer = np.zeros((40, 40), dtype=bool)
        outer[5:30, 5:30] = True
        inner = np.zeros((40, 40), dtype=bool)
        inner[12:20, 12:20] = True

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = process_candidates(
                [
                    candidate("cup", "coffee cup", 0.98, outer, (5, 5, 30, 30)),
                    candidate("spoon", "metal spoon", 0.95, inner, (12, 12, 20, 20)),
                ],
                image=image,
                source_image_id="src_cross_prompt_containment",
                run_id="run_cross_prompt_containment",
                paths=MemoryPaths(Path(temporary_directory) / "assets"),
                settings=Sam3PipelineConfig(),
            )

        self.assertEqual(len(result.kept), 2)
        self.assertNotIn("contained_mask", result.filter_counts)

    def test_identical_masks_are_deduplicated_across_text_concepts(self) -> None:
        image = Image.new("RGB", (40, 40), (220, 220, 220))
        shared_mask = np.zeros((40, 40), dtype=bool)
        shared_mask[5:25, 5:25] = True

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = process_candidates(
                [
                    candidate(
                        "cup",
                        "coffee cup",
                        0.98,
                        shared_mask,
                        (5, 5, 25, 25),
                    ),
                    candidate(
                        "container",
                        "drink container",
                        0.95,
                        shared_mask.copy(),
                        (5, 5, 25, 25),
                    ),
                ],
                image=image,
                source_image_id="src_cross_prompt_duplicate",
                run_id="run_cross_prompt_duplicate",
                paths=MemoryPaths(Path(temporary_directory) / "assets"),
                settings=Sam3PipelineConfig(),
            )

        self.assertEqual(len(result.kept), 1)
        self.assertEqual(result.kept[0].prompt, "coffee cup")
        self.assertEqual(result.filter_counts["duplicate_mask"], 1)

    def test_text_guided_candidates_filter_large_regions_and_apply_limit(self) -> None:
        image = Image.new("RGB", (20, 20), (220, 220, 220))
        large_mask = np.ones((20, 20), dtype=bool)
        first_mask = np.zeros((20, 20), dtype=bool)
        first_mask[1:6, 1:6] = True
        second_mask = np.zeros((20, 20), dtype=bool)
        second_mask[12:18, 12:18] = True
        candidates = [
            candidate(
                "large",
                "coffee cup",
                0.99,
                large_mask,
                (0, 0, 20, 20),
            ),
            candidate(
                "first",
                "coffee cup",
                0.95,
                first_mask,
                (1, 1, 6, 6),
            ),
            candidate(
                "second",
                "coffee cup",
                0.90,
                second_mask,
                (12, 12, 18, 18),
            ),
        ]
        settings = Sam3PipelineConfig(
            confidence_threshold=0.5,
            max_mask_area_ratio=0.8,
            max_candidates_per_image=1,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = process_candidates(
                candidates,
                image=image,
                source_image_id="src_automatic",
                run_id="run_automatic",
                paths=MemoryPaths(Path(temporary_directory) / "assets"),
                settings=settings,
            )

        self.assertEqual(len(result.kept), 1)
        self.assertEqual(result.filter_counts["mask_too_large"], 1)
        self.assertEqual(result.filter_counts["candidate_limit"], 1)


if __name__ == "__main__":
    unittest.main()
