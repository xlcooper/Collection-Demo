"""Deterministic tests for text-guided SAM3 candidate extraction."""

from __future__ import annotations

from contextlib import nullcontext
import unittest

import numpy as np
from PIL import Image

from object_memory.sam3_adapter import Sam3Adapter


def sam_output(*, count: int, height: int = 6, width: int = 8) -> dict[str, np.ndarray]:
    masks = np.zeros((count, 1, height, width), dtype=np.float32)
    boxes = np.empty((count, 4), dtype=np.float32)
    scores = np.empty((count,), dtype=np.float32)
    for index in range(count):
        masks[index, 0, 1:4, 2 + index : 5 + index] = 1.0
        boxes[index] = (2 + index, 1, 5 + index, 4)
        scores[index] = 0.95 - index * 0.05
    return {"masks": masks, "boxes": boxes, "scores": scores}


class FakeProcessor:
    def __init__(self) -> None:
        self.state = {"encoded": True}
        self.set_image_calls = 0
        self.prompt_calls: list[tuple[object, str]] = []

    def set_image(self, image: Image.Image) -> dict[str, bool]:
        self.set_image_calls += 1
        self.image_size = image.size
        return self.state

    def set_text_prompt(
        self,
        *,
        state: object,
        prompt: str,
    ) -> dict[str, np.ndarray]:
        self.prompt_calls.append((state, prompt))
        return sam_output(count=2 if prompt == "water bottle" else 0)


class FakeCuda:
    @staticmethod
    def synchronize() -> None:
        return None


class FakeTorch:
    bfloat16 = object()
    cuda = FakeCuda()

    @staticmethod
    def inference_mode() -> object:
        return nullcontext()

    @staticmethod
    def autocast(**_: object) -> object:
        return nullcontext()


class FloatConversionProbe:
    def __init__(self) -> None:
        self.events: list[str] = []

    def detach(self) -> "FloatConversionProbe":
        self.events.append("detach")
        return self

    def cpu(self) -> "FloatConversionProbe":
        self.events.append("cpu")
        return self

    def is_floating_point(self) -> bool:
        return True

    def float(self) -> np.ndarray:
        self.events.append("float")
        return np.array([0.75], dtype=np.float32)


class TextGuidedCandidateTests(unittest.TestCase):
    def test_predict_encodes_image_once_and_reuses_state_for_each_prompt(self) -> None:
        adapter = Sam3Adapter("unused-checkpoint.pt", 0.5)
        processor = FakeProcessor()
        adapter._processor = processor
        adapter._torch = FakeTorch()

        prediction = adapter.predict(
            Image.new("RGB", (8, 6)),
            [" Water   Bottle ", "computer mouse"],
        )

        self.assertEqual(processor.set_image_calls, 1)
        self.assertEqual(
            [prompt for _, prompt in processor.prompt_calls],
            ["water bottle", "computer mouse"],
        )
        self.assertTrue(
            all(state is processor.state for state, _ in processor.prompt_calls)
        )
        self.assertEqual(prediction.prompt_counts["water bottle"], 2)
        self.assertEqual(prediction.prompt_counts["computer mouse"], 0)
        self.assertEqual(len(prediction.candidates), 2)

    def test_predict_rejects_prompts_that_duplicate_after_normalization(self) -> None:
        adapter = Sam3Adapter("unused-checkpoint.pt", 0.5)
        with self.assertRaises(ValueError):
            adapter.predict(
                Image.new("RGB", (8, 6)),
                ["water bottle", " Water   Bottle "],
            )

    def test_float_tensor_is_moved_to_cpu_then_converted_to_float32(self) -> None:
        probe = FloatConversionProbe()

        array = Sam3Adapter._to_cpu_array(probe)

        self.assertEqual(probe.events, ["detach", "cpu", "float"])
        self.assertEqual(array.dtype, np.float32)
        self.assertAlmostEqual(float(array[0]), 0.75)

    def test_multiple_instances_keep_text_prompt_provenance(self) -> None:
        adapter = Sam3Adapter("unused-checkpoint.pt", 0.5)
        masks = np.zeros((2, 1, 6, 8), dtype=np.float32)
        masks[0, 0, 1:4, 2:6] = 1.0
        masks[1, 0, 3:5, 4:7] = 1.0
        output = {
            "masks": masks,
            "boxes": np.array(
                [[2.0, 1.0, 6.0, 4.0], [4.0, 3.0, 7.0, 5.0]],
                dtype=np.float32,
            ),
            "scores": np.array([0.95, 0.9], dtype=np.float32),
        }

        candidates = adapter._extract_text_candidates(
            output,
            prompt="water bottle",
            prompt_index=3,
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            candidates[0].raw_candidate_id,
            "text_003_candidate_000",
        )
        self.assertEqual(candidates[0].prompt, "water bottle")
        self.assertEqual(candidates[0].bbox_xyxy, (2.0, 1.0, 6.0, 4.0))
        self.assertAlmostEqual(candidates[0].score, 0.95)
        self.assertEqual(candidates[1].bbox_xyxy, (4.0, 3.0, 7.0, 5.0))

    def test_zero_instance_output_is_valid(self) -> None:
        adapter = Sam3Adapter("unused-checkpoint.pt", 0.5)
        candidates = adapter._extract_text_candidates(
            {
                "masks": np.empty((0, 1, 6, 8), dtype=np.float32),
                "boxes": np.empty((0, 4), dtype=np.float32),
                "scores": np.empty((0,), dtype=np.float32),
            },
            prompt="computer mouse",
            prompt_index=0,
        )

        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
