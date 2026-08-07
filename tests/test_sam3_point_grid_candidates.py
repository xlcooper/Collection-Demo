"""Deterministic tests for class-agnostic SAM3 point-grid extraction."""

from __future__ import annotations

from contextlib import nullcontext
import unittest

import numpy as np
from PIL import Image

from object_memory.sam3_adapter import AUTOMATIC_CANDIDATE_SOURCE, Sam3Adapter


class FakeProcessor:
    def __init__(self) -> None:
        self.state = {"encoded": True}
        self.set_image_calls = 0

    def set_image(self, image: Image.Image) -> dict[str, bool]:
        self.set_image_calls += 1
        self.image_size = image.size
        return self.state


class FakeModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.states: list[object] = []

    def predict_inst(
        self,
        state: object,
        *,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        multimask_output: bool,
    ) -> tuple[np.ndarray, np.ndarray, None]:
        count = len(point_coords)
        self.batch_sizes.append(count)
        self.states.append(state)
        self.assertion_payload = (point_labels.copy(), multimask_output)
        masks = np.zeros((count, 3, 6, 8), dtype=np.float32)
        scores = np.zeros((count, 3), dtype=np.float32)
        for index in range(count):
            masks[index, 0, 1:3, 1:3] = 1.0
            masks[index, 1, 2:5, 2:6] = 1.0
            scores[index] = (0.3, 0.95, 0.5)
        return masks, scores, None


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


class PointGridCandidateTests(unittest.TestCase):
    def test_predict_encodes_once_and_batches_the_complete_grid(self) -> None:
        adapter = Sam3Adapter(
            "unused-checkpoint.pt",
            0.5,
            points_per_side=2,
            points_per_batch=3,
        )
        processor = FakeProcessor()
        model = FakeModel()
        adapter._processor = processor
        adapter._model = model
        adapter._torch = FakeTorch()

        prediction = adapter.predict(Image.new("RGB", (8, 6)))

        self.assertEqual(processor.set_image_calls, 1)
        self.assertEqual(model.batch_sizes, [3, 1])
        self.assertTrue(all(state is processor.state for state in model.states))
        self.assertEqual(len(prediction.candidates), 4)
        self.assertEqual(
            [candidate.raw_candidate_id for candidate in prediction.candidates],
            [f"grid_point_{index:06d}" for index in range(4)],
        )
        self.assertTrue(
            all(candidate.prompt == AUTOMATIC_CANDIDATE_SOURCE for candidate in prediction.candidates)
        )

    def test_best_multimask_candidate_is_retained_per_point(self) -> None:
        adapter = Sam3Adapter("unused-checkpoint.pt", 0.5)
        masks = np.zeros((1, 3, 6, 8), dtype=np.float32)
        masks[0, 1, 2:5, 2:6] = 1.0
        candidates = adapter._extract_point_candidates(
            masks,
            np.asarray([[0.2, 0.9, 0.4]], dtype=np.float32),
            batch_start=7,
            expected_count=1,
        )
        self.assertEqual(candidates[0].raw_candidate_id, "grid_point_000007")
        self.assertAlmostEqual(candidates[0].score, 0.9)
        self.assertEqual(candidates[0].bbox_xyxy, (2.0, 2.0, 6.0, 5.0))

    def test_float_tensor_is_moved_to_cpu_then_converted_to_float32(self) -> None:
        probe = FloatConversionProbe()
        array = Sam3Adapter._to_cpu_array(probe)
        self.assertEqual(probe.events, ["detach", "cpu", "float"])
        self.assertEqual(array.dtype, np.float32)
        self.assertAlmostEqual(float(array[0]), 0.75)


if __name__ == "__main__":
    unittest.main()
