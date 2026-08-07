"""CPU-only tests for visual fingerprint persistence and matching math."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from object_memory.config import DEFAULT_CONFIG_PATH, load_config
from object_memory.dinov3_adapter import (
    FingerprintData,
    HistoricalFingerprint,
    match_fingerprint,
    read_fingerprint,
    write_fingerprint,
)
from object_memory.schemas import VisualMatchType


def data(global_values: tuple[float, float], local_values: tuple[float, float]) -> FingerprintData:
    global_array = np.asarray(global_values, dtype=np.float32)
    global_array /= np.linalg.norm(global_array)
    local_array = np.asarray([local_values], dtype=np.float32)
    local_array /= np.linalg.norm(local_array, axis=1, keepdims=True)
    return FingerprintData(
        global_embedding=global_array,
        local_embeddings=local_array,
        local_patch_indices=np.asarray([[0, 0]], dtype=np.int32),
    )


class FingerprintTests(unittest.TestCase):
    def test_npz_round_trip_records_hash_and_required_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "fingerprint.npz"
            metadata = write_fingerprint(
                path,
                data((1.0, 0.0), (1.0, 0.0)),
                relative_path="fingerprint.npz",
                model_id="dino",
                revision="a" * 40,
                feature_layer="last_hidden_state",
                input_size=512,
                storage_dtype="float16",
            )
            loaded = read_fingerprint(path, expected_sha256=metadata.sha256)
        self.assertEqual(loaded.global_embedding.shape, (2,))
        self.assertEqual(loaded.local_embeddings.shape, (1, 2))

    def test_match_aggregates_each_objects_best_historical_view(self) -> None:
        settings = load_config(DEFAULT_CONFIG_PATH).visual_fingerprint
        query = data((1.0, 0.0), (1.0, 0.0))
        result = match_fingerprint(
            query,
            [
                HistoricalFingerprint("obj_a", "obs_bad", data((0.0, 1.0), (0.0, 1.0))),
                HistoricalFingerprint("obj_a", "obs_good", data((1.0, 0.0), (1.0, 0.0))),
                HistoricalFingerprint("obj_b", "obs_b", data((0.2, 0.98), (0.2, 0.98))),
            ],
            settings,
        )
        self.assertEqual(result.result, VisualMatchType.MATCH)
        self.assertEqual(result.matched_object_id, "obj_a")
        self.assertEqual(result.matched_observation_id, "obs_good")

    def test_close_first_and_second_scores_are_ambiguous(self) -> None:
        base = load_config(DEFAULT_CONFIG_PATH)
        payload = base.visual_fingerprint.model_dump(mode="python")
        payload["match_threshold"] = 0.5
        payload["ambiguity_margin"] = 0.1
        settings = type(base.visual_fingerprint).model_validate(payload)
        query = data((1.0, 0.0), (1.0, 0.0))
        result = match_fingerprint(
            query,
            [
                HistoricalFingerprint("obj_a", "obs_a", data((1.0, 0.0), (1.0, 0.0))),
                HistoricalFingerprint("obj_b", "obs_b", data((0.99, 0.1), (0.99, 0.1))),
            ],
            settings,
        )
        self.assertEqual(result.result, VisualMatchType.AMBIGUOUS)

    def test_local_comparison_is_limited_to_globally_ranked_objects(self) -> None:
        base = load_config(DEFAULT_CONFIG_PATH)
        payload = base.visual_fingerprint.model_dump(mode="python")
        payload["local_top_k"] = 1
        payload["match_threshold"] = 0.0
        settings = type(base.visual_fingerprint).model_validate(payload)
        query = data((1.0, 0.0), (1.0, 0.0))
        result = match_fingerprint(
            query,
            [
                HistoricalFingerprint("obj_near", "obs_near", data((1.0, 0.0), (0.0, 1.0))),
                HistoricalFingerprint("obj_far", "obs_far", data((0.0, 1.0), (1.0, 0.0))),
            ],
            settings,
        )
        self.assertEqual([score.object_id for score in result.object_scores], ["obj_near"])


if __name__ == "__main__":
    unittest.main()
