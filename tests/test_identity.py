"""Tests for strict extraction of Qwen JSON responses."""

from __future__ import annotations

import unittest

from object_memory.identity import MllmOutputError, extract_json_object


class JsonExtractionTests(unittest.TestCase):
    def test_fenced_object_is_extracted(self) -> None:
        self.assertEqual(extract_json_object('```json\n{"a": 1}\n```'), {"a": 1})

    def test_non_object_is_rejected(self) -> None:
        with self.assertRaises(MllmOutputError):
            extract_json_object("[]")

    def test_prefix_text_may_precede_one_object(self) -> None:
        self.assertEqual(extract_json_object('result: {"reviews": []}'), {"reviews": []})


if __name__ == "__main__":
    unittest.main()
