"""Shared strict JSON extraction for the single Qwen response."""

from __future__ import annotations

import json
from typing import Any


class MllmOutputError(ValueError):
    """Raised when a model response is not safe to persist."""


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Extract one JSON object while tolerating a surrounding code fence."""

    stripped = raw_text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as original_error:
        start = stripped.find("{")
        if start < 0:
            raise MllmOutputError(
                "Qwen response contains no JSON object"
            ) from original_error
        try:
            payload, _ = json.JSONDecoder().raw_decode(stripped[start:])
        except json.JSONDecodeError as exc:
            raise MllmOutputError("Qwen response contains invalid JSON") from exc

    if not isinstance(payload, dict):
        raise MllmOutputError("Qwen response must be one JSON object")
    return payload
