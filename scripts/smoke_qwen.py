#!/usr/bin/env python3
"""Run one real Qwen3-VL inference and validate the required JSON fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "coarse_category",
    "fine_category",
    "material",
    "color",
    "shape",
    "description",
    "annotation_confidence",
}

DEFAULT_PROMPT = """Analyze the main object in this image.
Return exactly one JSON object and no Markdown.
Use concise Chinese values. Do not guess a brand or model when it is not visible.
Required fields:
- coarse_category: string
- fine_category: string; use "unknown" when uncertain
- material: array of strings
- color: array of strings
- shape: string
- description: string based only on visible evidence
- annotation_confidence: number from 0 to 1
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Qwen3-VL JSON output.")
    parser.add_argument("--image", required=True, help="Input object image path.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-8B-Instruct-FP8",
        help="Hugging Face model ID or local model path.",
    )
    parser.add_argument("--revision", help="Optional model revision.")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow Hugging Face network access instead of requiring local cache.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        default="runs/smoke/qwen",
        help="Ignored directory for the raw response.",
    )
    parser.add_argument(
        "--report",
        default="environment/qwen_smoke_report.json",
        help="Compact report path intended for Git.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def resolve_local_snapshot(model: str, revision: str | None) -> Path:
    """Resolve a model ID to one concrete Hugging Face cache snapshot."""
    direct_path = Path(model).expanduser()
    if direct_path.is_dir():
        snapshot_path = direct_path.resolve()
    else:
        cache_root_value = os.environ.get("HF_HUB_CACHE")
        if cache_root_value:
            cache_root = Path(cache_root_value).expanduser()
        else:
            hf_home = Path(
                os.environ.get("HF_HOME", "~/.cache/huggingface")
            ).expanduser()
            cache_root = hf_home / "hub"

        repository_cache = cache_root / f"models--{model.replace('/', '--')}"
        revision_name = revision or "main"
        reference_path = repository_cache / "refs" / revision_name

        if reference_path.is_file():
            snapshot_name = reference_path.read_text(encoding="utf-8").strip()
            snapshot_path = repository_cache / "snapshots" / snapshot_name
        else:
            requested_snapshot = repository_cache / "snapshots" / revision_name
            if requested_snapshot.is_dir():
                snapshot_path = requested_snapshot
            else:
                snapshots_root = repository_cache / "snapshots"
                available_snapshots = (
                    sorted(
                        path
                        for path in snapshots_root.iterdir()
                        if path.is_dir() and (path / "config.json").is_file()
                    )
                    if snapshots_root.is_dir()
                    else []
                )
                if len(available_snapshots) == 1:
                    snapshot_path = available_snapshots[0]
                else:
                    raise FileNotFoundError(
                        "Cannot resolve one local model snapshot. "
                        f"Expected a ref at {reference_path} or one complete snapshot "
                        f"under {snapshots_root}."
                    )

    if not (snapshot_path / "config.json").is_file():
        raise FileNotFoundError(
            f"Local model snapshot has no config.json: {snapshot_path}"
        )
    return snapshot_path.resolve()


def parse_json_object(raw_text: str) -> dict[str, Any]:
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Model output is not a JSON object.")
    return parsed


def run_smoke(args: argparse.Namespace, report: dict[str, Any]) -> int:
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA.")

    image_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.allow_network:
        model_source = args.model
        report["loading"].update({"source_type": "model_id_or_path"})
    else:
        snapshot_path = resolve_local_snapshot(args.model, args.revision)
        model_source = str(snapshot_path)
        report["loading"].update(
            {
                "source_type": "local_snapshot",
                "snapshot": snapshot_path.name,
            }
        )

    model_kwargs: dict[str, Any] = {
        "dtype": "auto",
        "device_map": "auto",
        "local_files_only": not args.allow_network,
    }
    if args.revision and args.allow_network:
        model_kwargs["revision"] = args.revision

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_source,
        **model_kwargs,
    )
    processor_kwargs: dict[str, Any] = {
        "local_files_only": not args.allow_network,
    }
    if args.revision and args.allow_network:
        processor_kwargs["revision"] = args.revision
    processor = AutoProcessor.from_pretrained(model_source, **processor_kwargs)
    model.eval()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path.as_uri(),
                    "max_pixels": args.max_pixels,
                },
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    rendered_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    images, videos = process_vision_info(messages, image_patch_size=16)
    inputs = processor(
        text=[rendered_prompt],
        images=images,
        videos=videos,
        do_resize=False,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    inference_started = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started

    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_response = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    (output_dir / "response.txt").write_text(raw_response + "\n", encoding="utf-8")

    parsed_response = parse_json_object(raw_response)
    missing_fields = sorted(REQUIRED_FIELDS - parsed_response.keys())
    status = "passed" if not missing_fields else "failed"

    report.update(
        {
            "status": status,
            "input": {
                "filename": image_path.name,
                "sha256": sha256_file(image_path),
            },
            "required_fields": sorted(REQUIRED_FIELDS),
            "missing_fields": missing_fields,
            "parsed_response": parsed_response,
            "timing_seconds": {
                "model_load": round(model_load_seconds, 3),
                "inference": round(inference_seconds, 3),
            },
            "cuda": {
                "device": torch.cuda.get_device_name(0),
                "peak_memory_mib": round(
                    torch.cuda.max_memory_allocated() / (1024**2), 2
                ),
            },
        }
    )

    if missing_fields:
        report["error"] = "Required JSON fields are missing."
        return 4
    return 0


def main() -> int:
    args = parse_args()
    report_path = Path(args.report).expanduser().resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test": "qwen3_vl_json_smoke",
        "status": "failed",
        "model": args.model,
        "revision": args.revision,
        "loading": {"local_files_only": not args.allow_network},
    }

    try:
        return_code = run_smoke(args, report)
    except Exception as exc:  # noqa: BLE001 - a smoke report must survive failures
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        return_code = 1

    write_json(report_path, report)
    print(f"Qwen smoke report: {report_path}")
    print(f"Status: {report['status']}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
