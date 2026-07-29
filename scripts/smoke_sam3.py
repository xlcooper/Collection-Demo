#!/usr/bin/env python3
"""Run one real SAM3 image inference and write a compact, Git-friendly report."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test SAM3 image inference.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument(
        "--checkpoint",
        default="weights/sam3/sam3.pt",
        help="SAM3 checkpoint path.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Concrete singular English category visibly present in the image.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/smoke/sam3",
        help="Ignored directory for generated masks.",
    )
    parser.add_argument(
        "--report",
        default="environment/sam3_smoke_report.json",
        help="Compact report path intended for Git.",
    )
    parser.add_argument("--max-saved-masks", type=int, default=8)
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


def run_smoke(args: argparse.Namespace, report: dict[str, Any]) -> int:
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The active CUDA device does not support bfloat16 autocast.")

    image_path = Path(args.image).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")

    builder_parameters = inspect.signature(build_sam3_image_model).parameters
    if "checkpoint_path" not in builder_parameters:
        raise RuntimeError("Installed SAM3 builder does not accept checkpoint_path.")

    builder_kwargs: dict[str, Any] = {"checkpoint_path": str(checkpoint_path)}
    optional_kwargs = {
        "load_from_HF": False,
        "device": "cuda",
        "eval_mode": True,
    }
    for name, value in optional_kwargs.items():
        if name in builder_parameters:
            builder_kwargs[name] = value

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = build_sam3_image_model(**builder_kwargs)
    processor = Sam3Processor(model)
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    inference_started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=args.prompt)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started

    masks = torch.as_tensor(output["masks"]).detach().cpu()
    boxes = torch.as_tensor(output["boxes"]).detach().cpu()
    scores = torch.as_tensor(output["scores"]).detach().cpu()

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)

    saved_masks: list[str] = []
    for index, mask in enumerate(masks[: max(args.max_saved_masks, 0)]):
        mask_image = Image.fromarray(((mask > 0.5).numpy() * 255).astype("uint8"))
        filename = f"mask_{index:03d}.png"
        mask_image.save(output_dir / filename)
        saved_masks.append(filename)

    proposal_count = int(masks.shape[0])
    report.update(
        {
            "status": "passed" if proposal_count > 0 else "failed",
            "input": {
                "filename": image_path.name,
                "sha256": sha256_file(image_path),
                "width": image.width,
                "height": image.height,
            },
            "checkpoint": {
                "filename": checkpoint_path.name,
                "size_gib": round(checkpoint_path.stat().st_size / (1024**3), 3),
            },
            "prompt": args.prompt,
            "precision": {"autocast": "bfloat16"},
            "proposal_count": proposal_count,
            "boxes_preview": boxes.tolist()[:20],
            "scores_preview": scores.flatten().tolist()[:20],
            "saved_masks": saved_masks,
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

    if proposal_count == 0:
        report["error"] = "SAM3 returned no proposals for the smoke-test prompt."
        return 4
    return 0


def main() -> int:
    args = parse_args()
    report_path = Path(args.report).expanduser().resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test": "sam3_image_smoke",
        "status": "failed",
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
    print(f"SAM3 smoke report: {report_path}")
    print(f"Status: {report['status']}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
