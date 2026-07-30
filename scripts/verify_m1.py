#!/usr/bin/env python3
"""Verify the M1 data skeleton and write one compact server report."""

from __future__ import annotations

import argparse
import json
import tempfile
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from object_memory.assets import MemoryPaths
from object_memory.cli import main as cli_main
from object_memory.config import DEFAULT_CONFIG_PATH, config_digest, load_config
from object_memory.memory_store import CORE_TABLES, SCHEMA_VERSION, MemoryStore
from object_memory.schemas import (
    BoundingBox,
    Decision,
    DecisionType,
    MemoryObject,
    Observation,
    Proposal,
    Run,
    SourceImage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the M1 data skeleton.")
    parser.add_argument(
        "--report",
        default="environment/m1_skeleton_report.json",
        help="Compact JSON report path intended for Git.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def validate_core_models(configuration_digest: str) -> list[str]:
    run = Run(
        config_digest=configuration_digest,
        sam_model_id="sam3",
        qwen_model_id="qwen3-vl",
    )
    source = SourceImage(
        run_id=run.id,
        sha256="b" * 64,
        relative_path="sources/example.jpg",
        width=640,
        height=480,
    )
    proposal = Proposal(
        source_image_id=source.id,
        raw_candidate_id="candidate-1",
        score=0.9,
        bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
    )
    memory_object = MemoryObject(
        coarse_category="杯子",
        fine_category="马克杯",
        material=["陶瓷"],
        color=["红色"],
        shape="圆柱形",
        description="带白色把手的红色杯子",
        annotation_confidence=0.9,
    )
    observation = Observation(
        object_id=memory_object.id,
        proposal_id=proposal.id,
        source_image_id=source.id,
        crop_path="objects/example/observations/one/crop.png",
        mask_path="objects/example/observations/one/mask.png",
        overlay_path="objects/example/observations/one/overlay.jpg",
        description="正面可见",
    )
    decision = Decision(
        proposal_id=proposal.id,
        decision=DecisionType.EXISTING,
        matched_object_id=memory_object.id,
        confidence=0.88,
        reason_code="visual_instance_match",
        short_reason="外观特征一致",
        prompt_version="identity-v1",
    )
    return [
        type(record).__name__
        for record in (run, source, proposal, memory_object, observation, decision)
    ]


def run_verification() -> dict[str, Any]:
    config = load_config(DEFAULT_CONFIG_PATH)
    validated_models = validate_core_models(config_digest(config))

    with tempfile.TemporaryDirectory() as temporary_directory:
        memory_root = Path(temporary_directory) / "memory"
        store = MemoryStore(
            MemoryPaths(memory_root, config.storage.database_filename)
        )
        first_status = store.initialize()
        second_status = store.initialize()

        output = StringIO()
        with redirect_stdout(output):
            cli_code = cli_main(
                ["status", "--memory-root", str(memory_root), "--json"]
            )
        cli_payload = json.loads(output.getvalue())

    checks = {
        "default_config_valid": config.schema_version == 1,
        "six_core_models_valid": len(validated_models) == 6,
        "database_reopens": first_status.schema_version
        == second_status.schema_version
        == SCHEMA_VERSION,
        "six_core_tables_present": set(second_status.counts) == set(CORE_TABLES),
        "empty_store_queryable": all(
            count == 0 for count in second_status.counts.values()
        ),
        "cli_status_passed": cli_code == 0 and cli_payload.get("status") == "ready",
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "validated_models": validated_models,
        "schema_version": second_status.schema_version,
        "empty_counts": second_status.counts,
    }


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test": "m1_data_skeleton",
        "status": "failed",
    }
    try:
        report.update(run_verification())
        return_code = 0 if report["status"] == "passed" else 4
    except Exception as exc:  # noqa: BLE001 - the report must survive failures
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        return_code = 1

    report_path = Path(args.report).expanduser().resolve()
    write_json(report_path, report)
    print(f"M1 verification report: {report_path}")
    print(f"Status: {report['status']}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
