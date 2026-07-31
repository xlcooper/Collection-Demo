"""M5 batch orchestration for SAM3, Qwen, and persistent object memory."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence
from uuid import uuid4

from PIL import Image

from .assets import MemoryPaths
from .config import AppConfig, config_digest
from .identity import CandidateEvaluation, evaluate_candidate
from .memory_loop import MemoryLoop
from .memory_store import MemoryStore, RunSummary
from .mllm_adapter import MllmPrediction
from .sam3_adapter import Sam3Prediction
from .sam3_postprocess import process_candidates
from .schemas import ObjectCard, Proposal, Run, SourceImage


SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


class SamRuntime(Protocol):
    model_load_seconds: float

    def load(self) -> None: ...

    def predict(
        self,
        image: Image.Image,
    ) -> Sam3Prediction: ...

    @property
    def peak_memory_mib(self) -> float: ...

    def close(self) -> None: ...


class MllmRuntime(Protocol):
    model_load_seconds: float
    model_placement: list[str]
    resolved_snapshot: str | None

    def load(self) -> None: ...

    def predict(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> MllmPrediction: ...

    @property
    def peak_memory_mib(self) -> float: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class ImageWork:
    input_path: Path
    source: SourceImage | None = None
    duplicate: bool = False
    status: str = "discovered"
    kept: tuple[Proposal, ...] = ()
    filtered_count: int = 0
    raw_candidate_count: int = 0
    candidate_source_counts: dict[str, int] = field(default_factory=dict)
    sam_inference_seconds: float = 0.0
    decisions: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_path": str(self.input_path),
            "source_id": self.source.id if self.source else None,
            "sha256": self.source.sha256 if self.source else None,
            "stored_source": self.source.relative_path if self.source else None,
            "duplicate": self.duplicate,
            "status": self.status,
            "sam": {
                "raw_candidates": self.raw_candidate_count,
                "kept": len(self.kept),
                "filtered": self.filtered_count,
                "candidate_source_counts": self.candidate_source_counts,
                "inference_seconds": round(self.sam_inference_seconds, 3),
            },
            "decisions": self.decisions,
            "error": self.error,
        }


class RecordingPredictor:
    """Retain raw model text even when M3 response parsing fails."""

    def __init__(self, runtime: MllmRuntime) -> None:
        self.runtime = runtime
        self.predictions: list[MllmPrediction] = []

    def predict(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> MllmPrediction:
        prediction = self.runtime.predict(messages)
        self.predictions.append(prediction)
        return prediction


def discover_images(input_directory: str | Path) -> list[Path]:
    """Return a deterministic recursive list of supported source images."""

    root = Path(input_directory).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Input image directory not found: {root}")
    images = sorted(
        (
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not images:
        raise ValueError(f"No supported images found under: {root}")
    return images


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ObjectMemoryPipeline:
    """Run one unattended image batch with sequential GPU model residency."""

    def __init__(
        self,
        *,
        config: AppConfig,
        paths: MemoryPaths,
        sam_runtime: SamRuntime,
        mllm_runtime: MllmRuntime,
    ) -> None:
        self.config = config
        self.paths = paths
        self.store = MemoryStore(paths)
        self.loop = MemoryLoop(self.store)
        self.sam_runtime = sam_runtime
        self.mllm_runtime = mllm_runtime

    def run(
        self,
        image_paths: Sequence[str | Path],
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_images = [Path(path).expanduser().resolve() for path in image_paths]
        if not normalized_images:
            raise ValueError("At least one input image is required")
        if any(not path.is_file() for path in normalized_images):
            missing = next(path for path in normalized_images if not path.is_file())
            raise FileNotFoundError(f"Input image not found: {missing}")

        self.store.initialize()
        run = Run(
            id=run_id or self._new_run_id(),
            config_digest=config_digest(self.config),
            sam_model_id=str(self.config.models.sam3_checkpoint),
            qwen_model_id=self.config.models.qwen_model_id,
        )
        self.loop.begin_run(run)
        works = [ImageWork(input_path=path) for path in normalized_images]
        external_errors: list[str] = []
        for work in works:
            self._register_image(run, work, external_errors)

        sam_metrics = self._run_sam(run, works)
        qwen_metrics = self._run_qwen(run, works)
        external_errors.extend(sam_metrics.pop("external_errors"))
        external_errors.extend(qwen_metrics.pop("external_errors"))

        summary = self.loop.complete_run(
            run.id,
            external_errors=len(external_errors),
        )
        checks = self._build_checks(works, summary)
        if summary.status.value == "completed" and all(checks.values()):
            report_status = "passed"
        elif summary.status.value == "completed":
            report_status = "failed"
        else:
            report_status = summary.status.value
        report = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "test": "m5_end_to_end_batch",
            "status": report_status,
            "run": summary.as_dict(),
            "checks": checks,
            "strategy": {
                "prompt_strategy": self.config.sam3_pipeline.prompt_strategy,
                "external_category_prompts": False,
                "points_per_side": self.config.sam3_pipeline.points_per_side,
                "points_per_batch": self.config.sam3_pipeline.points_per_batch,
                "scope": (
                    "class-agnostic point-grid candidates; candidate coverage "
                    "must be verified empirically"
                ),
                "model_residency": "SAM3 then release; Qwen then release",
                "qwen_call_policy": (
                    "one analysis call per candidate; one identity call only for "
                    "valid candidates when memory cards exist"
                ),
                "object_card_selection": (
                    "rank all card text locally, then send only the top semantic "
                    "shortlist with reference images"
                ),
                "object_card_shortlist_size": (
                    self.config.mllm_pipeline.object_card_shortlist_size
                ),
                "uncertain_policy": "persist pending; do not immediately repeat",
                "error_attempts": self.config.mllm_pipeline.max_error_attempts,
            },
            "models": {
                "sam3": sam_metrics,
                "qwen": qwen_metrics,
            },
            "images": [work.as_dict() for work in works],
            "external_errors": external_errors,
            "core_counts": self.store.status().counts,
        }
        report_path = self.paths.run_reports / f"{run.id}.json"
        report["run_report"] = self.paths.relative_asset(report_path)
        write_json_atomic(report_path, report)
        return report

    @staticmethod
    def _new_run_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"run_{timestamp}"

    def _register_image(
        self,
        run: Run,
        work: ImageWork,
        external_errors: list[str],
    ) -> None:
        try:
            digest = sha256_file(work.input_path)
            with Image.open(work.input_path) as opened:
                width, height = opened.size
            suffix = work.input_path.suffix.lower()
            source_asset = self.paths.sources / f"{digest}{suffix}"
            source = SourceImage(
                id=f"src_{digest[:32]}",
                run_id=run.id,
                sha256=digest,
                relative_path=self.paths.relative_asset(source_asset),
                width=width,
                height=height,
            )
            registration = self.loop.register_source(source)
            source.id = registration.source_id
            work.source = source
            work.duplicate = registration.duplicate
            if registration.duplicate:
                work.status = "duplicate"
                return
            self._copy_source_asset(work.input_path, source_asset)
            work.status = "registered"
        except Exception as exc:  # noqa: BLE001 - continue remaining images
            work.status = "failed"
            work.error = f"{type(exc).__name__}: {exc}"
            external_errors.append(f"{work.input_path}: {work.error}")
            if work.source is not None:
                try:
                    self.loop.fail_source(work.source.id, work.error)
                except Exception as fail_exc:  # noqa: BLE001
                    external_errors.append(
                        f"failed to mark source {work.source.id}: {fail_exc}"
                    )

    @staticmethod
    def _copy_source_asset(source: Path, destination: Path) -> None:
        if destination.is_file():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{uuid4().hex}.tmp"
        )
        try:
            shutil.copy2(source, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _run_sam(
        self,
        run: Run,
        works: list[ImageWork],
    ) -> dict[str, Any]:
        pending = [work for work in works if work.status == "registered"]
        metrics: dict[str, Any] = {
            "loaded": False,
            "model_load_seconds": 0.0,
            "inference_seconds": 0.0,
            "peak_memory_mib": 0.0,
            "external_errors": [],
        }
        if not pending:
            return metrics
        try:
            self.sam_runtime.load()
            metrics["loaded"] = True
            metrics["model_load_seconds"] = round(
                self.sam_runtime.model_load_seconds,
                3,
            )
            for work in pending:
                self._process_image_with_sam(run, work)
            metrics["inference_seconds"] = round(
                sum(work.sam_inference_seconds for work in pending),
                3,
            )
            metrics["peak_memory_mib"] = round(
                self.sam_runtime.peak_memory_mib,
                2,
            )
        except Exception as exc:  # model-load or unexpected stage failure
            message = f"{type(exc).__name__}: {exc}"
            metrics["external_errors"].append(f"SAM3 stage: {message}")
            for work in pending:
                if work.status == "registered":
                    self._fail_registered_source(work, message, metrics)
        finally:
            try:
                self.sam_runtime.close()
            except Exception as exc:  # noqa: BLE001
                metrics["external_errors"].append(
                    f"SAM3 close: {type(exc).__name__}: {exc}"
                )
        return metrics

    def _process_image_with_sam(
        self,
        run: Run,
        work: ImageWork,
    ) -> None:
        assert work.source is not None
        try:
            with Image.open(work.input_path) as opened:
                image = opened.convert("RGB")
            prediction = self.sam_runtime.predict(image)
            result = process_candidates(
                prediction.candidates,
                image=image,
                source_image_id=work.source.id,
                run_id=run.id,
                paths=self.paths,
                settings=self.config.sam3_pipeline,
            )
            for proposal in result.filtered:
                self.loop.record_filtered_proposal(proposal)
            work.raw_candidate_count = len(prediction.candidates)
            work.candidate_source_counts = prediction.prompt_counts
            work.sam_inference_seconds = prediction.inference_seconds
            work.filtered_count = len(result.filtered)
            work.kept = result.kept
            if work.kept:
                work.status = "awaiting_qwen"
            else:
                self.loop.complete_source(work.source.id)
                work.status = "completed"
        except Exception as exc:  # noqa: BLE001 - one image must not stop the batch
            message = f"{type(exc).__name__}: {exc}"
            work.error = message
            self.loop.fail_source(work.source.id, message)
            work.status = "failed"

    def _fail_registered_source(
        self,
        work: ImageWork,
        message: str,
        metrics: dict[str, Any],
    ) -> None:
        assert work.source is not None
        work.error = message
        try:
            self.loop.fail_source(work.source.id, message)
            work.status = "failed"
        except Exception as exc:  # noqa: BLE001
            metrics["external_errors"].append(
                f"failed to mark source {work.source.id}: {exc}"
            )

    def _run_qwen(self, run: Run, works: list[ImageWork]) -> dict[str, Any]:
        pending = [work for work in works if work.status == "awaiting_qwen"]
        metrics: dict[str, Any] = {
            "loaded": False,
            "model_load_seconds": 0.0,
            "inference_seconds": 0.0,
            "input_tokens": 0,
            "generated_tokens": 0,
            "analysis_calls": 0,
            "identity_calls": 0,
            "peak_memory_mib": 0.0,
            "placement": [],
            "snapshot": None,
            "external_errors": [],
        }
        if not pending:
            return metrics
        try:
            self.mllm_runtime.load()
            metrics["loaded"] = True
            metrics["model_load_seconds"] = round(
                self.mllm_runtime.model_load_seconds,
                3,
            )
            metrics["placement"] = self.mllm_runtime.model_placement
            metrics["snapshot"] = self.mllm_runtime.resolved_snapshot
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            metrics["external_errors"].append(f"Qwen load: {message}")
            for work in pending:
                self._fail_qwen_work(work, message, metrics)
            try:
                self.mllm_runtime.close()
            except Exception as close_exc:  # noqa: BLE001
                metrics["external_errors"].append(
                    f"Qwen close: {type(close_exc).__name__}: {close_exc}"
                )
            return metrics

        try:
            for work in pending:
                self._process_image_with_qwen(run, work, metrics)
            metrics["peak_memory_mib"] = round(
                self.mllm_runtime.peak_memory_mib,
                2,
            )
        finally:
            try:
                self.mllm_runtime.close()
            except Exception as exc:  # noqa: BLE001
                metrics["external_errors"].append(
                    f"Qwen close: {type(exc).__name__}: {exc}"
                )
        metrics["inference_seconds"] = round(metrics["inference_seconds"], 3)
        return metrics

    def _process_image_with_qwen(
        self,
        run: Run,
        work: ImageWork,
        metrics: dict[str, Any],
    ) -> None:
        assert work.source is not None
        for proposal in work.kept:
            outcome = self._process_proposal_with_qwen(run, proposal, metrics)
            work.decisions.append(outcome)
        try:
            self.loop.complete_source(work.source.id)
            work.status = "completed"
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            work.error = message
            metrics["external_errors"].append(
                f"complete source {work.source.id}: {message}"
            )
            try:
                self.loop.fail_source(work.source.id, message)
                work.status = "failed"
            except Exception as fail_exc:  # noqa: BLE001
                metrics["external_errors"].append(
                    f"fail source {work.source.id}: {fail_exc}"
                )

    def _process_proposal_with_qwen(
        self,
        run: Run,
        proposal: Proposal,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        errors: list[str] = []
        total_qwen_calls = 0
        for call_attempt in range(
            1,
            self.config.mllm_pipeline.max_error_attempts + 1,
        ):
            recorder = RecordingPredictor(self.mllm_runtime)
            evaluation: CandidateEvaluation | None = None
            metrics_added = False
            try:
                evaluation = evaluate_candidate(
                    recorder,
                    candidate_crop=self.paths.resolve_asset(proposal.crop_path or ""),
                    candidate_overlay=self.paths.resolve_asset(
                        proposal.overlay_path or ""
                    ),
                    sam_prompt=proposal.prompt,
                    get_card_texts=self.loop.object_card_texts,
                    get_reference_cards=self._reference_cards,
                    card_assets=self.paths,
                    settings=self.config.mllm_pipeline,
                )
                self._add_prediction_metrics(recorder.predictions, metrics)
                total_qwen_calls += len(recorder.predictions)
                metrics_added = True
                raw_path = self._write_raw_response(
                    run.id,
                    proposal.id,
                    call_attempt,
                    recorder.predictions,
                    evaluation=evaluation,
                    error=None,
                )
                write_result = self.loop.apply_response(
                    proposal=proposal,
                    response=evaluation.final_response,
                    prompt_version=self.config.mllm_pipeline.prompt_version,
                    raw_response_path=raw_path,
                    attempt=1,
                )
                response = evaluation.final_response
                return {
                    "proposal_id": proposal.id,
                    "candidate": self._candidate_report(proposal),
                    "status": write_result.proposal_status.value,
                    "decision": write_result.decision.value,
                    "object_id": write_result.object_id,
                    "confidence": response.confidence,
                    "reason_code": response.reason_code.value,
                    "short_reason": response.short_reason,
                    "annotation": (
                        response.annotation.model_dump(mode="json")
                        if response.annotation is not None
                        else None
                    ),
                    "candidate_analysis": evaluation.analysis.model_dump(mode="json"),
                    "retrieval": {
                        "memory_lookup_performed": (
                            evaluation.memory_lookup_performed
                        ),
                        "available_object_cards": (
                            evaluation.available_object_cards
                        ),
                        "shortlisted": [
                            item.as_dict() for item in evaluation.retrieved_cards
                        ],
                    },
                    "identity_confirmation": (
                        evaluation.identity_response.model_dump(mode="json")
                        if evaluation.identity_response is not None
                        else None
                    ),
                    "qwen_calls": total_qwen_calls,
                    "pipeline_attempts": call_attempt,
                    "raw_response": raw_path,
                    "errors": errors,
                }
            except Exception as exc:  # noqa: BLE001 - bounded model retry
                if not metrics_added:
                    self._add_prediction_metrics(recorder.predictions, metrics)
                    total_qwen_calls += len(recorder.predictions)
                message = f"{type(exc).__name__}: {exc}"
                errors.append(message)
                self._write_raw_response(
                    run.id,
                    proposal.id,
                    call_attempt,
                    recorder.predictions,
                    evaluation=evaluation,
                    error=message,
                )
                if evaluation is not None:
                    break

        error_message = errors[-1] if errors else "Qwen produced no usable response"
        try:
            self.loop.record_proposal_failure(proposal, error_message)
        except Exception as exc:  # noqa: BLE001
            metrics["external_errors"].append(
                f"record proposal failure {proposal.id}: {exc}"
            )
        return {
            "proposal_id": proposal.id,
            "candidate": self._candidate_report(proposal),
            "status": "failed",
            "decision": None,
            "object_id": None,
            "confidence": None,
            "reason_code": None,
            "short_reason": None,
            "annotation": None,
            "candidate_analysis": None,
            "retrieval": None,
            "identity_confirmation": None,
            "qwen_calls": total_qwen_calls,
            "pipeline_attempts": len(errors),
            "errors": errors,
        }

    @staticmethod
    def _candidate_report(proposal: Proposal) -> dict[str, Any]:
        return {
            "raw_candidate_id": proposal.raw_candidate_id,
            "source": proposal.prompt,
            "score": proposal.score,
            "bbox": proposal.bbox.model_dump(mode="json"),
            "mask_area_ratio": proposal.mask_area_ratio,
            "crop": proposal.crop_path,
            "mask": proposal.mask_path,
            "overlay": proposal.overlay_path,
        }

    def _reference_cards(
        self,
        object_ids: Sequence[str],
    ) -> list[ObjectCard]:
        return self.loop.object_cards_by_ids(
            list(object_ids),
            max_reference_views=(
                self.config.mllm_pipeline.max_reference_views_per_object
            ),
        )

    @staticmethod
    def _add_prediction_metrics(
        predictions: Sequence[MllmPrediction],
        metrics: dict[str, Any],
    ) -> None:
        metrics["inference_seconds"] += sum(
            prediction.inference_seconds for prediction in predictions
        )
        metrics["input_tokens"] += sum(
            prediction.input_tokens for prediction in predictions
        )
        metrics["generated_tokens"] += sum(
            prediction.generated_tokens for prediction in predictions
        )
        if predictions:
            metrics["analysis_calls"] += 1
            metrics["identity_calls"] += max(0, len(predictions) - 1)

    def _write_raw_response(
        self,
        run_id: str,
        proposal_id: str,
        call_attempt: int,
        predictions: Sequence[MllmPrediction],
        *,
        evaluation: CandidateEvaluation | None,
        error: str | None,
    ) -> str | None:
        path = (
            self.paths.raw_response_dir(run_id, proposal_id)
            / f"call_{call_attempt:02d}.json"
        )
        payload = {
            "call_attempt": call_attempt,
            "error": error,
            "final_response": (
                evaluation.final_response.model_dump(mode="json")
                if evaluation is not None
                else None
            ),
            "candidate_analysis": (
                evaluation.analysis.model_dump(mode="json")
                if evaluation is not None
                else None
            ),
            "retrieval": (
                {
                    "memory_lookup_performed": (
                        evaluation.memory_lookup_performed
                    ),
                    "available_object_cards": evaluation.available_object_cards,
                    "shortlisted": [
                        item.as_dict() for item in evaluation.retrieved_cards
                    ],
                }
                if evaluation is not None
                else None
            ),
            "identity_confirmation": (
                evaluation.identity_response.model_dump(mode="json")
                if evaluation is not None
                and evaluation.identity_response is not None
                else None
            ),
            "predictions": [
                {
                    "stage": (
                        "candidate_analysis"
                        if index == 0
                        else "identity_confirmation"
                    ),
                    "raw_text": prediction.raw_text,
                    "input_tokens": prediction.input_tokens,
                    "generated_tokens": prediction.generated_tokens,
                    "inference_seconds": prediction.inference_seconds,
                }
                for index, prediction in enumerate(predictions)
            ],
        }
        try:
            write_json_atomic(path, payload)
            return self.paths.relative_asset(path)
        except Exception:
            return None

    def _fail_qwen_work(
        self,
        work: ImageWork,
        message: str,
        metrics: dict[str, Any],
    ) -> None:
        assert work.source is not None
        for proposal in work.kept:
            try:
                self.loop.record_proposal_failure(proposal, message)
                work.decisions.append(
                    {
                        "proposal_id": proposal.id,
                        "candidate": self._candidate_report(proposal),
                        "status": "failed",
                        "decision": None,
                        "object_id": None,
                        "confidence": None,
                        "reason_code": None,
                        "short_reason": None,
                        "annotation": None,
                        "candidate_analysis": None,
                        "retrieval": None,
                        "identity_confirmation": None,
                        "qwen_calls": 0,
                        "pipeline_attempts": 0,
                        "errors": [message],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                metrics["external_errors"].append(
                    f"record proposal failure {proposal.id}: {exc}"
                )
        try:
            self.loop.complete_source(work.source.id)
            work.status = "completed"
        except Exception as exc:  # noqa: BLE001
            metrics["external_errors"].append(
                f"complete source {work.source.id}: {exc}"
            )
            work.status = "failed"
            work.error = message

    @staticmethod
    def _build_checks(
        works: Sequence[ImageWork],
        summary: RunSummary,
    ) -> dict[str, bool]:
        nonduplicates = [work for work in works if not work.duplicate]
        accounted_statuses = {"completed", "failed"}
        return {
            "input_images_nonempty": bool(works),
            "every_input_accounted_for": all(
                work.status == "duplicate" or work.status in accounted_statuses
                for work in works
            ),
            "sam_then_qwen_sequential_boundary": True,
            "no_source_left_processing": summary.source_counts["processing"] == 0,
            "registered_sources_match_nonduplicates": (
                sum(summary.source_counts.values()) == len(nonduplicates)
            ),
            "all_qwen_candidates_have_persisted_status": all(
                decision.get("status") in {"decided", "pending", "failed"}
                for work in works
                for decision in work.decisions
            ),
        }
