"""Batch orchestration for SAM3, Qwen, and persistent object memory."""

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
from .identity import (
    BatchCandidateInput,
    ImageBatchEvaluation,
    evaluate_image_batch,
)
from .memory_loop import MemoryLoop
from .memory_store import MemoryStore, RunSummary
from .mllm_adapter import MllmPrediction
from .sam3_adapter import Sam3Prediction
from .sam3_postprocess import process_candidates
from .scene_guidance import (
    SceneGuidanceEvaluation,
    SceneImageInput,
    evaluate_scene_guidance_batch,
)
from .schemas import BatchCandidateDecision, Proposal, Run, SourceImage


SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


class SamRuntime(Protocol):
    model_load_seconds: float

    def load(self) -> None: ...

    def predict(
        self,
        image: Image.Image,
        prompts: Sequence[str],
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
    scene_prompts: tuple[str, ...] = ()
    scene_guidance: dict[str, Any] | None = None
    kept: tuple[Proposal, ...] = ()
    filtered_count: int = 0
    raw_candidate_count: int = 0
    candidate_source_counts: dict[str, int] = field(default_factory=dict)
    sam_inference_seconds: float = 0.0
    decisions: list[dict[str, Any]] = field(default_factory=list)
    candidate_reasoning: dict[str, Any] | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_path": str(self.input_path),
            "source_id": self.source.id if self.source else None,
            "sha256": self.source.sha256 if self.source else None,
            "stored_source": self.source.relative_path if self.source else None,
            "duplicate": self.duplicate,
            "status": self.status,
            "scene_guidance": self.scene_guidance,
            "sam": {
                "model_thresholded_candidates": self.raw_candidate_count,
                "kept": len(self.kept),
                "filtered": self.filtered_count,
                "prompt_detection_counts": self.candidate_source_counts,
                "inference_seconds": round(self.sam_inference_seconds, 3),
            },
            "candidate_reasoning": self.candidate_reasoning,
            "decisions": self.decisions,
            "error": self.error,
        }


class RecordingPredictor:
    """Retain raw model text even when response parsing fails."""

    def __init__(self, runtime: MllmRuntime) -> None:
        self.runtime = runtime
        self.predictions: list[MllmPrediction] = []
        self.attempted_calls = 0

    def predict(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> MllmPrediction:
        self.attempted_calls += 1
        prediction = self.runtime.predict(messages)
        self.predictions.append(prediction)
        return prediction


@dataclass(frozen=True, slots=True)
class SceneGuidanceCallResult:
    """One audited scene-guidance model call for one configured batch."""

    scope_id: str
    evaluation: SceneGuidanceEvaluation | None
    qwen_calls: int
    raw_response: str | None
    errors: tuple[str, ...]


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

        scene_metrics = self._run_scene_guidance(run, works)
        sam_metrics = self._run_sam(run, works)
        candidate_metrics = self._run_candidate_reasoning(run, works)
        external_errors.extend(scene_metrics.pop("external_errors"))
        external_errors.extend(sam_metrics.pop("external_errors"))
        external_errors.extend(candidate_metrics.pop("external_errors"))
        qwen_metrics = self._combine_qwen_metrics(
            scene_metrics,
            candidate_metrics,
        )

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
            "schema_version": 4,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "test": "object_memory_demo_batch",
            "status": report_status,
            "run": summary.as_dict(),
            "checks": checks,
            "strategy": {
                "prompt_strategy": "qwen_scene_guidance_to_sam3_text_concepts",
                "external_category_prompts": False,
                "model_generated_category_prompts": True,
                "scene_guidance_batch_size": (
                    self.config.mllm_pipeline.scene_batch_size
                ),
                "max_scene_targets_per_image": (
                    self.config.mllm_pipeline.max_scene_targets_per_image
                ),
                "scope": (
                    "robot-oriented scene triage followed by open-vocabulary "
                    "concept segmentation; first-pass recall must be audited"
                ),
                "model_residency": (
                    "Qwen scene guidance then release; SAM3 text guidance then "
                    "release; Qwen candidate reasoning then release"
                ),
                "qwen_call_policy": (
                    "one scene-guidance call per configured source-image batch, "
                    "then one detailed call per source image containing every "
                    "retained candidate and every active memory object card"
                ),
                "scene_guidance_memory_context": (
                    "none; discovery is driven only by each new scene view"
                ),
                "qwen_error_policy": (
                    "one model call per logical scope; persist the raw result "
                    "and fail the affected scope without automatic retry, "
                    "single-source rescue, normalization, or fallback"
                ),
                "sam_candidate_semantics": (
                    "SAM3 detections already above the processor text-confidence "
                    "threshold, followed by script area, duplicate, same-prompt "
                    "containment, and fair per-prompt capacity filtering"
                ),
                "object_card_selection": (
                    "all active cards with configured recent reference views; "
                    "no script-side similarity ranking or shortlist"
                ),
                "uncertain_policy": "persist pending; do not immediately repeat",
                "scene_prompt_version": (
                    self.config.mllm_pipeline.scene_prompt_version
                ),
                "candidate_prompt_version": (
                    self.config.mllm_pipeline.prompt_version
                ),
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

    def _run_scene_guidance(
        self,
        run: Run,
        works: list[ImageWork],
    ) -> dict[str, Any]:
        """Use Qwen after the hash gate to plan text-guided SAM3 targets."""

        pending = [work for work in works if work.status == "registered"]
        metrics: dict[str, Any] = {
            "loaded": False,
            "model_load_seconds": 0.0,
            "inference_seconds": 0.0,
            "input_tokens": 0,
            "generated_tokens": 0,
            "scene_batch_calls": 0,
            "scene_batches": 0,
            "images_analyzed": 0,
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
            metrics["external_errors"].append(
                f"Qwen scene-guidance load: {message}"
            )
            for work in pending:
                self._fail_source_work(work, message, metrics)
            try:
                self.mllm_runtime.close()
            except Exception as close_exc:  # noqa: BLE001
                metrics["external_errors"].append(
                    "Qwen scene-guidance close: "
                    f"{type(close_exc).__name__}: {close_exc}"
                )
            return metrics

        batch_size = self.config.mllm_pipeline.scene_batch_size
        try:
            for batch_offset in range(0, len(pending), batch_size):
                batch = pending[batch_offset : batch_offset + batch_size]
                self._process_scene_guidance_batch(
                    run,
                    batch,
                    batch_index=batch_offset // batch_size + 1,
                    metrics=metrics,
                )
            metrics["peak_memory_mib"] = round(
                self.mllm_runtime.peak_memory_mib,
                2,
            )
        finally:
            try:
                self.mllm_runtime.close()
            except Exception as exc:  # noqa: BLE001
                metrics["external_errors"].append(
                    "Qwen scene-guidance close: "
                    f"{type(exc).__name__}: {exc}"
                )
        metrics["inference_seconds"] = round(metrics["inference_seconds"], 3)
        return metrics

    def _process_scene_guidance_batch(
        self,
        run: Run,
        works: Sequence[ImageWork],
        *,
        batch_index: int,
        metrics: dict[str, Any],
    ) -> None:
        batch = list(works)
        if not batch:
            return
        if any(work.source is None for work in batch):
            raise ValueError("Scene-guidance work requires registered sources")
        scope_id = f"scene_batch_{batch_index:04d}"
        metrics["scene_batches"] += 1
        result = self._call_scene_guidance_scope(
            run,
            batch,
            scope_id=scope_id,
            batch_index=batch_index,
            metrics=metrics,
        )
        if result.evaluation is not None:
            self._apply_scene_guidance(
                batch,
                result,
                batch_index=batch_index,
                metrics=metrics,
            )
            return
        for work in batch:
            self._record_scene_guidance_failure(
                work,
                result,
                batch_index=batch_index,
                metrics=metrics,
            )

    def _call_scene_guidance_scope(
        self,
        run: Run,
        works: Sequence[ImageWork],
        *,
        scope_id: str,
        batch_index: int,
        metrics: dict[str, Any],
    ) -> SceneGuidanceCallResult:
        scene_inputs = [
            SceneImageInput(
                source_id=work.source.id,
                image_path=self.paths.resolve_asset(work.source.relative_path),
            )
            for work in works
            if work.source is not None
        ]
        errors: list[str] = []
        recorder = RecordingPredictor(self.mllm_runtime)
        evaluation: SceneGuidanceEvaluation | None = None
        call_error: str | None = None
        try:
            evaluation = evaluate_scene_guidance_batch(
                recorder,
                images=scene_inputs,
                settings=self.config.mllm_pipeline,
            )
        except Exception as exc:  # noqa: BLE001 - expose the model/protocol failure
            call_error = f"{type(exc).__name__}: {exc}"
            errors.append(call_error)

        self._add_prediction_metrics(
            recorder.predictions,
            metrics,
            call_field="scene_batch_calls",
            attempted_calls=recorder.attempted_calls,
        )
        raw_response: str | None = None
        try:
            raw_response = self._write_scene_guidance_raw_response(
                run.id,
                scope_id,
                batch_index,
                scene_inputs,
                recorder.predictions,
                evaluation=evaluation,
                error=call_error,
            )
        except Exception as exc:  # noqa: BLE001 - audit data is mandatory
            storage_error = (
                "raw response persistence failed: "
                f"{type(exc).__name__}: {exc}"
            )
            metrics["external_errors"].append(
                f"Qwen scene-guidance {scope_id}: {storage_error}"
            )
            errors.append(storage_error)
            evaluation = None

        return SceneGuidanceCallResult(
            scope_id=scope_id,
            evaluation=evaluation,
            qwen_calls=recorder.attempted_calls,
            raw_response=raw_response,
            errors=tuple(errors),
        )

    def _apply_scene_guidance(
        self,
        works: Sequence[ImageWork],
        result: SceneGuidanceCallResult,
        *,
        batch_index: int,
        metrics: dict[str, Any],
    ) -> None:
        assert result.evaluation is not None
        guidance_by_source = {
            guidance.source_id: guidance
            for guidance in result.evaluation.response.images
        }
        metrics["images_analyzed"] += len(works)
        for work in works:
            assert work.source is not None
            guidance = guidance_by_source[work.source.id]
            work.scene_prompts = tuple(
                target.sam_text_prompt for target in guidance.targets
            )
            work.scene_guidance = {
                "prompt_version": self.config.mllm_pipeline.scene_prompt_version,
                "batch_index": batch_index,
                "scope_id": result.scope_id,
                "scene_summary": guidance.scene_summary,
                "target_count": len(guidance.targets),
                "targets": [
                    target.model_dump(mode="json")
                    for target in guidance.targets
                ],
                "no_target_reason": guidance.no_target_reason,
                "qwen_calls": result.qwen_calls,
                "raw_response": result.raw_response,
                "errors": list(result.errors),
            }
            if work.scene_prompts:
                work.status = "awaiting_sam"
                continue
            try:
                self.loop.complete_source(work.source.id)
                work.status = "completed"
            except Exception as exc:  # noqa: BLE001
                message = f"{type(exc).__name__}: {exc}"
                metrics["external_errors"].append(
                    f"complete no-target source {work.source.id}: {message}"
                )
                self._fail_source_work(work, message, metrics)

    def _record_scene_guidance_failure(
        self,
        work: ImageWork,
        result: SceneGuidanceCallResult,
        *,
        batch_index: int,
        metrics: dict[str, Any],
    ) -> None:
        all_errors = list(result.errors)
        error_message = (
            all_errors[-1]
            if all_errors
            else "Qwen produced no usable scene guidance"
        )
        work.scene_guidance = {
            "prompt_version": self.config.mllm_pipeline.scene_prompt_version,
            "batch_index": batch_index,
            "scope_id": result.scope_id,
            "target_count": None,
            "targets": None,
            "qwen_calls": result.qwen_calls,
            "raw_response": result.raw_response,
            "errors": all_errors,
        }
        self._fail_source_work(work, error_message, metrics)

    def _run_sam(
        self,
        run: Run,
        works: list[ImageWork],
    ) -> dict[str, Any]:
        pending = [work for work in works if work.status == "awaiting_sam"]
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
                if work.status == "awaiting_sam":
                    self._fail_source_work(work, message, metrics)
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
            canonical_source = self.paths.resolve_asset(work.source.relative_path)
            with Image.open(canonical_source) as opened:
                image = opened.convert("RGB")
            prediction = self.sam_runtime.predict(image, work.scene_prompts)
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
                work.status = "awaiting_candidate_reasoning"
            else:
                self.loop.complete_source(work.source.id)
                work.status = "completed"
        except Exception as exc:  # noqa: BLE001 - one image must not stop the batch
            message = f"{type(exc).__name__}: {exc}"
            work.error = message
            self.loop.fail_source(work.source.id, message)
            work.status = "failed"

    def _fail_source_work(
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

    def _run_candidate_reasoning(
        self,
        run: Run,
        works: list[ImageWork],
    ) -> dict[str, Any]:
        pending = [
            work
            for work in works
            if work.status == "awaiting_candidate_reasoning"
        ]
        metrics: dict[str, Any] = {
            "loaded": False,
            "model_load_seconds": 0.0,
            "inference_seconds": 0.0,
            "input_tokens": 0,
            "generated_tokens": 0,
            "candidate_reasoning_calls": 0,
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
            metrics["external_errors"].append(
                f"Qwen candidate-reasoning load: {message}"
            )
            for work in pending:
                self._fail_qwen_work(work, message, metrics)
            try:
                self.mllm_runtime.close()
            except Exception as close_exc:  # noqa: BLE001
                metrics["external_errors"].append(
                    "Qwen candidate-reasoning close: "
                    f"{type(close_exc).__name__}: {close_exc}"
                )
            return metrics

        try:
            for work in pending:
                self._process_image_candidate_reasoning(run, work, metrics)
            metrics["peak_memory_mib"] = round(
                self.mllm_runtime.peak_memory_mib,
                2,
            )
        finally:
            try:
                self.mllm_runtime.close()
            except Exception as exc:  # noqa: BLE001
                metrics["external_errors"].append(
                    "Qwen candidate-reasoning close: "
                    f"{type(exc).__name__}: {exc}"
                )
        metrics["inference_seconds"] = round(metrics["inference_seconds"], 3)
        return metrics

    def _process_image_candidate_reasoning(
        self,
        run: Run,
        work: ImageWork,
        metrics: dict[str, Any],
    ) -> None:
        assert work.source is not None
        errors: list[str] = []
        evaluation: ImageBatchEvaluation | None = None
        raw_path: str | None = None
        try:
            cards = self.loop.object_cards(
                max_reference_views=(
                    self.config.mllm_pipeline.max_reference_views_per_object
                )
            )
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            metrics["external_errors"].append(
                f"load object cards for {work.source.id}: {message}"
            )
            self._fail_qwen_work(work, message, metrics)
            return
        try:
            candidates = [
                BatchCandidateInput(
                    proposal_id=proposal.id,
                    crop_path=self.paths.resolve_asset(proposal.crop_path or ""),
                    overlay_path=self.paths.resolve_asset(
                        proposal.overlay_path or ""
                    ),
                    sam_prompt=proposal.prompt,
                )
                for proposal in work.kept
            ]
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            metrics["external_errors"].append(
                f"resolve candidate assets for {work.source.id}: {message}"
            )
            self._fail_qwen_work(work, message, metrics)
            return
        recorder = RecordingPredictor(self.mllm_runtime)
        call_error: str | None = None
        try:
            evaluation = evaluate_image_batch(
                recorder,
                candidates=candidates,
                cards=cards,
                card_assets=self.paths,
                settings=self.config.mllm_pipeline,
            )
        except Exception as exc:  # noqa: BLE001 - expose the model/protocol failure
            call_error = f"{type(exc).__name__}: {exc}"
            errors.append(call_error)

        self._add_prediction_metrics(
            recorder.predictions,
            metrics,
            call_field="candidate_reasoning_calls",
            attempted_calls=recorder.attempted_calls,
        )
        qwen_calls = recorder.attempted_calls
        try:
            raw_path = self._write_batch_raw_response(
                run.id,
                work.source.id,
                recorder.predictions,
                expected_proposal_ids=[
                    candidate.proposal_id for candidate in candidates
                ],
                object_card_ids=[card.object_id for card in cards],
                reference_image_count=sum(
                    len(card.representative_view_paths) for card in cards
                ),
                evaluation=evaluation,
                error=call_error,
            )
        except Exception as exc:  # noqa: BLE001 - audit data is mandatory
            storage_error = (
                "raw response persistence failed: "
                f"{type(exc).__name__}: {exc}"
            )
            metrics["external_errors"].append(
                f"Qwen candidate-reasoning {work.source.id}: {storage_error}"
            )
            errors.append(storage_error)
            evaluation = None

        if evaluation is None:
            error_message = (
                errors[-1] if errors else "Qwen produced no usable batch response"
            )
            work.candidate_reasoning = {
                "candidate_count": len(work.kept),
                "object_card_count": len(cards),
                "object_card_ids": [card.object_id for card in cards],
                "reference_image_count": sum(
                    len(card.representative_view_paths) for card in cards
                ),
                "qwen_calls": qwen_calls,
                "raw_response": raw_path,
                "errors": errors,
            }
            for proposal in work.kept:
                work.decisions.append(
                    self._record_failed_proposal(
                        proposal,
                        error_message,
                        errors=errors,
                        raw_path=raw_path,
                        metrics=metrics,
                    )
                )
        else:
            results_by_id = {
                item.proposal_id: item for item in evaluation.response.candidates
            }
            work.candidate_reasoning = {
                "candidate_count": len(work.kept),
                "object_card_count": evaluation.object_card_count,
                "object_card_ids": list(evaluation.object_card_ids),
                "reference_image_count": evaluation.reference_image_count,
                "qwen_calls": qwen_calls,
                "raw_response": raw_path,
                "errors": errors,
            }
            for proposal in work.kept:
                result = results_by_id[proposal.id]
                work.decisions.append(
                    self._persist_batch_candidate(
                        proposal,
                        result,
                        raw_path=raw_path,
                        errors=errors,
                        metrics=metrics,
                    )
                )

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

    def _persist_batch_candidate(
        self,
        proposal: Proposal,
        result: BatchCandidateDecision,
        *,
        raw_path: str | None,
        errors: list[str],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            response = result.to_mllm_response()
            write_result = self.loop.apply_response(
                proposal=proposal,
                response=response,
                prompt_version=self.config.mllm_pipeline.prompt_version,
                raw_response_path=raw_path,
                attempt=1,
            )
            return {
                "proposal_id": proposal.id,
                "candidate": self._candidate_report(proposal),
                "status": write_result.proposal_status.value,
                "decision": write_result.decision.value,
                "object_id": write_result.object_id,
                "confidence": response.confidence,
                "reason_code": response.reason_code.value,
                "short_reason": response.short_reason,
                "validity": result.validity.value,
                "validity_confidence": result.validity_confidence,
                "validity_reason_code": result.validity_reason_code.value,
                "validity_short_reason": result.validity_short_reason,
                "temporary_annotation": (
                    result.temporary_annotation.model_dump(mode="json")
                    if result.temporary_annotation is not None
                    else None
                ),
                "final_annotation": (
                    result.final_annotation.model_dump(mode="json")
                    if result.final_annotation is not None
                    else None
                ),
                "matched_object_id": result.matched_object_id,
                "raw_response": raw_path,
                "errors": errors,
            }
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            metrics["external_errors"].append(
                f"persist batch candidate {proposal.id}: {message}"
            )
            return self._record_failed_proposal(
                proposal,
                message,
                errors=[*errors, message],
                raw_path=raw_path,
                metrics=metrics,
            )

    def _record_failed_proposal(
        self,
        proposal: Proposal,
        error_message: str,
        *,
        errors: Sequence[str],
        raw_path: str | None = None,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
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
            "validity": None,
            "validity_confidence": None,
            "validity_reason_code": None,
            "validity_short_reason": None,
            "temporary_annotation": None,
            "final_annotation": None,
            "matched_object_id": None,
            "raw_response": raw_path,
            "errors": list(errors),
        }

    @staticmethod
    def _candidate_report(proposal: Proposal) -> dict[str, Any]:
        return {
            "raw_candidate_id": proposal.raw_candidate_id,
            "sam_text_prompt": proposal.prompt,
            "score": proposal.score,
            "bbox": proposal.bbox.model_dump(mode="json"),
            "mask_area_ratio": proposal.mask_area_ratio,
            "crop": proposal.crop_path,
            "mask": proposal.mask_path,
            "overlay": proposal.overlay_path,
        }

    @staticmethod
    def _add_prediction_metrics(
        predictions: Sequence[MllmPrediction],
        metrics: dict[str, Any],
        *,
        call_field: str,
        attempted_calls: int,
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
        metrics[call_field] += attempted_calls

    def _write_scene_guidance_raw_response(
        self,
        run_id: str,
        scope_id: str,
        batch_index: int,
        scene_inputs: Sequence[SceneImageInput],
        predictions: Sequence[MllmPrediction],
        *,
        evaluation: SceneGuidanceEvaluation | None,
        error: str | None,
    ) -> str:
        path = self.paths.raw_response_dir(run_id, scope_id) / "response.json"
        payload = {
            "stage": "scene_guidance",
            "prompt_version": self.config.mllm_pipeline.scene_prompt_version,
            "scope_id": scope_id,
            "batch_index": batch_index,
            "expected_source_ids": [item.source_id for item in scene_inputs],
            "error": error,
            "scene_guidance_response": (
                evaluation.response.model_dump(mode="json")
                if evaluation is not None
                else None
            ),
            "predictions": [
                {
                    "raw_text": prediction.raw_text,
                    "input_tokens": prediction.input_tokens,
                    "generated_tokens": prediction.generated_tokens,
                    "inference_seconds": prediction.inference_seconds,
                }
                for prediction in predictions
            ],
        }
        write_json_atomic(path, payload)
        return self.paths.relative_asset(path)

    def _write_batch_raw_response(
        self,
        run_id: str,
        source_id: str,
        predictions: Sequence[MllmPrediction],
        *,
        expected_proposal_ids: Sequence[str],
        object_card_ids: Sequence[str],
        reference_image_count: int,
        evaluation: ImageBatchEvaluation | None,
        error: str | None,
    ) -> str:
        path = self.paths.raw_response_dir(run_id, source_id) / "response.json"
        payload = {
            "stage": "candidate_reasoning",
            "prompt_version": self.config.mllm_pipeline.prompt_version,
            "expected_proposal_ids": list(expected_proposal_ids),
            "error": error,
            "candidate_reasoning_response": (
                evaluation.response.model_dump(mode="json")
                if evaluation is not None
                else None
            ),
            "memory_context": {
                "object_card_count": len(object_card_ids),
                "object_card_ids": list(object_card_ids),
                "reference_image_count": reference_image_count,
                "selection": "all_active_objects",
            },
            "predictions": [
                {
                    "raw_text": prediction.raw_text,
                    "input_tokens": prediction.input_tokens,
                    "generated_tokens": prediction.generated_tokens,
                    "inference_seconds": prediction.inference_seconds,
                }
                for prediction in predictions
            ],
        }
        write_json_atomic(path, payload)
        return self.paths.relative_asset(path)

    def _fail_qwen_work(
        self,
        work: ImageWork,
        message: str,
        metrics: dict[str, Any],
    ) -> None:
        assert work.source is not None
        work.candidate_reasoning = {
            "candidate_count": len(work.kept),
            "object_card_count": None,
            "object_card_ids": None,
            "reference_image_count": None,
            "qwen_calls": 0,
            "raw_response": None,
            "errors": [message],
        }
        for proposal in work.kept:
            work.decisions.append(
                self._record_failed_proposal(
                    proposal,
                    message,
                    errors=[message],
                    metrics=metrics,
                )
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
    def _combine_qwen_metrics(
        scene_metrics: dict[str, Any],
        candidate_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Combine two separately resident Qwen phases without hiding either."""

        placement = (
            candidate_metrics["placement"]
            if candidate_metrics["placement"]
            else scene_metrics["placement"]
        )
        snapshot = candidate_metrics["snapshot"] or scene_metrics["snapshot"]
        scene_calls = scene_metrics["scene_batch_calls"]
        candidate_calls = candidate_metrics["candidate_reasoning_calls"]
        return {
            "loaded": scene_metrics["loaded"] or candidate_metrics["loaded"],
            "load_count": int(scene_metrics["loaded"])
            + int(candidate_metrics["loaded"]),
            "model_load_seconds": round(
                scene_metrics["model_load_seconds"]
                + candidate_metrics["model_load_seconds"],
                3,
            ),
            "inference_seconds": round(
                scene_metrics["inference_seconds"]
                + candidate_metrics["inference_seconds"],
                3,
            ),
            "input_tokens": (
                scene_metrics["input_tokens"] + candidate_metrics["input_tokens"]
            ),
            "generated_tokens": (
                scene_metrics["generated_tokens"]
                + candidate_metrics["generated_tokens"]
            ),
            "scene_batch_calls": scene_calls,
            "candidate_reasoning_calls": candidate_calls,
            "total_calls": scene_calls + candidate_calls,
            "peak_memory_mib": max(
                scene_metrics["peak_memory_mib"],
                candidate_metrics["peak_memory_mib"],
            ),
            "placement": placement,
            "snapshot": snapshot,
            "phases": {
                "scene_guidance": scene_metrics,
                "candidate_reasoning": candidate_metrics,
            },
        }

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
            "qwen_then_sam_then_qwen_sequential_boundary": True,
            "every_processed_source_has_scene_guidance": all(
                work.duplicate
                or work.scene_guidance is not None
                or work.status == "failed"
                for work in works
            ),
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
