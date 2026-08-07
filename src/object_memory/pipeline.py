"""Per-image Qwen -> SAM3 -> DINOv3 -> memory orchestration."""

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
from .dinov3_adapter import (
    FingerprintData,
    HistoricalFingerprint,
    match_fingerprint,
    read_fingerprint,
    write_fingerprint,
)
from .identity_decision import (
    associate_targets,
    decide_identity,
    unmatched_proposal_decision,
)
from .memory_loop import MemoryLoop
from .memory_store import MemoryStore, RunSummary
from .mllm_adapter import MllmPrediction
from .progress import ProgressReporter
from .sam3_adapter import Sam3Prediction
from .sam3_postprocess import process_candidates
from .scene_guidance import (
    SceneGuidanceEvaluation,
    SceneImageInput,
    evaluate_scene_guidance,
)
from .schemas import Proposal, Run, SceneTarget, SourceImage


SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


class SamRuntime(Protocol):
    model_load_seconds: float

    def load(self) -> None: ...

    def predict(self, image: Image.Image, prompts: Sequence[str]) -> Sam3Prediction: ...

    @property
    def peak_memory_mib(self) -> float: ...

    def close(self) -> None: ...


class MllmRuntime(Protocol):
    model_load_seconds: float
    model_placement: list[str]
    resolved_snapshot: str | None

    def load(self) -> None: ...

    def predict(self, messages: Sequence[dict[str, Any]]) -> MllmPrediction: ...

    @property
    def peak_memory_mib(self) -> float: ...

    def close(self) -> None: ...


class DinoRuntime(Protocol):
    model_load_seconds: float
    model_placement: list[str]
    resolved_snapshot: str | None
    feature_layer: str
    last_inference_seconds: float

    def load(self) -> None: ...

    def extract(
        self,
        *,
        crop_path: Path,
        mask_path: Path,
        bbox: Any,
        crop_padding_pixels: int,
    ) -> FingerprintData: ...

    @property
    def peak_memory_mib(self) -> float: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class ImageWork:
    input_path: Path
    source: SourceImage | None = None
    duplicate: bool = False
    status: str = "discovered"
    qwen: dict[str, Any] | None = None
    kept: tuple[Proposal, ...] = ()
    filtered: tuple[Proposal, ...] = ()
    filter_counts: dict[str, int] = field(default_factory=dict)
    raw_candidate_count: int = 0
    prompt_detection_counts: dict[str, int] = field(default_factory=dict)
    sam_inference_seconds: float = 0.0
    fingerprint_count: int = 0
    fingerprint_inference_seconds: float = 0.0
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
            "qwen": self.qwen,
            "scene_guidance": self.qwen.get("response") if self.qwen else None,
            "sam": {
                "above_confidence_threshold_candidates": self.raw_candidate_count,
                "kept": len(self.kept),
                "filtered": len(self.filtered),
                "filter_counts": self.filter_counts,
                "prompt_detection_counts": self.prompt_detection_counts,
                "zero_candidate_prompts": [
                    prompt
                    for prompt, count in self.prompt_detection_counts.items()
                    if count == 0
                ],
                "inference_seconds": round(self.sam_inference_seconds, 3),
            },
            "kept_proposals": [
                proposal.model_dump(mode="json") for proposal in self.kept
            ],
            "filtered_proposals": [
                proposal.model_dump(mode="json") for proposal in self.filtered
            ],
            "dinov3": {
                "fingerprints": self.fingerprint_count,
                "inference_seconds": round(self.fingerprint_inference_seconds, 3),
            },
            "decisions": self.decisions,
            "error": self.error,
        }


class RecordingPredictor:
    """Retain the only raw Qwen response even when schema validation fails."""

    def __init__(self, runtime: MllmRuntime) -> None:
        self.runtime = runtime
        self.predictions: list[MllmPrediction] = []
        self.attempted_calls = 0

    def predict(self, messages: Sequence[dict[str, Any]]) -> MllmPrediction:
        self.attempted_calls += 1
        prediction = self.runtime.predict(messages)
        self.predictions.append(prediction)
        return prediction


def discover_images(input_directory: str | Path) -> list[Path]:
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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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
    """Run a state-dependent per-image loop with all three models resident."""

    def __init__(
        self,
        *,
        config: AppConfig,
        paths: MemoryPaths,
        sam_runtime: SamRuntime,
        mllm_runtime: MllmRuntime,
        dino_runtime: DinoRuntime,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.config = config
        self.paths = paths
        self.store = MemoryStore(paths)
        self.loop = MemoryLoop(self.store)
        self.sam_runtime = sam_runtime
        self.mllm_runtime = mllm_runtime
        self.dino_runtime = dino_runtime
        self.progress = progress
        self._qwen_metrics = {
            "loaded": False,
            "calls": 0,
            "input_tokens": 0,
            "generated_tokens": 0,
            "inference_seconds": 0.0,
        }
        self._sam_metrics = {"loaded": False, "inference_seconds": 0.0}
        self._dino_metrics = {
            "loaded": False,
            "fingerprints": 0,
            "inference_seconds": 0.0,
        }
        self._peak_memory_mib = {
            "qwen": 0.0,
            "sam3": 0.0,
            "dinov3": 0.0,
            "joint": 0.0,
        }

    def run(
        self,
        image_paths: Sequence[str | Path],
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        normalized = [Path(path).expanduser().resolve() for path in image_paths]
        if not normalized:
            raise ValueError("At least one input image is required")
        missing = next((path for path in normalized if not path.is_file()), None)
        if missing is not None:
            raise FileNotFoundError(f"Input image not found: {missing}")
        resolved_run_id = run_id or self._new_run_id()
        if self.progress is not None:
            self.progress.set_run_id(resolved_run_id)
        self.store.initialize()
        run = Run(
            id=resolved_run_id,
            config_digest=config_digest(self.config),
            sam_model_id=str(self.config.models.sam3_checkpoint),
            qwen_model_id=self.config.models.qwen_model_id,
            dinov3_model_id=self.config.models.dinov3_model_id,
        )
        works = [ImageWork(input_path=path) for path in normalized]
        self.loop.begin_run(run)
        try:
            return self._run_active(run, works)
        except BaseException as exc:
            self._close_interrupted_run(run, works, exc)
            raise

    def _run_active(self, run: Run, works: list[ImageWork]) -> dict[str, Any]:
        self._emit(
            event="run_started",
            stage="run",
            status="running",
            current=0,
            total=len(works),
            message="Object-memory run started",
            data={"memory_root": str(self.paths.root), "input_count": len(works)},
            overall_percent=0.0,
        )
        errors: list[str] = []
        for index, work in enumerate(works, start=1):
            self._register_image(run, work, errors)
            self._emit(
                event="input_registered",
                stage="input_registration",
                status=("skipped" if work.duplicate else work.status),
                current=index,
                total=len(works),
                message=f"Input registration finished: {work.input_path.name}",
                data={
                    "input_path": str(work.input_path),
                    "filename": work.input_path.name,
                    "source_id": work.source.id if work.source else None,
                    "sha256": work.source.sha256 if work.source else None,
                    "duplicate": work.duplicate,
                    "work_status": work.status,
                    "error": work.error,
                },
            )

        pending = [work for work in works if work.status == "registered"]
        if pending:
            self._load_models(pending, errors)
        try:
            for index, work in enumerate(pending, start=1):
                self._emit(
                    event="image_loop_started",
                    stage="per_image_loop",
                    status="running",
                    current=index - 1,
                    total=len(pending),
                    message=f"Processing {work.input_path.name}",
                    data={"source_id": work.source.id if work.source else None},
                    overall_percent=self._image_progress(
                        index=index,
                        total=len(pending),
                        fraction=0.0,
                    ),
                )
                self._process_image(
                    run,
                    work,
                    errors,
                    image_index=index,
                    image_total=len(pending),
                )
                self._emit(
                    event="image_loop_completed",
                    stage="per_image_loop",
                    status=work.status,
                    current=index,
                    total=len(pending),
                    message=f"Finished {work.input_path.name}: {work.status}",
                    data=work.as_dict(),
                    overall_percent=self._image_progress(
                        index=index,
                        total=len(pending),
                        fraction=1.0,
                    ),
                )
        finally:
            self._capture_peak_memory()
            self._close_models()

        summary = self.loop.complete_run(run.id, external_errors=len(errors))
        checks = self._build_checks(works, summary)
        status = (
            "passed"
            if summary.status.value == "completed" and all(checks.values())
            else summary.status.value
            if summary.status.value != "completed"
            else "failed"
        )
        report = self._build_report(run, works, summary, checks, status, errors)
        report_path = self.paths.run_reports / f"{run.id}.json"
        report["run_report"] = self.paths.relative_asset(report_path)
        write_json_atomic(report_path, report)
        self._emit(
            event="run_completed",
            stage="run",
            status=status,
            current=1,
            total=1,
            message=f"Object-memory run completed with status={status}",
            data={"run_report": report["run_report"], "run": report["run"]},
            overall_percent=100.0,
        )
        return report

    def _load_models(self, works: Sequence[ImageWork], errors: list[str]) -> None:
        self._emit(
            event="models_load_started",
            stage="models",
            status="running",
            current=0,
            total=3,
            message="Loading Qwen, SAM3, and DINOv3 for joint residency",
            data={},
            overall_percent=10.0,
        )
        try:
            self.mllm_runtime.load()
            self._qwen_metrics["loaded"] = True
            self.sam_runtime.load()
            self._sam_metrics["loaded"] = True
            self.dino_runtime.load()
            self._dino_metrics["loaded"] = True
        except Exception as exc:  # noqa: BLE001 - joint residency is an experiment result
            message = f"{type(exc).__name__}: {exc}"
            errors.append(f"joint model residency: {message}")
            for work in works:
                if work.source is not None and work.status == "registered":
                    self._fail_source(work, message, errors)
            self._capture_peak_memory()
            self._close_models()
            return
        self._emit(
            event="models_load_completed",
            stage="models",
            status="completed",
            current=3,
            total=3,
            message="All three models are resident",
            data={
                "qwen_load_seconds": self.mllm_runtime.model_load_seconds,
                "sam3_load_seconds": self.sam_runtime.model_load_seconds,
                "dinov3_load_seconds": self.dino_runtime.model_load_seconds,
                "combined_peak_memory_mib": self._combined_peak_memory(),
            },
            overall_percent=10.0,
        )

    def _process_image(
        self,
        run: Run,
        work: ImageWork,
        errors: list[str],
        *,
        image_index: int,
        image_total: int,
    ) -> None:
        if work.status != "registered" or work.source is None:
            return
        if not all(
            (
                self._qwen_metrics["loaded"],
                self._sam_metrics["loaded"],
                self._dino_metrics["loaded"],
            )
        ):
            return
        source_asset = self.paths.resolve_asset(work.source.relative_path)
        self._emit(
            event="scene_guidance_image_started",
            stage="scene_guidance",
            status="running",
            current=0,
            total=1,
            message=f"Qwen is reading {work.input_path.name} and current text memory",
            data={"input_path": str(work.input_path), "source_id": work.source.id},
            overall_percent=self._image_progress(
                index=image_index,
                total=image_total,
                fraction=0.0,
            ),
        )
        evaluation, raw_path = self._run_qwen_once(run, work, source_asset, errors)
        if evaluation is None or raw_path is None:
            return
        self._emit(
            event="scene_guidance_image_completed",
            stage="scene_guidance",
            status="completed",
            current=1,
            total=1,
            message=f"Qwen single-pass response completed for {work.input_path.name}",
            data={
                "input_path": str(work.input_path),
                "source_id": work.source.id,
                "work_status": work.status,
                "qwen": work.qwen,
                "scene_guidance": evaluation.response.model_dump(mode="json"),
            },
            overall_percent=self._image_progress(
                index=image_index,
                total=image_total,
                fraction=0.25,
            ),
        )
        targets = list(evaluation.response.targets)
        if not targets:
            self._complete_source(work, errors)
            return

        try:
            self._emit(
                event="sam3_image_started",
                stage="sam3",
                status="running",
                current=0,
                total=1,
                message=f"SAM3 is segmenting {work.input_path.name}",
                data={"input_path": str(work.input_path), "source_id": work.source.id},
                overall_percent=self._image_progress(
                    index=image_index,
                    total=image_total,
                    fraction=0.25,
                ),
            )
            prompts = list(dict.fromkeys(target.sam_text_prompt for target in targets))
            with Image.open(source_asset) as opened:
                image = opened.convert("RGB")
            prediction = self.sam_runtime.predict(image, prompts)
            self._sam_metrics["inference_seconds"] += prediction.inference_seconds
            work.sam_inference_seconds = prediction.inference_seconds
            work.raw_candidate_count = len(prediction.candidates)
            work.prompt_detection_counts = prediction.prompt_counts
            processed = process_candidates(
                prediction.candidates,
                image=image,
                source_image_id=work.source.id,
                run_id=run.id,
                paths=self.paths,
                settings=self.config.sam3_pipeline,
            )
            work.kept = processed.kept
            work.filtered = processed.filtered
            work.filter_counts = processed.filter_counts
            for proposal in work.filtered:
                self.loop.record_filtered_proposal(proposal)
            self._emit(
                event="sam3_image_completed",
                stage="sam3",
                status="completed",
                current=1,
                total=1,
                message=f"SAM3 completed {work.input_path.name}",
                data={
                    "input_path": str(work.input_path),
                    "source_id": work.source.id,
                    "work_status": work.status,
                    "sam": work.as_dict()["sam"],
                    "kept_proposals": [
                        proposal.model_dump(mode="json") for proposal in work.kept
                    ],
                    "filtered_proposals": [
                        proposal.model_dump(mode="json") for proposal in work.filtered
                    ],
                },
                overall_percent=self._image_progress(
                    index=image_index,
                    total=image_total,
                    fraction=0.6,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self._fail_source(work, f"{type(exc).__name__}: {exc}", errors)
            return

        assignments = associate_targets(
            work.kept,
            targets,
            image_width=work.source.width,
            image_height=work.source.height,
            minimum_iou=self.config.mllm_pipeline.target_proposal_iou_threshold,
        )
        try:
            historical = [
                HistoricalFingerprint(
                    object_id=record.object_id,
                    observation_id=record.observation_id,
                    data=read_fingerprint(
                        self.paths.resolve_asset(record.path),
                        expected_sha256=record.sha256,
                    ),
                )
                for record in self.loop.fingerprint_records()
            ]
        except Exception as exc:  # noqa: BLE001 - corrupt history stops this source
            message = f"{type(exc).__name__}: {exc}"
            for proposal in work.kept:
                self._record_proposal_failure(proposal, message, work, errors)
            self._fail_source(work, message, errors, append_error=False)
            return
        self._emit(
            event="visual_identity_started",
            stage="candidate_reasoning",
            status="running",
            current=0,
            total=len(work.kept),
            message=f"DINOv3 is fingerprinting {len(work.kept)} proposals",
            data={"input_path": str(work.input_path), "source_id": work.source.id},
            overall_percent=self._image_progress(
                index=image_index,
                total=image_total,
                fraction=0.6,
            ),
        )
        for index, proposal in enumerate(work.kept):
            try:
                decision_report = self._process_proposal(
                    proposal,
                    assignments.get(proposal.id),
                    historical=historical,
                    raw_path=raw_path,
                )
                work.decisions.append(decision_report)
                work.fingerprint_count += 1
                work.fingerprint_inference_seconds += float(
                    getattr(self.dino_runtime, "last_inference_seconds", 0.0)
                )
            except Exception as exc:  # noqa: BLE001 - stop this source, no rescue
                message = f"{type(exc).__name__}: {exc}"
                errors.append(f"proposal {proposal.id}: {message}")
                self._record_proposal_failure(proposal, message, work, errors)
                for remaining in work.kept[index + 1 :]:
                    self._record_proposal_failure(
                        remaining,
                        f"source stopped after proposal failure: {proposal.id}",
                        work,
                        errors,
                    )
                self._fail_source(work, message, errors, append_error=False)
                return
        self._emit(
            event="visual_identity_completed",
            stage="candidate_reasoning",
            status="completed",
            current=len(work.kept),
            total=len(work.kept),
            message=f"Visual identity decisions completed for {work.input_path.name}",
            data={
                "input_path": str(work.input_path),
                "source_id": work.source.id,
                "work_status": work.status,
                "dinov3": work.as_dict()["dinov3"],
                "decisions": work.decisions,
            },
            overall_percent=self._image_progress(
                index=image_index,
                total=image_total,
                fraction=1.0,
            ),
        )
        self._complete_source(work, errors)

    def _run_qwen_once(
        self,
        run: Run,
        work: ImageWork,
        source_asset: Path,
        errors: list[str],
    ) -> tuple[SceneGuidanceEvaluation | None, str | None]:
        assert work.source is not None
        recorder = RecordingPredictor(self.mllm_runtime)
        evaluation: SceneGuidanceEvaluation | None = None
        call_error: str | None = None
        cards = self.loop.object_cards()
        try:
            evaluation = evaluate_scene_guidance(
                recorder,
                image=SceneImageInput(work.source.id, source_asset),
                cards=cards,
                settings=self.config.mllm_pipeline,
            )
        except Exception as exc:  # noqa: BLE001 - one call, failure is evidence
            call_error = f"{type(exc).__name__}: {exc}"
        self._qwen_metrics["calls"] += recorder.attempted_calls
        self._qwen_metrics["input_tokens"] += sum(
            prediction.input_tokens for prediction in recorder.predictions
        )
        self._qwen_metrics["generated_tokens"] += sum(
            prediction.generated_tokens for prediction in recorder.predictions
        )
        self._qwen_metrics["inference_seconds"] += sum(
            prediction.inference_seconds for prediction in recorder.predictions
        )
        raw_path: str | None = None
        try:
            path = self.paths.raw_response_dir(run.id, work.source.id) / "response.json"
            payload = {
                "stage": "single_qwen_object_memory",
                "prompt_version": self.config.mllm_pipeline.prompt_version,
                "source_id": work.source.id,
                "memory_context": {
                    "object_card_count": len(cards),
                    "object_card_ids": [card.object_id for card in cards],
                    "selection": "all_active_text_summaries",
                    "historical_images_supplied": False,
                },
                "error": call_error,
                "response": (
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
                    for prediction in recorder.predictions
                ],
            }
            write_json_atomic(path, payload)
            raw_path = self.paths.relative_asset(path)
        except Exception as exc:  # noqa: BLE001 - raw evidence is mandatory
            call_error = f"raw response persistence failed: {type(exc).__name__}: {exc}"
            evaluation = None
        work.qwen = {
            "calls": recorder.attempted_calls,
            "object_card_count": len(cards),
            "object_card_ids": [card.object_id for card in cards],
            "raw_response": raw_path,
            "response": (
                evaluation.response.model_dump(mode="json")
                if evaluation is not None
                else None
            ),
            "error": call_error,
        }
        if evaluation is None:
            message = call_error or "Qwen produced no usable single-pass response"
            self._fail_source(work, message, errors)
        return evaluation, raw_path

    def _process_proposal(
        self,
        proposal: Proposal,
        target: SceneTarget | None,
        *,
        historical: Sequence[HistoricalFingerprint],
        raw_path: str,
    ) -> dict[str, Any]:
        if not proposal.crop_path or not proposal.mask_path:
            raise ValueError("Kept proposal has no crop or mask asset")
        fingerprint_data = self.dino_runtime.extract(
            crop_path=self.paths.resolve_asset(proposal.crop_path),
            mask_path=self.paths.resolve_asset(proposal.mask_path),
            bbox=proposal.bbox,
            crop_padding_pixels=self.config.sam3_pipeline.crop_padding_pixels,
        )
        fingerprint_path = self.paths.resolve_asset(proposal.crop_path).parent / "fingerprint.npz"
        fingerprint = write_fingerprint(
            fingerprint_path,
            fingerprint_data,
            relative_path=self.paths.relative_asset(fingerprint_path),
            model_id=self.config.models.dinov3_model_id,
            revision=self.config.models.dinov3_revision,
            feature_layer=self.dino_runtime.feature_layer,
            input_size=self.config.visual_fingerprint.input_size,
            storage_dtype=self.config.visual_fingerprint.storage_dtype,
        )
        proposal.fingerprint = fingerprint
        visual = match_fingerprint(
            fingerprint_data,
            historical,
            self.config.visual_fingerprint,
        )
        result = (
            decide_identity(target, visual)
            if target is not None
            else unmatched_proposal_decision(visual)
        )
        write_result = self.loop.apply_decision(
            proposal=proposal,
            result=result,
            fingerprint=fingerprint,
            prompt_version=self.config.mllm_pipeline.prompt_version,
            raw_response_path=raw_path,
        )
        self._dino_metrics["fingerprints"] += 1
        self._dino_metrics["inference_seconds"] += float(
            getattr(self.dino_runtime, "last_inference_seconds", 0.0)
        )
        return {
            "proposal_id": proposal.id,
            "target_id": target.target_id if target else None,
            "target_object_name_zh": target.object_name_zh if target else None,
            "sam_text_prompt": proposal.prompt,
            "status": write_result.proposal_status.value,
            "decision": write_result.decision.value,
            "object_id": write_result.object_id,
            "qwen_identity_hypothesis": result.qwen_hypothesis.value,
            "qwen_matched_object_id": result.qwen_matched_object_id,
            "visual_evidence": result.visual_evidence.model_dump(mode="json"),
            "confidence": result.confidence,
            "reason_code": result.reason_code.value,
            "short_reason": result.short_reason,
            "fingerprint": fingerprint.model_dump(mode="json"),
            "candidate": self._candidate_report(proposal),
            "raw_response": raw_path,
            "errors": [],
        }

    def _register_image(
        self,
        run: Run,
        work: ImageWork,
        errors: list[str],
    ) -> None:
        try:
            digest = sha256_file(work.input_path)
            with Image.open(work.input_path) as opened:
                width, height = opened.size
            source_asset = self.paths.sources / f"{digest}{work.input_path.suffix.lower()}"
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
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            work.status = "failed"
            work.error = message
            errors.append(f"register {work.input_path}: {message}")

    @staticmethod
    def _copy_source_asset(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            return
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            shutil.copy2(source, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _record_proposal_failure(
        self,
        proposal: Proposal,
        message: str,
        work: ImageWork,
        errors: list[str],
    ) -> None:
        try:
            self.loop.record_proposal_failure(proposal, message)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"record proposal failure {proposal.id}: {exc}")
        work.decisions.append(
            {
                "proposal_id": proposal.id,
                "target_id": proposal.target_id,
                "status": "failed",
                "decision": None,
                "candidate": self._candidate_report(proposal),
                "errors": [message],
            }
        )

    def _complete_source(self, work: ImageWork, errors: list[str]) -> None:
        assert work.source is not None
        try:
            self.loop.complete_source(work.source.id)
            work.status = "completed"
        except Exception as exc:  # noqa: BLE001
            self._fail_source(work, f"{type(exc).__name__}: {exc}", errors)

    def _fail_source(
        self,
        work: ImageWork,
        message: str,
        errors: list[str],
        *,
        append_error: bool = True,
    ) -> None:
        if append_error:
            errors.append(
                f"source {work.source.id if work.source else work.input_path}: {message}"
            )
        work.error = message
        if work.source is not None and work.status not in {"failed", "duplicate"}:
            try:
                self.loop.fail_source(work.source.id, message)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"fail source {work.source.id}: {exc}")
        work.status = "failed"

    def _close_models(self) -> None:
        for runtime in (self.dino_runtime, self.sam_runtime, self.mllm_runtime):
            try:
                runtime.close()
            except Exception:
                pass

    def _capture_peak_memory(self) -> None:
        qwen_peak = self.mllm_runtime.peak_memory_mib
        sam_peak = self.sam_runtime.peak_memory_mib
        dino_peak = self.dino_runtime.peak_memory_mib
        self._peak_memory_mib = {
            "qwen": max(self._peak_memory_mib["qwen"], qwen_peak),
            "sam3": max(self._peak_memory_mib["sam3"], sam_peak),
            "dinov3": max(self._peak_memory_mib["dinov3"], dino_peak),
            "joint": max(
                self._peak_memory_mib["joint"], qwen_peak, sam_peak, dino_peak
            ),
        }

    def _combined_peak_memory(self) -> float:
        return max(
            self._peak_memory_mib["joint"],
            self.mllm_runtime.peak_memory_mib,
            self.sam_runtime.peak_memory_mib,
            self.dino_runtime.peak_memory_mib,
        )

    @staticmethod
    def _image_progress(*, index: int, total: int, fraction: float) -> float:
        if total <= 0 or index < 1 or index > total:
            raise ValueError("Image progress requires one valid 1-based index")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("Image progress fraction must be between 0 and 1")
        span = 85.0 / total
        return 10.0 + span * ((index - 1) + fraction)

    def _build_report(
        self,
        run: Run,
        works: Sequence[ImageWork],
        summary: RunSummary,
        checks: dict[str, bool],
        status: str,
        errors: list[str],
    ) -> dict[str, Any]:
        visual_scores = [
            decision["visual_evidence"].get("visual_score")
            for work in works
            for decision in work.decisions
            if isinstance(decision.get("visual_evidence"), dict)
            and decision["visual_evidence"].get("visual_score") is not None
        ]
        visual_result_counts = {result: 0 for result in ("match", "no_match", "ambiguous")}
        for work in works:
            for decision in work.decisions:
                evidence = decision.get("visual_evidence")
                if not isinstance(evidence, dict):
                    continue
                result = evidence.get("result")
                if result in visual_result_counts:
                    visual_result_counts[result] += 1
        return {
            "schema_version": 7,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "test": "object_memory_demo_single_pass_dinov3",
            "status": status,
            "run": summary.as_dict(),
            "checks": checks,
            "strategy": {
                "workflow": "per_image_qwen_sam3_dinov3_commit",
                "qwen_call_policy": "exactly one call per unique new source image",
                "qwen_memory_context": "all active object text summaries only",
                "second_qwen_stage": False,
                "sam_prompt_deduplication": True,
                "target_proposal_association": "same prompt plus normalized bbox IoU",
                "automatic_identity_assets": "DINOv3 fingerprints only",
                "uncertain_policy": "terminal; no retry or second Qwen call",
                "object_text_policy": "one iterated structured summary per object",
                "prompt_version": self.config.mllm_pipeline.prompt_version,
            },
            "models": {
                "qwen": {
                    **self._qwen_metrics,
                    "model_id": self.config.models.qwen_model_id,
                    "model_load_seconds": round(self.mllm_runtime.model_load_seconds, 3),
                    "peak_memory_mib": round(self._peak_memory_mib["qwen"], 2),
                    "placement": self.mllm_runtime.model_placement,
                    "snapshot": self.mllm_runtime.resolved_snapshot,
                },
                "sam3": {
                    **self._sam_metrics,
                    "checkpoint": str(self.config.models.sam3_checkpoint),
                    "confidence_threshold": self.config.sam3_pipeline.confidence_threshold,
                    "model_load_seconds": round(self.sam_runtime.model_load_seconds, 3),
                    "peak_memory_mib": round(self._peak_memory_mib["sam3"], 2),
                },
                "dinov3": {
                    **self._dino_metrics,
                    "model_id": self.config.models.dinov3_model_id,
                    "revision": self.config.models.dinov3_revision,
                    "model_path": str(self.config.models.dinov3_model_path),
                    "feature_layer": self.dino_runtime.feature_layer,
                    "input_size": self.config.visual_fingerprint.input_size,
                    "model_load_seconds": round(self.dino_runtime.model_load_seconds, 3),
                    "peak_memory_mib": round(self._peak_memory_mib["dinov3"], 2),
                    "placement": self.dino_runtime.model_placement,
                    "visual_score_count": len(visual_scores),
                    "visual_score_min": min(visual_scores) if visual_scores else None,
                    "visual_score_max": max(visual_scores) if visual_scores else None,
                    "visual_score_mean": (
                        sum(visual_scores) / len(visual_scores)
                        if visual_scores
                        else None
                    ),
                    "result_counts": visual_result_counts,
                },
                "joint_peak_memory_mib": round(self._combined_peak_memory(), 2),
            },
            "visual_fingerprint_config": self.config.visual_fingerprint.model_dump(
                mode="json"
            ),
            "images": [work.as_dict() for work in works],
            "external_errors": errors,
            "core_counts": self.store.status().counts,
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

    def _build_checks(
        self,
        works: Sequence[ImageWork], summary: RunSummary
    ) -> dict[str, bool]:
        processed = [work for work in works if not work.duplicate and work.source]
        successful = [work for work in processed if work.status == "completed"]
        decided = [
            decision
            for work in works
            for decision in work.decisions
            if decision.get("status") == "decided"
        ]

        def fingerprint_asset_is_valid(decision: dict[str, Any]) -> bool:
            fingerprint = decision.get("fingerprint")
            if not isinstance(fingerprint, dict):
                return False
            path = fingerprint.get("path")
            expected_hash = fingerprint.get("sha256")
            if not isinstance(path, str) or not isinstance(expected_hash, str):
                return False
            asset = self.paths.resolve_asset(path)
            return asset.is_file() and sha256_file(asset) == expected_hash

        return {
            "input_images_nonempty": bool(works),
            "every_input_accounted_for": all(
                work.duplicate or work.status in {"completed", "failed"}
                for work in works
            ),
            "single_qwen_call_per_completed_source": all(
                work.qwen is not None and work.qwen.get("calls") == 1
                for work in successful
            ),
            "second_qwen_stage_absent": True,
            "every_decided_proposal_has_fingerprint": all(
                bool(decision.get("fingerprint")) for decision in decided
            ),
            "fingerprint_assets_exist_and_match_hashes": all(
                fingerprint_asset_is_valid(decision) for decision in decided
            ),
            "no_source_left_processing": summary.source_counts["processing"] == 0,
            "no_pending_proposals": summary.proposal_counts["pending"] == 0,
        }

    def _close_interrupted_run(
        self,
        run: Run,
        works: Sequence[ImageWork],
        original_error: BaseException,
    ) -> None:
        self._capture_peak_memory()
        self._close_models()
        message = f"{type(original_error).__name__}: {original_error}"
        cleanup_errors: list[str] = []
        for work in works:
            if work.source is None or work.duplicate or work.status in {"completed", "failed"}:
                continue
            try:
                self.loop.fail_source(work.source.id, message)
                work.status = "failed"
            except BaseException as exc:  # noqa: BLE001
                cleanup_errors.append(f"fail source {work.source.id}: {exc}")
        try:
            self.store.complete_run(run.id, error_message=message)
        except BaseException as exc:  # noqa: BLE001
            cleanup_errors.append(f"complete run {run.id}: {exc}")
        if cleanup_errors:
            try:
                original_error.add_note("Cleanup also failed: " + "; ".join(cleanup_errors))
            except BaseException:
                pass

    def _emit(
        self,
        *,
        event: str,
        stage: str,
        status: str,
        current: int,
        total: int,
        message: str,
        data: dict[str, Any],
        overall_percent: float | None = None,
    ) -> None:
        if self.progress is not None:
            self.progress.emit(
                event=event,
                stage=stage,
                status=status,
                current=current,
                total=total,
                message=message,
                data=data,
                overall_percent=overall_percent,
            )

    @staticmethod
    def _new_run_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"run_{timestamp}"
