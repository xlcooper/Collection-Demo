"""SAM3 point grid -> DINOv3 clustering -> Qwen cluster review pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence
from uuid import uuid4

from PIL import Image

from .assets import MemoryPaths
from .candidate_clustering import (
    CandidateCluster,
    FingerprintedCandidate,
    cluster_candidates,
    cluster_historical_evidence,
    write_cluster_contact_sheet,
)
from .cluster_review import (
    ClusterReviewEvaluation,
    ClusterReviewInput,
    evaluate_cluster_reviews,
)
from .config import AppConfig, config_digest
from .dinov3_adapter import (
    FingerprintData,
    HistoricalFingerprint,
    read_fingerprint,
    write_fingerprint,
)
from .memory_loop import MemoryLoop
from .memory_store import MemoryStore, RunSummary
from .mllm_adapter import MllmPrediction
from .progress import ProgressReporter
from .sam3_adapter import Sam3Prediction
from .sam3_postprocess import process_candidates
from .schemas import (
    ClusterReview,
    ClusterVerdict,
    DecisionReasonCode,
    DecisionType,
    IdentityHypothesis,
    Proposal,
    ProposalStatus,
    Run,
    SourceImage,
    VisualEvidence,
    VisualMatchType,
)


SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


class SamRuntime(Protocol):
    model_load_seconds: float

    def load(self) -> None: ...

    def predict(self, image: Image.Image) -> Sam3Prediction: ...

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
    kept: tuple[Proposal, ...] = ()
    filtered: tuple[Proposal, ...] = ()
    filter_counts: dict[str, int] = field(default_factory=dict)
    raw_candidate_count: int = 0
    sam_inference_seconds: float = 0.0
    fingerprint_count: int = 0
    fingerprint_inference_seconds: float = 0.0
    cluster_ids: list[str] = field(default_factory=list)
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
                "candidate_source": "automatic_point_grid",
                "grid_points": self.raw_candidate_count,
                "raw_candidates": self.raw_candidate_count,
                "kept": len(self.kept),
                "filtered": len(self.filtered),
                "filter_counts": self.filter_counts,
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
                "cluster_ids": self.cluster_ids,
            },
            "decisions": self.decisions,
            "error": self.error,
        }


@dataclass(slots=True)
class ClusterWork:
    cluster: CandidateCluster
    contact_sheet: str
    historical_evidence: VisualEvidence | None = None
    qwen_review: ClusterReview | None = None
    raw_response: str | None = None
    final_decision: str | None = None
    object_id: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.cluster.report(),
            "contact_sheet": self.contact_sheet,
            "historical_visual_evidence": (
                self.historical_evidence.model_dump(mode="json")
                if self.historical_evidence is not None
                else None
            ),
            "qwen_review": (
                self.qwen_review.model_dump(mode="json")
                if self.qwen_review is not None
                else None
            ),
            "raw_response": self.raw_response,
            "final_decision": self.final_decision,
            "object_id": self.object_id,
            "error": self.error,
        }


class RecordingPredictor:
    """Retain the only raw response for one Qwen cluster batch."""

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
    """Run the staged batch workflow while preserving explicit evidence."""

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
        self._sam_metrics = {
            "loaded": False,
            "inference_seconds": 0.0,
            "raw_candidates": 0,
        }
        self._dino_metrics = {
            "loaded": False,
            "fingerprints": 0,
            "inference_seconds": 0.0,
            "clusters": 0,
        }
        self._peak_memory_mib = {"qwen": 0.0, "sam3": 0.0, "dinov3": 0.0}

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
            message="Object-memory batch run started",
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
                overall_percent=10.0 * index / len(works),
            )

        pending = [work for work in works if work.status == "registered"]
        cluster_works: list[ClusterWork] = []
        if pending:
            self._run_sam_stage(run, pending, errors)
            fingerprinted = self._run_dino_stage(run, pending, errors)
            if fingerprinted:
                cluster_works = self._build_clusters(run, fingerprinted)
                self._run_qwen_stage(run, cluster_works, works, errors)
        self._finalize_sources(works, errors)

        summary = self.loop.complete_run(run.id, external_errors=len(errors))
        checks = self._build_checks(works, cluster_works, summary)
        status = (
            "passed"
            if summary.status.value == "completed" and all(checks.values())
            else summary.status.value
            if summary.status.value != "completed"
            else "failed"
        )
        report = self._build_report(
            run,
            works,
            cluster_works,
            summary,
            checks,
            status,
            errors,
        )
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

    def _run_sam_stage(
        self,
        run: Run,
        works: Sequence[ImageWork],
        errors: list[str],
    ) -> None:
        active = [work for work in works if work.status == "registered"]
        if not active:
            return
        self._emit(
            event="sam3_batch_started",
            stage="sam3",
            status="running",
            current=0,
            total=len(active),
            message="SAM3 point-grid candidate generation started",
            data={},
            overall_percent=10.0,
        )
        try:
            self.sam_runtime.load()
            self._sam_metrics["loaded"] = True
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            errors.append(f"SAM3 load: {message}")
            self.sam_runtime.close()
            for work in active:
                self._fail_source(work, message, errors, append_error=False)
            return
        try:
            for index, work in enumerate(active, start=1):
                if work.source is None or work.status != "registered":
                    continue
                try:
                    source_asset = self.paths.resolve_asset(work.source.relative_path)
                    with Image.open(source_asset) as opened:
                        image = opened.convert("RGB")
                    prediction = self.sam_runtime.predict(image)
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
                    work.raw_candidate_count = len(prediction.candidates)
                    work.sam_inference_seconds = prediction.inference_seconds
                    self._sam_metrics["inference_seconds"] += prediction.inference_seconds
                    self._sam_metrics["raw_candidates"] += len(prediction.candidates)
                    for proposal in work.filtered:
                        self.loop.record_filtered_proposal(proposal)
                    self._emit(
                        event="sam3_image_completed",
                        stage="sam3",
                        status="completed",
                        current=index,
                        total=len(active),
                        message=f"SAM3 point grid completed: {work.input_path.name}",
                        data={
                            "input_path": str(work.input_path),
                            "source_id": work.source.id,
                            "work_status": work.status,
                            "sam": work.as_dict()["sam"],
                            "kept_proposals": [
                                proposal.model_dump(mode="json")
                                for proposal in work.kept
                            ],
                            "filtered_proposals": [
                                proposal.model_dump(mode="json")
                                for proposal in work.filtered
                            ],
                        },
                        overall_percent=10.0 + 30.0 * index / len(active),
                    )
                except Exception as exc:  # noqa: BLE001
                    self._fail_source(
                        work,
                        f"{type(exc).__name__}: {exc}",
                        errors,
                    )
        finally:
            self._peak_memory_mib["sam3"] = max(
                self._peak_memory_mib["sam3"], self.sam_runtime.peak_memory_mib
            )
            self.sam_runtime.close()

    def _run_dino_stage(
        self,
        run: Run,
        works: Sequence[ImageWork],
        errors: list[str],
    ) -> list[FingerprintedCandidate]:
        active = [
            work
            for work in works
            if work.status == "registered" and work.kept
        ]
        if not active:
            return []
        total_candidates = sum(len(work.kept) for work in active)
        self._emit(
            event="dinov3_batch_started",
            stage="dinov3",
            status="running",
            current=0,
            total=total_candidates,
            message="DINOv3 fingerprint extraction started",
            data={},
            overall_percent=40.0,
        )
        try:
            self.dino_runtime.load()
            self._dino_metrics["loaded"] = True
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            errors.append(f"DINOv3 load: {message}")
            self.dino_runtime.close()
            for work in active:
                for proposal in work.kept:
                    self._record_proposal_failure(proposal, message, work, errors)
                self._fail_source(work, message, errors, append_error=False)
            return []

        completed = 0
        fingerprinted: list[FingerprintedCandidate] = []
        try:
            for work in active:
                source_candidates: list[FingerprintedCandidate] = []
                source_error: str | None = None
                for proposal in work.kept:
                    try:
                        if not proposal.crop_path or not proposal.mask_path:
                            raise ValueError("Kept proposal has no crop or mask asset")
                        data = self.dino_runtime.extract(
                            crop_path=self.paths.resolve_asset(proposal.crop_path),
                            mask_path=self.paths.resolve_asset(proposal.mask_path),
                            bbox=proposal.bbox,
                            crop_padding_pixels=(
                                self.config.sam3_pipeline.crop_padding_pixels
                            ),
                        )
                        fingerprint_path = (
                            self.paths.resolve_asset(proposal.crop_path).parent
                            / "fingerprint.npz"
                        )
                        proposal.fingerprint = write_fingerprint(
                            fingerprint_path,
                            data,
                            relative_path=self.paths.relative_asset(fingerprint_path),
                            model_id=self.config.models.dinov3_model_id,
                            revision=self.config.models.dinov3_revision,
                            feature_layer=self.dino_runtime.feature_layer,
                            input_size=self.config.visual_fingerprint.input_size,
                            storage_dtype=(
                                self.config.visual_fingerprint.storage_dtype
                            ),
                        )
                        inference_seconds = float(
                            getattr(
                                self.dino_runtime,
                                "last_inference_seconds",
                                0.0,
                            )
                        )
                        work.fingerprint_count += 1
                        work.fingerprint_inference_seconds += inference_seconds
                        self._dino_metrics["fingerprints"] += 1
                        self._dino_metrics["inference_seconds"] += inference_seconds
                        source_candidates.append(
                            FingerprintedCandidate(proposal=proposal, data=data)
                        )
                        completed += 1
                        self._emit(
                            event="dinov3_candidate_completed",
                            stage="dinov3",
                            status="completed",
                            current=completed,
                            total=total_candidates,
                            message=f"DINOv3 fingerprinted {proposal.id}",
                            data={
                                "source_id": proposal.source_image_id,
                                "proposal_id": proposal.id,
                                "fingerprint": proposal.fingerprint.model_dump(
                                    mode="json"
                                ),
                            },
                            overall_percent=(
                                40.0 + 30.0 * completed / total_candidates
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001
                        source_error = f"{type(exc).__name__}: {exc}"
                        break
                if source_error is not None:
                    for proposal in work.kept:
                        if proposal.status is ProposalStatus.PENDING:
                            self._record_proposal_failure(
                                proposal,
                                source_error,
                                work,
                                errors,
                            )
                    self._fail_source(
                        work,
                        source_error,
                        errors,
                        append_error=False,
                    )
                    continue
                fingerprinted.extend(source_candidates)
        finally:
            self._peak_memory_mib["dinov3"] = max(
                self._peak_memory_mib["dinov3"],
                self.dino_runtime.peak_memory_mib,
            )
            self.dino_runtime.close()
        return fingerprinted

    def _build_clusters(
        self,
        run: Run,
        candidates: Sequence[FingerprintedCandidate],
    ) -> list[ClusterWork]:
        clusters = cluster_candidates(candidates, self.config.visual_fingerprint)
        self._dino_metrics["clusters"] = len(clusters)
        cluster_works: list[ClusterWork] = []
        for cluster in clusters:
            for member in cluster.members:
                member.proposal.target_id = cluster.id
            contact_sheet = write_cluster_contact_sheet(
                cluster,
                run_id=run.id,
                paths=self.paths,
                cell_size=self.config.visual_fingerprint.contact_sheet_cell_size,
            )
            cluster_works.append(
                ClusterWork(cluster=cluster, contact_sheet=contact_sheet)
            )
        self._emit(
            event="dinov3_clustering_completed",
            stage="clustering",
            status="completed",
            current=len(clusters),
            total=len(clusters),
            message=f"DINOv3 formed {len(clusters)} candidate clusters",
            data={
                "clusters": [cluster_work.as_dict() for cluster_work in cluster_works]
            },
            overall_percent=75.0,
        )
        return cluster_works

    def _run_qwen_stage(
        self,
        run: Run,
        cluster_works: list[ClusterWork],
        image_works: Sequence[ImageWork],
        errors: list[str],
    ) -> None:
        if not cluster_works:
            return
        batch_size = self.config.mllm_pipeline.max_clusters_per_batch
        total_batches = math.ceil(len(cluster_works) / batch_size)
        self._emit(
            event="cluster_review_started",
            stage="cluster_review",
            status="running",
            current=0,
            total=total_batches,
            message="Qwen cluster review started",
            data={"cluster_count": len(cluster_works)},
            overall_percent=75.0,
        )
        try:
            self.mllm_runtime.load()
            self._qwen_metrics["loaded"] = True
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            errors.append(f"Qwen load: {message}")
            self.mllm_runtime.close()
            self._fail_cluster_works(cluster_works, image_works, message, errors)
            return

        try:
            for batch_index in range(total_batches):
                start = batch_index * batch_size
                batch = cluster_works[start : start + batch_size]
                historical = self._historical_fingerprints()
                for cluster_work in batch:
                    cluster_work.historical_evidence = cluster_historical_evidence(
                        cluster_work.cluster,
                        historical,
                        self.config.visual_fingerprint,
                    )
                evaluation, raw_path, error = self._run_qwen_batch(
                    run,
                    batch,
                    batch_index=batch_index + 1,
                )
                if raw_path is not None:
                    for cluster_work in batch:
                        cluster_work.raw_response = raw_path
                if evaluation is None or raw_path is None:
                    message = error or "Qwen cluster review produced no usable response"
                    remaining = cluster_works[start:]
                    self._fail_cluster_works(
                        remaining,
                        image_works,
                        message,
                        errors,
                    )
                    break
                reviews = {review.cluster_id: review for review in evaluation.reviews}
                for cluster_work in batch:
                    cluster_work.qwen_review = reviews[cluster_work.cluster.id]
                    try:
                        self._apply_cluster_review(
                            cluster_work,
                            image_works=image_works,
                            raw_path=raw_path,
                        )
                    except Exception as exc:  # noqa: BLE001
                        message = f"{type(exc).__name__}: {exc}"
                        cluster_work.error = message
                        cluster_work.final_decision = "failed"
                        self._fail_cluster_works(
                            [cluster_work],
                            image_works,
                            message,
                            errors,
                        )
                self._emit(
                    event="cluster_review_batch_completed",
                    stage="cluster_review",
                    status="completed",
                    current=batch_index + 1,
                    total=total_batches,
                    message=(
                        f"Qwen reviewed cluster batch {batch_index + 1}/"
                        f"{total_batches}"
                    ),
                    data={"clusters": [item.as_dict() for item in batch]},
                    overall_percent=(
                        75.0 + 20.0 * (batch_index + 1) / total_batches
                    ),
                )
        finally:
            self._peak_memory_mib["qwen"] = max(
                self._peak_memory_mib["qwen"],
                self.mllm_runtime.peak_memory_mib,
            )
            self.mllm_runtime.close()

    def _run_qwen_batch(
        self,
        run: Run,
        batch: Sequence[ClusterWork],
        *,
        batch_index: int,
    ) -> tuple[ClusterReviewEvaluation | None, str | None, str | None]:
        cards = self.loop.object_cards()
        recorder = RecordingPredictor(self.mllm_runtime)
        evaluation: ClusterReviewEvaluation | None = None
        call_error: str | None = None
        try:
            inputs = [
                ClusterReviewInput(
                    cluster=item.cluster,
                    contact_sheet_path=self.paths.resolve_asset(item.contact_sheet),
                    historical_evidence=(
                        item.historical_evidence
                        if item.historical_evidence is not None
                        else VisualEvidence(result=VisualMatchType.NO_MATCH)
                    ),
                )
                for item in batch
            ]
            evaluation = evaluate_cluster_reviews(
                recorder,
                inputs=inputs,
                cards=cards,
                settings=self.config.mllm_pipeline,
            )
        except Exception as exc:  # noqa: BLE001
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
            scope_id = f"cluster_batch_{batch_index:04d}"
            path = self.paths.raw_response_dir(run.id, scope_id) / "response.json"
            payload = {
                "stage": "cluster_semantic_review",
                "prompt_version": self.config.mllm_pipeline.prompt_version,
                "batch_id": scope_id,
                "cluster_ids": [item.cluster.id for item in batch],
                "memory_context": {
                    "object_card_count": len(cards),
                    "object_card_ids": [card.object_id for card in cards],
                    "selection": "all_active_text_summaries",
                },
                "cluster_inputs": [item.as_dict() for item in batch],
                "error": call_error,
                "response": (
                    {
                        "reviews": [
                            review.model_dump(mode="json")
                            for review in evaluation.reviews
                        ]
                    }
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
        except Exception as exc:  # noqa: BLE001
            call_error = f"raw response persistence failed: {type(exc).__name__}: {exc}"
            evaluation = None
        return evaluation, raw_path, call_error

    def _apply_cluster_review(
        self,
        cluster_work: ClusterWork,
        *,
        image_works: Sequence[ImageWork],
        raw_path: str,
    ) -> None:
        review = cluster_work.qwen_review
        evidence = cluster_work.historical_evidence
        if review is None or evidence is None:
            raise ValueError("Cluster review and visual evidence are required")
        proposals = [member.proposal for member in cluster_work.cluster.members]
        if review.verdict is ClusterVerdict.IGNORE:
            self.loop.record_filtered_cluster(
                proposals,
                cluster_id=cluster_work.cluster.id,
                reason=review.short_reason,
            )
            cluster_work.final_decision = "ignored"
            self._record_cluster_results(
                cluster_work,
                image_works,
                proposal_results=None,
            )
            return

        decision_type, reason_code, short_reason = self._resolve_cluster_identity(
            review,
            evidence,
        )
        write_result = self.loop.apply_cluster_decision(
            proposals=proposals,
            review=review,
            decision_type=decision_type,
            visual_evidence=evidence,
            prompt_version=self.config.mllm_pipeline.prompt_version,
            raw_response_path=raw_path,
            reason_code=reason_code,
            short_reason=short_reason,
        )
        cluster_work.final_decision = decision_type.value
        cluster_work.object_id = write_result.object_id
        self._record_cluster_results(
            cluster_work,
            image_works,
            proposal_results=write_result.proposal_results,
        )

    @staticmethod
    def _resolve_cluster_identity(
        review: ClusterReview,
        evidence: VisualEvidence,
    ) -> tuple[DecisionType, DecisionReasonCode, str]:
        if review.verdict is ClusterVerdict.UNCERTAIN:
            return (
                DecisionType.UNCERTAIN,
                DecisionReasonCode.INSUFFICIENT_EVIDENCE,
                "Qwen无法确认该视觉聚类是否为一个完整且一致的物体。",
            )
        if evidence.result is VisualMatchType.AMBIGUOUS:
            return (
                DecisionType.UNCERTAIN,
                DecisionReasonCode.AMBIGUOUS_MATCH,
                "聚类对历史对象的第一、第二视觉匹配不足以安全区分。",
            )
        if review.identity_hypothesis is IdentityHypothesis.NEW:
            if evidence.result is VisualMatchType.NO_MATCH:
                return (
                    DecisionType.NEW,
                    DecisionReasonCode.NEW_OBJECT,
                    "Qwen确认其为完整对象，且DINOv3未匹配已有对象。",
                )
            return (
                DecisionType.UNCERTAIN,
                DecisionReasonCode.INSUFFICIENT_EVIDENCE,
                "Qwen判断为新对象，但DINOv3聚类证据匹配已有对象。",
            )
        if (
            review.identity_hypothesis is IdentityHypothesis.EXISTING
            and evidence.result is VisualMatchType.MATCH
            and evidence.matched_object_id == review.matched_object_id
        ):
            return (
                DecisionType.EXISTING,
                DecisionReasonCode.VISUAL_INSTANCE_MATCH,
                "Qwen对象判断与DINOv3历史视角匹配一致。",
            )
        return (
            DecisionType.UNCERTAIN,
            DecisionReasonCode.INSUFFICIENT_EVIDENCE,
            "Qwen已有对象判断未得到同一DINOv3对象匹配支持。",
        )

    def _record_cluster_results(
        self,
        cluster_work: ClusterWork,
        image_works: Sequence[ImageWork],
        *,
        proposal_results: Sequence[Any] | None,
    ) -> None:
        work_by_source = {
            work.source.id: work
            for work in image_works
            if work.source is not None
        }
        result_by_proposal = {
            result.proposal_id: result for result in (proposal_results or [])
        }
        review = cluster_work.qwen_review
        evidence = cluster_work.historical_evidence
        for member in cluster_work.cluster.members:
            proposal = member.proposal
            work = work_by_source[proposal.source_image_id]
            if cluster_work.cluster.id not in work.cluster_ids:
                work.cluster_ids.append(cluster_work.cluster.id)
            result = result_by_proposal.get(proposal.id)
            work.decisions.append(
                {
                    "proposal_id": proposal.id,
                    "cluster_id": cluster_work.cluster.id,
                    "status": proposal.status.value,
                    "decision": (
                        result.decision.value
                        if result is not None
                        else cluster_work.final_decision
                    ),
                    "object_id": (
                        result.object_id if result is not None else None
                    ),
                    "qwen_cluster_verdict": (
                        review.verdict.value if review is not None else None
                    ),
                    "qwen_identity_hypothesis": (
                        review.identity_hypothesis.value
                        if review is not None
                        else None
                    ),
                    "qwen_matched_object_id": (
                        review.matched_object_id if review is not None else None
                    ),
                    "visual_evidence": (
                        evidence.model_dump(mode="json")
                        if evidence is not None
                        else None
                    ),
                    "fingerprint": (
                        proposal.fingerprint.model_dump(mode="json")
                        if proposal.fingerprint is not None
                        else None
                    ),
                    "candidate": self._candidate_report(proposal),
                    "raw_response": cluster_work.raw_response,
                    "errors": [],
                }
            )

    def _historical_fingerprints(self) -> list[HistoricalFingerprint]:
        return [
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

    def _fail_cluster_works(
        self,
        cluster_works: Sequence[ClusterWork],
        image_works: Sequence[ImageWork],
        message: str,
        errors: list[str],
    ) -> None:
        work_by_source = {
            work.source.id: work
            for work in image_works
            if work.source is not None
        }
        for cluster_work in cluster_works:
            cluster_work.error = cluster_work.error or message
            cluster_work.final_decision = "failed"
            for member in cluster_work.cluster.members:
                proposal = member.proposal
                if proposal.status is not ProposalStatus.PENDING:
                    continue
                work = work_by_source[proposal.source_image_id]
                self._record_proposal_failure(proposal, message, work, errors)
        errors.append(f"cluster review: {message}")

    def _finalize_sources(
        self,
        works: Sequence[ImageWork],
        errors: list[str],
    ) -> None:
        for work in works:
            if work.status != "registered" or work.source is None:
                continue
            pending = [
                proposal
                for proposal in work.kept
                if proposal.status is ProposalStatus.PENDING
            ]
            if pending:
                message = "Source contains proposals without a terminal cluster result"
                for proposal in pending:
                    self._record_proposal_failure(proposal, message, work, errors)
                self._fail_source(work, message, errors)
                continue
            if any(
                proposal.status is ProposalStatus.FAILED
                for proposal in work.kept
            ):
                self._fail_source(
                    work,
                    "Source contains failed cluster proposals",
                    errors,
                )
                continue
            self._complete_source(work, errors)

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
        if proposal.status is not ProposalStatus.PENDING:
            return
        try:
            self.loop.record_proposal_failure(proposal, message)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"record proposal failure {proposal.id}: {exc}")
        work.decisions.append(
            {
                "proposal_id": proposal.id,
                "cluster_id": proposal.target_id,
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

    def _build_report(
        self,
        run: Run,
        works: Sequence[ImageWork],
        cluster_works: Sequence[ClusterWork],
        summary: RunSummary,
        checks: dict[str, bool],
        status: str,
        errors: list[str],
    ) -> dict[str, Any]:
        final_cluster_counts: dict[str, int] = {}
        for cluster_work in cluster_works:
            key = cluster_work.final_decision or "pending"
            final_cluster_counts[key] = final_cluster_counts.get(key, 0) + 1
        return {
            "schema_version": 8,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "test": "object_memory_demo_sam_grid_dino_cluster_review",
            "status": status,
            "run": summary.as_dict(),
            "checks": checks,
            "strategy": {
                "workflow": "sam3_point_grid_dinov3_cluster_qwen_review",
                "sam_candidate_source": (
                    f"automatic {self.config.sam3_pipeline.points_per_side}x"
                    f"{self.config.sam3_pipeline.points_per_side} positive point grid"
                ),
                "script_filtering": "score, area, IoU, containment, per-image cap",
                "candidate_clustering": "DINOv3 CLS cosine across different sources",
                "same_source_cluster_merge": False,
                "qwen_call_policy": "one call per cluster batch",
                "qwen_input": "cluster contact sheets plus active text summaries",
                "second_qwen_stage": False,
                "object_text_policy": "one cluster-level structured summary per object",
                "prompt_version": self.config.mllm_pipeline.prompt_version,
            },
            "models": {
                "qwen": {
                    **self._qwen_metrics,
                    "model_id": self.config.models.qwen_model_id,
                    "model_load_seconds": round(
                        self.mllm_runtime.model_load_seconds, 3
                    ),
                    "peak_memory_mib": round(self._peak_memory_mib["qwen"], 2),
                    "placement": self.mllm_runtime.model_placement,
                    "snapshot": self.mllm_runtime.resolved_snapshot,
                },
                "sam3": {
                    **self._sam_metrics,
                    "checkpoint": str(self.config.models.sam3_checkpoint),
                    "points_per_side": self.config.sam3_pipeline.points_per_side,
                    "points_per_batch": self.config.sam3_pipeline.points_per_batch,
                    "confidence_threshold": (
                        self.config.sam3_pipeline.confidence_threshold
                    ),
                    "model_load_seconds": round(
                        self.sam_runtime.model_load_seconds, 3
                    ),
                    "peak_memory_mib": round(self._peak_memory_mib["sam3"], 2),
                },
                "dinov3": {
                    **self._dino_metrics,
                    "model_id": self.config.models.dinov3_model_id,
                    "revision": self.config.models.dinov3_revision,
                    "model_path": str(self.config.models.dinov3_model_path),
                    "feature_layer": self.dino_runtime.feature_layer,
                    "input_size": self.config.visual_fingerprint.input_size,
                    "model_load_seconds": round(
                        self.dino_runtime.model_load_seconds, 3
                    ),
                    "peak_memory_mib": round(self._peak_memory_mib["dinov3"], 2),
                    "placement": self.dino_runtime.model_placement,
                },
                "execution_peak_memory_mib": round(
                    max(self._peak_memory_mib.values()), 2
                ),
                "residency_policy": "sequential SAM3 then DINOv3 then Qwen",
            },
            "visual_fingerprint_config": self.config.visual_fingerprint.model_dump(
                mode="json"
            ),
            "images": [work.as_dict() for work in works],
            "clusters": [cluster_work.as_dict() for cluster_work in cluster_works],
            "cluster_counts": final_cluster_counts,
            "external_errors": errors,
            "core_counts": self.store.status().counts,
        }

    @staticmethod
    def _candidate_report(proposal: Proposal) -> dict[str, Any]:
        return {
            "raw_candidate_id": proposal.raw_candidate_id,
            "candidate_source": proposal.prompt,
            "score": proposal.score,
            "bbox": proposal.bbox.model_dump(mode="json"),
            "mask_area_ratio": proposal.mask_area_ratio,
            "crop": proposal.crop_path,
            "mask": proposal.mask_path,
            "overlay": proposal.overlay_path,
        }

    def _build_checks(
        self,
        works: Sequence[ImageWork],
        cluster_works: Sequence[ClusterWork],
        summary: RunSummary,
    ) -> dict[str, bool]:
        decided_or_filtered = {
            ProposalStatus.DECIDED,
            ProposalStatus.FILTERED,
            ProposalStatus.FAILED,
        }
        return {
            "input_images_nonempty": bool(works),
            "every_input_accounted_for": all(
                work.duplicate or work.status in {"completed", "failed"}
                for work in works
            ),
            "automatic_candidate_source_recorded": all(
                proposal.prompt == "automatic_point_grid"
                for work in works
                for proposal in (*work.kept, *work.filtered)
            ),
            "every_kept_proposal_has_fingerprint_or_failure": all(
                proposal.fingerprint is not None
                or proposal.status is ProposalStatus.FAILED
                for work in works
                for proposal in work.kept
            ),
            "every_candidate_has_terminal_status": all(
                proposal.status in decided_or_filtered
                for work in works
                for proposal in work.kept
            ),
            "every_cluster_has_qwen_result_or_failure": all(
                cluster_work.final_decision is not None
                for cluster_work in cluster_works
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
        for runtime in (self.mllm_runtime, self.dino_runtime, self.sam_runtime):
            try:
                runtime.close()
            except Exception:
                pass
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

    @staticmethod
    def _new_run_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"run_{timestamp}"

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
