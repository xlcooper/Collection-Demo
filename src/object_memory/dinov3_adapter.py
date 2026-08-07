"""DINOv3 extraction, fingerprint persistence, and view-to-view matching."""

from __future__ import annotations

import gc
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
import time
from types import TracebackType
from typing import Any, Sequence
from uuid import uuid4

import numpy as np
from PIL import Image

from .config import VisualFingerprintConfig
from .schemas import (
    BoundingBox,
    VisualEvidence,
    VisualFingerprint,
    VisualMatchType,
    VisualObjectScore,
)


@dataclass(frozen=True, slots=True)
class FingerprintData:
    global_embedding: np.ndarray
    local_embeddings: np.ndarray
    local_patch_indices: np.ndarray

    def __post_init__(self) -> None:
        global_embedding = np.asarray(self.global_embedding, dtype=np.float32)
        local_embeddings = np.asarray(self.local_embeddings, dtype=np.float32)
        patch_indices = np.asarray(self.local_patch_indices, dtype=np.int32)
        if global_embedding.ndim != 1 or not global_embedding.size:
            raise ValueError("global_embedding must be one non-empty vector")
        if local_embeddings.ndim != 2:
            raise ValueError("local_embeddings must be a two-dimensional matrix")
        if local_embeddings.shape[1] != global_embedding.shape[0]:
            raise ValueError("global and local embedding dimensions must agree")
        if patch_indices.shape != (local_embeddings.shape[0], 2):
            raise ValueError("local_patch_indices must contain one row/column pair")
        object.__setattr__(self, "global_embedding", global_embedding)
        object.__setattr__(self, "local_embeddings", local_embeddings)
        object.__setattr__(self, "local_patch_indices", patch_indices)


@dataclass(frozen=True, slots=True)
class HistoricalFingerprint:
    object_id: str
    observation_id: str
    data: FingerprintData


def _l2_normalize(values: np.ndarray, *, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_fingerprint(
    path: Path,
    data: FingerprintData,
    *,
    relative_path: str,
    model_id: str,
    revision: str,
    feature_layer: str,
    input_size: int,
    storage_dtype: str,
) -> VisualFingerprint:
    """Write one immutable NPZ fingerprint and return audited metadata."""

    if storage_dtype != "float16":
        raise ValueError("The first implementation stores only float16 fingerprints")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                global_embedding=data.global_embedding.astype(np.float16),
                local_embeddings=data.local_embeddings.astype(np.float16),
                local_patch_indices=data.local_patch_indices.astype(np.int32),
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return VisualFingerprint(
        path=relative_path,
        sha256=sha256_file(path),
        model_id=model_id,
        revision=revision,
        feature_layer=feature_layer,
        input_size=input_size,
        storage_dtype=storage_dtype,
        global_dimension=int(data.global_embedding.shape[0]),
        local_count=int(data.local_embeddings.shape[0]),
        l2_normalized=True,
    )


def read_fingerprint(path: Path, *, expected_sha256: str | None = None) -> FingerprintData:
    """Load and validate one persisted fingerprint without consulting image assets."""

    if not path.is_file():
        raise FileNotFoundError(f"Visual fingerprint not found: {path}")
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f"Visual fingerprint hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "global_embedding",
            "local_embeddings",
            "local_patch_indices",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"Visual fingerprint is missing arrays: {missing}")
        return FingerprintData(
            global_embedding=_l2_normalize(payload["global_embedding"], axis=0),
            local_embeddings=_l2_normalize(payload["local_embeddings"], axis=1),
            local_patch_indices=payload["local_patch_indices"],
        )


def local_match_ratio(
    first: np.ndarray,
    second: np.ndarray,
    *,
    similarity_threshold: float,
) -> float:
    """Return a symmetric ratio of patches with a strong cross-view counterpart."""

    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1:] != second.shape[1:]:
        raise ValueError("Local fingerprint matrices must share one embedding width")
    if not len(first) or not len(second):
        return 0.0
    similarities = first @ second.T
    forward = float(np.mean(np.max(similarities, axis=1) >= similarity_threshold))
    backward = float(np.mean(np.max(similarities, axis=0) >= similarity_threshold))
    return (forward + backward) / 2.0


def compare_fingerprints(
    first: FingerprintData,
    second: FingerprintData,
    settings: VisualFingerprintConfig,
) -> tuple[float, float, float]:
    """Return global cosine, local correspondence ratio, and weighted score."""

    global_similarity = float(
        np.clip(first.global_embedding @ second.global_embedding, -1.0, 1.0)
    )
    local_ratio = local_match_ratio(
        first.local_embeddings,
        second.local_embeddings,
        similarity_threshold=settings.local_patch_similarity_threshold,
    )
    visual_score = (
        settings.global_weight * global_similarity
        + settings.local_weight * local_ratio
    )
    return global_similarity, local_ratio, float(np.clip(visual_score, -1.0, 1.0))


def match_fingerprint(
    query: FingerprintData,
    historical: Sequence[HistoricalFingerprint],
    settings: VisualFingerprintConfig,
) -> VisualEvidence:
    """Globally rank objects, then compare local patches for the top objects."""

    views_by_object: dict[str, list[HistoricalFingerprint]] = {}
    for item in historical:
        views_by_object.setdefault(item.object_id, []).append(item)

    ranked_objects: list[tuple[float, str]] = []
    for object_id, views in views_by_object.items():
        best_global = max(
            float(
                np.clip(
                    query.global_embedding @ view.data.global_embedding,
                    -1.0,
                    1.0,
                )
            )
            for view in views
        )
        ranked_objects.append((best_global, object_id))
    ranked_objects.sort(key=lambda item: (-item[0], item[1]))

    best_by_object: dict[str, VisualObjectScore] = {}
    for _, object_id in ranked_objects[: settings.local_top_k]:
        for item in views_by_object[object_id]:
            global_similarity, local_ratio, visual_score = compare_fingerprints(
                query,
                item.data,
                settings,
            )
            score = VisualObjectScore(
                object_id=item.object_id,
                observation_id=item.observation_id,
                global_similarity=global_similarity,
                local_match_ratio=local_ratio,
                visual_score=visual_score,
            )
            previous = best_by_object.get(item.object_id)
            if previous is None or (score.visual_score, score.observation_id) > (
                previous.visual_score,
                previous.observation_id,
            ):
                best_by_object[item.object_id] = score

    scores = sorted(
        best_by_object.values(),
        key=lambda item: (-item.visual_score, item.object_id, item.observation_id),
    )
    if not scores:
        return VisualEvidence(result=VisualMatchType.NO_MATCH)

    best = scores[0]
    second_score = scores[1].visual_score if len(scores) > 1 else None
    margin = (
        max(0.0, best.visual_score - second_score)
        if second_score is not None
        else None
    )
    common = {
        "global_similarity": best.global_similarity,
        "local_match_ratio": best.local_match_ratio,
        "visual_score": best.visual_score,
        "second_best_score": second_score,
        "score_margin": margin,
        "object_scores": scores,
    }
    if best.visual_score < settings.match_threshold:
        return VisualEvidence(result=VisualMatchType.NO_MATCH, **common)
    if margin is not None and margin < settings.ambiguity_margin:
        return VisualEvidence(result=VisualMatchType.AMBIGUOUS, **common)
    return VisualEvidence(
        result=VisualMatchType.MATCH,
        matched_object_id=best.object_id,
        matched_observation_id=best.observation_id,
        **common,
    )


class DinoV3Adapter:
    """Load the pinned local DINOv3 model and extract CLS plus masked patches."""

    feature_layer = "last_hidden_state"

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_id: str,
        revision: str,
        settings: VisualFingerprintConfig,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.model_id = model_id
        self.revision = revision
        self.settings = settings
        self.model_load_seconds = 0.0
        self.last_inference_seconds = 0.0
        self.model_placement: list[str] = []
        self.resolved_snapshot: str | None = None
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._patch_size = 16
        self._num_register_tokens = 0

    def load(self) -> None:
        if self._model is not None:
            return
        if not (self.model_path / "config.json").is_file():
            raise FileNotFoundError(
                f"Local DINOv3 snapshot has no config.json: {self.model_path}"
            )
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access CUDA for DINOv3.")
        load_started = time.perf_counter()
        model = AutoModel.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            dtype=torch.float16,
        ).to("cuda")
        processor = AutoImageProcessor.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )
        model.eval()
        torch.cuda.synchronize()
        self.model_load_seconds = time.perf_counter() - load_started
        self.model_placement = [str(next(model.parameters()).device)]
        self.resolved_snapshot = self.revision
        configured_patch_size = getattr(model.config, "patch_size", 16)
        if isinstance(configured_patch_size, (list, tuple)):
            if len(configured_patch_size) != 2 or len(set(configured_patch_size)) != 1:
                raise ValueError("DINOv3 requires one square patch size")
            configured_patch_size = configured_patch_size[0]
        self._patch_size = int(configured_patch_size)
        self._num_register_tokens = int(
            getattr(model.config, "num_register_tokens", 0)
        )
        if self.settings.input_size % self._patch_size:
            raise ValueError("DINOv3 input_size must be divisible by model patch_size")
        self._model = model
        self._processor = processor
        self._torch = torch

    def extract(
        self,
        *,
        crop_path: Path,
        mask_path: Path,
        bbox: BoundingBox,
        crop_padding_pixels: int,
    ) -> FingerprintData:
        if self._model is None:
            self.load()
        model = self._model
        processor = self._processor
        torch = self._torch
        if model is None or processor is None or torch is None:
            raise RuntimeError("DINOv3 adapter failed to initialize")

        with Image.open(crop_path) as opened:
            crop = opened.convert("RGB")
        with Image.open(mask_path) as opened:
            mask = opened.convert("L")
        left = max(0, math.floor(bbox.x_min) - crop_padding_pixels)
        top = max(0, math.floor(bbox.y_min) - crop_padding_pixels)
        right = min(mask.width, math.ceil(bbox.x_max) + crop_padding_pixels)
        bottom = min(mask.height, math.ceil(bbox.y_max) + crop_padding_pixels)
        crop_mask = mask.crop((left, top, right, bottom))
        if crop_mask.size != crop.size:
            raise ValueError(
                "Proposal crop and bbox-derived mask crop have different dimensions"
            )

        size = self.settings.input_size
        resized_crop = crop.resize((size, size), Image.Resampling.BICUBIC)
        resized_mask = crop_mask.resize((size, size), Image.Resampling.NEAREST)
        inputs = processor(
            images=resized_crop,
            do_resize=False,
            do_center_crop=False,
            return_tensors="pt",
        )
        if tuple(inputs["pixel_values"].shape[-2:]) != (size, size):
            raise RuntimeError("DINOv3 processor changed the configured input size")
        pixel_values = inputs["pixel_values"].to(
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )
        inference_started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(pixel_values=pixel_values)
        torch.cuda.synchronize()
        self.last_inference_seconds = time.perf_counter() - inference_started

        hidden = outputs.last_hidden_state[0].detach().float().cpu().numpy()
        grid = size // self._patch_size
        patch_count = grid * grid
        first_patch = 1 + self._num_register_tokens
        expected_tokens = first_patch + patch_count
        if hidden.shape[0] != expected_tokens:
            raise RuntimeError(
                "DINOv3 hidden state does not contain the expected CLS, register, "
                "and patch tokens"
            )
        global_embedding = _l2_normalize(hidden[0], axis=0)
        patch_embeddings = _l2_normalize(
            hidden[first_patch:expected_tokens],
            axis=1,
        )
        mask_array = np.asarray(resized_mask, dtype=np.float32) / 255.0
        coverage = mask_array.reshape(
            grid,
            self._patch_size,
            grid,
            self._patch_size,
        ).mean(axis=(1, 3))
        valid_indices = np.argwhere(
            coverage >= self.settings.min_patch_mask_coverage
        ).astype(np.int32)
        flat_indices = valid_indices[:, 0] * grid + valid_indices[:, 1]
        local_embeddings = patch_embeddings[flat_indices]
        return FingerprintData(
            global_embedding=global_embedding,
            local_embeddings=local_embeddings,
            local_patch_indices=valid_indices,
        )

    @property
    def peak_memory_mib(self) -> float:
        if self._torch is None:
            return 0.0
        return float(self._torch.cuda.max_memory_allocated() / (1024**2))

    def close(self) -> None:
        torch = self._torch
        self._processor = None
        self._model = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._torch = None

    def __enter__(self) -> "DinoV3Adapter":
        self.load()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
