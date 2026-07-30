"""Thin adapter around the pinned SAM3 image processor."""

from __future__ import annotations

import gc
import inspect
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Sequence

import numpy as np
from PIL import Image


AUTOMATIC_CANDIDATE_SOURCE = "automatic_point_grid"


@dataclass(frozen=True, slots=True)
class RawSamCandidate:
    """One SDK-independent SAM3 candidate kept on CPU."""

    raw_candidate_id: str
    prompt: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    mask: np.ndarray

    def __post_init__(self) -> None:
        if not self.raw_candidate_id:
            raise ValueError("raw_candidate_id must not be empty")
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be finite and between 0 and 1")
        if len(self.bbox_xyxy) != 4 or not all(
            math.isfinite(value) for value in self.bbox_xyxy
        ):
            raise ValueError("bbox_xyxy must contain four finite values")
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError("SAM3 candidate masks must be two-dimensional")
        object.__setattr__(self, "prompt", self.prompt.strip())
        object.__setattr__(self, "mask", mask)


@dataclass(frozen=True, slots=True)
class Sam3Prediction:
    candidates: tuple[RawSamCandidate, ...]
    prompt_counts: dict[str, int]
    inference_seconds: float


class Sam3Adapter:
    """Load SAM3 once and expose class-agnostic CPU candidates."""

    def __init__(
        self,
        checkpoint: str | Path,
        confidence_threshold: float,
        *,
        points_per_side: int = 16,
        points_per_batch: int = 32,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.confidence_threshold = confidence_threshold
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.model_load_seconds = 0.0
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None

    def load(self) -> None:
        if self._processor is not None:
            return
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.points_per_side < 2:
            raise ValueError("points_per_side must be at least 2")
        if self.points_per_batch < 1:
            raise ValueError("points_per_batch must be positive")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {self.checkpoint}")

        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access CUDA.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "The active CUDA device does not support bfloat16 autocast."
            )

        builder_parameters = inspect.signature(build_sam3_image_model).parameters
        if "checkpoint_path" not in builder_parameters:
            raise RuntimeError("Installed SAM3 builder does not accept checkpoint_path.")
        builder_kwargs: dict[str, Any] = {"checkpoint_path": str(self.checkpoint)}
        for name, value in {
            "load_from_HF": False,
            "device": "cuda",
            "eval_mode": True,
            "enable_inst_interactivity": True,
        }.items():
            if name in builder_parameters:
                builder_kwargs[name] = value

        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        model = build_sam3_image_model(**builder_kwargs)
        processor = Sam3Processor(
            model,
            confidence_threshold=self.confidence_threshold,
        )
        model.eval()
        torch.cuda.synchronize()

        self.model_load_seconds = time.perf_counter() - load_started
        self._model = model
        self._processor = processor
        self._torch = torch

    def predict(
        self,
        image: Image.Image,
        prompts: Sequence[str] | None = None,
    ) -> Sam3Prediction:
        """Generate candidates automatically, or retain explicit prompts for M2 replay."""

        if prompts is None:
            return self._predict_automatic(image)
        return self._predict_with_text_prompts(image, prompts)

    def _predict_automatic(self, image: Image.Image) -> Sam3Prediction:
        if self._processor is None:
            self.load()

        torch = self._torch
        processor = self._processor
        model = self._model
        if torch is None or processor is None or model is None:
            raise RuntimeError("SAM3 adapter failed to initialize")
        if not hasattr(model, "predict_inst"):
            raise RuntimeError(
                "Installed SAM3 model does not expose interactive point prediction"
            )

        rgb_image = image.convert("RGB")
        grid_points = self._point_grid(rgb_image.width, rgb_image.height)
        candidates: list[RawSamCandidate] = []
        inference_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            state = processor.set_image(rgb_image)
            for batch_start in range(0, len(grid_points), self.points_per_batch):
                point_batch = grid_points[
                    batch_start : batch_start + self.points_per_batch
                ]
                point_coords = point_batch[:, None, :]
                point_labels = np.ones((len(point_batch), 1), dtype=np.int64)
                masks, scores, _ = model.predict_inst(
                    state,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
                candidates.extend(
                    self._extract_point_candidates(
                        masks,
                        scores,
                        batch_start=batch_start,
                        expected_count=len(point_batch),
                    )
                )
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started
        return Sam3Prediction(
            candidates=tuple(candidates),
            prompt_counts={AUTOMATIC_CANDIDATE_SOURCE: len(candidates)},
            inference_seconds=inference_seconds,
        )

    def _predict_with_text_prompts(
        self,
        image: Image.Image,
        prompts: Sequence[str],
    ) -> Sam3Prediction:
        normalized_prompts = [prompt.strip() for prompt in prompts]
        if not normalized_prompts or any(not prompt for prompt in normalized_prompts):
            raise ValueError("At least one non-empty SAM3 prompt is required")
        if len(set(normalized_prompts)) != len(normalized_prompts):
            raise ValueError("SAM3 prompts must be unique")
        if self._processor is None:
            self.load()

        torch = self._torch
        processor = self._processor
        if torch is None or processor is None:
            raise RuntimeError("SAM3 adapter failed to initialize")

        candidates: list[RawSamCandidate] = []
        prompt_counts: dict[str, int] = {}
        inference_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            state = processor.set_image(image.convert("RGB"))
            for prompt_index, prompt in enumerate(normalized_prompts):
                output = processor.set_text_prompt(state=state, prompt=prompt)
                prompt_candidates = self._extract_candidates(
                    output,
                    prompt=prompt,
                    prompt_index=prompt_index,
                )
                prompt_counts[prompt] = len(prompt_candidates)
                candidates.extend(prompt_candidates)
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started
        return Sam3Prediction(
            candidates=tuple(candidates),
            prompt_counts=prompt_counts,
            inference_seconds=inference_seconds,
        )

    def _point_grid(self, width: int, height: int) -> np.ndarray:
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        x_coords = (
            np.arange(self.points_per_side, dtype=np.float32) + 0.5
        ) * (width / self.points_per_side)
        y_coords = (
            np.arange(self.points_per_side, dtype=np.float32) + 0.5
        ) * (height / self.points_per_side)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        return np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)

    def _extract_point_candidates(
        self,
        masks: Any,
        scores: Any,
        *,
        batch_start: int,
        expected_count: int,
    ) -> list[RawSamCandidate]:
        mask_array = np.asarray(masks)
        score_array = np.asarray(scores)
        if mask_array.ndim == 3:
            mask_array = mask_array[None, ...]
        if score_array.ndim == 1:
            score_array = score_array[None, ...]
        if mask_array.ndim != 4 or score_array.ndim != 2:
            raise ValueError(
                "Unexpected SAM3 point-prediction mask or score dimensions"
            )
        if (
            mask_array.shape[0] != expected_count
            or score_array.shape[0] != expected_count
            or mask_array.shape[1] != score_array.shape[1]
        ):
            raise ValueError(
                "SAM3 returned a different number of point prompts, masks, or scores"
            )

        candidates: list[RawSamCandidate] = []
        for point_offset in range(expected_count):
            best_mask_index = int(np.argmax(score_array[point_offset]))
            score = float(
                np.clip(score_array[point_offset, best_mask_index], 0.0, 1.0)
            )
            mask = np.asarray(
                mask_array[point_offset, best_mask_index],
                dtype=bool,
            )
            bbox = self._mask_bbox(mask)
            if bbox is None:
                bbox = (0.0, 0.0, 0.0, 0.0)
            point_index = batch_start + point_offset
            candidates.append(
                RawSamCandidate(
                    raw_candidate_id=f"grid_point_{point_index:06d}",
                    prompt=AUTOMATIC_CANDIDATE_SOURCE,
                    score=score,
                    bbox_xyxy=bbox,
                    mask=mask,
                )
            )
        return candidates

    @staticmethod
    def _mask_bbox(
        mask: np.ndarray,
    ) -> tuple[float, float, float, float] | None:
        y_indices, x_indices = np.nonzero(mask)
        if not len(x_indices):
            return None
        return (
            float(x_indices.min()),
            float(y_indices.min()),
            float(x_indices.max() + 1),
            float(y_indices.max() + 1),
        )

    def _extract_candidates(
        self,
        output: dict[str, Any],
        *,
        prompt: str,
        prompt_index: int,
    ) -> list[RawSamCandidate]:
        torch = self._torch
        if torch is None:
            raise RuntimeError("SAM3 adapter is not loaded")

        masks = torch.as_tensor(output["masks"]).detach().cpu()
        boxes = torch.as_tensor(output["boxes"]).detach().cpu().reshape(-1, 4)
        scores = torch.as_tensor(output["scores"]).detach().cpu().flatten()
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if masks.ndim != 3:
            raise ValueError(f"Unexpected SAM3 mask shape: {tuple(masks.shape)}")

        count = int(masks.shape[0])
        if int(boxes.shape[0]) != count or int(scores.shape[0]) != count:
            raise ValueError(
                "SAM3 returned different numbers of masks, boxes, and scores"
            )

        candidates: list[RawSamCandidate] = []
        for candidate_index in range(count):
            box = tuple(float(value) for value in boxes[candidate_index].tolist())
            candidate = RawSamCandidate(
                raw_candidate_id=(
                    f"prompt_{prompt_index:03d}_candidate_{candidate_index:03d}"
                ),
                prompt=prompt,
                score=float(scores[candidate_index].item()),
                bbox_xyxy=box,
                mask=(masks[candidate_index] > 0.5).numpy(),
            )
            candidates.append(candidate)
        return candidates

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

    def __enter__(self) -> "Sam3Adapter":
        self.load()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
