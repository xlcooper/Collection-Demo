"""Thin Qwen3-VL adapter for constrained object-memory decisions."""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class MllmPrediction:
    """One SDK-independent model response with runtime measurements."""

    raw_text: str
    input_tokens: int
    generated_tokens: int
    inference_seconds: float


def resolve_local_snapshot(model: str, revision: str | None = None) -> Path:
    """Resolve a model ID or path to one concrete local HF snapshot."""

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
                available = (
                    sorted(
                        path
                        for path in snapshots_root.iterdir()
                        if path.is_dir() and (path / "config.json").is_file()
                    )
                    if snapshots_root.is_dir()
                    else []
                )
                if len(available) != 1:
                    raise FileNotFoundError(
                        "Cannot resolve one local Qwen snapshot under "
                        f"{snapshots_root}."
                    )
                snapshot_path = available[0]

    if not (snapshot_path / "config.json").is_file():
        raise FileNotFoundError(
            f"Local model snapshot has no config.json: {snapshot_path}"
        )
    return snapshot_path.resolve()


class QwenMllmAdapter:
    """Load Qwen once and return raw text without leaking SDK types."""

    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        allow_network: bool = False,
        max_pixels: int = 1024 * 1024,
        max_new_tokens: int = 512,
    ) -> None:
        self.model_id = model
        self.revision = revision
        self.allow_network = allow_network
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.model_load_seconds = 0.0
        self.model_placement: list[str] = []
        self.resolved_snapshot: str | None = None
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._process_vision_info: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        if self.max_pixels <= 0 or self.max_new_tokens <= 0:
            raise ValueError("Qwen pixel and token limits must be positive")

        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access CUDA.")

        if self.allow_network:
            model_source = self.model_id
        else:
            snapshot = resolve_local_snapshot(self.model_id, self.revision)
            model_source = str(snapshot)
            self.resolved_snapshot = snapshot.name

        model_kwargs: dict[str, Any] = {
            "dtype": "auto",
            "device_map": "auto",
            "local_files_only": not self.allow_network,
        }
        processor_kwargs: dict[str, Any] = {
            "local_files_only": not self.allow_network,
        }
        if self.revision and self.allow_network:
            model_kwargs["revision"] = self.revision
            processor_kwargs["revision"] = self.revision

        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_source,
            **model_kwargs,
        )
        processor = AutoProcessor.from_pretrained(
            model_source,
            **processor_kwargs,
        )
        model.eval()
        device_map = getattr(model, "hf_device_map", None)
        self.model_placement = (
            sorted({str(device) for device in device_map.values()})
            if isinstance(device_map, dict)
            else [str(model.device)]
        )
        torch.cuda.synchronize()

        self.model_load_seconds = time.perf_counter() - load_started
        self._model = model
        self._processor = processor
        self._torch = torch
        self._process_vision_info = process_vision_info

    def predict(self, messages: Sequence[dict[str, Any]]) -> MllmPrediction:
        if self._model is None:
            self.load()
        model = self._model
        processor = self._processor
        torch = self._torch
        process_vision_info = self._process_vision_info
        if any(value is None for value in (model, processor, torch, process_vision_info)):
            raise RuntimeError("Qwen adapter failed to initialize")

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
        input_tokens = int(inputs.input_ids.numel())

        inference_started = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started

        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_tokens = sum(int(token_ids.numel()) for token_ids in trimmed_ids)
        raw_text = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return MllmPrediction(
            raw_text=raw_text,
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            inference_seconds=inference_seconds,
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
        self._process_vision_info = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._torch = None

    def __enter__(self) -> "QwenMllmAdapter":
        self.load()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
