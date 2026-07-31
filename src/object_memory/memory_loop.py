"""Business boundary between validated MLLM output and persistent memory."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from .assets import MemoryPaths
from .memory_store import (
    DecisionWriteResult,
    MemoryStore,
    RunSummary,
    SourceRegistration,
)
from .schemas import (
    Decision,
    DecisionType,
    MemoryObject,
    MllmResponse,
    ObjectCard,
    Observation,
    Proposal,
    Run,
    SourceImage,
    new_id,
)


@dataclass(frozen=True, slots=True)
class ObservationAssets:
    """Canonical object-observation paths created from proposal assets."""

    directory: Path
    crop_path: str
    mask_path: str
    overlay_path: str


class MemoryLoop:
    """Expose readable M4 operations while keeping SQL inside MemoryStore."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.paths: MemoryPaths = store.paths
        self._duplicate_counts: dict[str, int] = {}

    def begin_run(self, run: Run) -> None:
        self.store.begin_run(run)

    def register_source(self, source: SourceImage) -> SourceRegistration:
        registration = self.store.register_source(source)
        if registration.duplicate:
            self._duplicate_counts[source.run_id] = (
                self._duplicate_counts.get(source.run_id, 0) + 1
            )
        return registration

    def record_filtered_proposal(self, proposal: Proposal) -> None:
        self.store.record_filtered_proposal(proposal)

    def object_cards(self, *, max_reference_views: int = 2) -> list[ObjectCard]:
        return self.store.list_object_cards(
            max_reference_views=max_reference_views
        )

    def apply_response(
        self,
        *,
        proposal: Proposal,
        response: MllmResponse,
        prompt_version: str,
        raw_response_path: str | None = None,
        attempt: int = 1,
    ) -> DecisionWriteResult:
        """Turn one validated four-way MLLM response into durable records."""

        decision = Decision(
            proposal_id=proposal.id,
            decision=response.decision,
            matched_object_id=response.matched_object_id,
            confidence=response.confidence,
            reason_code=response.reason_code.value,
            short_reason=response.short_reason,
            prompt_version=prompt_version,
            raw_response_path=raw_response_path,
            attempt=attempt,
        )
        memory_object: MemoryObject | None = None
        observation: Observation | None = None
        promoted_assets: ObservationAssets | None = None

        if response.decision in {DecisionType.NEW, DecisionType.EXISTING}:
            if response.annotation is None:  # also enforced by MllmResponse
                raise ValueError("new and existing responses require annotation")
            object_id = response.matched_object_id or new_id("obj")
            observation_id = new_id("obs")
            promoted_assets = self._promote_observation_assets(
                proposal=proposal,
                object_id=object_id,
                observation_id=observation_id,
            )
            if response.decision is DecisionType.NEW:
                memory_object = MemoryObject(
                    id=object_id,
                    coarse_category=response.annotation.coarse_category,
                    fine_category=response.annotation.fine_category,
                    material=response.annotation.material,
                    color=response.annotation.color,
                    shape=response.annotation.shape,
                    description=response.annotation.description,
                    annotation_confidence=response.annotation.annotation_confidence,
                )
            observation = Observation(
                id=observation_id,
                object_id=object_id,
                proposal_id=proposal.id,
                source_image_id=proposal.source_image_id,
                crop_path=promoted_assets.crop_path,
                mask_path=promoted_assets.mask_path,
                overlay_path=promoted_assets.overlay_path,
                description=response.annotation.description,
            )

        try:
            return self.store.commit_decision(
                proposal=proposal,
                decision=decision,
                memory_object=memory_object,
                object_annotation=(
                    response.annotation
                    if response.decision is DecisionType.EXISTING
                    else None
                ),
                observation=observation,
            )
        except Exception:
            if promoted_assets is not None:
                self._remove_promoted_assets(promoted_assets.directory)
            raise

    def record_proposal_failure(
        self,
        proposal: Proposal,
        error_message: str,
    ) -> None:
        self.store.record_proposal_failure(proposal, error_message)

    def complete_source(self, source_id: str) -> None:
        self.store.complete_source(source_id)

    def fail_source(self, source_id: str, error_message: str) -> None:
        self.store.fail_source(source_id, error_message)

    def complete_run(
        self,
        run_id: str,
        *,
        external_errors: int = 0,
    ) -> RunSummary:
        if external_errors < 0:
            raise ValueError("external_errors must not be negative")
        error_message = (
            f"{external_errors} error(s) occurred outside registered records"
            if external_errors
            else None
        )
        summary = self.store.complete_run(
            run_id,
            error_message=error_message,
        )
        return replace(
            summary,
            duplicate_sources_skipped=self._duplicate_counts.get(run_id, 0),
            external_errors=external_errors,
        )

    def _promote_observation_assets(
        self,
        *,
        proposal: Proposal,
        object_id: str,
        observation_id: str,
    ) -> ObservationAssets:
        source_paths = {
            "crop": proposal.crop_path,
            "mask": proposal.mask_path,
            "overlay": proposal.overlay_path,
        }
        if any(relative_path is None for relative_path in source_paths.values()):
            raise ValueError(
                "new and existing proposals require crop, mask, and overlay assets"
            )

        destination_directory = self.paths.observation_dir(
            object_id,
            observation_id,
        )
        if destination_directory.exists():
            raise FileExistsError(
                f"Observation asset directory already exists: {destination_directory}"
            )
        destination_directory.mkdir(parents=True)
        destinations: dict[str, Path] = {}
        try:
            for role, relative_path in source_paths.items():
                assert relative_path is not None
                source = self.paths.resolve_asset(relative_path)
                if not source.is_file():
                    raise FileNotFoundError(f"Proposal {role} asset not found: {source}")
                suffix = source.suffix.lower() or ".bin"
                destination = destination_directory / f"{role}{suffix}"
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                shutil.copy2(source, temporary)
                temporary.replace(destination)
                destinations[role] = destination
        except Exception:
            self._remove_promoted_assets(destination_directory)
            raise

        return ObservationAssets(
            directory=destination_directory,
            crop_path=self.paths.relative_asset(destinations["crop"]),
            mask_path=self.paths.relative_asset(destinations["mask"]),
            overlay_path=self.paths.relative_asset(destinations["overlay"]),
        )

    def _remove_promoted_assets(self, directory: Path) -> None:
        if directory.is_dir():
            shutil.rmtree(directory)
        stop = self.paths.objects
        parent = directory.parent
        while parent != stop and parent.is_relative_to(stop):
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
