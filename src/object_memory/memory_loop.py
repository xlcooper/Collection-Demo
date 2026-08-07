"""Business boundary between visual/text identity evidence and durable memory."""

from __future__ import annotations

from dataclasses import replace

from .assets import MemoryPaths
from .memory_store import (
    DecisionWriteResult,
    FingerprintRecord,
    MemoryStore,
    RunSummary,
    SourceRegistration,
)
from .schemas import (
    Decision,
    DecisionType,
    FinalIdentityDecision,
    MemoryObject,
    ObjectCard,
    Observation,
    Proposal,
    ProposalStatus,
    Run,
    SourceImage,
    VisualFingerprint,
    new_id,
    utc_now,
)


class MemoryLoop:
    """Expose readable memory operations while keeping SQL inside MemoryStore."""

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

    def object_cards(self) -> list[ObjectCard]:
        return self.store.list_object_cards()

    def fingerprint_records(self) -> list[FingerprintRecord]:
        return self.store.list_fingerprint_records()

    def apply_decision(
        self,
        *,
        proposal: Proposal,
        result: FinalIdentityDecision,
        fingerprint: VisualFingerprint,
        prompt_version: str,
        raw_response_path: str | None,
    ) -> DecisionWriteResult:
        """Persist one final Qwen-plus-DINOv3 decision without copying assets."""

        proposal.fingerprint = fingerprint
        decision = Decision(
            proposal_id=proposal.id,
            decision=result.decision,
            matched_object_id=result.matched_object_id,
            confidence=result.confidence,
            reason_code=result.reason_code.value,
            short_reason=result.short_reason,
            prompt_version=prompt_version,
            qwen_hypothesis=result.qwen_hypothesis,
            qwen_matched_object_id=result.qwen_matched_object_id,
            visual_evidence=result.visual_evidence,
            raw_response_path=raw_response_path,
        )
        memory_object: MemoryObject | None = None
        observation: Observation | None = None
        object_summary = None
        if result.decision in {DecisionType.NEW, DecisionType.EXISTING}:
            if result.object_summary is None:
                raise ValueError("new and existing decisions require object_summary")
            object_id = result.matched_object_id or new_id("obj")
            if result.decision is DecisionType.NEW:
                memory_object = MemoryObject(
                    id=object_id,
                    summary=result.object_summary,
                )
            else:
                object_summary = result.object_summary
            observation = Observation(
                object_id=object_id,
                proposal_id=proposal.id,
                source_image_id=proposal.source_image_id,
                fingerprint=fingerprint,
            )
        write_result = self.store.commit_decision(
            proposal=proposal,
            decision=decision,
            memory_object=memory_object,
            object_summary=object_summary,
            observation=observation,
        )
        proposal.status = write_result.proposal_status
        proposal.updated_at = utc_now()
        return write_result

    def record_proposal_failure(self, proposal: Proposal, error_message: str) -> None:
        self.store.record_proposal_failure(proposal, error_message)
        proposal.status = ProposalStatus.FAILED
        proposal.error_message = error_message.strip()
        proposal.updated_at = utc_now()

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
        summary = self.store.complete_run(
            run_id,
            error_message=(
                f"{external_errors} error(s) occurred outside registered records"
                if external_errors
                else None
            ),
        )
        return replace(
            summary,
            duplicate_sources_skipped=self._duplicate_counts.get(run_id, 0),
            external_errors=external_errors,
        )
