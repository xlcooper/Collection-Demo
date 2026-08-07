"""Business boundary between visual/text identity evidence and durable memory."""

from __future__ import annotations

from dataclasses import replace

from .assets import MemoryPaths
from .memory_store import (
    ClusterWriteResult,
    FingerprintRecord,
    MemoryStore,
    RunSummary,
    SourceRegistration,
)
from .schemas import (
    ClusterReview,
    Decision,
    DecisionReasonCode,
    DecisionType,
    MemoryObject,
    ObjectCard,
    Observation,
    Proposal,
    ProposalStatus,
    Run,
    SourceImage,
    VisualEvidence,
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

    def record_filtered_cluster(
        self,
        proposals: list[Proposal],
        *,
        cluster_id: str,
        reason: str,
    ) -> None:
        message = reason.strip()
        if not message:
            raise ValueError("cluster filter reason must not be empty")
        for proposal in proposals:
            proposal.target_id = cluster_id
            proposal.status = ProposalStatus.FILTERED
            proposal.filter_reason = f"qwen_cluster_ignore:{message}"
            proposal.updated_at = utc_now()
        self.store.record_filtered_proposals(proposals)

    def object_cards(self) -> list[ObjectCard]:
        return self.store.list_object_cards()

    def fingerprint_records(self) -> list[FingerprintRecord]:
        return self.store.list_fingerprint_records()

    def apply_cluster_decision(
        self,
        *,
        proposals: list[Proposal],
        review: ClusterReview,
        decision_type: DecisionType,
        visual_evidence: VisualEvidence,
        prompt_version: str,
        raw_response_path: str,
        reason_code: DecisionReasonCode,
        short_reason: str,
    ) -> ClusterWriteResult:
        """Persist one cluster atomically and create at most one object."""

        if not proposals:
            raise ValueError("A cluster decision requires proposals")
        proposal_list = sorted(proposals, key=lambda proposal: proposal.id)
        for proposal in proposal_list:
            if proposal.fingerprint is None:
                raise ValueError("Every reviewed proposal requires a fingerprint")
            proposal.target_id = review.cluster_id
            proposal.target_object_name_zh = (
                review.object_summary.object_name_zh
                if review.object_summary is not None
                else None
            )

        memory_object: MemoryObject | None = None
        object_summary = None
        object_id: str | None = None
        if decision_type is DecisionType.NEW:
            if review.object_summary is None:
                raise ValueError("A new cluster requires an object summary")
            object_id = new_id("obj")
            memory_object = MemoryObject(id=object_id, summary=review.object_summary)
        elif decision_type is DecisionType.EXISTING:
            if review.object_summary is None or review.matched_object_id is None:
                raise ValueError("An existing cluster requires object ID and summary")
            object_id = review.matched_object_id
            object_summary = review.object_summary

        decisions: list[Decision] = []
        observations: list[Observation] = []
        first_new = True
        for proposal in proposal_list:
            proposal_decision = decision_type
            matched_object_id = object_id if decision_type is DecisionType.EXISTING else None
            proposal_reason_code = reason_code
            proposal_short_reason = short_reason
            if decision_type is DecisionType.NEW and not first_new:
                proposal_decision = DecisionType.EXISTING
                matched_object_id = object_id
                proposal_reason_code = DecisionReasonCode.CLUSTER_MEMBER
                proposal_short_reason = (
                    "该视角与本批次首个新对象候选属于同一DINOv3聚类。"
                )
            decisions.append(
                Decision(
                    proposal_id=proposal.id,
                    decision=proposal_decision,
                    matched_object_id=matched_object_id,
                    confidence=(
                        review.object_summary.summary_confidence
                        if (
                            review.object_summary is not None
                            and decision_type is not DecisionType.UNCERTAIN
                        )
                        else 0.0
                    ),
                    reason_code=proposal_reason_code.value,
                    short_reason=proposal_short_reason,
                    prompt_version=prompt_version,
                    qwen_hypothesis=review.identity_hypothesis,
                    qwen_matched_object_id=review.matched_object_id,
                    visual_evidence=visual_evidence,
                    raw_response_path=raw_response_path,
                )
            )
            if object_id is not None:
                observations.append(
                    Observation(
                        object_id=object_id,
                        proposal_id=proposal.id,
                        source_image_id=proposal.source_image_id,
                        fingerprint=proposal.fingerprint,
                    )
                )
            first_new = False

        result = self.store.commit_cluster_decisions(
            proposals=proposal_list,
            decisions=decisions,
            memory_object=memory_object,
            object_summary=object_summary,
            observations=observations,
        )
        for proposal in proposal_list:
            proposal.status = ProposalStatus.DECIDED
            proposal.updated_at = utc_now()
        return result

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
