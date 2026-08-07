"""Cross-image DINOv3 clustering and cluster explanation assets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from statistics import fmean
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .assets import MemoryPaths
from .config import VisualFingerprintConfig
from .dinov3_adapter import FingerprintData, HistoricalFingerprint, match_fingerprint
from .schemas import Proposal, VisualEvidence, VisualMatchType


@dataclass(frozen=True, slots=True)
class FingerprintedCandidate:
    proposal: Proposal
    data: FingerprintData


@dataclass(frozen=True, slots=True)
class CandidateCluster:
    id: str
    members: tuple[FingerprintedCandidate, ...]
    representative_proposal_ids: tuple[str, ...]
    global_similarity_min: float
    global_similarity_mean: float
    global_similarity_max: float

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(item.proposal.source_image_id for item in self.members)

    @property
    def proposal_ids(self) -> tuple[str, ...]:
        return tuple(item.proposal.id for item in self.members)

    def report(self) -> dict[str, object]:
        return {
            "cluster_id": self.id,
            "member_proposal_ids": list(self.proposal_ids),
            "source_ids": list(self.source_ids),
            "member_count": len(self.members),
            "source_count": len(set(self.source_ids)),
            "representative_proposal_ids": list(
                self.representative_proposal_ids
            ),
            "global_similarity": {
                "min": self.global_similarity_min,
                "mean": self.global_similarity_mean,
                "max": self.global_similarity_max,
            },
        }


def _cluster_id(member_ids: Sequence[str]) -> str:
    digest = hashlib.sha256("\n".join(sorted(member_ids)).encode("utf-8")).hexdigest()
    return f"clu_{digest[:24]}"


def _global_similarity(
    first: FingerprintedCandidate,
    second: FingerprintedCandidate,
) -> float:
    return float(
        np.clip(
            first.data.global_embedding @ second.data.global_embedding,
            -1.0,
            1.0,
        )
    )


def cluster_candidates(
    candidates: Sequence[FingerprintedCandidate],
    settings: VisualFingerprintConfig,
) -> list[CandidateCluster]:
    """Greedily merge strong cross-image similarities without source collisions."""

    items = sorted(candidates, key=lambda item: item.proposal.id)
    if not items:
        return []
    parents = list(range(len(items)))
    source_sets = [{item.proposal.source_image_id} for item in items]

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    edges: list[tuple[float, int, int]] = []
    for first_index, first in enumerate(items):
        for second_index in range(first_index + 1, len(items)):
            second = items[second_index]
            if first.proposal.source_image_id == second.proposal.source_image_id:
                continue
            similarity = _global_similarity(first, second)
            if similarity >= settings.cluster_global_similarity_threshold:
                edges.append((similarity, first_index, second_index))
    edges.sort(
        key=lambda item: (
            -item[0],
            items[item[1]].proposal.id,
            items[item[2]].proposal.id,
        )
    )
    for _, first_index, second_index in edges:
        first_root = find(first_index)
        second_root = find(second_index)
        if first_root == second_root:
            continue
        if source_sets[first_root].intersection(source_sets[second_root]):
            continue
        parents[second_root] = first_root
        source_sets[first_root].update(source_sets[second_root])

    grouped: dict[int, list[FingerprintedCandidate]] = {}
    for index, item in enumerate(items):
        grouped.setdefault(find(index), []).append(item)

    clusters: list[CandidateCluster] = []
    for members in grouped.values():
        members.sort(key=lambda item: item.proposal.id)
        pairwise: list[float] = []
        centrality: dict[str, list[float]] = {
            item.proposal.id: [] for item in members
        }
        for first_index, first in enumerate(members):
            for second in members[first_index + 1 :]:
                similarity = _global_similarity(first, second)
                pairwise.append(similarity)
                centrality[first.proposal.id].append(similarity)
                centrality[second.proposal.id].append(similarity)
        ranked = sorted(
            members,
            key=lambda item: (
                -(
                    fmean(centrality[item.proposal.id])
                    if centrality[item.proposal.id]
                    else 1.0
                ),
                -item.proposal.score,
                item.proposal.id,
            ),
        )
        similarities = pairwise or [1.0]
        clusters.append(
            CandidateCluster(
                id=_cluster_id([item.proposal.id for item in members]),
                members=tuple(members),
                representative_proposal_ids=tuple(
                    item.proposal.id
                    for item in ranked[: settings.max_cluster_representatives]
                ),
                global_similarity_min=float(min(similarities)),
                global_similarity_mean=float(fmean(similarities)),
                global_similarity_max=float(max(similarities)),
            )
        )
    clusters.sort(key=lambda cluster: (-len(cluster.members), cluster.id))
    return clusters


def cluster_historical_evidence(
    cluster: CandidateCluster,
    historical: Sequence[HistoricalFingerprint],
    settings: VisualFingerprintConfig,
) -> VisualEvidence:
    """Use the strongest non-conflicting member match as cluster history evidence."""

    evidences = [
        match_fingerprint(member.data, historical, settings)
        for member in cluster.members
    ]
    matches = [
        evidence
        for evidence in evidences
        if evidence.result is VisualMatchType.MATCH
    ]
    matched_object_ids = {
        evidence.matched_object_id for evidence in matches if evidence.matched_object_id
    }
    if len(matched_object_ids) > 1:
        best = max(
            matches,
            key=lambda item: (
                item.visual_score if item.visual_score is not None else -1.0
            ),
        )
        selected = VisualEvidence(
            result=VisualMatchType.AMBIGUOUS,
            global_similarity=best.global_similarity,
            local_match_ratio=best.local_match_ratio,
            visual_score=best.visual_score,
            second_best_score=best.second_best_score,
            score_margin=best.score_margin,
            object_scores=best.object_scores,
        )
    elif matches:
        selected = max(
            matches,
            key=lambda item: (
                item.visual_score if item.visual_score is not None else -1.0
            ),
        )
    else:
        ambiguous = [
            evidence
            for evidence in evidences
            if evidence.result is VisualMatchType.AMBIGUOUS
        ]
        selected = max(
            ambiguous or evidences,
            key=lambda item: item.visual_score if item.visual_score is not None else -1.0,
        )
    return selected.model_copy(
        update={
            "cluster_id": cluster.id,
            "cluster_member_proposal_ids": list(cluster.proposal_ids),
            "cluster_global_similarity_min": cluster.global_similarity_min,
            "cluster_global_similarity_mean": cluster.global_similarity_mean,
            "cluster_global_similarity_max": cluster.global_similarity_max,
        }
    )


def write_cluster_contact_sheet(
    cluster: CandidateCluster,
    *,
    run_id: str,
    paths: MemoryPaths,
    cell_size: int,
) -> str:
    """Create one Qwen/Web board with isolated crops and source context."""

    representatives = {
        item.proposal.id: item for item in cluster.members
    }
    selected = [
        representatives[proposal_id]
        for proposal_id in cluster.representative_proposal_ids
    ]
    margin = 12
    label_height = 28
    row_height = cell_size + label_height
    width = cell_size * 2 + margin * 3
    height = margin + row_height * len(selected) + margin
    sheet = Image.new("RGB", (width, height), (238, 240, 243))
    draw = ImageDraw.Draw(sheet)
    for row_index, item in enumerate(selected):
        proposal = item.proposal
        if not proposal.crop_path or not proposal.overlay_path:
            raise ValueError("Cluster representative lacks crop or overlay asset")
        top = margin + row_index * row_height
        label = (
            f"{proposal.id} | source={proposal.source_image_id} | "
            f"sam={proposal.score:.3f}"
        )
        draw.text((margin, top), label, fill=(28, 32, 38))
        for column, relative_path in enumerate(
            (proposal.crop_path, proposal.overlay_path)
        ):
            with Image.open(paths.resolve_asset(relative_path)) as opened:
                image = opened.convert("RGB")
                contained = ImageOps.contain(
                    image,
                    (cell_size, cell_size),
                    Image.Resampling.LANCZOS,
                )
            cell = Image.new("RGB", (cell_size, cell_size), (127, 127, 127))
            left = (cell_size - contained.width) // 2
            image_top = (cell_size - contained.height) // 2
            cell.paste(contained, (left, image_top))
            x = margin + column * (cell_size + margin)
            sheet.paste(cell, (x, top + label_height))
    output_dir = paths.cluster_dir(run_id, cluster.id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "contact_sheet.jpg"
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    try:
        sheet.save(temporary, format="JPEG", quality=92)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return paths.relative_asset(output)
