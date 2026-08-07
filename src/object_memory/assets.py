"""Safe, portable paths for memory assets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def _validate_segment(value: str, label: str) -> str:
    if not _SAFE_SEGMENT.fullmatch(value):
        raise ValueError(f"{label} contains unsafe path characters: {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class MemoryPaths:
    """Canonical directories below one movable memory root."""

    root: Path
    database_filename: str = "memory.sqlite"

    def __post_init__(self) -> None:
        resolved_root = self.root.expanduser().resolve()
        object.__setattr__(self, "root", resolved_root)
        if Path(self.database_filename).name != self.database_filename:
            raise ValueError("database_filename must not contain directories")

    @property
    def database(self) -> Path:
        return self.root / self.database_filename

    @property
    def sources(self) -> Path:
        return self.root / "sources"

    @property
    def proposals(self) -> Path:
        return self.root / "proposals"

    @property
    def objects(self) -> Path:
        return self.root / "objects"

    @property
    def clusters(self) -> Path:
        return self.root / "clusters"

    @property
    def raw_responses(self) -> Path:
        return self.root / "raw_responses"

    @property
    def run_reports(self) -> Path:
        return self.root / "run_reports"

    def ensure_layout(self) -> None:
        """Create the fixed directory skeleton without deleting existing data."""

        for directory in (
            self.root,
            self.sources,
            self.proposals,
            self.objects,
            self.clusters,
            self.raw_responses,
            self.run_reports,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def proposal_dir(self, run_id: str, proposal_id: str) -> Path:
        run_id = _validate_segment(run_id, "run_id")
        proposal_id = _validate_segment(proposal_id, "proposal_id")
        return self.proposals / run_id / proposal_id

    def observation_dir(self, object_id: str, observation_id: str) -> Path:
        object_id = _validate_segment(object_id, "object_id")
        observation_id = _validate_segment(observation_id, "observation_id")
        return self.objects / object_id / "observations" / observation_id

    def cluster_dir(self, run_id: str, cluster_id: str) -> Path:
        run_id = _validate_segment(run_id, "run_id")
        cluster_id = _validate_segment(cluster_id, "cluster_id")
        return self.clusters / run_id / cluster_id

    def raw_response_dir(self, run_id: str, scope_id: str) -> Path:
        run_id = _validate_segment(run_id, "run_id")
        scope_id = _validate_segment(scope_id, "scope_id")
        return self.raw_responses / run_id / scope_id

    def resolve_asset(self, relative_path: str) -> Path:
        """Resolve a stored POSIX relative path and prevent root traversal."""

        if "\\" in relative_path:
            raise ValueError("Stored asset paths must use forward slashes")
        candidate = PurePosixPath(relative_path)
        if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
            raise ValueError(f"Invalid relative asset path: {relative_path!r}")
        resolved = (self.root / Path(*candidate.parts)).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"Asset path escapes the memory root: {relative_path!r}")
        return resolved

    def relative_asset(self, path: str | Path) -> str:
        """Convert a path below the root to its portable POSIX representation."""

        resolved = Path(path).expanduser().resolve()
        try:
            relative = resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Asset is outside the memory root: {resolved}") from exc
        return relative.as_posix()
