"""Structured progress events for feature extraction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ProgressEvent:
    """A single progress event emitted during extraction."""

    kind: str
    timestamp: float
    rank: int
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProgressEvent:
        return cls(**data)


@runtime_checkable
class ProgressReporter(Protocol):
    """Protocol for progress reporting."""

    def emit(self, event: ProgressEvent) -> None: ...


class JsonlProgressReporter:
    """Writes one JSON line per event to {output_dir}/.progress/rank_{rank}.jsonl."""

    def __init__(self, output_dir: Path, rank: int) -> None:
        progress_dir = Path(output_dir) / ".progress"
        progress_dir.mkdir(parents=True, exist_ok=True)
        self._path = progress_dir / f"rank_{rank}.jsonl"
        self._file = open(self._path, "a")

    def emit(self, event: ProgressEvent) -> None:
        self._file.write(json.dumps(event.to_dict()) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class NullProgressReporter:
    """No-op reporter."""

    def emit(self, event: ProgressEvent) -> None:
        pass
