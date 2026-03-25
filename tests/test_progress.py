"""Tests for soma.encoders.progress — structured progress events."""

from __future__ import annotations

import json
from pathlib import Path

from soma.encoders.progress import (
    JsonlProgressReporter,
    NullProgressReporter,
    ProgressEvent,
)


class TestProgressEvent:
    def test_serialization_roundtrip(self):
        event = ProgressEvent(
            kind="extraction.slide.completed",
            timestamp=1234567890.123,
            rank=0,
            payload={"slide_id": "slide_001", "num_tiles": 64, "duration_s": 1.5},
        )
        data = event.to_dict()
        restored = ProgressEvent.from_dict(data)
        assert restored.kind == event.kind
        assert restored.timestamp == event.timestamp
        assert restored.rank == event.rank
        assert restored.payload == event.payload


class TestJsonlProgressReporter:
    def test_writes_events(self, tmp_path: Path):
        reporter = JsonlProgressReporter(tmp_path, rank=0)
        for i in range(3):
            reporter.emit(
                ProgressEvent(
                    kind=f"test.event.{i}",
                    timestamp=float(i),
                    rank=0,
                    payload={"index": i},
                )
            )
        log_path = tmp_path / ".progress" / "rank_0.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3
        for i, line in enumerate(lines):
            data = json.loads(line)
            assert data["kind"] == f"test.event.{i}"
            assert data["payload"]["index"] == i


class TestNullProgressReporter:
    def test_is_noop(self):
        reporter = NullProgressReporter()
        # Should not raise
        reporter.emit(
            ProgressEvent(
                kind="test.event",
                timestamp=0.0,
                rank=0,
                payload={},
            )
        )
