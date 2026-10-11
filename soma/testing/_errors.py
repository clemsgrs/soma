"""The conformance failure type shared by every ``check_*`` suite."""

from __future__ import annotations


class ConformanceError(AssertionError):
    """A component violates its protocol; ``method`` and ``sample_id`` locate the breach."""

    def __init__(self, method: str, message: str, *, sample_id: str | None = None) -> None:
        self.method = method
        self.sample_id = sample_id
        where = f"{method}({sample_id!r})" if sample_id is not None else method
        super().__init__(f"{where}: {message}")
