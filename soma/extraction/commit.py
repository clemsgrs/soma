"""Chunked cache commits shared by the extraction paths.

slide2vec persists every embedding as it is produced and resumes on sidecar
existence, but soma only trusts a payload once its identity signature is
recorded in the cache metadata — and unsigned payloads are deleted on the next
run. Committing signatures once per *chunk* of work, rather than once after the
whole extraction, bounds what an interrupted run has to redo to one chunk.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeVar

T = TypeVar("T")

#: Pre-cropped images per commit on the tile-image path (``Model.embed_images``).
DEFAULT_IMAGE_COMMIT_EVERY = 1024
#: Slides per commit *per GPU* on the WSI tile/hierarchical paths: each chunk is one
#: slide2vec pipeline call (one model load), so committing every slide would reload
#: the encoder per slide.
DEFAULT_SLIDES_PER_GPU_COMMIT_EVERY = 8


def resolve_commit_every(commit_every: int | None, *, default: int) -> int:
    """Return the configured chunk size, or ``default`` when unset."""
    if commit_every is None:
        return int(default)
    if int(commit_every) < 1:
        raise ValueError(f"cache.commit_every must be >= 1, got {commit_every!r}")
    return int(commit_every)


def commit_chunks(items: Sequence[T], chunk_size: int) -> Iterator[list[T]]:
    """Yield ``items`` in consecutive chunks of at most ``chunk_size``."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    for start in range(0, len(items), chunk_size):
        yield list(items[start : start + chunk_size])
