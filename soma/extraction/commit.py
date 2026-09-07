"""Chunked commits for pre-cropped images extracted with ``Model.embed_images``.

Image signatures commit once per chunk to bound interruption loss. WSI tile and
hierarchical extraction instead commit each slide through ``on_slide_persisted``
within one pipeline call and do not use these chunk settings.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeVar

T = TypeVar("T")

#: Pre-cropped images per commit on the tile-image path (``Model.embed_images``).
DEFAULT_IMAGE_COMMIT_EVERY = 1024


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
