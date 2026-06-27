"""Dense-mode description helper (soma-only).

The sliding-window dense-encoding primitives (``cover_origins``,
``resolve_window_geometry``, ``encode_dense_sliding``) now live in slide2vec's unified
streaming dense module :mod:`slide2vec.runtime.dense_sliding`; soma imports them from
there. Only :func:`describe_dense_mode` — a soma-side logging helper — remains here.
"""

from __future__ import annotations

__all__ = ["describe_dense_mode"]


def describe_dense_mode(window_size: int | None, overlap: float) -> str:
    """Human-readable resolved dense-input mode, for logging (never silent)."""
    if window_size is None:
        return "whole (single padded forward)"
    return f"sliding_window (window={int(window_size)}, overlap={float(overlap)})"
