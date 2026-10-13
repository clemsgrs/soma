"""Soma-owned dense-mode and outer prediction-canvas helpers.

Encoder-window geometry and encoding live behind slide2vec's public ``DenseEncodeKit``.
The outer segmentation prediction canvas remains Soma's responsibility, including its
small cover rule below.
"""

from __future__ import annotations

__all__ = ["cover_origins", "describe_dense_mode"]


def cover_origins(extent: int, size: int, stride: int) -> list[int]:
    """Window starts covering an outer pixel canvas, including a flush final edge."""
    if size >= extent:
        return [0]
    starts = list(range(0, extent - size + 1, stride))
    if starts[-1] + size < extent:
        starts.append(extent - size)
    return starts


def describe_dense_mode(window_size: int | None, overlap: float) -> str:
    """Human-readable resolved dense-input mode, for logging (never silent)."""
    if window_size is None:
        return "whole (single padded forward)"
    return f"sliding_window (window={int(window_size)}, overlap={float(overlap)})"
