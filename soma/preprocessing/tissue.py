"""Thin wrappers over hs2p's shared tissue preprocessing core."""

from hs2p.preprocessing import ContourResult, detect_contours, segment_tissue

__all__ = [
    "ContourResult",
    "detect_contours",
    "segment_tissue",
]
