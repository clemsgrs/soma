"""Detection (cell / nucleus point detection) — dense-feature heatmap path.

detection-v1 reuses the segmentation front half verbatim (frozen encoder →
``encode_tiles_dense`` → ``(d, h, w)`` cache → ``decoder`` → dense head); only the
output representation is detection-specific. This package owns that new machinery,
kept free of the torch ``TaskHead`` so each piece unit-tests on tiny explicit inputs:

* :mod:`soma.detection.encode` — level-0 → target-frame coordinate transform and the
  point → per-class peak-heatmap target encoder (design §4, §5).
* :mod:`soma.detection.peaks` — local-maxima + NMS + threshold peak extraction that
  turns a predicted heatmap back into ``(x, y, class, score)`` points (design §6).
* :mod:`soma.detection.matching` — class-aware F1@δ point matching (Hungarian /
  greedy), per-image TP/FP/FN accumulation, the global + per-image-macro reduce, and
  the per-class tune-split threshold sweep (design §7).

The head that wires these into the soma task contract is
:class:`soma.tasks.detection.DetectionHead`.
"""

from __future__ import annotations

from soma.detection.encode import (
    render_peak_heatmap,
    transform_points_to_level0,
    transform_points_to_target,
)
from soma.detection.matching import (
    detection_counts,
    match_points,
    reduce_f1,
    sweep_score_thresholds,
)
from soma.detection.peaks import extract_peaks

__all__ = [
    "transform_points_to_target",
    "transform_points_to_level0",
    "render_peak_heatmap",
    "extract_peaks",
    "match_points",
    "detection_counts",
    "reduce_f1",
    "sweep_score_thresholds",
]
