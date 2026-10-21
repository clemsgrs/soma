"""Detection (cell / nucleus point detection) — the grid shape's point scoring.

Pure, weight-free pieces the :class:`soma.models.heads.detection.DetectionHead` and
:func:`soma.evaluate` compose, each unit-testable on tiny explicit inputs:

* :mod:`soma.evaluation.detection.peaks` — local-maxima + NMS + threshold peak extraction
  that turns a predicted heatmap back into ``(x, y, class, score)`` points.
* :mod:`soma.evaluation.detection.matching` — class-aware F1@δ point matching (Hungarian /
  greedy), per-image TP/FP/FN accumulation, the global + per-image-macro reduce, the
  per-class threshold sweep, and the one-pass greedy sweep (per-image score histograms
  read at every threshold) that training-time validation uses.
* :mod:`soma.evaluation.detection.io` — point-file and ignore-mask readers.

The target side (level-0 <-> target-frame coordinate transforms and the point -> per-class
peak-heatmap encoder) is :mod:`soma.models.detection_targets`.

Protocol scorers (MIDOG F1, Monkey FROC) and the attention-map rung live with their
benchmarks under :mod:`soma.benchmark`.
"""

from __future__ import annotations

from soma.evaluation.detection.matching import (
    SCORE_BINS,
    ClassMatch,
    counts_at_thresholds,
    detection_counts,
    greedy_score_histogram,
    histogram_thresholds,
    match_assignment,
    match_points,
    reduce_f1,
    sweep_score_histograms,
    sweep_score_thresholds,
)
from soma.evaluation.detection.peaks import extract_peaks

__all__ = [
    "extract_peaks",
    "ClassMatch",
    "match_assignment",
    "match_points",
    "detection_counts",
    "reduce_f1",
    "sweep_score_thresholds",
    "SCORE_BINS",
    "greedy_score_histogram",
    "counts_at_thresholds",
    "sweep_score_histograms",
    "histogram_thresholds",
]
