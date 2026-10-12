"""Detection (cell / nucleus point detection) — the grid shape's point machinery.

Pure, weight-free pieces the :class:`soma.tasks.detection.DetectionHead` and
:func:`soma.evaluate` compose, each unit-testable on tiny explicit inputs:

* :mod:`soma.detection.encode` — level-0 <-> target-frame coordinate transforms and the
  point -> per-class peak-heatmap target encoder.
* :mod:`soma.detection.peaks` — local-maxima + NMS + threshold peak extraction that turns
  a predicted heatmap back into ``(x, y, class, score)`` points.
* :mod:`soma.detection.matching` — class-aware F1@δ point matching (Hungarian / greedy),
  per-image TP/FP/FN accumulation, the global + per-image-macro reduce, and the per-class
  threshold sweep.
* :mod:`soma.detection.io` — point-file and ignore-mask readers.

Protocol scorers (MIDOG F1, Monkey FROC) and the attention-map rung live with their
benchmarks under :mod:`soma.benchmarks`.
"""

from __future__ import annotations

from soma.detection.encode import (
    render_peak_heatmap,
    transform_points_to_level0,
    transform_points_to_target,
)
from soma.detection.matching import (
    ClassMatch,
    detection_counts,
    match_assignment,
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
    "ClassMatch",
    "match_assignment",
    "match_points",
    "detection_counts",
    "reduce_f1",
    "sweep_score_thresholds",
]
