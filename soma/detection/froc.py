"""FROC — Free-response ROC, the Monkey challenge's native detection metric.

FROC generalizes the F1@δ point matcher already in :mod:`soma.detection.matching`:
instead of freezing one score threshold and reporting a single F1, it **sweeps** the
detection-confidence threshold and plots *sensitivity* (recall) against the mean number
of **false positives per mm²**, then reports the mean sensitivity read off that curve at
a fixed set of FP/mm² operating points. F1@δ is the one-threshold special case.

The Monkey challenge (kidney biopsies, PAS stain; lymphocytes / monocytes + the merged
"inflammatory-cells" / MNL class) scores each class with this curve and averages the
per-class FROC for the ranking. Because the x-axis is *per mm²*, the scorer needs the
physical area of the evaluated region — threaded in as ``area_mm2`` (derived from the run
spacing, see :func:`patch_area_mm2`).

Everything operates on plain numpy arrays per image so it unit-tests against hand-built
point sets with known sensitivity / FP-per-mm². Matching reuses the greedy-by-confidence
matcher (:func:`soma.detection.matching._match_single_class`) — the canonical FROC
assignment (CAMELYON16 / LUNA16), where each prediction, in descending-score order,
claims the nearest unclaimed ground-truth point within δ. A matched prediction is a true
positive, an unmatched one a false positive; the labelling is threshold-independent
(sweeping only *removes* the lowest-scoring predictions), which is exactly what lets the
curve be built by one sweep.

The curve math (:func:`froc_curve` / :func:`froc_score_at_thresholds`) mirrors MONAI's
``compute_froc_curve_data`` / ``compute_froc_score`` (the Monkey reference evaluator's
backend) so the produced number tracks the published leaderboard band — which the design
treats as a *reference band* ("different test set"), not an aligned leaderboard row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soma.detection.matching import _match_single_class

__all__ = [
    "DEFAULT_FP_THRESHOLDS",
    "label_detections",
    "froc_curve",
    "froc_score_at_thresholds",
    "compute_froc",
    "FrocResult",
    "patch_area_mm2",
    "score_monkey_froc",
    "MONKEY_CLASS_NAMES",
    "MONKEY_MATCH_UM",
    "MNL_CLASS_NAME",
    "MNL_MATCH_UM",
]

# Monkey's FROC operating points: the FP/mm² values at which the swept sensitivity is
# read off and averaged into the single FROC score (challenge evaluate.py).
DEFAULT_FP_THRESHOLDS: tuple[float, ...] = (10.0, 20.0, 50.0, 100.0, 200.0, 300.0)

# Monkey's two object classes (0-based ids match the curated points CSV) and their
# per-class matching margins in µm (cell-size-based: lymphocytes 4µm, monocytes 5µm).
MONKEY_CLASS_NAMES: tuple[str, ...] = ("lymphocytes", "monocytes")
MONKEY_MATCH_UM: tuple[float, ...] = (4.0, 5.0)

# The merged "MNL" / inflammatory-cells leaderboard: both classes pooled into one, scored
# at the 5µm inflammation margin.
MNL_CLASS_NAME = "inflammatory-cells"
MNL_MATCH_UM = 5.0


def label_detections(
    pred_xy: np.ndarray,
    pred_score: np.ndarray,
    gt_xy: np.ndarray,
    *,
    delta: float,
    method: str = "greedy",
) -> tuple[np.ndarray, np.ndarray]:
    """Label one class's predictions as TP/FP against its GT; return their scores.

    Greedy-by-confidence matching within ``delta`` (same length unit as the points):
    a matched prediction's score joins ``tp_scores``, an unmatched one's joins
    ``fp_scores``. Scores are returned in input order. ``method`` is passed through to
    the shared matcher (``"greedy"`` is FROC-canonical; ``"hungarian"`` also accepted).
    """
    pred_xy = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
    gt_xy = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    pred_score = np.asarray(pred_score, dtype=np.float64).reshape(-1)
    if pred_xy.shape[0] != pred_score.shape[0]:
        raise ValueError("pred_xy and pred_score must share length.")

    match = _match_single_class(pred_xy, pred_score, gt_xy, delta=float(delta), method=method)
    matched = np.zeros(pred_xy.shape[0], dtype=bool)
    if match.pairs.shape[0]:
        matched[match.pairs[:, 0]] = True
    return pred_score[matched], pred_score[~matched]


def froc_curve(
    tp_scores: np.ndarray,
    fp_scores: np.ndarray,
    *,
    num_targets: int,
    area_mm2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the (FP/mm², sensitivity) sweep from labelled TP/FP prediction scores.

    Follows MONAI's ``compute_froc_curve_data`` (the Monkey reference backend): for each
    unique score threshold, count predictions scoring ``>= threshold`` as TP or FP, then
    anchor the most-exclusive end with a trailing ``(0, 0)`` point. FP counts are divided
    by ``area_mm2`` (so the x-axis is per mm²) and TP counts by ``num_targets`` (so the
    y-axis is sensitivity). Both arrays are returned in descending-threshold order (the
    most-inclusive, highest-sensitivity operating point first).

    One deliberate deviation from MONAI: it sweeps **every** unique threshold, whereas
    MONAI drops the single lowest score (``all_probs[1:]``) and then special-cases the
    resulting degenerate one-point curve. Including the most-inclusive point makes a lone
    detection behave correctly (a single true positive → sensitivity 1; a single false
    positive → sensitivity 0) with no special case, and shifts the aggregate on real data
    by at most one detection — immaterial against the reference *band*.

    ``num_targets`` is the total ground-truth count; ``area_mm2`` the physical area of the
    evaluated region. Both must be positive.
    """
    tp_scores = np.asarray(tp_scores, dtype=np.float64).reshape(-1)
    fp_scores = np.asarray(fp_scores, dtype=np.float64).reshape(-1)
    if num_targets <= 0:
        raise ValueError(f"num_targets must be > 0, got {num_targets}.")
    if area_mm2 <= 0:
        raise ValueError(f"area_mm2 must be > 0, got {area_mm2}.")

    all_probs = sorted(set(fp_scores.tolist()) | set(tp_scores.tolist()))
    total_fps: list[float] = []
    total_tps: list[float] = []
    for thresh in all_probs:
        total_fps.append(float((fp_scores >= thresh).sum()))
        total_tps.append(float((tp_scores >= thresh).sum()))
    total_fps.append(0.0)  # most-exclusive anchor: threshold above every score
    total_tps.append(0.0)

    fp_per_mm2 = np.asarray(total_fps, dtype=np.float64) / float(area_mm2)
    sensitivity = np.asarray(total_tps, dtype=np.float64) / float(num_targets)
    return fp_per_mm2, sensitivity


def froc_score_at_thresholds(
    fp_per_mm2: np.ndarray,
    sensitivity: np.ndarray,
    *,
    fp_thresholds: tuple[float, ...] = DEFAULT_FP_THRESHOLDS,
) -> float:
    """Mean sensitivity read off the FROC curve at each FP/mm² operating point.

    Mirrors MONAI's ``compute_froc_score``: linearly interpolate ``sensitivity`` at each
    value in ``fp_thresholds`` along the FP/mm² axis, then average. ``fp_per_mm2`` /
    ``sensitivity`` are the descending-threshold arrays :func:`froc_curve` returns; they
    are reversed here to the ascending x-axis :func:`numpy.interp` requires. Operating
    points beyond the curve's max FP/mm² clamp to the highest sensitivity attained.
    """
    fp_per_mm2 = np.asarray(fp_per_mm2, dtype=np.float64).reshape(-1)
    sensitivity = np.asarray(sensitivity, dtype=np.float64).reshape(-1)
    interp = np.interp(np.asarray(fp_thresholds, dtype=np.float64), fp_per_mm2[::-1], sensitivity[::-1])
    return float(np.mean(interp))


@dataclass(frozen=True)
class FrocResult:
    """One class's FROC outcome: the headline ``score`` plus the curve it was read off.

    ``fp_per_mm2`` / ``sensitivity`` are the descending-threshold sweep; ``num_targets``
    the pooled ground-truth count and ``area_mm2`` the pooled evaluated area (both over
    all images), retained so a caller can re-derive the score at other operating points
    or re-aggregate.
    """

    score: float
    fp_per_mm2: np.ndarray
    sensitivity: np.ndarray
    num_targets: int
    area_mm2: float


def compute_froc(
    per_image_pred_xy: list[np.ndarray],
    per_image_pred_score: list[np.ndarray],
    per_image_gt_xy: list[np.ndarray],
    *,
    delta: float,
    per_image_area_mm2: list[float],
    method: str = "greedy",
    fp_thresholds: tuple[float, ...] = DEFAULT_FP_THRESHOLDS,
) -> FrocResult:
    """Single-class FROC over a set of images (the Monkey aggregation is dataset-global).

    Each image is matched independently (its own point frame), then the TP/FP scores are
    **pooled** across images and the ground-truth counts and areas summed, so one curve is
    built over the whole set (micro-averaging, matching the Monkey reference evaluator).
    ``delta`` is the matching distance in the points' length unit; ``per_image_area_mm2``
    the evaluated area of each image in mm².

    An empty set (no ground truth anywhere) scores ``0`` with a flat ``(0, 0)`` curve.
    """
    n = len(per_image_pred_xy)
    if not (len(per_image_pred_score) == len(per_image_gt_xy) == len(per_image_area_mm2) == n):
        raise ValueError("per-image inputs must all share the same length.")

    tp_chunks: list[np.ndarray] = []
    fp_chunks: list[np.ndarray] = []
    num_targets = 0
    for pxy, ps, gxy in zip(per_image_pred_xy, per_image_pred_score, per_image_gt_xy):
        tp, fp = label_detections(pxy, ps, gxy, delta=float(delta), method=method)
        tp_chunks.append(tp)
        fp_chunks.append(fp)
        num_targets += int(np.asarray(gxy, dtype=np.float64).reshape(-1, 2).shape[0])
    area_mm2 = float(sum(per_image_area_mm2))

    empty = np.zeros((1,), dtype=np.float64)
    if num_targets == 0 or area_mm2 <= 0:
        return FrocResult(
            score=0.0, fp_per_mm2=empty * 0.0, sensitivity=empty * 0.0,
            num_targets=num_targets, area_mm2=area_mm2,
        )

    tp_scores = np.concatenate(tp_chunks) if tp_chunks else np.zeros((0,))
    fp_scores = np.concatenate(fp_chunks) if fp_chunks else np.zeros((0,))
    fp_per_mm2, sensitivity = froc_curve(
        tp_scores, fp_scores, num_targets=num_targets, area_mm2=area_mm2
    )
    score = froc_score_at_thresholds(fp_per_mm2, sensitivity, fp_thresholds=fp_thresholds)
    return FrocResult(
        score=score, fp_per_mm2=fp_per_mm2, sensitivity=sensitivity,
        num_targets=num_targets, area_mm2=area_mm2,
    )


def patch_area_mm2(width_px: int, height_px: int, spacing_um: float) -> float:
    """Physical area of a ``width_px × height_px`` region at ``spacing_um`` µm/px, in mm².

    The FROC x-axis is per mm², so a caller evaluating flat patches (no ROI polygon)
    derives each patch's area this way from the run spacing; ROI-based datasets pass the
    annotated ROI area directly.
    """
    return (float(width_px) * float(spacing_um) / 1000.0) * (float(height_px) * float(spacing_um) / 1000.0)


def score_monkey_froc(
    per_image_pred_xy: list[np.ndarray],
    per_image_pred_class: list[np.ndarray],
    per_image_pred_score: list[np.ndarray],
    per_image_gt_xy: list[np.ndarray],
    per_image_gt_class: list[np.ndarray],
    per_image_area_mm2: list[float],
    *,
    spacing_um: float,
    class_names: tuple[str, ...] = MONKEY_CLASS_NAMES,
    match_um: tuple[float, ...] = MONKEY_MATCH_UM,
    mnl_match_um: float = MNL_MATCH_UM,
    fp_thresholds: tuple[float, ...] = DEFAULT_FP_THRESHOLDS,
) -> dict[str, FrocResult | float]:
    """Score Monkey detection with FROC: the 2-class **and** merged-MNL paths at once.

    Points live in the level-0 pixel frame; ``spacing_um`` (µm/px) converts each class's
    µm matching margin to the pixel ``delta`` the matcher uses (threading the physical
    spacing FROC's per-mm² axis needs). ``per_image_pred_class`` / ``per_image_gt_class``
    hold 0-based class ids aligned to :data:`MONKEY_CLASS_NAMES`.

    Returns a dict with one :class:`FrocResult` per class name, a ``mean_froc`` float (the
    Task-2 ranking = mean of the per-class scores), and one :class:`FrocResult` under
    :data:`MNL_CLASS_NAME` for the merged inflammatory-cells (Task-1) leaderboard.
    """
    if spacing_um <= 0:
        raise ValueError(f"spacing_um must be > 0, got {spacing_um}.")

    def _filter(per_image_xy, per_image_cls, keep):
        return [xy[np.asarray(cls).reshape(-1) == keep] for xy, cls in zip(per_image_xy, per_image_cls)]

    result: dict[str, FrocResult | float] = {}
    per_class_scores: list[float] = []
    for cls_idx, name in enumerate(class_names):
        delta_px = float(match_um[cls_idx]) / float(spacing_um)
        froc = compute_froc(
            per_image_pred_xy=_filter(per_image_pred_xy, per_image_pred_class, cls_idx),
            per_image_pred_score=[
                np.asarray(s).reshape(-1)[np.asarray(c).reshape(-1) == cls_idx]
                for s, c in zip(per_image_pred_score, per_image_pred_class)
            ],
            per_image_gt_xy=_filter(per_image_gt_xy, per_image_gt_class, cls_idx),
            delta=delta_px,
            per_image_area_mm2=per_image_area_mm2,
            fp_thresholds=fp_thresholds,
        )
        result[name] = froc
        per_class_scores.append(froc.score)
    result["mean_froc"] = float(np.mean(per_class_scores)) if per_class_scores else 0.0

    # MNL: pool both classes into one merged inflammatory-cells detection problem.
    result[MNL_CLASS_NAME] = compute_froc(
        per_image_pred_xy=per_image_pred_xy,
        per_image_pred_score=per_image_pred_score,
        per_image_gt_xy=per_image_gt_xy,
        delta=float(mnl_match_um) / float(spacing_um),
        per_image_area_mm2=per_image_area_mm2,
        fp_thresholds=fp_thresholds,
    )
    return result
