"""F1 at matching distance δ — point-matching detection evaluator (design §7).

The genuinely novel evaluator. Predicted points are matched to ground-truth points
**per class** (class-aware: a prediction only matches a GT of the same class within
δ) using either optimal one-to-one **Hungarian** assignment (default) or
**greedy-by-confidence** (OCELOT's official scorer; emit on demand for a comparable
leaderboard number). Matched pairs are TP, unmatched predictions FP, unmatched GT FN.

Everything operates on plain numpy arrays per image so it unit-tests against
hand-built point sets with known TP/FP/FN. :func:`detection_counts` produces the
compact ``(C, 3)`` per-image stat the streaming evaluator accumulates;
:func:`reduce_f1` reduces the concatenated ``(N, C, 3)`` to the global-headline +
per-image-macro mF1; :func:`sweep_score_thresholds` picks the per-class score
threshold on the tune split.

**One-pass greedy sweep.** Under greedy matching, whether a prediction is a hit does not
depend on the threshold: lowering it only appends lower-scored predictions after the
higher ones. So one matching pass over every prediction, binned by score
(:func:`greedy_score_histogram`), gives the exact counts at every threshold
(:func:`counts_at_thresholds`) and the per-class best F1 (:func:`sweep_score_histograms`).

**Sparse.** Only the prediction and annotation pairs within ``delta`` are listed (a
KD-tree query), so matching a whole slide's points never builds the full distance matrix.
Hungarian matching solves each connected group of these pairs on its own; greedy matching
walks each prediction's pairs in the dense matcher's order, so it makes the same pairs.

All distances are in the same (target-frame) pixel units as ``delta`` — the head
resolves ``delta`` from µm/px via the run spacing before calling in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, min_weight_full_bipartite_matching
from scipy.spatial import cKDTree
from torch import Tensor

__all__ = [
    "ClassMatch",
    "match_assignment",
    "match_points",
    "match_single_class",
    "detection_counts",
    "reduce_f1",
    "sweep_score_thresholds",
    "greedy_score_histogram",
    "counts_at_thresholds",
    "sweep_score_histograms",
    "histogram_thresholds",
    "SCORE_BINS",
    "VALID_MATCHING",
]

VALID_MATCHING = ("hungarian", "greedy")

#: Score bins of the one-pass greedy sweep: its candidate thresholds are ``k / SCORE_BINS``.
SCORE_BINS = 1000


@dataclass(frozen=True)
class ClassMatch:
    """One class's point-matching assignment — the primitive the counts derive from.

    ``pairs`` is an ``(L, 2)`` int array of matched ``(pred_idx, gt_idx)`` pairs and
    ``n_pred`` / ``n_gt`` the number of predictions / GT in the class. ``tp`` is the
    number of matched pairs, ``fp`` the unmatched predictions, ``fn`` the unmatched GT —
    derived here, in one place, so a ``(tp, fp, fn)`` count and a match-status overlay
    that reads ``pairs`` cannot disagree. The match overlay reads ``pairs`` directly;
    :func:`match_assignment` returns global indices, so an index points back into the
    original ``pred_*`` / ``gt_*`` arrays (e.g. recovering which class channel fired).
    """

    pairs: np.ndarray  # (L, 2) int64 matched (pred_idx, gt_idx)
    n_pred: int
    n_gt: int

    @property
    def tp(self) -> int:
        return int(self.pairs.shape[0])

    @property
    def fp(self) -> int:
        return self.n_pred - self.tp

    @property
    def fn(self) -> int:
        return self.n_gt - self.tp

    @property
    def counts(self) -> tuple[int, int, int]:
        return self.tp, self.fp, self.fn


def match_single_class(
    pred_xy: np.ndarray,
    pred_score: np.ndarray,
    gt_xy: np.ndarray,
    *,
    delta: float,
    method: str,
) -> ClassMatch:
    """Match one class's predictions to its GT; return the :class:`ClassMatch`.

    ``pairs`` hold indices **local** to the passed (already class-filtered) arrays;
    :func:`match_assignment` lifts them to global indices. ``pred_score`` orders the
    ``greedy`` claims; ``hungarian`` ignores it.
    """
    n_pred = int(pred_xy.shape[0])
    n_gt = int(gt_xy.shape[0])
    empty = np.zeros((0, 2), dtype=np.int64)
    if n_pred == 0 or n_gt == 0:
        return ClassMatch(pairs=empty, n_pred=n_pred, n_gt=n_gt)

    if method not in VALID_MATCHING:
        raise ValueError(f"unknown matching method {method!r}; use one of {VALID_MATCHING}.")
    pred_xy = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
    gt_xy = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    rows, cols, dist = _pairs_in_reach(pred_xy, gt_xy, float(delta))
    if method == "hungarian":
        pairs = _hungarian_pairs(rows, cols, dist, n_pred, n_gt, float(delta))
    else:
        pairs = _greedy_pairs(rows, cols, dist, np.asarray(pred_score, dtype=np.float64).reshape(-1))
    return ClassMatch(pairs=pairs, n_pred=n_pred, n_gt=n_gt)


def _pairs_in_reach(
    pred_xy: np.ndarray, gt_xy: np.ndarray, delta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every ``(pred, gt)`` pair at most ``delta`` apart, and its distance.

    A KD-tree lists the candidates within a radius a hair wider than ``delta``; the
    distance is then computed as ``sqrt(dx**2 + dy**2)``, the dense matrix's arithmetic,
    and the strict ``<= delta`` test is applied to it. So a pair exactly ``delta`` apart
    is kept or dropped exactly as the dense matrix would. Rows are sorted by prediction,
    then annotation.
    """
    radius = delta * (1 + 1e-9) + 1e-9
    found = cKDTree(pred_xy).sparse_distance_matrix(cKDTree(gt_xy), radius, output_type="ndarray")
    rows = found["i"].astype(np.int64)
    cols = found["j"].astype(np.int64)
    dist = np.sqrt(((pred_xy[rows] - gt_xy[cols]) ** 2).sum(axis=1))
    keep = dist <= delta
    rows, cols, dist = rows[keep], cols[keep], dist[keep]
    order = np.lexsort((cols, rows))
    return rows[order], cols[order], dist[order]


#: A connected group of pairs in reach with more ``predictions x annotations`` than this
#: is solved on its sparse pairs; a smaller one on its dense cost matrix.
_DENSE_GROUP_ENTRIES = 1_000_000


def _hungarian_pairs(
    rows: np.ndarray, cols: np.ndarray, dist: np.ndarray, n_pred: int, n_gt: int, delta: float
) -> np.ndarray:
    """The most pairs in reach that a one-to-one matching can make, and among those the
    ones of least total distance, solved per connected group of pairs in reach.

    The dense matcher solved one assignment over the full matrix, with a cost beyond
    ``delta`` so large that it first maximizes the number of pairs in reach. Pairs of two
    groups never share a point, so that optimum is the union of the groups' optima.
    """
    if rows.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    graph = coo_matrix((np.ones(rows.size), (rows, n_pred + cols)), shape=(n_pred + n_gt,) * 2)
    _, labels = connected_components(graph, directed=False)
    group = labels[rows]
    order = np.argsort(group, kind="stable")
    bounds = np.flatnonzero(np.diff(group[order])) + 1
    pairs: list[np.ndarray] = []
    lone: list[int] = []  # groups of a single pair: one prediction, one annotation
    big = float(delta) * 1e6 + 1.0
    for members in np.split(order, bounds):
        if members.size == 1:
            lone.append(int(members[0]))
            continue
        preds, local_rows = np.unique(rows[members], return_inverse=True)
        gts, local_cols = np.unique(cols[members], return_inverse=True)
        if preds.size * gts.size <= _DENSE_GROUP_ENTRIES:
            cost = np.full((preds.size, gts.size), big)
            cost[local_rows, local_cols] = dist[members]
            r, c = linear_sum_assignment(cost)
            kept = cost[r, c] < big
            r, c = r[kept], c[kept]
        else:
            r, c = _sparse_assignment(local_rows, local_cols, dist[members], preds.size, gts.size, delta)
        pairs.append(np.stack([preds[r], gts[c]], axis=1))
    if lone:
        pairs.append(np.stack([rows[lone], cols[lone]], axis=1))
    out = np.concatenate(pairs).astype(np.int64)
    return out[np.argsort(out[:, 0], kind="stable")]


def _sparse_assignment(
    rows: np.ndarray, cols: np.ndarray, dist: np.ndarray, n_pred: int, n_gt: int, delta: float
) -> tuple[np.ndarray, np.ndarray]:
    """The dense objective on sparse pairs: each prediction may also take a dummy
    annotation of its own at a cost beyond any matching's total distance, so the least
    cost full matching makes the most pairs in reach, then the shortest ones. Every
    weight is shifted by 1, as the solver reads an explicit zero weight as no edge."""
    dummy = n_pred * (float(delta) + 1.0) + 2.0
    graph = csr_matrix(
        (
            np.concatenate([dist + 1.0, np.full(n_pred, dummy)]),
            (np.concatenate([rows, np.arange(n_pred)]), np.concatenate([cols, n_gt + np.arange(n_pred)])),
        ),
        shape=(n_pred, n_gt + n_pred),
    )
    r, c = min_weight_full_bipartite_matching(graph)
    kept = c < n_gt
    return r[kept], c[kept]


def _greedy_pairs(
    rows: np.ndarray, cols: np.ndarray, dist: np.ndarray, pred_score: np.ndarray
) -> np.ndarray:
    """Predictions claim by descending score (a stable sort: ties in input order), each the
    nearest unclaimed annotation in reach (ties: the lowest index)."""
    rank = np.empty(pred_score.shape[0], dtype=np.int64)
    rank[np.argsort(-pred_score, kind="stable")] = np.arange(pred_score.shape[0])
    order = np.lexsort((cols, dist, rank[rows]))
    claimed: set[int] = set()
    done: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for r, g in zip(rows[order].tolist(), cols[order].tolist()):
        if r in done or g in claimed:
            continue
        done.add(r)
        claimed.add(g)
        pairs.append((r, g))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _near_any(xy: np.ndarray, gt_xy: np.ndarray, delta_sq: float) -> np.ndarray:
    """Which points have an annotation within ``sqrt(delta_sq)``: ``dx**2 + dy**2 <=
    delta_sq`` on the KD-tree's candidates."""
    radius = float(np.sqrt(delta_sq)) * (1 + 1e-9) + 1e-9
    found = cKDTree(xy).sparse_distance_matrix(cKDTree(gt_xy), radius, output_type="ndarray")
    rows = found["i"].astype(np.int64)
    d_sq = ((xy[rows] - gt_xy[found["j"].astype(np.int64)]) ** 2).sum(axis=1)
    near = np.zeros(xy.shape[0], dtype=bool)
    near[rows[d_sq <= delta_sq]] = True
    return near


def match_assignment(
    pred_xy: np.ndarray,
    pred_class: np.ndarray,
    pred_score: np.ndarray,
    gt_xy: np.ndarray,
    gt_class: np.ndarray,
    *,
    num_classes: int,
    delta: float,
    method: str = "hungarian",
) -> list[ClassMatch]:
    """Class-aware point matching for one image -> per-class :class:`ClassMatch`.

    The assignment primitive: a length-``C`` list, one :class:`ClassMatch` per object
    class, with ``pairs`` holding **global** ``(pred_idx, gt_idx)`` indices into the
    original ``pred_*`` / ``gt_*`` arrays. :func:`match_points` derives its ``(C, 3)``
    counts from these and the match overlay reads the same ``pairs``, so the headline
    F1 and the overlay are the same computation.

    Args:
        pred_xy / pred_class / pred_score: predicted points ``(M, 2)``, class ids
            ``(M,)``, scores ``(M,)`` (scores used only by ``greedy``).
        gt_xy / gt_class: ground-truth points ``(K, 2)`` and class ids ``(K,)``.
        num_classes: number of object classes ``C``.
        delta: matching distance in target-frame pixels.
        method: ``"hungarian"`` (optimal one-to-one) or ``"greedy"`` (by confidence).
    """
    if float(delta) <= 0.0:
        raise ValueError(f"delta must be > 0, got {delta}.")
    pred_xy = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
    gt_xy = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    pred_class = np.asarray(pred_class).reshape(-1).astype(np.int64)
    gt_class = np.asarray(gt_class).reshape(-1).astype(np.int64)
    pred_score = np.asarray(pred_score, dtype=np.float64).reshape(-1)
    if pred_xy.shape[0] != pred_class.shape[0] or pred_xy.shape[0] != pred_score.shape[0]:
        raise ValueError("pred_xy, pred_class, pred_score must share length.")
    if gt_xy.shape[0] != gt_class.shape[0]:
        raise ValueError("gt_xy and gt_class must share length.")

    matches: list[ClassMatch] = []
    for c in range(int(num_classes)):
        p = np.nonzero(pred_class == c)[0]  # global predicted indices for this class
        g = np.nonzero(gt_class == c)[0]  # global GT indices for this class
        local = match_single_class(
            pred_xy[p], pred_score[p], gt_xy[g], delta=float(delta), method=method
        )
        if local.pairs.shape[0]:
            global_pairs = np.stack([p[local.pairs[:, 0]], g[local.pairs[:, 1]]], axis=1)
        else:
            global_pairs = np.zeros((0, 2), dtype=np.int64)
        matches.append(ClassMatch(pairs=global_pairs, n_pred=local.n_pred, n_gt=local.n_gt))
    return matches


def match_points(
    pred_xy: np.ndarray,
    pred_class: np.ndarray,
    pred_score: np.ndarray,
    gt_xy: np.ndarray,
    gt_class: np.ndarray,
    *,
    num_classes: int,
    delta: float,
    method: str = "hungarian",
) -> np.ndarray:
    """Class-aware point matching for one image -> ``(C, 3)`` ``(tp, fp, fn)`` counts.

    A thin reduction of :func:`match_assignment`: the counts are derived from each
    class's :class:`ClassMatch` (``tp = #pairs``), never computed on a separate path.
    """
    matches = match_assignment(
        pred_xy, pred_class, pred_score, gt_xy, gt_class,
        num_classes=num_classes, delta=delta, method=method,
    )
    counts = np.zeros((int(num_classes), 3), dtype=np.int64)
    for c, m in enumerate(matches):
        counts[c] = m.counts
    return counts


def detection_counts(
    pred_xy: np.ndarray,
    pred_class: np.ndarray,
    pred_score: np.ndarray,
    gt_xy: np.ndarray,
    gt_class: np.ndarray,
    *,
    num_classes: int,
    delta: float,
    method: str = "hungarian",
) -> Tensor:
    """:func:`match_points` as a ``(1, C, 3)`` long tensor (one per-image stat row).

    The detection analogue of ``dense_confusion_counts`` — the streaming evaluator
    concatenates these along dim 0 into ``(N, C, 3)`` and reduces once via
    :func:`reduce_f1`, keeping the per-image axis (needed for per-image-macro).
    """
    counts = match_points(
        pred_xy, pred_class, pred_score, gt_xy, gt_class,
        num_classes=num_classes, delta=delta, method=method,
    )
    return torch.from_numpy(counts).to(torch.long).unsqueeze(0)


def _prf1(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    # F1 = 2·TP / (2·TP + FP + FN). Undefined (NaN) only when the class is absent from
    # both prediction and GT (tp=fp=fn=0); any class with predictions or GT but no true
    # positives is a real failure and scores 0, so it is not dropped from the macro mean.
    denom = 2.0 * tp + fp + fn
    f1 = float("nan") if denom == 0 else 2.0 * tp / denom
    return precision, recall, f1


def _nanmean(values: list[float]) -> float:
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=np.float64)
    return float(arr.mean()) if arr.size else 0.0


def reduce_f1(
    counts: Tensor,
    *,
    num_classes: int,
    aggregation: str = "dataset_global",
) -> dict[str, float]:
    """Reduce ``(N, C, 3)`` ``(tp, fp, fn)`` counts to F1 metrics.

    ``dataset_global`` (the headline, OCELOT-faithful): pool counts over all images
    per class, one F1 per class, then mean = ``mean_f1``. ``per_image_macro``: F1 per
    image per class (a class absent from both pred and GT in an image → undefined →
    excluded), mean over its defined classes, then mean over images. Per-class F1
    (``f1_class_{c}``) and precision/recall are always the dataset-global values.
    """
    if counts.ndim != 3 or counts.shape[1] != int(num_classes) or counts.shape[2] != 3:
        raise ValueError(
            f"counts must be (N, C={num_classes}, 3), got {tuple(counts.shape)}."
        )
    c64 = counts.to(torch.float64).cpu().numpy()  # (N, C, 3)

    # Dataset-global per class.
    g = c64.sum(axis=0)  # (C, 3)
    per_class_f1: list[float] = []
    result: dict[str, float] = {}
    for c in range(int(num_classes)):
        precision, recall, f1 = _prf1(g[c, 0], g[c, 1], g[c, 2])
        per_class_f1.append(f1)
        result[f"f1_class_{c}"] = 0.0 if np.isnan(f1) else f1
        result[f"precision_class_{c}"] = 0.0 if np.isnan(precision) else precision
        result[f"recall_class_{c}"] = 0.0 if np.isnan(recall) else recall

    if aggregation == "dataset_global":
        result["mean_f1"] = _nanmean(per_class_f1)
    elif aggregation == "per_image_macro":
        per_image: list[float] = []
        for n in range(c64.shape[0]):
            f1s = [_prf1(c64[n, c, 0], c64[n, c, 1], c64[n, c, 2])[2] for c in range(int(num_classes))]
            defined = [v for v in f1s if not np.isnan(v)]
            if defined:
                per_image.append(float(np.mean(defined)))
        result["mean_f1"] = float(np.mean(per_image)) if per_image else 0.0
    else:
        raise ValueError(f"unknown aggregation {aggregation!r}.")
    return result


def sweep_score_thresholds(
    per_image_pred_xy: list[np.ndarray],
    per_image_pred_class: list[np.ndarray],
    per_image_pred_score: list[np.ndarray],
    per_image_gt_xy: list[np.ndarray],
    per_image_gt_class: list[np.ndarray],
    *,
    num_classes: int,
    delta: float,
    method: str = "hungarian",
    num_candidates: int = 50,
) -> list[float]:
    """Pick the per-class score threshold maximizing dataset-global per-class F1.

    Run on the **tune** split. With ``method="greedy"`` this is the exact one-pass sweep
    (:func:`sweep_score_histograms`, scores in ``[0, 1]``, thresholds on its
    ``1 / SCORE_BINS`` grid): the same function the detection head's training-time
    validation reads its monitor from. With ``hungarian``, for each class the candidate
    grid unions three sources —
    quantiles of that class's predicted scores, a uniform grid over the observed score
    range, and quantiles of the scores of predictions lying within ``delta`` of a
    same-class GT point. At each candidate, predictions below it are dropped (all
    images), the chosen ``method`` is re-run, and the threshold with the best pooled F1
    is frozen. Returns a length-``C`` list applied unchanged at test.

    Score quantiles alone starve when the candidate pool is decoded at threshold 0 and
    junk local maxima dominate it (MIDOG-style heatmaps yield hundreds of near-zero
    peaks per tile): every quantile then lands in the junk mass, the mid-range where F1
    peaks is never evaluated, and the sweep freezes a near-zero threshold that floods
    test with false positives. The uniform grid covers the value range regardless of
    where the probability mass sits, and the GT-anchored quantiles target the scores
    where the F1 curve actually breaks.

    Tuning each class independently is exact for the dataset-global metric because the
    per-class F1 only depends on that class's own predictions (class-aware matching).
    """
    n_images = len(per_image_pred_xy)
    if method == "greedy":
        if n_images == 0:
            return [0.0] * int(num_classes)
        histograms = np.stack(
            [
                greedy_score_histogram(*image, num_classes=num_classes, delta=delta)
                for image in zip(
                    per_image_pred_xy, per_image_pred_class, per_image_pred_score,
                    per_image_gt_xy, per_image_gt_class,
                )
            ]
        )
        swept, _ = sweep_score_histograms(histograms)
        return [float(t) for t in swept]
    delta_sq = float(delta) * float(delta)
    thresholds: list[float] = []
    for c in range(int(num_classes)):
        # Gather this class's scores across all tune images for candidate thresholds.
        class_scores = np.concatenate(
            [
                np.asarray(s, dtype=np.float64).reshape(-1)[
                    np.asarray(cl).reshape(-1).astype(np.int64) == c
                ]
                for s, cl in zip(per_image_pred_score, per_image_pred_class)
            ]
        ) if n_images else np.zeros((0,))
        if class_scores.size == 0:
            thresholds.append(0.0)
            continue
        # Scores of this class's predictions within delta of a same-class GT point: the
        # potential true positives whose scores are the F1 curve's breakpoints.
        near_gt_scores: list[np.ndarray] = []
        for i in range(n_images):
            cls_i = np.asarray(per_image_pred_class[i]).reshape(-1).astype(np.int64)
            sel = cls_i == c
            if not np.any(sel):
                continue
            gcls = np.asarray(per_image_gt_class[i]).reshape(-1).astype(np.int64)
            gxy = np.asarray(per_image_gt_xy[i], dtype=np.float64).reshape(-1, 2)[gcls == c]
            if gxy.shape[0] == 0:
                continue
            xy_i = np.asarray(per_image_pred_xy[i], dtype=np.float64).reshape(-1, 2)[sel]
            sc_i = np.asarray(per_image_pred_score[i], dtype=np.float64).reshape(-1)[sel]
            near_gt_scores.append(sc_i[_near_any(xy_i, gxy, delta_sq)])
        quantiles = np.linspace(0.0, 1.0, int(num_candidates))
        lo, hi = float(class_scores.min()), float(class_scores.max())
        grids = [
            np.quantile(class_scores, quantiles),
            np.linspace(lo, hi, int(num_candidates)),
        ]
        near = (
            np.concatenate(near_gt_scores) if near_gt_scores else np.zeros((0,))
        )
        if near.size:
            grids.append(np.quantile(near, quantiles))
        candidates = np.unique(np.concatenate(grids))
        # Bracket the observed scores (candidates are ascending): a just-below-min option
        # keeps everything, a just-above-max option suppresses the class entirely (keep is
        # ``score >= thr``). The latter lets a class whose tune predictions are all false
        # positives pick the zero-FP operating point instead of being forced to emit its
        # top peak.
        candidates = np.concatenate([[lo - 1e-6], candidates, [hi + 1e-6]])

        best_thr, best_f1 = float(candidates[0]), -1.0
        for thr in candidates:
            tp = fp = fn = 0
            for i in range(n_images):
                cls_i = np.asarray(per_image_pred_class[i]).reshape(-1).astype(np.int64)
                score_i = np.asarray(per_image_pred_score[i], dtype=np.float64).reshape(-1)
                xy_i = np.asarray(per_image_pred_xy[i], dtype=np.float64).reshape(-1, 2)
                keep = (cls_i == c) & (score_i >= thr)
                gcls = np.asarray(per_image_gt_class[i]).reshape(-1).astype(np.int64)
                gxy = np.asarray(per_image_gt_xy[i], dtype=np.float64).reshape(-1, 2)
                m = match_single_class(
                    xy_i[keep], score_i[keep], gxy[gcls == c],
                    delta=float(delta), method=method,
                )
                tp += m.tp
                fp += m.fp
                fn += m.fn
            _, _, f1 = _prf1(tp, fp, fn)
            f1 = 0.0 if np.isnan(f1) else f1
            # ``>=`` over ascending candidates tie-breaks toward the higher threshold
            # (fewer false positives at equal F1).
            if f1 >= best_f1:
                best_f1, best_thr = f1, float(thr)
        thresholds.append(best_thr)
    return thresholds


# --------------------------------------------------------------------------------------
# One-pass greedy sweep
# --------------------------------------------------------------------------------------


def histogram_thresholds(bins: int = SCORE_BINS) -> np.ndarray:
    """The thresholds the one-pass sweep reads: ``k / bins`` for ``k = 0 .. bins + 1``.

    A prediction is kept at threshold ``t`` when its score is ``>= t``. Scores lie in
    ``[0, 1]``, so the last threshold (``1 + 1 / bins``) keeps no prediction.
    """
    return np.arange(int(bins) + 2, dtype=np.float64) / int(bins)


def greedy_score_histogram(
    pred_xy: np.ndarray,
    pred_class: np.ndarray,
    pred_score: np.ndarray,
    gt_xy: np.ndarray,
    gt_class: np.ndarray,
    *,
    num_classes: int,
    delta: float,
    bins: int = SCORE_BINS,
) -> np.ndarray:
    """One image's greedy matching as per-class score histograms, ``(C, 2 * S + 1)``.

    ``S = bins + 1`` score slots: slot ``k`` holds the scores in ``[t_k, t_(k+1))`` of
    :func:`histogram_thresholds` (the last slot holds a score of exactly 1). Per class,
    the row is ``[predictions per slot | hits per slot | annotated points]``. Every
    prediction is matched once, greedily; summed over images and read by
    :func:`counts_at_thresholds`, the rows give the exact counts at every threshold.
    Scores must lie in ``[0, 1]`` (the head's peaks are sigmoid outputs).
    """
    pred_score = np.asarray(pred_score, dtype=np.float64).reshape(-1)
    if pred_score.size and not (np.all(pred_score >= 0.0) and np.all(pred_score <= 1.0)):
        raise ValueError(
            "the one-pass greedy sweep bins scores over [0, 1]; got scores in "
            f"[{np.nanmin(pred_score)}, {np.nanmax(pred_score)}]"
        )
    pred_class = np.asarray(pred_class).reshape(-1).astype(np.int64)
    num_classes = int(num_classes)
    matches = match_assignment(
        pred_xy, pred_class, pred_score, gt_xy, gt_class,
        num_classes=num_classes, delta=delta, method="greedy",
    )
    slots = int(bins) + 1
    hit = np.zeros(pred_score.shape[0], dtype=bool)
    for m in matches:
        hit[m.pairs[:, 0]] = True
    known = (pred_class >= 0) & (pred_class < num_classes)
    slot = np.searchsorted(histogram_thresholds(bins), pred_score, side="right") - 1
    flat = pred_class * slots + slot
    size = num_classes * slots
    out = np.zeros((num_classes, 2 * slots + 1), dtype=np.int64)
    out[:, :slots] = np.bincount(flat[known], minlength=size).reshape(num_classes, slots)
    out[:, slots : 2 * slots] = np.bincount(flat[known & hit], minlength=size).reshape(num_classes, slots)
    out[:, -1] = [m.n_gt for m in matches]
    return out


def _slots(histogram: np.ndarray) -> int:
    width = int(histogram.shape[-1])
    if width < 3 or (width - 1) % 2:
        raise ValueError(f"a score histogram row is 2 * S + 1 wide, got {width}")
    return (width - 1) // 2


def counts_at_thresholds(histogram) -> np.ndarray:
    """Score histograms ``(..., C, 2 * S + 1)`` -> TP/FP/FN at every threshold.

    Returns ``(..., C, S + 1, 3)``: entry ``k`` holds the ``(tp, fp, fn)`` of the
    predictions scoring at least ``histogram_thresholds()[k]`` (reverse cumulative sums
    over the score slots). Sum the per-image rows first for the pooled counts.
    """
    h = np.asarray(histogram, dtype=np.int64)
    slots = _slots(h)
    zero = np.zeros(h.shape[:-1] + (1,), dtype=np.int64)

    def at_or_above(counts: np.ndarray) -> np.ndarray:
        return np.concatenate([np.cumsum(counts[..., ::-1], axis=-1)[..., ::-1], zero], axis=-1)

    kept = at_or_above(h[..., :slots])
    tp = at_or_above(h[..., slots : 2 * slots])
    return np.stack([tp, kept - tp, h[..., -1:] - tp], axis=-1)


def sweep_score_histograms(histograms) -> tuple[np.ndarray, Tensor]:
    """Per-class best threshold of per-image score histograms ``(N, C, 2 * S + 1)``.

    For each class, the threshold of :func:`histogram_thresholds` whose dataset-global
    F1 is highest (ties keep the higher threshold: fewer false positives at equal F1; a
    class whose every threshold scores 0 keeps no prediction). A class with no prediction
    at all has nothing to tune and gets threshold 0. Tuning each class alone is
    exact for the dataset-global metric (class-aware matching). Returns the ``(C,)``
    thresholds and the per-image ``(N, C, 3)`` TP/FP/FN at them, ready for
    :func:`reduce_f1`.
    """
    h = np.asarray(histograms, dtype=np.int64)
    if h.ndim != 3:
        raise ValueError(f"histograms must be (N, C, 2 * S + 1), got {tuple(h.shape)}")
    slots = _slots(h)
    pooled = counts_at_thresholds(h.sum(axis=0))  # (C, S + 1, 3)
    tp, fp, fn = pooled[..., 0], pooled[..., 1], pooled[..., 2]
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1), 0.0)
    best = f1.shape[1] - 1 - np.argmax(f1[:, ::-1], axis=1)  # the last (highest) maximum
    best = np.where(tp[:, 0] + fp[:, 0] == 0, 0, best)
    counts = np.zeros((h.shape[0], h.shape[1], 3), dtype=np.int64)
    for c, k in enumerate(best):
        kept = h[:, c, k:slots].sum(axis=1)
        hits = h[:, c, slots + k : 2 * slots].sum(axis=1)
        counts[:, c] = np.stack([hits, kept - hits, h[:, c, -1] - hits], axis=1)
    return histogram_thresholds(slots - 1)[best], torch.from_numpy(counts)
