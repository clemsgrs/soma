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

All distances are in the same (target-frame) pixel units as ``delta`` — the head
resolves ``delta`` from µm/px via the run spacing before calling in.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

__all__ = [
    "match_points",
    "detection_counts",
    "reduce_f1",
    "sweep_score_thresholds",
    "VALID_MATCHING",
]

VALID_MATCHING = ("hungarian", "greedy")


def _match_single_class(
    pred_xy: np.ndarray,
    pred_score: np.ndarray,
    gt_xy: np.ndarray,
    *,
    delta: float,
    method: str,
) -> tuple[int, int, int]:
    """Match one class's predictions to its GT; return ``(tp, fp, fn)``."""
    n_pred = int(pred_xy.shape[0])
    n_gt = int(gt_xy.shape[0])
    if n_pred == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_pred, 0

    # Pairwise Euclidean distance (n_pred, n_gt).
    diff = pred_xy[:, None, :] - gt_xy[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=2))
    within = dist <= float(delta)

    if method == "hungarian":
        # Optimal one-to-one over a cost matrix; forbid > δ pairs with a large finite
        # cost, then drop any assigned pair that ended up beyond δ.
        big = float(delta) * 1e6 + 1.0
        cost = np.where(within, dist, big)
        rows, cols = linear_sum_assignment(cost)
        tp = int(sum(1 for r, c in zip(rows, cols) if within[r, c]))
    elif method == "greedy":
        # Sort predictions by descending score; each claims the nearest unclaimed
        # in-δ GT. OCELOT's official convention.
        order = np.argsort(-pred_score)
        claimed = np.zeros(n_gt, dtype=bool)
        tp = 0
        for r in order:
            in_range = within[r] & ~claimed
            if not in_range.any():
                continue
            masked = np.where(in_range, dist[r], np.inf)
            g = int(np.argmin(masked))
            claimed[g] = True
            tp += 1
    else:
        raise ValueError(f"unknown matching method {method!r}; use one of {VALID_MATCHING}.")

    fp = n_pred - tp
    fn = n_gt - tp
    return tp, fp, fn


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

    counts = np.zeros((int(num_classes), 3), dtype=np.int64)
    for c in range(int(num_classes)):
        p = pred_class == c
        g = gt_class == c
        tp, fp, fn = _match_single_class(
            pred_xy[p], pred_score[p], gt_xy[g], delta=float(delta), method=method
        )
        counts[c] = (tp, fp, fn)
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

    Run on the **tune** split: for each class, the candidate thresholds are quantiles
    of that class's predicted scores; at each candidate, predictions below it are
    dropped (all images), the chosen ``method`` is re-run, and the threshold with the
    best pooled F1 is frozen. Returns a length-``C`` list applied unchanged at test.

    Tuning each class independently is exact for the dataset-global metric because the
    per-class F1 only depends on that class's own predictions (class-aware matching).
    """
    n_images = len(per_image_pred_xy)
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
        quantiles = np.linspace(0.0, 1.0, int(num_candidates))
        candidates = np.unique(np.quantile(class_scores, quantiles))
        # Bracket the observed scores (candidates are ascending): a just-below-min option
        # keeps everything, a just-above-max option suppresses the class entirely (keep is
        # ``score >= thr``). The latter lets a class whose tune predictions are all false
        # positives pick the zero-FP operating point instead of being forced to emit its
        # top peak.
        candidates = np.concatenate(
            [[float(class_scores.min()) - 1e-6], candidates, [float(class_scores.max()) + 1e-6]]
        )

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
                t, f, n = _match_single_class(
                    xy_i[keep], score_i[keep], gxy[gcls == c],
                    delta=float(delta), method=method,
                )
                tp += t
                fp += f
                fn += n
            _, _, f1 = _prf1(tp, fp, fn)
            f1 = 0.0 if np.isnan(f1) else f1
            # ``>=`` over ascending candidates tie-breaks toward the higher threshold
            # (fewer false positives at equal F1).
            if f1 >= best_f1:
                best_f1, best_thr = f1, float(thr)
        thresholds.append(best_thr)
    return thresholds
