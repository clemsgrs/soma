"""Peak extraction — predicted heatmap ``(C, H, W)`` -> discrete points (design §6).

The postprocess inverse of :func:`soma.detection.encode.render_peak_heatmap`: turn a
per-class probability map back into ``(x, y, class, score)`` detections. Two stages,
per channel independently (a location may fire in two class channels → two candidate
detections, which the class-aware matcher then resolves):

1. **Local-maxima prefilter** — a pixel is a candidate iff it equals the maximum in
   its ``(2*min_distance+1)`` neighbourhood (the CenterNet max-pool trick), cutting
   the field to a handful of modes before the (quadratic) NMS.
2. **Greedy NMS + threshold** — keep candidates ``>= score_threshold``, sort by score
   descending, accept a peak only if no already-accepted peak of the same class lies
   within ``min_distance`` pixels.

Pure / weight-free so it unit-tests on tiny hand-built maps.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["extract_peaks"]


#: Hard ceiling on candidates per class fed to the quadratic NMS loop. Real heatmaps
#: yield a handful of modes; this only bites on degenerate near-uniform predictions
#: (e.g. an untrained net's flat output), where it bounds the loop instead of letting it
#: hang/OOM. The highest-scoring candidates are kept.
MAX_CANDIDATES_PER_CLASS = 10000


def extract_peaks(
    heatmap: Tensor,
    *,
    min_distance: float,
    score_threshold: float | list[float],
    max_candidates: int = MAX_CANDIDATES_PER_CLASS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract peaks from a ``(C, H, W)`` heatmap.

    Args:
        heatmap: Per-class probability map ``(C, H, W)`` in ``[0, 1]``.
        min_distance: NMS / local-maxima radius in pixels (``> 0``). Two accepted
            peaks of one class cannot be closer than this.
        score_threshold: Minimum peak score to keep — a scalar (shared) or a
            per-class list of length ``C`` (the tune-split-frozen thresholds).
        max_candidates: Per-class ceiling on candidates entering NMS; the top-scoring
            ones are kept (guards the quadratic loop on degenerate flat predictions).

    Returns:
        ``(points_xy (M, 2), classes (M,), scores (M,))``: detected peak centres
        ``(x, y)``, their class ids, and heatmap scores, ordered by descending score
        within each class. ``M`` may be 0.
    """
    if heatmap.ndim != 3:
        raise ValueError(f"heatmap must be (C, H, W), got shape {tuple(heatmap.shape)}.")
    num_classes = int(heatmap.shape[0])
    if float(min_distance) <= 0.0:
        raise ValueError(f"min_distance must be > 0, got {min_distance}.")
    if isinstance(score_threshold, (list, tuple)):
        if len(score_threshold) != num_classes:
            raise ValueError(
                f"per-class score_threshold must have C={num_classes} entries, "
                f"got {len(score_threshold)}."
            )
        thresholds = [float(t) for t in score_threshold]
    else:
        thresholds = [float(score_threshold)] * num_classes

    hm = heatmap.detach().float()
    radius = int(round(float(min_distance)))
    kernel = 2 * radius + 1
    # Local-maxima mask: a pixel that is the max over its (kernel x kernel) window AND
    # strictly above the window minimum. The second clause is plateau-safe: on a flat
    # window (e.g. a uniform 0.5 heatmap from sigmoid(0) on an untrained/all-background
    # tile) max == min, so no pixel qualifies — otherwise every pixel would be a "peak"
    # and flood the quadratic NMS loop.
    pooled = F.max_pool2d(hm.unsqueeze(0), kernel_size=kernel, stride=1, padding=radius).squeeze(0)
    min_pooled = -F.max_pool2d((-hm).unsqueeze(0), kernel_size=kernel, stride=1, padding=radius).squeeze(0)
    is_local_max = (hm == pooled) & (hm > min_pooled)  # (C, H, W)

    all_xy: list[np.ndarray] = []
    all_cls: list[np.ndarray] = []
    all_score: list[np.ndarray] = []
    md_sq = float(min_distance) ** 2
    for c in range(num_classes):
        cand = is_local_max[c] & (hm[c] >= thresholds[c])
        ys, xs = torch.nonzero(cand, as_tuple=True)
        if ys.numel() == 0:
            continue
        scores = hm[c, ys, xs]
        order = torch.argsort(scores, descending=True)
        if int(max_candidates) > 0 and order.numel() > int(max_candidates):
            order = order[: int(max_candidates)]  # keep the top-scoring candidates only
        xs_n = xs[order].cpu().numpy().astype(np.float64)
        ys_n = ys[order].cpu().numpy().astype(np.float64)
        scores_n = scores[order].cpu().numpy().astype(np.float64)
        kept_x: list[float] = []
        kept_y: list[float] = []
        kept_s: list[float] = []
        for x, y, s in zip(xs_n, ys_n, scores_n):
            ok = True
            for kx, ky in zip(kept_x, kept_y):
                if (x - kx) ** 2 + (y - ky) ** 2 < md_sq:
                    ok = False
                    break
            if ok:
                kept_x.append(float(x))
                kept_y.append(float(y))
                kept_s.append(float(s))
        if kept_x:
            all_xy.append(np.stack([kept_x, kept_y], axis=1))
            all_cls.append(np.full(len(kept_x), c, dtype=np.int64))
            all_score.append(np.asarray(kept_s, dtype=np.float64))

    if not all_xy:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)
    return (
        np.concatenate(all_xy, axis=0),
        np.concatenate(all_cls, axis=0),
        np.concatenate(all_score, axis=0),
    )
