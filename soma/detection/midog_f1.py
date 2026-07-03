"""MIDOG-native F1 scorer — the single-class mitosis-detection metric (design §2.2).

MIDOG 2022 scores mitotic-figure detection with a single **F1**: a predicted point is a
true positive if it matches a ground-truth mitosis within the challenge tolerance
(:data:`MIDOG_MATCH_DISTANCE_UM` = 7.5 µm, roughly one nucleus) under **one-to-one**
(Hungarian) matching; unmatched predictions are false positives, unmatched GT are false
negatives. Counts are pooled over **every** image (dataset-global) and reduced to
precision / recall / F1 for the lone mitosis class.

This is deliberately a thin **adapter** over :mod:`soma.detection.matching` with
``num_classes=1``: it reuses the exact TP/FP/FN matcher and F1 reduction the OCELOT path
uses, so the two datasets' metrics are one implementation, not two that can drift. All
distances are in the point coordinate frame, the same units as ``delta`` — a run resolves
the µm tolerance to frame pixels via its spacing before calling in (wiring is a run-time
step). The reference band vs. the published MIDOG leaderboard is a *different test set*
(the official test is held out; soma reports on a local held-out split), so it is shown as
a band, never an aligned row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from soma.detection.matching import detection_counts, reduce_f1

__all__ = ["MIDOG_MATCH_DISTANCE_UM", "MidogF1Score", "midog_f1"]

# MIDOG 2022's native hit tolerance: a detection matches a GT mitosis within 7.5 µm
# (~one nucleus). Kept as a µm constant because the challenge defines it physically; a run
# converts it to point-frame pixels via its spacing before scoring, exactly as the head
# resolves OCELOT's delta=3 µm.
MIDOG_MATCH_DISTANCE_UM = 7.5


@dataclass(frozen=True)
class MidogF1Score:
    """Dataset-global mitosis F1 with its precision / recall and pooled TP/FP/FN."""

    f1: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int

    def as_dict(self) -> dict[str, float]:
        """Flatten to a ``{metric: value}`` mapping (for reports / leaderboard rows)."""
        return {
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "tp": float(self.tp),
            "fp": float(self.fp),
            "fn": float(self.fn),
        }


def midog_f1(
    per_image_pred_xy: Sequence[np.ndarray],
    per_image_pred_score: Sequence[np.ndarray],
    per_image_gt_xy: Sequence[np.ndarray],
    *,
    delta: float,
    method: str = "hungarian",
) -> MidogF1Score:
    """Pool per-image mitosis TP/FP/FN and reduce to the MIDOG-native F1.

    Args:
        per_image_pred_xy: one ``(M, 2)`` array of predicted points per image.
        per_image_pred_score: one ``(M,)`` score array per image (used only by the
            greedy matcher; still required so lengths line up with ``per_image_pred_xy``).
        per_image_gt_xy: one ``(K, 2)`` array of ground-truth mitosis points per image.
        delta: matching distance in the point coordinate frame (see
            :data:`MIDOG_MATCH_DISTANCE_UM` for the challenge's µm value).
        method: ``"hungarian"`` (MIDOG-native one-to-one, the default) or ``"greedy"``.

    Returns:
        A :class:`MidogF1Score` — mitosis F1 with precision / recall and pooled counts.
    """
    n_images = len(per_image_pred_xy)
    if len(per_image_pred_score) != n_images or len(per_image_gt_xy) != n_images:
        raise ValueError(
            "midog_f1 needs the same number of images in per_image_pred_xy, "
            f"per_image_pred_score and per_image_gt_xy; got "
            f"{n_images}, {len(per_image_pred_score)}, {len(per_image_gt_xy)}."
        )

    rows: list[torch.Tensor] = []
    for pred_xy, pred_score, gt_xy in zip(
        per_image_pred_xy, per_image_pred_score, per_image_gt_xy
    ):
        pred_xy = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
        gt_xy = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
        pred_score = np.asarray(pred_score, dtype=np.float64).reshape(-1)
        # Single mitosis class -> every point is class 0.
        rows.append(
            detection_counts(
                pred_xy,
                np.zeros(pred_xy.shape[0], dtype=np.int64),
                pred_score,
                gt_xy,
                np.zeros(gt_xy.shape[0], dtype=np.int64),
                num_classes=1,
                delta=delta,
                method=method,
            )
        )

    counts = torch.cat(rows, dim=0) if rows else torch.zeros((0, 1, 3), dtype=torch.long)
    metrics = reduce_f1(counts, num_classes=1, aggregation="dataset_global")
    tp, fp, fn = (int(v) for v in counts.to(torch.long).sum(dim=(0, 1)).tolist())
    return MidogF1Score(
        f1=metrics["mean_f1"],
        precision=metrics["precision_class_0"],
        recall=metrics["recall_class_0"],
        tp=tp,
        fp=fp,
        fn=fn,
    )
