"""Pixel-classifier segmentation glue: pixel sampling, dense predict, evaluation.

Pure orchestration helpers shared by ``train_one_pixel_classifier_fold`` (in
``soma.pipeline``). They reuse the segmentation **data plane** verbatim — the
``SegmentationHead``'s geometry (``forward`` upsamples a ``(C, gh, gw)`` grid to the
mask's pixel resolution; ``extract_targets`` reads the spacing-aware mask), the
``dense_confusion_counts`` / streaming Dice-IoU reduction, and the
``DenseArtifactWriter`` — so a pixel-classifier run produces byte-identical metrics and
artifacts to the neural-decoder run. Only the model differs.

Design: attention-pixel segmentation §9 (pixel-resolution, sample-train / predict-all).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from soma.evaluation.dense_artifacts import DenseArtifactWriter
from soma.evaluation.report import EvaluationReport
from soma.training.segmentation_dataset import SegmentationBatch

if TYPE_CHECKING:
    from soma.dataset import SampleRecord, SegmentationManifest
    from soma.dense import DenseFeatureSource
    from soma.pixel_classifiers.base import PixelClassifier
    from soma.tasks.segmentation import SegmentationHead

logger = logging.getLogger(__name__)

_LOG_EPS = 1e-8


def grid_to_target_features(grid: torch.Tensor, head: "SegmentationHead") -> torch.Tensor:
    """Upsample a dense ``(K, gh, gw)`` grid to per-pixel features ``(K, H, W)``.

    Reuses the head's geometry (``forward`` interpolates the token grid to the padded
    encoded size, then crops ``crop_box`` to the mask ``target_size``) — the same map a
    decoder's logits take, so classifier and decoder localize identically. The head
    forward is channel-agnostic, so feeding it ``K`` feature channels (instead of ``C``
    logits) yields the bilinearly-upsampled attention maps the paper classifies.
    """
    with torch.no_grad():
        return head.forward(grid.unsqueeze(0).float())[0]  # (K, H, W)


def _valid_pixel_features(
    record: "SampleRecord",
    feature_store: "DenseFeatureSource",
    head: "SegmentationHead",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(features (P, K), mask (P,))`` flattened over all pixels of one tile."""
    grid = feature_store.load(record.sample_id)
    feat = grid_to_target_features(grid, head)  # (K, H, W)
    k = feat.shape[0]
    feat = feat.reshape(k, -1).transpose(0, 1).contiguous().numpy()  # (H*W, K)
    mask = head.extract_targets(record)["mask"].reshape(-1).numpy()  # (H*W,)
    return feat, mask


def build_training_matrix(
    records: list["SampleRecord"],
    feature_store: "DenseFeatureSource",
    head: "SegmentationHead",
    *,
    max_pixels: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Class-stratified pixel sampling across ``records`` → ``(X (N, K), y (N,))``.

    Two passes: pass 1 counts per-class supervised pixels (masks only, cheap); pass 2
    allocates each present class an equal share of ``max_pixels`` (capped by
    availability) and draws each tile's proportional slice without replacement,
    extracting only the sampled feature rows. ``ignore_index`` pixels are excluded.
    Balanced-by-construction, so a rare class is not swamped at fit time.
    """
    num_classes = head.num_classes
    # Pass 1: per-class totals (re-reads masks in pass 2 to keep memory bounded).
    totals = np.zeros(num_classes, dtype=np.int64)
    for record in records:
        mask = head.extract_targets(record)["mask"].reshape(-1).numpy()
        # ignore_index may be negative (e.g. torch's -100) — those pass `< num_classes`,
        # so filter to in-range labels [0, num_classes) before bincount (which rejects
        # negatives). Pass 2 uses `mask == c` for c in [0, num_classes) and is already safe.
        supervised = mask[(mask >= 0) & (mask < num_classes)]
        totals += np.bincount(supervised, minlength=num_classes)[:num_classes]
    present = totals > 0
    n_present = int(present.sum())
    if n_present == 0:
        raise ValueError(
            "pixel-classifier training found no supervised pixels across the train split "
            "(every pixel is ignore_index)."
        )
    per_class = max(1, int(max_pixels) // n_present)
    budget = np.where(present, np.minimum(totals, per_class), 0)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for record in records:
        feat = None
        mask = head.extract_targets(record)["mask"].reshape(-1).numpy()
        for c in range(num_classes):
            if budget[c] == 0:
                continue
            idx_c = np.flatnonzero(mask == c)
            if idx_c.size == 0:
                continue
            take = min(int(round(budget[c] * idx_c.size / totals[c])), idx_c.size)
            if take <= 0:
                continue
            if feat is None:
                grid = feature_store.load(record.sample_id)
                k = grid.shape[0]
                feat = grid_to_target_features(grid, head).reshape(k, -1).transpose(0, 1).contiguous().numpy()
            sel = rng.choice(idx_c, size=take, replace=False)
            x_parts.append(feat[sel])
            y_parts.append(np.full(take, c, dtype=np.int64))
    if not x_parts:
        raise ValueError("pixel-classifier training sampled zero pixels; check max_train_pixels.")
    return np.concatenate(x_parts).astype(np.float32), np.concatenate(y_parts)


def inverse_frequency_sample_weight(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Per-pixel inverse-frequency weights (mean ~1), the tree/linear analog of CE class
    weights. Absent classes get no rows, so they never enter the weighting."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid div-by-zero; absent classes carry no pixels anyway
    inv = 1.0 / counts
    weight = inv[y]
    return (weight * (len(y) / weight.sum())).astype(np.float32)


def predict_tile_pseudologits(
    clf: "PixelClassifier",
    feat: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Predict per-pixel class probabilities and return ``(1, C, H, W)`` pseudo-logits.

    The downstream dense machinery (``dense_confusion_counts``, ``DenseArtifactWriter``)
    consumes *logits* (argmax + softmax). Feeding ``log(prob)`` makes ``argmax`` exact
    and ``softmax(log prob) == prob`` (so an opt-in probability sidecar is faithful, not
    a softmax-of-probabilities).
    """
    k, height, width = feat.shape
    x = feat.reshape(k, -1).transpose(0, 1).contiguous().numpy()  # (H*W, K)
    proba = clf.predict_proba(x)  # (H*W, C)
    log_proba = np.log(np.clip(proba, _LOG_EPS, 1.0)).astype(np.float32)
    return (
        torch.from_numpy(log_proba)
        .transpose(0, 1)
        .reshape(1, num_classes, height, width)
        .contiguous()
    )


def evaluate_pixel_classifier(
    clf: "PixelClassifier",
    records: list["SampleRecord"],
    feature_store: "DenseFeatureSource",
    head: "SegmentationHead",
    split_name: str,
    *,
    dataset: "SegmentationManifest | None" = None,
    output_dir: Path | None = None,
    save_probabilities: bool = False,
) -> EvaluationReport:
    """Predict every pixel of every tile and reduce to dense metrics + artifacts.

    Mirrors ``soma.pipeline._evaluate_segmentation`` but iterates records directly (no
    DataLoader / nn.Module): per tile it builds pseudo-logits, accumulates the per-image
    confusion row (``head.dense_stats``), and streams rasters/overlays/CSV through the
    shared ``DenseArtifactWriter``.
    """
    writer = (
        DenseArtifactWriter(
            head=head,
            split=split_name,
            output_dir=output_dir,
            dataset=dataset,
            save_probabilities=save_probabilities,
        )
        if output_dir is not None
        else None
    )
    stat_rows: list[torch.Tensor] = []
    for record in records:
        grid = feature_store.load(record.sample_id)
        feat = grid_to_target_features(grid, head)  # (K, H, W)
        targets = head.extract_targets(record)
        mask = targets["mask"].unsqueeze(0)  # (1, H, W)
        logits = predict_tile_pseudologits(clf, feat, head.num_classes)  # (1, C, H, W)
        stat_row = head.dense_stats(logits, {"mask": mask})  # (1, C, 3)
        stat_rows.append(stat_row)
        if writer is not None:
            batch = SegmentationBatch(
                features=feat.unsqueeze(0),
                targets={"mask": mask},
                sample_ids=(record.sample_id,),
            )
            writer(batch, logits, stat_row)
    metrics = head.finalize_eval_metrics(torch.cat(stat_rows, dim=0)) if stat_rows else {}
    if writer is not None:
        writer.finalize()
    return EvaluationReport(split=split_name, metrics=metrics, predictions=[])
