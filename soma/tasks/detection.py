"""Detection task head — dense peak-heatmap regression (design §6, §7).

The detection counterpart of :class:`~soma.tasks.segmentation.SegmentationHead`. Like
it, the head is parameter-free (the learnable weights live in the ``decoder``) and owns
the geometry: it interpolates the decoder's ``(B, C, h', w')`` output to ``encoded_size``,
crops to ``target_size`` via ``crop_box`` (copied from the segmentation head so the two
cannot drift), then applies a sigmoid so the heatmap is a bounded ``[0, 1]`` regression
target. Beyond geometry it owns the detection-specific contract:

* **targets** (``extract_targets``) — read the level-0 points, map them to the target
  frame, render per-class peak Gaussians (the loss target) and keep the in-frame GT
  points (for matching).
* **loss** (``compute_loss``) — foreground-weighted MSE against the peak heatmap.
* **eval** (``dense_stats`` / ``finalize_eval_metrics``) — extract predicted peaks,
  class-aware F1@δ match to GT, stream per-image ``(C, 3)`` TP/FP/FN, reduce to the
  global mF1 headline + per-image-macro secondary.
* **postprocess** — predicted ``(x, y, class, score)`` points per image.

The matching distance, Gaussian σ, and NMS radius are passed in already resolved to
target-frame pixels (the pipeline converts from µm/px via the run spacing).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from soma.detection.encode import render_peak_heatmap, transform_points_to_target
from soma.detection.io import read_points
from soma.detection.matching import VALID_MATCHING, match_points, reduce_f1
from soma.detection.peaks import extract_peaks
from soma.evaluation.metrics import resolve_metrics
from soma.tasks.base import TaskHead
from soma.tasks.registry import task_registry

if TYPE_CHECKING:
    from soma.dataset import SampleRecord
    from soma.dense import DenseSampleSpacing
    from soma.dense.geometry import DenseGridGeometry


class DetectionHead(TaskHead):
    """Dense peak-heatmap detection head (parameter-free; the decoder learns).

    Args:
        num_classes: Number of object classes ``C`` (one heatmap channel each;
            background is the absence of a peak).
        geometry: The run's :class:`DenseGridGeometry` — supplies ``encoded_size`` and
            ``crop_box`` (same source the extractor used).
        delta_px: F1 matching distance δ in target-frame pixels.
        sigma_px: Target Gaussian σ in target-frame pixels.
        nms_distance_px: NMS / local-maxima radius for peak extraction (defaults to
            ``delta_px`` so two detections cannot both satisfy one GT).
        score_threshold: Peak score threshold — a scalar (monitor default) or a
            per-class list frozen from the tune-split sweep, set before test eval.
        foreground_weight: Up-weights the MSE on non-zero (near-peak) target pixels to
            fight the heavy background imbalance; per-pixel weight ``1 + fw * target``.
        matching: ``"hungarian"`` (default, optimal one-to-one) or ``"greedy"``
            (OCELOT-official by confidence).
        sample_spacings: Resolved source and effective grid spacing for every sample,
            read from dense extraction sidecars before the head is built.
        truncate: Render each target Gaussian within ``truncate * sigma_px``.
        metrics: Metric names (validated against the ``detection`` family).
    """

    target_dtypes = {"heatmap": torch.float32, "gt_points": torch.float32}
    task_family = "detection"
    accumulates_eval_metrics = True

    def __init__(
        self,
        *,
        num_classes: int,
        geometry: "DenseGridGeometry",
        delta_px: float,
        sigma_px: float,
        nms_distance_px: float | None = None,
        score_threshold: float | list[float] = 0.5,
        foreground_weight: float = 10.0,
        matching: str = "hungarian",
        sample_spacings: Mapping[str, "DenseSampleSpacing"],
        truncate: float = 3.0,
        metrics: list[str] | None = None,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if float(delta_px) <= 0.0:
            raise ValueError(f"delta_px must be > 0, got {delta_px}")
        if float(sigma_px) <= 0.0:
            raise ValueError(f"sigma_px must be > 0, got {sigma_px}")
        if matching not in VALID_MATCHING:
            raise ValueError(f"matching must be one of {VALID_MATCHING}, got {matching!r}")
        if float(foreground_weight) < 0.0:
            raise ValueError(f"foreground_weight must be >= 0, got {foreground_weight}")
        self.num_classes = int(num_classes)
        self.delta_px = float(delta_px)
        self.sigma_px = float(sigma_px)
        self.nms_distance_px = float(nms_distance_px) if nms_distance_px is not None else float(delta_px)
        self.score_threshold: float | list[float] = score_threshold
        self.foreground_weight = float(foreground_weight)
        self.matching = matching
        self._sample_spacings = {
            str(sample_id): spacing for sample_id, spacing in sample_spacings.items()
        }
        self.truncate = float(truncate)
        self._encoded_size = tuple(int(s) for s in geometry.encoded_size)
        self._crop_box = tuple(int(v) for v in geometry.crop_box)
        self.metrics = resolve_metrics("detection", metrics or [])

    # --- geometry ---------------------------------------------------------- #

    def forward(self, X: Tensor) -> Tensor:
        """Decoder output ``(B, C, h', w')`` -> target-res heatmap ``(B, C, H, W)`` in [0, 1]."""
        if X.ndim != 4:
            raise ValueError(f"DetectionHead expects (B, C, h', w'), got {tuple(X.shape)}")
        upsampled = F.interpolate(X, size=self._encoded_size, mode="bilinear", align_corners=False)
        top, left, height, width = self._crop_box
        cropped = upsampled[:, :, top : top + height, left : left + width]
        return torch.sigmoid(cropped)

    # --- targets ----------------------------------------------------------- #

    def spacing_for_sample(self, sample_id: str) -> "DenseSampleSpacing":
        try:
            return self._sample_spacings[str(sample_id)]
        except KeyError:
            raise ValueError(
                f"Detection sample '{sample_id}' has no resolved dense spacing provenance."
            ) from None

    def _sample_target_points(self, record: "SampleRecord") -> tuple[np.ndarray, np.ndarray]:
        """Read + transform a sample's points into in-frame ``(xy, classes)``."""
        if getattr(record, "points_path", None) is None:
            raise ValueError(f"detection sample '{record.sample_id}' has no points_path")
        xy_l0, classes = read_points(record.points_path)
        spacing = self.spacing_for_sample(record.sample_id)
        top, left, height, width = self._crop_box
        xy = transform_points_to_target(
            xy_l0,
            source_spacing_um=spacing.source_spacing_um,
            effective_spacing_um=spacing.effective_spacing_um,
            crop_top=top, crop_left=left,
        )
        # Keep only points landing inside this tile's target frame (the supervised set).
        if xy.shape[0]:
            inside = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
            xy, classes = xy[inside], classes[inside]
        invalid = sorted({int(c) for c in classes if not 0 <= int(c) < self.num_classes})
        if invalid:
            raise ValueError(
                f"points for '{record.sample_id}' have class id(s) {invalid} outside "
                f"[0, num_classes={self.num_classes}). Map annotation classes to 0-based ids."
            )
        return xy, classes

    def extract_targets(self, record: "SampleRecord") -> dict[str, Tensor]:
        xy, classes = self._sample_target_points(record)
        _, _, height, width = self._crop_box
        heatmap = render_peak_heatmap(
            xy, classes, target_size=(height, width),
            num_classes=self.num_classes, sigma=self.sigma_px, truncate=self.truncate,
        )
        if xy.shape[0]:
            gt = np.concatenate([xy, classes.reshape(-1, 1).astype(np.float64)], axis=1)
        else:
            gt = np.zeros((0, 3), dtype=np.float64)
        return {"heatmap": heatmap, "gt_points": torch.from_numpy(gt).to(torch.float32)}

    # --- loss -------------------------------------------------------------- #

    def compute_loss(self, predictions: Tensor, targets: dict[str, Tensor]) -> Tensor:
        target = targets["heatmap"]
        weight = 1.0 + self.foreground_weight * target
        return (weight * (predictions - target) ** 2).mean()

    # --- eval -------------------------------------------------------------- #

    @staticmethod
    def _strip_padding(gt_points: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """``(K, 3)`` padded GT rows -> ``(xy (k, 2), class (k,))`` (NaN rows dropped)."""
        arr = gt_points.detach().cpu().numpy()
        if arr.size == 0:
            return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.int64)
        valid = ~np.isnan(arr).any(axis=1)
        arr = arr[valid]
        return arr[:, :2].astype(np.float64), arr[:, 2].astype(np.int64)

    def _predict_points(self, heatmap: Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return extract_peaks(
            heatmap, min_distance=self.nms_distance_px, score_threshold=self.score_threshold
        )

    def dense_stats(self, raw_output: Tensor, targets: dict[str, Tensor]) -> Tensor:
        """Per-image, per-class ``(B, C, 3)`` TP/FP/FN at the current ``score_threshold``."""
        gt_points = targets["gt_points"]
        rows = []
        for b in range(raw_output.shape[0]):
            pred_xy, pred_cls, pred_score = self._predict_points(raw_output[b])
            gt_xy, gt_cls = self._strip_padding(gt_points[b])
            counts = match_points(
                pred_xy, pred_cls, pred_score, gt_xy, gt_cls,
                num_classes=self.num_classes, delta=self.delta_px, method=self.matching,
            )
            rows.append(torch.from_numpy(counts).to(torch.long))
        return torch.stack(rows, dim=0)

    def finalize_eval_metrics(self, counts: Tensor) -> dict[str, float]:
        """Reduce accumulated ``(N, C, 3)`` counts to the selected detection metrics."""
        glob = reduce_f1(counts, num_classes=self.num_classes, aggregation="dataset_global")
        selected: dict[str, float] = {}
        if "mean_f1" in self.metrics:
            selected["mean_f1"] = glob["mean_f1"]
        if "f1_per_class" in self.metrics:
            for c in range(self.num_classes):
                selected[f"f1_class_{c}"] = glob[f"f1_class_{c}"]
        if "precision" in self.metrics:
            for c in range(self.num_classes):
                selected[f"precision_class_{c}"] = glob[f"precision_class_{c}"]
        if "recall" in self.metrics:
            for c in range(self.num_classes):
                selected[f"recall_class_{c}"] = glob[f"recall_class_{c}"]
        if "mean_f1_per_image" in self.metrics:
            macro = reduce_f1(counts, num_classes=self.num_classes, aggregation="per_image_macro")
            selected["mean_f1_per_image"] = macro["mean_f1"]
        return selected

    def compute_metrics(self, raw_output: Tensor, targets: dict[str, Tensor]) -> dict[str, float]:
        return self.finalize_eval_metrics(self.dense_stats(raw_output, targets))

    def postprocess(self, raw_output: Tensor) -> dict[str, Any]:
        points_xy, points_class, points_score = [], [], []
        for b in range(raw_output.shape[0]):
            xy, cls, score = self._predict_points(raw_output[b])
            points_xy.append(xy)
            points_class.append(cls)
            points_score.append(score)
        return {"points_xy": points_xy, "points_class": points_class, "points_score": points_score}


task_registry.register("detection", DetectionHead)
