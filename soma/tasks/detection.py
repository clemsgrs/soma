"""Detection task head — dense peak-heatmap regression over a token grid.

The detection counterpart of :class:`~soma.tasks.segmentation.SegmentationHead`. Like
it, the head is parameter-free (the learnable weights live in the ``decoder``) and owns
the geometry: it interpolates the decoder's ``(B, C, h', w')`` output to ``encoded_size``,
crops to ``target_size`` via ``crop_box``, then applies a sigmoid so the heatmap is a
bounded ``[0, 1]`` regression target. Beyond geometry it owns the detection contract:

* **targets** — the batch carries raw supervision: ``points`` ``(B, K, 3)`` level-0
  ``(x, y, class)`` rows (NaN-padded) and an optional ``ignore`` ``(B, H, W)`` map in the
  target frame. The head maps the points into its bound frame and renders the per-class
  peak Gaussians on the logits' device at loss time; nothing is precomputed per sample.
* **loss** (``compute_loss``) — foreground-weighted MSE against the peak heatmap,
  averaged over the supervised pixels.
* **training-time validation** (``dense_stats`` / ``finalize_eval_metrics``) — peaks at
  the fixed ``validation_threshold``, class-aware F1@δ against the in-frame points, so
  ``EarlyStopping``'s hold-at-floor logic sees a stable operating point.
* **predictions** (``postprocess``) — every peak above ``min_score``, mapped back to
  level-0 pixels. Choosing an operating threshold is a frame operation on the
  predictions (:meth:`~soma.evaluation.PointPredictions.with_threshold`,
  :func:`~soma.evaluation.sweep_threshold`), never head state.

Distances are configured in µm and resolved to frame pixels with ``spacing_um`` (the
grid frame's µm/px): µm is spacing-invariant, px would silently change the physical
tolerance across encoders and spacings.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from soma.detection.encode import points_on_valid
from soma.detection.matching import VALID_MATCHING, match_points, reduce_f1
from soma.detection.peaks import extract_peaks
from soma.evaluation.metrics import resolve_metrics
from soma.params import check_choice, check_non_negative, check_positive, check_probability
from soma.tasks.base import TaskHead, as_grid_geometry
from soma.tasks.registry import task_registry

if TYPE_CHECKING:
    from soma.data import SampleRecord
    from soma.evaluation.grid import PointPredictions


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


@dataclass
class DetectionParams:
    """:class:`DetectionHead`'s params. ``num_classes`` and the class names follow from
    the task's class scheme; ``geometry`` and ``spacing_um`` from the grid source;
    ``metrics`` from evaluation. ``match_distance`` (µm) must be set."""

    num_classes: int | None = None
    match_distance: float | None = None
    sigma: float | None = None
    nms_distance: float | None = None
    min_score: float = 0.01
    validation_threshold: float = 0.5
    foreground_weight: float = 10.0
    matching: str = "hungarian"
    truncate: float = 3.0

    def __post_init__(self) -> None:
        check_positive("num_classes", self.num_classes, integer=True, optional=True)
        for name in ("match_distance", "sigma", "nms_distance"):
            check_positive(name, getattr(self, name), optional=True)
        check_probability("min_score", self.min_score)
        check_probability("validation_threshold", self.validation_threshold)
        check_non_negative("foreground_weight", self.foreground_weight)
        check_choice("matching", self.matching, VALID_MATCHING)


class DetectionHead(TaskHead):
    """Dense peak-heatmap detection head (parameter-free; the decoder learns).

    Args:
        num_classes: Number of object classes ``C`` (one heatmap channel each;
            background is the absence of a peak).
        geometry: The run's :class:`~soma.data.GridGeometry`: ``encoded_size`` and
            ``crop_box`` for the logits, and the level-0 anchor (origin, level-0 pixels
            per frame pixel) mapping point annotations into the frame.
        spacing_um: µm/px of the frame the heatmap lives in (the grid's effective
            spacing); converts the µm distances below to frame pixels.
        match_distance: F1 matching distance δ in µm.
        sigma: Target Gaussian σ in µm (default δ/3).
        nms_distance: NMS / local-maxima radius in µm (default δ, so two detections
            cannot both satisfy one ground-truth point).
        min_score: Floor on the peaks :meth:`postprocess` emits. Keep it low: thresholds
            are chosen afterwards on the predictions.
        validation_threshold: Fixed peak threshold of the training-time validation
            metrics (``dense_stats``).
        foreground_weight: Up-weights the MSE on near-peak target pixels to fight the
            background imbalance; per-pixel weight ``1 + fw * target``.
        matching: ``"hungarian"`` (optimal one-to-one) or ``"greedy"`` (by confidence,
            OCELOT-official); the training-time validation matcher and the default the
            pipeline scores with.
        truncate: Render each target Gaussian within ``truncate * sigma``.
        metrics: Metric names (validated against the ``detection`` family).
        class_names: One name per class index, carried by the predictions.
    """

    target_dtypes = {"points": torch.float32, "ignore": torch.bool}
    target_schema: dict[str, type] = {}
    task_family = "detection"
    target_kind = "points"
    accumulates_eval_metrics = True

    def __init__(
        self,
        *,
        num_classes: int,
        geometry,
        spacing_um: float,
        match_distance: float,
        sigma: float | None = None,
        nms_distance: float | None = None,
        min_score: float = 0.01,
        validation_threshold: float = 0.5,
        foreground_weight: float = 10.0,
        matching: str = "hungarian",
        truncate: float = 3.0,
        metrics: list[str] | None = None,
        class_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if matching not in VALID_MATCHING:
            raise ValueError(f"matching must be one of {VALID_MATCHING}, got {matching!r}")
        if float(foreground_weight) < 0.0:
            raise ValueError(f"foreground_weight must be >= 0, got {foreground_weight}")
        if not 0.0 <= float(min_score) <= 1.0:
            raise ValueError(f"min_score must be in [0, 1], got {min_score}")
        if not 0.0 <= float(validation_threshold) <= 1.0:
            raise ValueError(f"validation_threshold must be in [0, 1], got {validation_threshold}")
        self.num_classes = int(num_classes)
        self.spacing_um = _positive("spacing_um", spacing_um)
        self.match_distance = _positive("match_distance", match_distance)
        self.delta_px = self.match_distance / self.spacing_um
        self.sigma_px = (
            self.delta_px / 3.0 if sigma is None else _positive("sigma", sigma) / self.spacing_um
        )
        self.nms_distance_px = (
            self.delta_px
            if nms_distance is None
            else _positive("nms_distance", nms_distance) / self.spacing_um
        )
        self.min_score = float(min_score)
        self.validation_threshold = float(validation_threshold)
        self.foreground_weight = float(foreground_weight)
        self.matching = matching
        self.truncate = float(truncate)
        self._geometry = as_grid_geometry(geometry)
        self._encoded_size = tuple(int(s) for s in self._geometry.encoded_size)
        self._crop_box = tuple(int(v) for v in self._geometry.crop_box)
        self.metrics = resolve_metrics("detection", metrics or [])
        if class_names is None:
            class_names = [f"class_{index}" for index in range(self.num_classes)]
        if len(class_names) != self.num_classes:
            raise ValueError(
                f"class_names has {len(class_names)} entries for num_classes={self.num_classes}."
            )
        self.class_names = tuple(str(name) for name in class_names)

    @property
    def geometry(self):
        """The bound :class:`~soma.data.GridGeometry` (frame layout + level-0 anchor)."""
        return self._geometry

    @property
    def classes(self) -> tuple[str, ...]:
        return self.class_names

    @property
    def match_distance_px(self) -> float:
        """The matching tolerance in level-0 pixels (the predictions' frame)."""
        return self.delta_px * float(self._geometry.level0_px_per_token_px)

    # --- geometry ---------------------------------------------------------- #

    def forward(self, X: Tensor) -> Tensor:
        """Decoder output ``(B, C, h', w')`` -> target-res heatmap ``(B, C, H, W)`` in [0, 1]."""
        if X.ndim != 4:
            raise ValueError(f"DetectionHead expects (B, C, h', w'), got {tuple(X.shape)}")
        upsampled = F.interpolate(X, size=self._encoded_size, mode="bilinear", align_corners=False)
        top, left, height, width = self._crop_box
        cropped = upsampled[:, :, top : top + height, left : left + width]
        return torch.sigmoid(cropped)

    def to_frame(self, xy_level0: Tensor) -> Tensor:
        """Level-0 ``(N, 2)`` ``(x, y)`` points -> target-frame pixels (float)."""
        top, left = self._crop_box[0], self._crop_box[1]
        origin = torch.as_tensor(self._geometry.origin_level0, dtype=xy_level0.dtype, device=xy_level0.device)
        frame = (xy_level0 - origin) / float(self._geometry.level0_px_per_token_px)
        offset = torch.as_tensor([left, top], dtype=xy_level0.dtype, device=xy_level0.device)
        return frame - offset

    def to_level0(self, xy_frame: np.ndarray) -> np.ndarray:
        """Target-frame ``(N, 2)`` points -> level-0 pixels (the inverse of :meth:`to_frame`)."""
        top, left = self._crop_box[0], self._crop_box[1]
        xy = np.asarray(xy_frame, dtype=np.float64).reshape(-1, 2) + np.array([left, top], dtype=np.float64)
        return xy * float(self._geometry.level0_px_per_token_px) + np.asarray(
            self._geometry.origin_level0, dtype=np.float64
        )

    # --- targets ----------------------------------------------------------- #

    def extract_targets(self, record: "SampleRecord") -> dict[str, Tensor]:
        raise TypeError(
            "DetectionHead reads its points from a TargetSource, not from records: pass "
            "targets=PointSource(...) to soma.learn.loaders"
        )

    def check_target(self, sample_id: str, points) -> None:
        """Refuse points whose class id is outside ``[0, num_classes)``, naming the sample."""
        invalid = sorted({int(c) for c in points.classes.tolist() if not 0 <= int(c) < self.num_classes})
        if invalid:
            raise ValueError(
                f"points for '{sample_id}' have class id(s) {invalid} outside "
                f"[0, num_classes={self.num_classes}). Name the annotated ids in "
                "task.params.classes (PointSource class_remap), or map them to 0-based ids."
            )

    def frame_points(self, points: Tensor, ignore: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """One sample's supervised points in the frame: ``(xy (k, 2) float, class (k,) long)``.

        Strips the NaN padding, maps level-0 rows into the frame and keeps the points
        that land inside it and, with an ``ignore`` map, on a supervised pixel.
        """
        rows = points[~torch.isnan(points).any(dim=1)]
        classes = rows[:, 2].round().to(torch.long)
        if rows.shape[0]:
            invalid = sorted({int(c) for c in classes.tolist() if not 0 <= int(c) < self.num_classes})
            if invalid:
                raise ValueError(
                    f"point class id(s) {invalid} outside [0, num_classes={self.num_classes}); "
                    "remap annotated ids through PointSource(class_remap=...)"
                )
        xy = self.to_frame(rows[:, :2].to(torch.float64)).to(torch.float32)
        height, width = self._crop_box[2], self._crop_box[3]
        keep = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        if ignore is not None and xy.shape[0]:
            cols = xy[:, 0].round().clamp(0, width - 1).to(torch.long)
            rows_ = xy[:, 1].round().clamp(0, height - 1).to(torch.long)
            keep &= ~ignore[rows_, cols]
        return xy[keep], classes[keep]

    def render_heatmap(self, xy: Tensor, classes: Tensor) -> Tensor:
        """Per-class peak Gaussians ``(C, H, W)`` for frame points, on their device.

        Peak value 1 at each point, overlaps of one class merged by element-wise max; a
        point whose rounded centre leaves the frame is dropped.
        """
        height, width = self._crop_box[2], self._crop_box[3]
        heat = torch.zeros(self.num_classes * height * width, dtype=torch.float32, device=xy.device)
        if xy.shape[0] == 0:
            return heat.view(self.num_classes, height, width)
        radius = int(math.ceil(self.truncate * self.sigma_px))
        two_sigma_sq = 2.0 * self.sigma_px * self.sigma_px
        cx, cy = xy[:, 0].round().to(torch.long), xy[:, 1].round().to(torch.long)
        inside = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)
        xy, classes, cx, cy = xy[inside], classes[inside], cx[inside], cy[inside]
        offsets = torch.arange(-radius, radius + 1, device=xy.device)
        xs = cx[:, None] + offsets[None, :]  # (k, w)
        ys = cy[:, None] + offsets[None, :]  # (k, w)
        dx2 = (xs.to(torch.float32) - xy[:, 0:1]) ** 2
        dy2 = (ys.to(torch.float32) - xy[:, 1:2]) ** 2
        values = torch.exp(-(dy2[:, :, None] + dx2[:, None, :]) / two_sigma_sq)  # (k, w, w)
        on_canvas = ((ys >= 0) & (ys < height))[:, :, None] & ((xs >= 0) & (xs < width))[:, None, :]
        index = (
            classes[:, None, None] * height * width
            + ys.clamp(0, height - 1)[:, :, None] * width
            + xs.clamp(0, width - 1)[:, None, :]
        )
        heat.scatter_reduce_(0, index[on_canvas], values[on_canvas], reduce="amax")
        return heat.view(self.num_classes, height, width)

    def encode_targets(self, targets: dict[str, Tensor]) -> tuple[Tensor, Tensor, list[tuple[Tensor, Tensor]]]:
        """``(heatmap (B, C, H, W), valid (B, H, W), frame points per image)`` on device."""
        points = targets["points"]
        ignore = targets.get("ignore")
        height, width = self._crop_box[2], self._crop_box[3]
        heatmaps, frames = [], []
        for b in range(points.shape[0]):
            xy, classes = self.frame_points(points[b], None if ignore is None else ignore[b])
            frames.append((xy, classes))
            heatmaps.append(self.render_heatmap(xy, classes))
        if heatmaps:
            heatmap = torch.stack(heatmaps)
        else:
            heatmap = torch.zeros(0, self.num_classes, height, width, device=points.device)
        valid = (
            torch.ones(points.shape[0], height, width, dtype=torch.bool, device=points.device)
            if ignore is None
            else ~ignore.to(torch.bool)
        )
        return heatmap, valid, frames

    # --- loss -------------------------------------------------------------- #

    def compute_loss(self, predictions: Tensor, targets: dict[str, Tensor]) -> Tensor:
        target, valid, _ = self.encode_targets(targets)
        target = target.to(predictions.dtype)
        weight = 1.0 + self.foreground_weight * target
        per_pixel = weight * (predictions - target) ** 2
        if bool(valid.all()):
            return per_pixel.mean()
        # Average over the supervised pixels of every channel; ignored pixels carry no loss.
        mask = valid.unsqueeze(1).expand_as(per_pixel)
        return per_pixel[mask].sum() / mask.sum().clamp(min=1)

    # --- training-time validation ------------------------------------------ #

    def _peaks(self, heatmap: Tensor, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return extract_peaks(heatmap, min_distance=self.nms_distance_px, score_threshold=threshold)

    def dense_stats(self, raw_output: Tensor, targets: dict[str, Tensor]) -> Tensor:
        """Per-image, per-class ``(B, C, 3)`` TP/FP/FN at ``validation_threshold``."""
        _, valid, frames = self.encode_targets(targets)
        rows = []
        for b in range(raw_output.shape[0]):
            pred_xy, pred_cls, pred_score = self._peaks(raw_output[b], self.validation_threshold)
            if pred_xy.shape[0]:
                keep = points_on_valid(pred_xy, valid[b].detach().cpu().numpy())
                pred_xy, pred_cls, pred_score = pred_xy[keep], pred_cls[keep], pred_score[keep]
            gt_xy, gt_cls = frames[b]
            counts = match_points(
                pred_xy, pred_cls, pred_score,
                gt_xy.detach().cpu().numpy().astype(np.float64),
                gt_cls.detach().cpu().numpy().astype(np.int64),
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

    # --- predictions ------------------------------------------------------- #

    def postprocess(self, raw_output: Tensor) -> dict[str, Any]:
        """Every peak above ``min_score`` per image as ``(k, 4)`` level-0 ``(x, y, class, score)``."""
        points = []
        for b in range(raw_output.shape[0]):
            xy, cls, score = self._peaks(raw_output[b], self.min_score)
            rows = np.zeros((xy.shape[0], 4), dtype=np.float64)
            if xy.shape[0]:
                rows[:, :2] = self.to_level0(xy)
                rows[:, 2] = cls
                rows[:, 3] = score
            points.append(rows)
        return {"points": points}

    @property
    def prediction_columns(self) -> tuple[str, ...]:
        return ("x", "y", "class", "score")

    def to_columns(self, processed: dict[str, Any]) -> dict[str, Any]:
        """One ``(k, 4)`` level-0 point array per sample."""
        return {"points": processed["points"]}

    def assemble_predictions(
        self, sample_ids: Sequence[str], columns: dict[str, list[Any]]
    ) -> "PointPredictions":
        """The noun :func:`soma.predict` returns for this head."""
        import pandas as pd

        from soma.evaluation.grid import PointPredictions

        frames = []
        for sample_id, rows in zip(sample_ids, columns.get("points", []), strict=True):
            rows = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
            frames.append(
                pd.DataFrame(
                    {
                        "sample_id": [str(sample_id)] * rows.shape[0],
                        "x": rows[:, 0],
                        "y": rows[:, 1],
                        "class": rows[:, 2].astype(np.int64),
                        "score": rows[:, 3],
                    }
                )
            )
        frame = pd.concat(frames, ignore_index=True) if frames else None
        return PointPredictions(
            frame,
            classes=self.classes,
            sample_ids=list(sample_ids),
            match_distance_px=self.match_distance_px,
        )


task_registry.register("detection", DetectionHead, params=DetectionParams)
