"""Segmentation task head — dense per-pixel classification.

Unlike the scalar heads, the segmentation head is parameter-free: the learnable
weights live in the ``decoder``. The head owns the *geometry* (resize the decoder
logits to ``encoded_size`` and crop to the mask's ``target_size`` via ``crop_box``)
plus the target/loss/metric/postprocess contract. The resize+crop happens once, in
``forward``, so ``compute_loss``/``compute_metrics``/``postprocess`` all receive
target-resolution logits — no chance of three divergent resize paths.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from soma.dense.reader import accepted_mask_values
from soma.dense.reader import load_mask as load_mask
from soma.evaluation.metrics import resolve_metrics
from soma.tasks.base import TaskHead, as_grid_geometry
from soma.tasks.dense_metrics import (
    dense_confusion_counts,
    reduce_dice_iou,
    reduce_foreground_dice,
    segmentation_loss,
)
from soma.tasks.registry import task_registry

if TYPE_CHECKING:
    from soma.data import SampleRecord
    from soma.dense.geometry import DenseGridGeometry
    from soma.evaluation.grid import LabelMapPredictions


class SegmentationHead(TaskHead):
    """Dense per-pixel classification head (parameter-free; the decoder learns).

    Args:
        num_classes: Number of segmentation classes ``C``.
        geometry: The run's :class:`DenseGridGeometry` — supplies ``encoded_size``
            and ``crop_box`` so ``forward`` maps decoder logits to the mask's
            ``target_size``. The same geometry the extractor used (single source).
        ignore_index: Mask value excluded from loss and metrics.
        dice_weight: Weight on the overlap (soft-Dice / Tversky) term added to the
            region (cross-entropy) term.
        class_weights: Optional per-class ``(num_classes,)`` weights for the
            cross-entropy term (e.g. inverse frequency) up-weighting rare classes.
            ``None`` = unweighted.
        ce_gamma: Focal exponent on cross-entropy. ``0.0`` = plain (optionally
            class-weighted) CE; ``> 0`` down-weights easy, well-classified pixels.
        tversky_alpha / tversky_beta / tversky_gamma: Overlap-term shape. The
            default ``(0.5, 0.5, 1.0)`` *is* soft-Dice; ``beta > alpha`` penalizes
            false negatives more (recall-oriented for small structures), ``gamma > 1``
            focuses on hard, low-overlap classes (focal-Tversky).
        metrics: Metric names (validated against the ``segmentation`` family);
            empty uses the default ``[mean_dice, mean_iou]``. Request
            ``dataset_global_mean_dice`` explicitly to sum counts over the split
            before averaging class Dice, or ``foreground_mean_dice`` for the
            background-free, empty-target-skipping per-image mean.

        class_names: One name per class index, carried by the predictions.

    Supervision is a raw ``(B, H, W)`` label map at the target size (``ignore_index``
    folded in), served by a :class:`~soma.data.LabelMapSource`; ``target_kind`` declares
    it so :func:`soma.learn.loaders` refuses a points source. The mask-read arguments
    (``spacing_um`` ... ``pixel_mapping``) describe how the pipeline builds that source
    (:attr:`label_map_read_params`); the head itself never reads a file.
    """

    target_dtypes = {"mask": torch.long}
    target_schema: dict[str, type] = {}
    task_family = "segmentation"
    target_kind = "label_map"
    # Eval streams compact per-image confusion counts instead of full logits — dense
    # logits (N, C, H, W) would OOM if concatenated across a cohort. The trainer /
    # evaluator accumulate `dense_stats` rows and call `finalize_eval_metrics`.
    accumulates_eval_metrics = True

    def __init__(
        self,
        *,
        num_classes: int,
        geometry: "DenseGridGeometry",
        ignore_index: int = 255,
        dice_weight: float = 1.0,
        class_weights: list[float] | None = None,
        ce_gamma: float = 0.0,
        tversky_alpha: float = 0.5,
        tversky_beta: float = 0.5,
        tversky_gamma: float = 1.0,
        metrics: list[str] | None = None,
        spacing_um: float | None = None,
        spacing_policy: str = "strict",
        backend: str = "auto",
        image_backend: str = "auto",
        tolerance: float = 0.05,
        label_remap: "np.ndarray | None" = None,
        pixel_mapping: Mapping[str, int | list[int]] | None = None,
        class_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if 0 <= ignore_index < num_classes:
            raise ValueError(
                f"ignore_index {ignore_index} must be outside [0, num_classes={num_classes})."
            )
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.dice_weight = float(dice_weight)
        if class_weights is not None:
            class_weights = [float(w) for w in class_weights]
            if len(class_weights) != self.num_classes:
                raise ValueError(
                    f"class_weights must have num_classes={self.num_classes} entries, "
                    f"got {len(class_weights)}."
                )
            if any(w < 0.0 for w in class_weights):
                raise ValueError(f"class_weights must be non-negative, got {class_weights}.")
        self.class_weights = class_weights
        if float(ce_gamma) < 0.0:
            raise ValueError(f"ce_gamma must be >= 0, got {ce_gamma}.")
        self.ce_gamma = float(ce_gamma)
        if float(tversky_alpha) < 0.0 or float(tversky_beta) < 0.0:
            raise ValueError(
                f"tversky_alpha/beta must be >= 0, got ({tversky_alpha}, {tversky_beta})."
            )
        if float(tversky_gamma) <= 0.0:
            raise ValueError(f"tversky_gamma must be > 0, got {tversky_gamma}.")
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.tversky_gamma = float(tversky_gamma)
        # When set, masks are read spacing-aware (hs2p) at this µm/px to register
        # against the dense grid; None falls back to a flat page-0 load_mask read.
        self._spacing_um = float(spacing_um) if spacing_um is not None else None
        if spacing_policy not in {"strict", "native_if_coarser"}:
            raise ValueError(
                "spacing_policy must be 'strict' or 'native_if_coarser', got "
                f"{spacing_policy!r}"
            )
        self._spacing_policy = spacing_policy
        self._backend = backend
        # Reads each mask's image (the slide for an ROI) to align the mask to it.
        self._image_backend = image_backend
        self._tolerance = float(tolerance)
        # Optional raw-pixel → class-index LUT built from task.params.classes / ignore
        # (see soma.dense.reader.build_label_remap). None ⇒ masks are already contiguous
        # class indices.
        if label_remap is not None:
            label_remap = np.asarray(label_remap)
            if label_remap.shape != (256,):
                raise ValueError(
                    f"label_remap must be a length-256 LUT, got shape {label_remap.shape}."
                )
        self._label_remap = label_remap
        # The label vocabulary hs2p reads spacing-aware masks against: the sampling
        # pixel_mapping for slide-manifest ROIs, otherwise the values this head accepts.
        self._mask_vocabulary = (
            dict(pixel_mapping)
            if pixel_mapping is not None
            else accepted_mask_values(
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
                label_remap=label_remap,
            )
        )
        self._geometry = as_grid_geometry(geometry)
        self._encoded_size = tuple(int(s) for s in self._geometry.encoded_size)
        self._crop_box = tuple(int(v) for v in self._geometry.crop_box)
        self.metrics = resolve_metrics("segmentation", metrics or [])
        if class_names is None:
            class_names = [f"class_{index}" for index in range(self.num_classes)]
        if len(class_names) != self.num_classes:
            raise ValueError(
                f"class_names has {len(class_names)} entries for num_classes={self.num_classes}."
            )
        self.class_names = tuple(str(name) for name in class_names)

    @property
    def geometry(self):
        """The bound :class:`~soma.data.GridGeometry` (crop box, target size)."""
        return self._geometry

    @property
    def classes(self) -> tuple[str, ...]:
        return self.class_names

    @property
    def label_remap(self) -> "np.ndarray | None":
        """The raw-pixel → class-index LUT from ``task.params.classes`` (``None`` without)."""
        return self._label_remap

    @property
    def mask_vocabulary(self) -> dict[str, int]:
        """The label vocabulary spacing-aware masks are read against."""
        return dict(self._mask_vocabulary)

    @property
    def target_identity(self) -> dict[str, object]:
        """Exact mask-read/remap contract used to key reusable ROI populations."""
        return {
            "num_classes": self.num_classes,
            "ignore_index": self.ignore_index,
            "encoded_size": list(self._encoded_size),
            "crop_box": list(self._crop_box),
            "spacing_um": self._spacing_um,
            "spacing_policy": self._spacing_policy,
            "backend": self._backend,
            "tolerance": self._tolerance,
            "label_remap": (
                None
                if self._label_remap is None
                else [int(value) for value in self._label_remap.tolist()]
            ),
        }

    def forward(self, X: Tensor) -> Tensor:
        """Decoder logits ``(B, C, h', w')`` -> target-res logits ``(B, C, H, W)``.

        Interpolate to ``encoded_size`` (the padded encoder input the grid spans),
        then crop ``crop_box`` back to the mask's ``target_size``.
        """
        if X.ndim != 4:
            raise ValueError(f"SegmentationHead expects (B, C, h', w') logits, got {tuple(X.shape)}")
        upsampled = F.interpolate(
            X, size=self._encoded_size, mode="bilinear", align_corners=False
        )
        top, left, height, width = self._crop_box
        return upsampled[:, :, top : top + height, left : left + width]

    @property
    def label_map_read_params(self) -> dict[str, Any]:
        """Constructor kwargs for a :class:`~soma.data.LabelMapSource` matching this head."""
        _, _, target_h, target_w = self._crop_box
        return {
            "size": (target_h, target_w),
            "mask_vocabulary": dict(self._mask_vocabulary),
            "label_remap": self._label_remap,
            "spacing_um": self._spacing_um,
            "spacing_policy": self._spacing_policy,
            "tolerance": self._tolerance,
            "backend": self._backend,
            "image_backend": self._image_backend,
        }

    def check_target(self, sample_id: str, mask: Tensor) -> None:
        """Refuse a label map holding a value outside the classes and ``ignore_index``.

        Caught here, naming the sample, an off-by-one labeling (classes ``{1, 2, 3}``) or a
        stray value would otherwise surface as a cryptic one_hot / cross-entropy index
        assert (device-side on CUDA) far from its cause.
        """
        # An elementwise bounds test, not ``torch.unique`` over the whole map: it runs on
        # every sample of every epoch, and sorting a megapixel map dominated data loading.
        bad = ((mask < 0) | (mask >= self.num_classes)) & (mask != self.ignore_index)
        if bool(bad.any()):
            invalid = sorted(torch.unique(mask[bad]).tolist())
            raise ValueError(
                f"mask for '{sample_id}' has label value(s) {invalid} outside "
                f"[0, num_classes={self.num_classes}) ∪ {{ignore_index={self.ignore_index}}}."
            )

    def extract_targets(self, record: "SampleRecord") -> dict[str, Tensor]:
        raise TypeError(
            "SegmentationHead reads its label maps from a TargetSource, not from records: "
            "pass targets=LabelMapSource(...) to soma.learn.loaders"
        )

    def compute_loss(self, predictions: Tensor, targets: dict[str, Tensor]) -> Tensor:
        return segmentation_loss(
            predictions,
            targets["mask"],
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            dice_weight=self.dice_weight,
            class_weights=self.class_weights,
            ce_gamma=self.ce_gamma,
            tversky_alpha=self.tversky_alpha,
            tversky_beta=self.tversky_beta,
            tversky_gamma=self.tversky_gamma,
        )

    def dense_stats(self, raw_output: Tensor, targets: dict[str, Tensor]) -> Tensor:
        """Per-image, per-class confusion counts ``(B, C, 3)`` for streaming eval."""
        return dense_confusion_counts(
            raw_output,
            targets["mask"],
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

    def finalize_eval_metrics(self, counts: Tensor) -> dict[str, float]:
        """Reduce accumulated per-image confusion counts ``(N, C, 3)`` to metrics.

        The single reduce+filter path shared by ``compute_metrics`` and the
        streaming evaluator, so a batched (concatenated-logits) and a streamed
        (concatenated-counts) evaluation cannot drift. ``counts`` keeps the
        per-image axis so legacy ``mean_dice`` remains per-image macro while
        ``dataset_global_mean_dice`` can sum the same rows before reduction.
        """
        per_image_metrics = reduce_dice_iou(counts, num_classes=self.num_classes)
        # Honor the configured metric selection (like the scalar heads). "dice_per_class"
        # expands to the per-class breakdown; otherwise only the requested scalars.
        selected: dict[str, float] = {}
        if "mean_dice" in self.metrics:
            selected["mean_dice"] = per_image_metrics["mean_dice"]
        if "dataset_global_mean_dice" in self.metrics:
            dataset_global = reduce_dice_iou(
                counts,
                num_classes=self.num_classes,
                aggregation="dataset_global",
            )
            selected["dataset_global_mean_dice"] = dataset_global["mean_dice"]
        if "foreground_mean_dice" in self.metrics:
            selected["foreground_mean_dice"] = reduce_foreground_dice(
                counts, num_classes=self.num_classes
            )
        if "mean_iou" in self.metrics:
            selected["mean_iou"] = per_image_metrics["mean_iou"]
        if "dice_per_class" in self.metrics:
            for c in range(self.num_classes):
                selected[f"dice_class_{c}"] = per_image_metrics[f"dice_class_{c}"]
        return selected

    def compute_metrics(self, raw_output: Tensor, targets: dict[str, Tensor]) -> dict[str, float]:
        return self.finalize_eval_metrics(self.dense_stats(raw_output, targets))

    def postprocess(self, raw_output: Tensor) -> dict[str, Any]:
        prediction = raw_output.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
        return {"prediction": prediction}

    # --- predictions ------------------------------------------------------ #

    @property
    def prediction_columns(self) -> tuple[str, ...]:
        return ("label_map",)

    def to_columns(self, processed: dict[str, Any]) -> dict[str, Any]:
        """One ``(H, W)`` uint8 label map per sample."""
        return {"label_map": processed["prediction"]}

    def assemble_predictions(
        self, sample_ids: Sequence[str], columns: dict[str, list[Any]]
    ) -> "LabelMapPredictions":
        """The noun :func:`soma.predict` returns for this head."""
        from soma.evaluation.grid import LabelMapPredictions

        return LabelMapPredictions(
            dict(zip(sample_ids, columns.get("label_map", []), strict=True)),
            classes=self.classes,
            ignore_index=self.ignore_index,
            geometry=self.geometry,
        )


task_registry.register("segmentation", SegmentationHead)
