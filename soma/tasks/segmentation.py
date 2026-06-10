"""Segmentation task head — dense per-pixel classification.

Unlike the scalar heads, the segmentation head is parameter-free: the learnable
weights live in the ``decoder``. The head owns the *geometry* (resize the decoder
logits to ``encoded_size`` and crop to the mask's ``target_size`` via ``crop_box``)
plus the target/loss/metric/postprocess contract. The resize+crop happens once, in
``forward``, so ``compute_loss``/``compute_metrics``/``postprocess`` all receive
target-resolution logits — no chance of three divergent resize paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from soma.evaluation.metrics import resolve_metrics
from soma.tasks.base import TaskHead
from soma.tasks.dense_metrics import (
    cross_entropy_dice_loss,
    dense_confusion_counts,
    reduce_dice_iou,
)
from soma.tasks.registry import task_registry

if TYPE_CHECKING:
    from soma.dataset import SampleRecord
    from soma.dense.geometry import DenseGridGeometry


def load_mask(path, *, expected_size: tuple[int, int] | None = None) -> Tensor:
    """Load a segmentation mask as a 2-D ``long`` tensor of class indices.

    Fails loud on RGB/palette masks (a classic silent corruption — class indices
    smeared across channels) and on non-integer dtypes. v1 loads at native
    resolution (no resize); when resizing arrives it must be nearest-neighbor only
    so class indices and ``ignore_index`` survive.
    """
    with Image.open(path) as image:
        # Palette ("P"/"PA") images load as a 2-D index array and would pass the
        # rank/dtype checks below, but palette indices are NOT class ids — reject
        # explicitly so a colormapped mask can't be silently misread.
        if image.mode in ("P", "PA"):
            raise ValueError(
                f"mask '{path}' is a palette image (mode '{image.mode}'); palette indices "
                "are not class ids. Save masks as single-channel integer (e.g. 'L'/'I') rasters."
            )
        array = np.array(image)
    if array.ndim != 2:
        raise ValueError(
            f"mask '{path}' must be a 2-D single-channel class-index raster, got shape "
            f"{array.shape}. RGB/palette masks are not supported."
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"mask '{path}' must have an integer dtype (class indices), got {array.dtype}."
        )
    tensor = torch.from_numpy(array.astype(np.int64))
    if expected_size is not None and tuple(int(s) for s in tensor.shape) != tuple(expected_size):
        raise ValueError(
            f"mask '{path}' is {tuple(int(s) for s in tensor.shape)}, expected {tuple(expected_size)}."
        )
    return tensor


class SegmentationHead(TaskHead):
    """Dense per-pixel classification head (parameter-free; the decoder learns).

    Args:
        num_classes: Number of segmentation classes ``C``.
        geometry: The run's :class:`DenseGridGeometry` — supplies ``encoded_size``
            and ``crop_box`` so ``forward`` maps decoder logits to the mask's
            ``target_size``. The same geometry the extractor used (single source).
        ignore_index: Mask value excluded from loss and metrics.
        dice_weight: Weight on the soft-Dice term added to cross-entropy.
        metrics: Metric names (validated against the ``segmentation`` family);
            empty uses the default ``[mean_dice, mean_iou]``.
    """

    target_dtypes = {"mask": torch.long}
    task_family = "segmentation"
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
        metrics: list[str] | None = None,
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
        self._encoded_size = tuple(int(s) for s in geometry.encoded_size)
        self._crop_box = tuple(int(v) for v in geometry.crop_box)
        self.metrics = resolve_metrics("segmentation", metrics or [])

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

    def extract_targets(self, record: "SampleRecord") -> dict[str, Tensor]:
        if record.mask_path is None:
            raise ValueError(f"segmentation sample '{record.sample_id}' has no mask_path")
        mask = load_mask(record.mask_path)
        # Catch off-by-one labelings (e.g. classes {1,2,3}) and stray values here,
        # with the sample_id — otherwise they surface as a cryptic one_hot/cross_entropy
        # index assert (a device-side async assert on CUDA) far from the cause.
        allowed = set(range(self.num_classes)) | {self.ignore_index}
        invalid = sorted(v for v in torch.unique(mask).tolist() if v not in allowed)
        if invalid:
            raise ValueError(
                f"mask for '{record.sample_id}' has label value(s) {invalid} outside "
                f"[0, num_classes={self.num_classes}) ∪ {{ignore_index={self.ignore_index}}}."
            )
        return {"mask": mask}

    def compute_loss(self, predictions: Tensor, targets: dict[str, Tensor]) -> Tensor:
        return cross_entropy_dice_loss(
            predictions,
            targets["mask"],
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            dice_weight=self.dice_weight,
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
        (concatenated-counts) evaluation cannot drift. ``counts`` must keep the
        per-image axis — summing it to ``(C, 3)`` would silently switch the
        per-image-macro monitor metric to dataset-global.
        """
        full = reduce_dice_iou(counts, num_classes=self.num_classes)
        # Honor the configured metric selection (like the scalar heads). "dice_per_class"
        # expands to the per-class breakdown; otherwise only the requested scalars.
        selected: dict[str, float] = {}
        if "mean_dice" in self.metrics:
            selected["mean_dice"] = full["mean_dice"]
        if "mean_iou" in self.metrics:
            selected["mean_iou"] = full["mean_iou"]
        if "dice_per_class" in self.metrics:
            for c in range(self.num_classes):
                selected[f"dice_class_{c}"] = full[f"dice_class_{c}"]
        return selected

    def compute_metrics(self, raw_output: Tensor, targets: dict[str, Tensor]) -> dict[str, float]:
        return self.finalize_eval_metrics(self.dense_stats(raw_output, targets))

    def postprocess(self, raw_output: Tensor) -> dict[str, Any]:
        prediction = raw_output.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
        return {"prediction": prediction}


task_registry.register("segmentation", SegmentationHead)
