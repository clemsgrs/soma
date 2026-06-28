"""Dense (segmentation) evaluation artifacts: prediction rasters + overlays + CSV.

Streaming side-effect, not a threaded report type. :class:`DenseArtifactWriter` is
invoked once per batch by the shared dense-eval loop
(:func:`soma.training.trainer.accumulate_dense_stats`) *before* the logits are
discarded, so it never holds a cohort of ``(N, C, H, W)`` logits in memory. The
metrics flow is unchanged: segmentation still returns a plain ``EvaluationReport``
(``predictions=[]``); these are disk artifacts the report concept (design's
"DenseEvaluationReport") refers to, not a new dataclass woven through ``FoldResult``.

Artifacts written under ``fold_dir``:
  - ``preds/<split>/<sample_id>.png``    argmax class-index raster (uint8, mode L) —
                                         always written; the canonical, exact, viewable
                                         prediction.
  - ``probs/<split>/<sample_id>.npz``    float16 ``(C, H, W)`` softmax probabilities —
                                         **opt-in** (``save_segmentation_probabilities``);
                                         ~C×/precision× larger than the argmax raster, for
                                         post-hoc soft-Dice / calibration / entropy / ensembling.
  - ``pred_overlays/<split>/<sample_id>.png`` predicted foreground color-blended over
                                         the source tile — **on by default**, suppressible
                                         via ``save_segmentation_overlays`` for a metrics-only
                                         run. Also **fail-soft**: skipped (logged) when the
                                         source image is unreadable, the pred raster is still
                                         written.
  - ``gt_overlays/<split>/<sample_id>.png`` ground-truth mask color-blended over the
                                         same source tile with the same palette/alpha,
                                         for side-by-side comparison with the prediction
                                         overlay. Same on-by-default/suppressible + fail-soft
                                         behavior; ``ignore_index`` pixels are treated as
                                         background (unblended).
  - ``predictions_<split>.csv``          one row per tile: raster/overlay/gt-overlay paths
                                         + per-tile Dice/IoU (from the same per-image
                                         confusion counts the metric monitor uses).
  - ``metrics_<split>.csv``              split-level per-class Dice + means, computed
                                         unconditionally (design §9 per-class breakdown
                                         even when the monitor metrics are just means).

The per-tile Dice/IoU reuse the per-image ``(C, 3)`` confusion row already computed
in the eval loop (``reduce_dice_iou`` over a single image), so they cannot drift
from the streamed split metric.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from soma.tasks.dense_metrics import reduce_dice_iou

if TYPE_CHECKING:
    from soma.training.segmentation_dataset import SegmentationBatch

logger = logging.getLogger(__name__)

__all__ = ["class_palette", "DenseArtifactWriter"]

# Distinct, high-contrast colors for foreground classes. Class 0 is treated as
# background (black raster value; left unblended in overlays). Colors cycle if a
# run has more classes than entries — fine for visualization, not identity.
_BASE_COLORS: list[tuple[int, int, int]] = [
    (0, 0, 0),        # 0: background
    (230, 25, 75),    # red
    (60, 180, 75),    # green
    (255, 225, 25),   # yellow
    (0, 130, 200),    # blue
    (245, 130, 48),   # orange
    (145, 30, 180),   # purple
    (70, 240, 240),   # cyan
    (240, 50, 230),   # magenta
    (210, 245, 60),   # lime
    (250, 190, 212),  # pink
    (0, 128, 128),    # teal
]


def class_palette(num_classes: int) -> np.ndarray:
    """Return a deterministic ``(num_classes, 3)`` uint8 RGB palette."""
    colors = [_BASE_COLORS[c % len(_BASE_COLORS)] for c in range(num_classes)]
    return np.asarray(colors, dtype=np.uint8)


class DenseArtifactWriter:
    """Per-batch callback that writes dense prediction rasters, overlays + a CSV.

    Construct one per split; pass it as ``on_batch_output`` to
    ``accumulate_dense_stats``; call :meth:`finalize` after the loop to flush the CSV.

    Args:
        head: the ``SegmentationHead`` (for ``num_classes``).
        split: split name (``"tune"``, ``"test"``, ...) — names the subdirs + CSV.
        output_dir: the fold directory; artifacts live under it.
        dataset: the ``SegmentationManifest`` (``samples[sample_id].image_path``) for
            overlays. ``None`` disables overlays entirely (rasters still written).
        overlay_alpha: blend weight of the predicted color over the source image.
        save_segmentation_overlays: write the pred/GT color overlay PNGs. On by default
            (a viewable prediction is cheap), but suppressible for a metrics-only run.
            Rasters/probabilities/CSV are unaffected by this toggle.
        save_segmentation_probabilities: also write a per-tile float16 ``(C, H, W)`` softmax
            sidecar under ``probs/``. Off by default — the always-written argmax raster covers
            the common case; this is for post-hoc soft-output analysis.
        spacing_um/backend/tolerance: the run's read geometry, used only for slide-manifest
            ROIs (``record.region`` set), whose ``image_path`` is a whole slide — the overlay
            reads just that ROI window at the run spacing rather than opening the gigapixel
            slide. ``None`` spacing ⇒ flat tiles only (the pre-cropped path).
    """

    def __init__(
        self,
        *,
        head,
        split: str,
        output_dir: Path | str,
        dataset=None,
        overlay_alpha: float = 0.5,
        save_segmentation_overlays: bool = True,
        save_segmentation_probabilities: bool = False,
        spacing_um: float | None = None,
        backend: str = "auto",
        tolerance: float = 0.05,
    ) -> None:
        self._num_classes = int(head.num_classes)
        # GT masks may carry ignore_index (outside [0, num_classes)); treat those
        # pixels as background in the GT overlay. Default mirrors SegmentationHead.
        self._ignore_index = int(getattr(head, "ignore_index", 255))
        if self._num_classes > 256:
            # The argmax raster is uint8 (mode "L"); >256 classes would silently wrap
            # class indices. v1 segmentation assumes ≤256 classes — fail loud instead.
            raise ValueError(
                f"DenseArtifactWriter writes a uint8 class-index raster (max 256 classes); "
                f"got num_classes={self._num_classes}."
            )
        self._split = split
        self._output_dir = Path(output_dir)
        self._preds_dir = self._output_dir / "preds" / split
        self._probs_dir = self._output_dir / "probs" / split
        self._pred_overlays_dir = self._output_dir / "pred_overlays" / split
        self._gt_overlays_dir = self._output_dir / "gt_overlays" / split
        self._save_segmentation_probabilities = bool(save_segmentation_probabilities)
        self._dataset = dataset
        # Overlays are written only when enabled (default) AND a dataset supplies the
        # source tiles; suppressing the flag yields a metrics-only run (rasters/CSV stay).
        self._overlays_enabled = dataset is not None and bool(save_segmentation_overlays)
        self._spacing_um = None if spacing_um is None else float(spacing_um)
        self._backend = backend
        self._tolerance = float(tolerance)
        self._palette = class_palette(self._num_classes)
        self._overlay_alpha = float(overlay_alpha)
        self._rows: list[dict] = []
        self._stat_rows: list[torch.Tensor] = []

    def __call__(
        self,
        batch: "SegmentationBatch",
        logits: torch.Tensor,
        stat_row: torch.Tensor,
    ) -> None:
        # logits are already target-res (the head cropped them); argmax per pixel.
        self._stat_rows.append(stat_row)
        preds = logits.argmax(dim=1).to(torch.uint8).cpu().numpy()  # (B, H, W)
        masks = batch.targets["mask"].cpu().numpy()  # (B, H, W) GT class indices
        # Softmax only when the sidecar is enabled (the eval loop discards `logits`
        # after this call, so the conversion has to happen here, not in finalize).
        probs = (
            logits.softmax(dim=1).to(torch.float16).cpu().numpy()  # (B, C, H, W)
            if self._save_segmentation_probabilities
            else None
        )
        for i, sample_id in enumerate(batch.sample_ids):
            pred = preds[i]
            pred_path = self._write_pred_raster(sample_id, pred)
            probs_path = self._write_probs(sample_id, probs[i]) if probs is not None else None
            # Load the source tile once and blend both overlays from it (the GT and
            # prediction overlays share the same image/palette/alpha, so a viewer can
            # compare them side by side). Returns (None, None) if overlays are disabled
            # or the source image is unreadable (fail-soft).
            overlay_path, gt_overlay_path = self._write_overlays(sample_id, pred, masks[i])
            # Per-tile Dice/IoU from this image's own confusion row (per_image_macro
            # over its defined classes) — the same reduction as the split metric.
            tile = reduce_dice_iou(stat_row[i : i + 1], num_classes=self._num_classes)
            self._rows.append(
                {
                    "sample_id": sample_id,
                    "pred_path": self._rel(pred_path),
                    "probs_path": "" if probs_path is None else self._rel(probs_path),
                    "pred_overlay_path": "" if overlay_path is None else self._rel(overlay_path),
                    "gt_overlay_path": "" if gt_overlay_path is None else self._rel(gt_overlay_path),
                    "dice": tile["mean_dice"],
                    "iou": tile["mean_iou"],
                }
            )

    def _rel(self, path: Path) -> str:
        return str(path.relative_to(self._output_dir))

    def _write_pred_raster(self, sample_id: str, pred: np.ndarray) -> Path:
        # mkdir lazily on first write so an empty split leaves no stray dir.
        self._preds_dir.mkdir(parents=True, exist_ok=True)
        path = self._preds_dir / f"{sample_id}.png"
        Image.fromarray(pred, mode="L").save(path)
        return path

    def _write_probs(self, sample_id: str, probs: np.ndarray) -> Path:
        # Compressed float16 (C, H, W) softmax sidecar. argmax(axis=0) recovers the
        # raster; keyed "probs" so readers don't depend on np.load's default arr_0.
        self._probs_dir.mkdir(parents=True, exist_ok=True)
        path = self._probs_dir / f"{sample_id}.npz"
        np.savez_compressed(path, probs=probs)
        return path

    def _write_overlays(
        self, sample_id: str, pred: np.ndarray, mask: np.ndarray
    ) -> tuple[Path | None, Path | None]:
        """Write the prediction and GT overlays for one tile from a single source read.

        Returns ``(pred_overlay_path, gt_overlay_path)``; both are ``None`` when
        overlays are disabled (no dataset) or the source image is unreadable
        (fail-soft — the pred raster is still written by the caller).
        """
        if not self._overlays_enabled:
            return None, None
        record = self._dataset.samples.get(sample_id)
        if record is None:
            return None, None
        height, width = pred.shape
        region = getattr(record, "region", None)
        try:
            if region is not None:
                # Slide-manifest ROI: image_path is a whole slide — read just this ROI
                # window at the run spacing (never open the gigapixel slide with PIL).
                if self._spacing_um is None:
                    raise ValueError("slide-manifest overlay needs a run spacing")
                from soma.dense.reader import read_image_region_at_spacing

                image_arr = read_image_region_at_spacing(
                    record.image_path,
                    location=(int(region[0]), int(region[1])),
                    size=(width, height),  # hs2p size is (w, h)
                    spacing_um=self._spacing_um,
                    backend=self._backend,
                    tolerance=self._tolerance,
                ).astype(np.float32)
            else:
                with Image.open(record.image_path) as img:
                    image = img.convert("RGB")
                if image.size != (width, height):
                    image = image.resize((width, height))  # PIL size is (W, H)
                image_arr = np.asarray(image, dtype=np.float32)
        except (FileNotFoundError, OSError, ValueError, Image.DecompressionBombError):
            # Fail-soft: cached-feature runs may not retain the source tile, and a slide
            # whose ROI window can't be read shouldn't sink the whole eval.
            logger.debug("overlays skipped for '%s': source image unreadable", sample_id)
            return None, None
        pred_path = self._blend_and_save(image_arr, pred, self._pred_overlays_dir, sample_id)
        gt_path = self._blend_and_save(image_arr, mask, self._gt_overlays_dir, sample_id)
        return pred_path, gt_path

    def _blend_and_save(
        self, image_arr: np.ndarray, raster: np.ndarray, out_dir: Path, sample_id: str
    ) -> Path:
        """Color-blend foreground classes of ``raster`` over ``image_arr`` and save.

        Class 0 (background) and any out-of-range value (e.g. ``ignore_index`` in a GT
        mask) show the raw tile; in-range foreground classes get palette color at
        ``overlay_alpha``. Predictions never contain out-of-range values, so the same
        path serves both the prediction and GT overlays.
        """
        in_range = (raster >= 0) & (raster < self._num_classes)
        safe = np.where(in_range, raster, 0)  # clamp so palette indexing never wraps/errors
        color = self._palette[safe].astype(np.float32)
        foreground = (in_range & (raster != 0))[..., None]
        blended = np.where(
            foreground,
            (1.0 - self._overlay_alpha) * image_arr + self._overlay_alpha * color,
            image_arr,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{sample_id}.png"
        Image.fromarray(blended.astype(np.uint8), mode="RGB").save(path)
        return path

    def finalize(self) -> Path:
        """Flush ``predictions_<split>.csv`` (per tile) + ``metrics_<split>.csv``
        (per-class Dice + means), returning the predictions CSV path.

        ``metrics_<split>.csv`` is computed unconditionally via ``reduce_dice_iou``
        over all accumulated per-image rows, **independent of the head's configured
        metric selection** — so the per-class breakdown (design §9) is always emitted,
        even when the run's monitor metrics are just the means (the default).
        """
        path = self._output_dir / f"predictions_{self._split}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_id",
                    "pred_path",
                    "probs_path",
                    "pred_overlay_path",
                    "gt_overlay_path",
                    "dice",
                    "iou",
                ],
            )
            writer.writeheader()
            writer.writerows(self._rows)
        self._write_metrics_csv()
        return path

    def _write_metrics_csv(self) -> Path | None:
        if not self._stat_rows:
            return None
        full = reduce_dice_iou(
            torch.cat(self._stat_rows, dim=0), num_classes=self._num_classes
        )
        # Long format: one (metric, aggregation, value) row. The two means are
        # per-image-macro (the monitor convention); the per-class ``dice_class_{c}``
        # are dataset-global. The explicit ``aggregation`` column makes that mix
        # self-describing — ``mean_dice`` is NOT mean(dice_class_*) when image sizes
        # vary, so the file must say which is which rather than leave it implied.
        ordered = [("mean_dice", "per_image_macro"), ("mean_iou", "per_image_macro")] + [
            (f"dice_class_{c}", "dataset_global") for c in range(self._num_classes)
        ]
        path = self._output_dir / f"metrics_{self._split}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "aggregation", "value"])
            writer.writerows([(name, agg, full[name]) for name, agg in ordered])
        return path
