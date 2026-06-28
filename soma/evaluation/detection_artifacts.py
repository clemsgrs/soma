"""Detection evaluation artifacts: plain point overlays + per-image manifest + metrics.

The detection counterpart of :class:`~soma.evaluation.dense_artifacts.DenseArtifactWriter`.
Where the segmentation writer is invoked per *batch* by the shared dense-eval loop, the
detection writer is invoked per *image* by the consolidated ``_evaluate_detection`` loop,
which decodes peaks and runs the class-aware F1@δ match **exactly once** and hands the
single result to all consumers. The writer receives that per-image payload
(``heatmap``, decoded ``pred`` points, the per-class match ``assignment``, ``gt`` points),
so the manifest's per-image counts/F1 come from the *same* reduction the headline metric
uses and cannot drift from it.

Artifacts written under ``fold_dir`` (the per-point ``predictions_<split>.csv`` is written
by the pipeline, not here, and is unchanged):
  - ``pred_overlays/<split>/<sample_id>.png`` all predicted points as filled dots,
                                         color = class (``class_palette``), over the
                                         source tile — **on by default**, suppressible via
                                         ``save_detection_overlays`` for a metrics-only run.
                                         **Fail-soft**: skipped (logged) when the source tile
                                         is unreadable; the manifest row is still written.
  - ``gt_overlays/<split>/<sample_id>.png`` all ground-truth points, same palette, for
                                         side-by-side comparison with the predictions.
  - ``detection_per_image_<split>.csv``  one row per tile: overlay paths + ``n_pred``,
                                         ``n_gt``, ``tp``, ``fp``, ``fn`` (summed over
                                         classes) + ``mean_f1`` (``reduce_f1`` on the
                                         image's own ``(1, C, 3)`` counts) — always written.
  - ``metrics_<split>.csv``              split-level per-class F1/precision/recall
                                         (dataset-global), always written (the per-class
                                         breakdown exists even when the monitor metric is
                                         just ``mean_f1``).

The per-class match-status overlays (slice #175) and the raw heatmap overlays + npz
sidecar (slice #176) extend this writer: the ``heatmap`` / ``assignment`` payload fields
are already threaded through ``add_image`` for them, and per-class artifacts use a
``class_<c>`` index subdir (matching the ``f1_class_<c>`` metric vocabulary).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image, ImageDraw

from soma.detection.matching import reduce_f1
from soma.evaluation.dense_artifacts import class_palette

if TYPE_CHECKING:
    from soma.detection.matching import ClassMatch

logger = logging.getLogger(__name__)

__all__ = ["DetectionArtifactWriter"]


class DetectionArtifactWriter:
    """Per-image callback that writes detection point overlays + a manifest + metrics.

    Construct one per split; call :meth:`add_image` once per image from the consolidated
    eval loop (with the single decode+match result), then :meth:`finalize` to flush the
    per-image manifest and split-level metrics CSVs.

    Args:
        head: the :class:`~soma.tasks.detection.DetectionHead` — supplies ``num_classes``
            (palette) and ``_crop_box`` (the target-frame size the points live in).
        split: split name (``"tune"``, ``"test"``, ...) — names the subdirs + CSVs.
        output_dir: the fold directory; artifacts live under it.
        dataset: the :class:`~soma.dataset.DetectionManifest`
            (``samples[sample_id].image_path``, flat tiles) for overlays. ``None`` disables
            overlays (manifest + metrics still written).
        save_detection_overlays: write the pred/GT point overlay PNGs. On by default (a
            viewable prediction is cheap), but suppressible for a metrics-only run — the
            manifest + metrics CSVs are unaffected.
        overlay_dot_radius: radius (target-frame px) of each plotted point dot.
    """

    def __init__(
        self,
        *,
        head,
        split: str,
        output_dir: Path | str,
        dataset=None,
        save_detection_overlays: bool = True,
        overlay_dot_radius: int = 4,
    ) -> None:
        self._num_classes = int(head.num_classes)
        # crop_box = (top, left, height, width); the target frame the points + heatmap
        # live in (predicted/GT points are already in it, no transform for overlays).
        _, _, height, width = (int(v) for v in head._crop_box)
        self._target_h = height
        self._target_w = width
        self._split = split
        self._output_dir = Path(output_dir)
        self._pred_overlays_dir = self._output_dir / "pred_overlays" / split
        self._gt_overlays_dir = self._output_dir / "gt_overlays" / split
        self._dataset = dataset
        # Overlays need both the flag (default on) and a dataset to read source tiles;
        # suppressing either yields a metrics-only run (manifest + metrics still written).
        self._overlays_enabled = dataset is not None and bool(save_detection_overlays)
        self._dot_radius = int(overlay_dot_radius)
        self._palette = class_palette(self._num_classes)
        self._rows: list[dict] = []
        self._stat_rows: list[torch.Tensor] = []

    def add_image(
        self,
        *,
        sample_id: str,
        heatmap: torch.Tensor,
        pred_xy: np.ndarray,
        pred_class: np.ndarray,
        pred_score: np.ndarray,
        assignment: "list[ClassMatch]",
        gt_xy: np.ndarray,
        gt_class: np.ndarray,
    ) -> None:
        """Accumulate one image's manifest row and render its plain pred/GT overlays.

        ``assignment`` is the per-class :class:`~soma.detection.matching.ClassMatch` list
        from the single upstream match — the manifest counts/F1 derive from it (never a
        second match). ``heatmap`` and ``pred_score`` are part of the per-image payload for
        the heatmap-overlay (#176) / match-status (#175) slices; this slice draws the plain
        predicted-point and ground-truth overlays, which need only the point xy + class.
        """
        # Per-class (tp, fp, fn) from the assignment — the same reduction match_points /
        # the headline metric uses, so per-image numbers can't drift from the split.
        counts = np.zeros((self._num_classes, 3), dtype=np.int64)
        for c, m in enumerate(assignment):
            counts[c] = m.counts
        stat_row = torch.from_numpy(counts).to(torch.long).unsqueeze(0)  # (1, C, 3)
        self._stat_rows.append(stat_row)
        n_pred = int(sum(m.n_pred for m in assignment))
        n_gt = int(sum(m.n_gt for m in assignment))
        tp, fp, fn = (int(v) for v in counts.sum(axis=0))
        mean_f1 = reduce_f1(
            stat_row, num_classes=self._num_classes, aggregation="dataset_global"
        )["mean_f1"]

        image_arr = self._read_source(sample_id)
        pred_overlay = (
            self._draw_points(image_arr, pred_xy, pred_class, self._pred_overlays_dir, sample_id)
            if image_arr is not None
            else None
        )
        gt_overlay = (
            self._draw_points(image_arr, gt_xy, gt_class, self._gt_overlays_dir, sample_id)
            if image_arr is not None
            else None
        )
        self._rows.append(
            {
                "sample_id": sample_id,
                "pred_overlay_path": "" if pred_overlay is None else self._rel(pred_overlay),
                "gt_overlay_path": "" if gt_overlay is None else self._rel(gt_overlay),
                "n_pred": n_pred,
                "n_gt": n_gt,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "mean_f1": mean_f1,
            }
        )

    def _rel(self, path: Path) -> str:
        return str(path.relative_to(self._output_dir))

    def _read_source(self, sample_id: str) -> np.ndarray | None:
        """Read the flat source tile and resize to the target frame, or ``None``.

        Detection is flat-tile only (no ``region`` on ``DetectionManifest``), so a plain
        ``Image.open`` + resize suffices — no slide-ROI/spacing handling (kept out of the
        shared seg reader deliberately). ``None`` when overlays are disabled, the record
        is unknown, or the tile is unreadable (fail-soft: cached-feature runs may not
        retain the source tile).
        """
        if not self._overlays_enabled:
            return None
        record = self._dataset.samples.get(sample_id)
        if record is None:
            return None
        try:
            with Image.open(record.image_path) as img:
                image = img.convert("RGB")
            if image.size != (self._target_w, self._target_h):
                image = image.resize((self._target_w, self._target_h))  # PIL size is (W, H)
            return np.asarray(image, dtype=np.uint8)
        except (FileNotFoundError, OSError, ValueError, Image.DecompressionBombError):
            logger.debug("overlays skipped for '%s': source image unreadable", sample_id)
            return None

    def _draw_points(
        self,
        image_arr: np.ndarray,
        points_xy: np.ndarray,
        points_class: np.ndarray,
        out_dir: Path,
        sample_id: str,
    ) -> Path:
        """Draw filled dots (color = class via the palette) over a copy of the tile."""
        image = Image.fromarray(image_arr, mode="RGB").copy()
        draw = ImageDraw.Draw(image)
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        cls = np.asarray(points_class, dtype=np.int64).reshape(-1)
        r = self._dot_radius
        for (x, y), c in zip(pts, cls):
            color = tuple(int(v) for v in self._palette[int(c) % self._num_classes])
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{sample_id}.png"
        image.save(path)
        return path

    def finalize(self) -> Path:
        """Flush ``detection_per_image_<split>.csv`` + ``metrics_<split>.csv``.

        Returns the per-image manifest CSV path. ``metrics_<split>.csv`` is written
        unconditionally (even for an empty split) so the per-class F1/precision/recall
        breakdown always exists, independent of the run's monitor-metric selection.
        """
        path = self._output_dir / f"detection_per_image_{self._split}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_id",
                    "pred_overlay_path",
                    "gt_overlay_path",
                    "n_pred",
                    "n_gt",
                    "tp",
                    "fp",
                    "fn",
                    "mean_f1",
                ],
            )
            writer.writeheader()
            writer.writerows(self._rows)
        self._write_metrics_csv()
        return path

    def _write_metrics_csv(self) -> Path:
        counts = (
            torch.cat(self._stat_rows, dim=0)
            if self._stat_rows
            else torch.zeros(0, self._num_classes, 3, dtype=torch.long)
        )
        full = reduce_f1(counts, num_classes=self._num_classes, aggregation="dataset_global")
        # Long format: one (metric, aggregation, value) row. Per-class F1/precision/recall
        # and mean_f1 are all dataset-global (the OCELOT-faithful headline reduction); the
        # explicit aggregation column mirrors the segmentation metrics CSV.
        ordered = [("mean_f1", "dataset_global")]
        for c in range(self._num_classes):
            ordered.append((f"f1_class_{c}", "dataset_global"))
        for c in range(self._num_classes):
            ordered.append((f"precision_class_{c}", "dataset_global"))
        for c in range(self._num_classes):
            ordered.append((f"recall_class_{c}", "dataset_global"))
        path = self._output_dir / f"metrics_{self._split}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "aggregation", "value"])
            writer.writerows([(name, agg, full[name]) for name, agg in ordered])
        return path
