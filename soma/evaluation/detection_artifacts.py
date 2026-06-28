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
  - ``match_overlays/class_<c>/<split>/<sample_id>.png`` per class, the headline
                                         match-status overlay: GT points as δ-radius
                                         rings, predictions as filled dots, hue = outcome
                                         (TP green / FP red / FN blue). TP/FP markers at
                                         the predicted xy, FN at the GT xy — drawn from
                                         the *same* per-class ``assignment`` the headline
                                         F1 reduces, so overlay and metric cannot drift.
                                         Same ``save_detection_overlays`` gate + fail-soft
                                         source read as the plain overlays.
  - ``detection_per_image_<split>.csv``  one row per tile: overlay paths + ``n_pred``,
                                         ``n_gt``, ``tp``, ``fp``, ``fn`` (summed over
                                         classes) + ``mean_f1`` (``reduce_f1`` on the
                                         image's own ``(1, C, 3)`` counts) — always written.
  - ``metrics_<split>.csv``              split-level per-class F1/precision/recall
                                         (dataset-global), always written (the per-class
                                         breakdown exists even when the monitor metric is
                                         just ``mean_f1``).

The raw heatmap overlays + npz sidecar (slice #176) extend this writer further: the
``heatmap`` payload field is already threaded through ``add_image`` for them, and
per-class artifacts use a ``class_<c>`` index subdir (matching the ``f1_class_<c>``
metric vocabulary).
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

# Match-overlay outcome hues (slice #175). Conventional green/red/blue; ring-vs-dot
# redundantly carries pred-vs-GT, so the red-green pair stays separable for colorblind
# viewers. Class is encoded by the filename, freeing hue + shape for the match structure.
_TP_COLOR = (0, 200, 0)  # true positive  — matched pred, green
_FP_COLOR = (220, 0, 0)  # false positive — unmatched pred, red
_FN_COLOR = (0, 0, 220)  # false negative — unmatched GT, blue
_RING_WIDTH = 2  # outline width (px) of the δ-radius GT ring


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
        # The matching tolerance δ (target-frame px) — the match-overlay GT-ring radius,
        # so the ring literally is the tolerance every TP fell within.
        self._delta = float(head.delta_px)
        self._split = split
        self._output_dir = Path(output_dir)
        self._pred_overlays_dir = self._output_dir / "pred_overlays" / split
        self._gt_overlays_dir = self._output_dir / "gt_overlays" / split
        self._match_overlays_dir = self._output_dir / "match_overlays"
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
        """Accumulate one image's manifest row and render its overlays (pred/GT + match).

        ``assignment`` is the per-class :class:`~soma.detection.matching.ClassMatch` list
        from the single upstream match — the manifest counts/F1 *and* the per-class match
        overlays both derive from it (never a second match). ``heatmap`` and ``pred_score``
        are part of the per-image payload for the heatmap-overlay (#176) slice; the plain
        pred/GT overlays need only the point xy + class, while the match overlays read each
        class's ``pairs`` to colour every point by its TP/FP/FN outcome.
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
        # One match-status overlay per class (slice #175). Skipped (None) wholesale when the
        # source tile is unreadable — same fail-soft gate as the plain overlays above.
        match_overlays = (
            [
                self._draw_match_overlay(
                    image_arr, c, assignment[c],
                    pred_xy, pred_class, gt_xy, gt_class, sample_id,
                )
                for c in range(self._num_classes)
            ]
            if image_arr is not None
            else [None] * self._num_classes
        )
        row = {
            "sample_id": sample_id,
            "pred_overlay_path": "" if pred_overlay is None else self._rel(pred_overlay),
            "gt_overlay_path": "" if gt_overlay is None else self._rel(gt_overlay),
        }
        for c, overlay in enumerate(match_overlays):
            row[f"match_overlay_class_{c}"] = "" if overlay is None else self._rel(overlay)
        row.update(
            {
                "n_pred": n_pred,
                "n_gt": n_gt,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "mean_f1": mean_f1,
            }
        )
        self._rows.append(row)

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

    def _draw_match_overlay(
        self,
        image_arr: np.ndarray,
        class_idx: int,
        match: "ClassMatch",
        pred_xy: np.ndarray,
        pred_class: np.ndarray,
        gt_xy: np.ndarray,
        gt_class: np.ndarray,
        sample_id: str,
    ) -> Path:
        """Draw one class's TP/FP/FN match-status overlay (slice #175).

        Reads ``match.pairs`` (global ``(pred_idx, gt_idx)`` for this class) — the *same*
        assignment the headline F1 reduces — so the overlay and the metric are one
        computation, never two that coincide. Each matched pair is a TP (green dot at the
        predicted xy + green δ-ring at the GT xy); each unmatched class-``c`` prediction is
        an FP (red dot at its xy); each unmatched class-``c`` GT is an FN (blue δ-ring at
        its xy). Ring-vs-dot carries pred-vs-GT, hue carries outcome, ring radius carries δ.
        """
        image = Image.fromarray(image_arr, mode="RGB").copy()
        draw = ImageDraw.Draw(image)
        pred_xy = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
        gt_xy = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
        pred_class = np.asarray(pred_class, dtype=np.int64).reshape(-1)
        gt_class = np.asarray(gt_class, dtype=np.int64).reshape(-1)
        matched_pred = {int(i) for i in match.pairs[:, 0]}
        matched_gt = {int(i) for i in match.pairs[:, 1]}
        # TP: a green dot at the prediction inside a green ring at the matched GT.
        for p_idx, g_idx in match.pairs:
            self._ring(draw, gt_xy[int(g_idx)], _TP_COLOR)
            self._dot(draw, pred_xy[int(p_idx)], _TP_COLOR)
        # FP: a lone red dot at each unmatched class-c prediction.
        for i in np.nonzero(pred_class == class_idx)[0]:
            if int(i) not in matched_pred:
                self._dot(draw, pred_xy[int(i)], _FP_COLOR)
        # FN: a lone blue ring at each unmatched class-c GT.
        for i in np.nonzero(gt_class == class_idx)[0]:
            if int(i) not in matched_gt:
                self._ring(draw, gt_xy[int(i)], _FN_COLOR)
        out_dir = self._match_overlays_dir / f"class_{class_idx}" / self._split
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{sample_id}.png"
        image.save(path)
        return path

    def _dot(self, draw: ImageDraw.ImageDraw, xy: np.ndarray, color: tuple) -> None:
        """Filled dot of radius ``overlay_dot_radius`` at ``xy`` — a predicted point."""
        x, y = float(xy[0]), float(xy[1])
        r = self._dot_radius
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    def _ring(self, draw: ImageDraw.ImageDraw, xy: np.ndarray, color: tuple) -> None:
        """Outline ring of radius δ at ``xy`` — a GT point + its matching tolerance."""
        x, y = float(xy[0]), float(xy[1])
        r = self._delta
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=_RING_WIDTH)

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
                    *(f"match_overlay_class_{c}" for c in range(self._num_classes)),
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
