"""Grid artefact writers as :func:`soma.predict` callbacks.

Each callback writes one kind of side artefact while ``predict`` runs, from the batch
and the learner's :class:`~soma.learn.PredictOutput` (``raw.logits`` is the model output
at the supervision frame); the logits are never accumulated. Predictions themselves are
the noun ``predict`` returns; these are views of them:

* :class:`SaveOverlays` — predicted classes (segmentation) or points (detection) blended
  over the sample's pixels, beside the same overlay of the supervision when the batch
  carries it.
* :class:`SaveProbabilities` — per-sample float16 ``(C, H, W)`` softmax (segmentation).
* :class:`SaveHeatmaps` — per-sample float16 ``(C, H, W)`` sigmoid heatmap (detection).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

__all__ = ["SaveHeatmaps", "SaveOverlays", "SaveProbabilities", "class_palette"]

logger = logging.getLogger(__name__)

# Segmentation: class 0 is background (left unblended); colours cycle past the last entry.
_SEGMENTATION_COLORS: list[tuple[int, int, int]] = [
    (0, 0, 0),
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
]
# Detection classes carry no good/bad meaning: neutral hues absent from H&E.
_DETECTION_COLORS: list[tuple[int, int, int]] = [
    (0, 200, 255),
    (255, 190, 0),
    (240, 50, 230),
    (255, 120, 0),
    (150, 80, 230),
    (0, 200, 180),
    (250, 150, 200),
    (255, 255, 255),
]
_OUTLINE = (15, 15, 15)


def class_palette(num_classes: int, *, detection: bool = False) -> np.ndarray:
    """A deterministic ``(num_classes, 3)`` uint8 palette (cycles past its entries)."""
    colors = _DETECTION_COLORS if detection else _SEGMENTATION_COLORS
    return np.asarray([colors[c % len(colors)] for c in range(num_classes)], dtype=np.uint8)


def _pixels_of(batch, index: int, images, sample_id: str, size: tuple[int, int]) -> np.ndarray | None:
    """The sample's RGB pixels ``(H, W, 3)`` at the frame size, or ``None`` (fail-soft)."""
    pixels = getattr(batch, "pixels", None)
    if pixels is not None and pixels.dtype == torch.uint8 and tuple(pixels.shape[-2:]) == size:
        return pixels[index].permute(1, 2, 0).detach().cpu().numpy()
    if images is None:
        return None
    try:
        array = images.load(sample_id).permute(1, 2, 0).detach().cpu().numpy()
    except (FileNotFoundError, OSError, ValueError, KeyError):
        logger.debug("overlay skipped for '%s': pixels unreadable", sample_id)
        return None
    if tuple(array.shape[:2]) != size:
        array = np.asarray(Image.fromarray(array).resize((size[1], size[0])))
    return array


class SaveOverlays:
    """Write overlays of each prediction (and its supervision) over the sample's pixels.

    Segmentation writes ``<directory>/pred/<id>.png`` (foreground classes blended at
    ``alpha``) and ``<directory>/gt/<id>.png`` from the batch's label map. Detection
    draws the points scoring at least ``threshold`` (default: the head's
    ``validation_threshold``) as class-coloured dots in ``pred/`` and the annotated
    points in ``gt/``. Pixels come from the batch when it carries raw uint8 pixels at the
    frame size, else from ``images`` (an :class:`~soma.data.ImageSource`); without
    either the overlay is skipped.
    """

    def __init__(self, directory: str | Path, *, images=None, alpha: float = 0.5, threshold: float | None = None) -> None:
        self.directory = Path(directory)
        self._images = images
        self._alpha = float(alpha)
        self._threshold = threshold
        self._head = None

    def on_predict_start(self, learner) -> None:
        self._head = learner.head

    def on_predict_batch(self, batch, output) -> None:
        head = self._head
        size = tuple(int(v) for v in head.geometry.target_size)
        for index, sample_id in enumerate(batch.sample_ids):
            pixels = _pixels_of(batch, index, self._images, sample_id, size)
            if pixels is None:
                continue
            if "label_map" in output.columns:
                self._label_maps(sample_id, pixels, output.columns["label_map"][index], batch, index)
            else:
                self._points(sample_id, pixels, output.columns["points"][index], batch, index)

    def _save(self, kind: str, sample_id: str, image: Image.Image) -> None:
        out = self.directory / kind
        out.mkdir(parents=True, exist_ok=True)
        image.save(out / f"{sample_id}.png")

    def _blend(self, pixels: np.ndarray, raster: np.ndarray) -> Image.Image:
        num_classes = int(self._head.num_classes)
        palette = class_palette(num_classes)
        in_range = (raster >= 0) & (raster < num_classes)
        color = palette[np.where(in_range, raster, 0)].astype(np.float32)
        foreground = (in_range & (raster != 0))[..., None]
        image = pixels.astype(np.float32)
        blended = np.where(foreground, (1.0 - self._alpha) * image + self._alpha * color, image)
        return Image.fromarray(blended.astype(np.uint8), mode="RGB")

    def _label_maps(self, sample_id, pixels, label_map, batch, index) -> None:
        self._save("pred", sample_id, self._blend(pixels, np.asarray(label_map)))
        mask = batch.targets.get("mask") if batch.targets else None
        if mask is not None:
            self._save("gt", sample_id, self._blend(pixels, mask[index].detach().cpu().numpy()))

    def _draw(self, pixels: np.ndarray, xy: np.ndarray, classes: np.ndarray) -> Image.Image:
        palette = class_palette(int(self._head.num_classes), detection=True)
        image = Image.fromarray(np.ascontiguousarray(pixels), mode="RGB").copy()
        draw = ImageDraw.Draw(image)
        radius = max(2.0, float(self._head.delta_px) / 3.0)
        for (x, y), c in zip(xy, classes):
            color = tuple(int(v) for v in palette[int(c)])
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=_OUTLINE)
        return image

    def _points(self, sample_id, pixels, rows, batch, index) -> None:
        head = self._head
        rows = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
        threshold = head.validation_threshold if self._threshold is None else float(self._threshold)
        rows = rows[rows[:, 3] >= threshold]
        xy = head.to_frame(torch.from_numpy(rows[:, :2])).numpy() if rows.shape[0] else np.zeros((0, 2))
        self._save("pred", sample_id, self._draw(pixels, xy, rows[:, 2].astype(np.int64)))
        points = batch.targets.get("points") if batch.targets else None
        if points is not None:
            gt_xy, gt_cls = head.frame_points(points[index].detach().cpu())
            self._save("gt", sample_id, self._draw(pixels, gt_xy.numpy(), gt_cls.numpy()))


class SaveProbabilities:
    """Write each sample's float16 ``(C, H, W)`` softmax as ``<directory>/<id>.npz`` (key
    ``probs``); ``argmax`` over it recovers the predicted label map."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def on_predict_batch(self, batch, output) -> None:
        probabilities = output.raw.logits.float().softmax(dim=1).to(torch.float16).cpu().numpy()
        self.directory.mkdir(parents=True, exist_ok=True)
        for index, sample_id in enumerate(batch.sample_ids):
            np.savez_compressed(self.directory / f"{sample_id}.npz", probs=probabilities[index])


class SaveHeatmaps:
    """Write each sample's float16 ``(C, H, W)`` sigmoid heatmap (the detection head's
    output before peak decoding) as ``<directory>/<id>.npz`` (key ``heatmap``)."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def on_predict_batch(self, batch, output: Any) -> None:
        heatmaps = output.raw.logits.float().to(torch.float16).cpu().numpy()
        self.directory.mkdir(parents=True, exist_ok=True)
        for index, sample_id in enumerate(batch.sample_ids):
            np.savez_compressed(self.directory / f"{sample_id}.npz", heatmap=heatmaps[index])
