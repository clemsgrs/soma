"""Detection points over whole slides, gathered while each slide streams tile by tile.

The tiles lie on hs2p's slide grid, and their outputs come from the seam the label maps
read (:class:`~soma.evaluation.whole_slide.TileOutputs`). Each slide gets one parquet per
output: every peak above the head's ``min_score``, with no operating threshold applied.

Per tile
--------
The head gives the tile's peaks in its own pixels
(:meth:`~soma.models.heads.detection.DetectionHead.frame_peaks`). A peak at tile pixel
``p`` lies at level-0 ``origin + p * scale``, where ``origin`` is the tile's level-0 origin
and ``scale`` the slide's read spacing over its level-0 spacing (the tile's
:meth:`~soma.data.images.RegionImageSource.geometry`). The head's own level-0 anchor, the
frame it was trained in, plays no part.

Stitching
---------
Pick one, by location, as for the label maps. ``margin`` is ``tile * overlap / 2`` pixels
and ``step`` is the grid's step in the tile's own pixels. A tile keeps the peaks in
``[margin, margin + step)`` on each axis, from 0 on an axis where its origin is 0 (the
slide's first row and column), and only those on the slide. ``step`` is usually
``tile - 2 * margin``. On a slide resampled to the read spacing it can differ by a fraction
of a pixel, because hs2p rounds the step in the pixels it reads. With ``overlap: 0`` a tile
keeps all its peaks. Then one per-class greedy NMS over the slide, at the head's NMS
distance in level-0 pixels, removes the duplicates the two sides of a seam found
(:func:`~soma.evaluation.detection.greedy_point_nms`). With an overlap the margin must be at
least the NMS distance, so a tile finds each peak it keeps with the peak's whole NMS
neighbourhood inside the tile.

Artefact
--------
A parquet with columns ``x`` and ``y`` (level-0 pixels, float64), ``class`` (int64) and
``score`` (float64). Its schema metadata (key ``soma``, JSON, ``kind: slide_points``)
records what scoring needs: the matching and NMS distances in µm, the slide's level-0
spacing, and what the caller adds: the class scheme the bundle was trained with
(``classes``, class name → the annotated ids that form it, in class order, and ``drop``,
the annotated ids dropped), the read spacing, the bundles and their tune threshold.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from soma._internal.atomic_io import atomic_write_bytes
from soma.evaluation.detection import greedy_point_nms
from soma.evaluation.whole_slide import SlideTiles, TileOutputs, stitch_margin_px, tile_id

__all__ = ["SLIDE_POINTS_KIND", "predict_slide_points"]

#: The ``kind`` a slide's points parquet records in its ``soma`` schema metadata.
SLIDE_POINTS_KIND = "slide_points"
_METADATA_KEY = b"soma"


def predict_slide_points(
    outputs: TileOutputs,
    *,
    tiles: SlideTiles,
    tile_px: int,
    step_px: float,
    overlap: float,
    slide_size: tuple[int, int],
    level0_spacing_um: float,
    paths: Sequence[str | Path],
    metadata: Sequence[Mapping[str, Any]],
    ensemble: bool = False,
    batch_size: int = 8,
) -> list[int]:
    """Write one slide's detection points as parquets at ``paths``.

    ``outputs`` gives each batch's results (:class:`~soma.evaluation.whole_slide.TileOutputs`)
    for the detection bundles it runs. ``tiles`` are the slide's kept tiles on the slide
    grid, and ``step_px`` is the grid's step in the pixels of a ``tile_px`` tile;
    ``slide_size`` is the slide's level-0 ``(width, height)`` at ``level0_spacing_um``.
    Without ``ensemble`` there is one path (and one ``metadata`` mapping, stored in the
    file) per bundle; with it there is one, and the peaks come from the bundles' mean
    heatmap, with the first bundle's head. Each ``metadata`` mapping holds the class
    scheme: ``classes`` (one entry per class of its head, in class order) and ``drop``. A
    slide without a kept tile gets a parquet without rows. Returns each parquet's point
    count.
    """
    bundle_heads = list(outputs.heads)
    paths = [Path(path) for path in paths]
    expected = 1 if ensemble else len(bundle_heads)
    if not bundle_heads:
        raise ValueError("predict_slide_points needs at least one bundle")
    if len(paths) != expected or len(metadata) != expected:
        raise ValueError(f"{expected} parquet(s) to write, got {len(paths)} path(s) and {len(metadata)} metadata")
    heads = bundle_heads[:1] if ensemble else bundle_heads
    margin = stitch_margin_px(int(tile_px), overlap)
    width, height = (int(v) for v in slide_size)
    kept: list[list[np.ndarray]] = [[] for _ in heads]
    if tiles.origins:
        source = tiles.source(heads[0].geometry.target_size)
        for batch_origins, results in outputs.batches(tiles, batch_size=batch_size):
            if ensemble:
                peaks = [heads[0].frame_peaks(_mean_heatmap(results))]
            else:
                peaks = [head.frame_peaks(output.logits.float()) for head, output in zip(heads, results, strict=True)]
            for found, per_tile in zip(kept, peaks):
                for origin, rows in zip(batch_origins, per_tile):
                    geometry = source.geometry(tile_id(origin))
                    found.append(
                        _contributed(
                            rows,
                            origin=origin,
                            scale=float(geometry.level0_px_per_token_px),
                            tile_px=int(tile_px),
                            step_px=float(step_px),
                            margin=margin,
                            slide_size=(width, height),
                        )
                    )
    counts = []
    for head, found, path, meta in zip(heads, kept, paths, metadata):
        rows = np.concatenate(found, axis=0) if found else np.zeros((0, 4), dtype=np.float64)
        nms_px = float(head.nms_distance) / float(level0_spacing_um)
        rows = rows[greedy_point_nms(rows[:, :2], rows[:, 3], rows[:, 2].astype(np.int64), nms_px)]
        atomic_write_bytes(path, _parquet_bytes(rows, _metadata(head, meta, level0_spacing_um)))
        counts.append(int(rows.shape[0]))
    return counts


def _mean_heatmap(outputs) -> torch.Tensor:
    """An ensemble's mean heatmap, ``(B, C, H, W)``. ``outputs`` is consumed one at a time,
    as :func:`~soma.models.model.mean_softmax` does, so only the running sum is kept."""
    total: torch.Tensor | None = None
    count = 0
    for output in outputs:
        heatmap = output.logits.float()
        total = heatmap if total is None else total + heatmap
        count += 1
        del output, heatmap  # not held while the next output is computed
    return total / count


def _contributed(
    rows: np.ndarray,
    *,
    origin: tuple[int, int],
    scale: float,
    tile_px: int,
    step_px: float,
    margin: int,
    slide_size: tuple[int, int],
) -> np.ndarray:
    """A tile's ``(k, 4)`` tile-pixel peaks that fall in the region it contributes, in
    level-0 pixels: ``[margin, margin + step_px)`` on each axis, from 0 where the origin is
    0, the whole tile without a margin, and on the slide."""
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
    keep = np.ones(rows.shape[0], dtype=bool)
    last = tile_px if margin == 0 else margin + step_px
    for axis, start in enumerate(origin):
        first = 0 if int(start) == 0 else margin
        keep &= (rows[:, axis] >= first) & (rows[:, axis] < last)
    rows = rows[keep].copy()
    rows[:, 0] = origin[0] + rows[:, 0] * scale
    rows[:, 1] = origin[1] + rows[:, 1] * scale
    return rows[(rows[:, 0] < slide_size[0]) & (rows[:, 1] < slide_size[1])]


def _metadata(head, extra: Mapping[str, Any], level0_spacing_um: float) -> dict[str, Any]:
    import soma

    return {
        "kind": SLIDE_POINTS_KIND,
        "task_family": "detection",
        "soma_version": str(soma.__version__),
        "match_distance_um": float(head.match_distance),
        "nms_distance_um": float(head.nms_distance),
        "min_score": float(head.min_score),
        "level0_spacing_um": float(level0_spacing_um),
        **dict(extra),
    }


def _parquet_bytes(rows: np.ndarray, metadata: Mapping[str, Any]) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "x": pa.array(rows[:, 0], type=pa.float64()),
            "y": pa.array(rows[:, 1], type=pa.float64()),
            "class": pa.array(rows[:, 2].astype(np.int64), type=pa.int64()),
            "score": pa.array(rows[:, 3], type=pa.float64()),
        }
    )
    table = table.replace_schema_metadata({_METADATA_KEY: json.dumps(dict(metadata)).encode("utf-8")})
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()

