"""Segmentation label maps over whole slides, written while each slide streams tile by tile.

A slide's tiles lie on hs2p's slide grid (``TilingConfig(grid="slide")``). Each kept tile
is read through a :class:`~soma.data.images.RegionImageSource`, encoded once by the frozen
encoder stage its models share, and decoded by every model
(:func:`~soma.models.model.forward_sharing_encoder`). The class of each pixel goes into a
rolling band of map rows, and full rows of the band go, in order, to hs2p's label-TIFF
writer (:func:`hs2p.write_label_tiff`). Neither a slide's map nor all of its tiles' outputs
are ever held in memory: only the band, a few rows of tiles in flight to the writer, and
the writer's own reduced level.

Output frame
------------
hs2p puts the tile origins of the slide grid at ``(i * step, j * step)`` level-0 pixels,
where ``step`` is its level-0 stride. The map is indexed by that lattice and never by
dividing a coordinate, so no coordinate is rounded, whatever the level downsample:

- ``tile`` is the model's tile side in pixels and ``margin`` is ``tile * overlap / 2``,
  a whole number of pixels (:func:`stitch_margin_px`); ``stride = tile - 2 * margin``.
- The tile at lattice index ``(i, j)``, level-0 origin ``(i * step, j * step)``, covers map
  columns ``[i * stride, i * stride + tile)`` and rows ``[j * stride, j * stride + tile)``.
- A ``width`` by ``height`` level-0 slide gives a map of ``ceil(width * stride / step)`` by
  ``ceil(height * stride / step)`` pixels, at ``step * level0_spacing / stride`` µm/px.

When the slide is read at a pyramid level within tolerance of the read spacing, the map
spacing is that level's spacing (up to hs2p's rounding of ``step``), and the map is that
level's pixel grid. When the tile is resampled from a finer level, hs2p rounds ``step``
from the read size, so the lattice scale (``step / stride`` level-0 pixels per map pixel)
and the scale inside a tile (``tile_size_lv0 / tile``) can differ by a fraction of a
pixel per tile. The error stays inside each tile: it never accumulates across the slide.

Stitching
---------
Pick one: each map pixel takes the class of exactly one tile. A tile contributes
``[origin + margin, origin + tile - margin)`` on each axis, and from ``0`` on an axis where
its lattice index is 0, so the map starts at pixel 0. With ``overlap: 0`` (no margin) each
tile contributes its whole output. Map pixels no kept tile contributes are 0, "not
predicted"; so are the margins a tile gives up next to a tile that was not kept.

Values: class ``k`` is written as ``k + 1``.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from soma.data.images import RegionImageSource
from soma.data.manifests import ImageEntry
from soma.models.model import forward_sharing_encoder, mean_softmax

__all__ = ["SlideFrame", "predict_slide_label_maps", "stitch_margin_px", "WRITE_TILE_PX"]

#: Side of the square tiles a label TIFF is stored in (hs2p needs a multiple of 16).
WRITE_TILE_PX = 256
#: The largest class count a uint8 map holds once class ``k`` is written as ``k + 1``.
MAX_CLASSES = 255
#: How often a blocked hand-over to a writer checks that the writer is still running.
_POLL_SECONDS = 0.5


def stitch_margin_px(tile_px: int, overlap: float) -> int:
    """The margin each tile gives up on each side: ``tile * overlap / 2`` pixels.

    Raises ``ValueError`` unless ``overlap`` is in ``[0, 1)`` and the margin is a whole
    number of pixels, so every map pixel comes from one tile with no rounding.
    """
    overlap = float(overlap)
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"extraction.tiling.overlap must be in [0, 1), got {overlap}")
    margin = tile_px * overlap / 2
    if abs(margin - round(margin)) > 1e-6:
        nearest = 2 * round(margin) / tile_px
        raise ValueError(
            f"extraction.tiling.overlap {overlap} gives {tile_px} px tiles a margin of "
            f"{margin:g} px (tile x overlap / 2), not a whole number of pixels. Pick an "
            f"overlap whose margin is whole, such as {nearest:g}."
        )
    return int(round(margin))


@dataclass(frozen=True)
class SlideFrame:
    """The pixel grid of one slide's label map (see the module docstring).

    ``tile_px`` and ``stride_px`` are in model pixels, ``step_lv0`` in level-0 pixels;
    ``width`` and ``height`` are the map's size and ``spacing_um`` its µm/px.
    """

    tile_px: int
    stride_px: int
    step_lv0: int
    width: int
    height: int
    spacing_um: float

    @classmethod
    def from_grid(
        cls,
        *,
        tile_px: int,
        overlap: float,
        step_lv0: int,
        slide_size: tuple[int, int],
        level0_spacing_um: float,
    ) -> "SlideFrame":
        """The frame of a slide grid with level-0 stride ``step_lv0`` over a level-0
        ``slide_size`` ``(width, height)`` slide."""
        stride = int(tile_px) - 2 * stitch_margin_px(int(tile_px), overlap)
        step = int(step_lv0)
        width, height = (int(v) for v in slide_size)
        return cls(
            tile_px=int(tile_px),
            stride_px=stride,
            step_lv0=step,
            width=-(-width * stride // step),
            height=-(-height * stride // step),
            spacing_um=step * float(level0_spacing_um) / stride,
        )

    @property
    def margin_px(self) -> int:
        return (self.tile_px - self.stride_px) // 2

    def origin(self, x: int, y: int) -> tuple[int, int]:
        """The map ``(column, row)`` of the tile whose level-0 origin is ``(x, y)``."""
        if x % self.step_lv0 or y % self.step_lv0:
            raise ValueError(
                f"tile origin ({x}, {y}) is not on the slide grid of level-0 stride {self.step_lv0}"
            )
        return (x // self.step_lv0) * self.stride_px, (y // self.step_lv0) * self.stride_px

    def contribution(self, x: int, y: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """The map rows ``(first, stop)`` and columns ``(first, stop)`` the tile at level-0
        origin ``(x, y)`` writes: its middle, from 0 on an axis where it is first."""
        column, row = self.origin(x, y)
        return self._span(row, self.height), self._span(column, self.width)

    def _span(self, start: int, extent: int) -> tuple[int, int]:
        first = start if start == 0 else start + self.margin_px
        return first, min(start + self.tile_px - self.margin_px, extent)


class _Aborted(Exception):
    """Stops a writer whose map will not be completed."""


_END = object()
_ABORT = object()


class _LabelTiffStream:
    """hs2p's label-TIFF writer in a thread of its own, fed one storage tile at a time.

    The writer pulls tiles from a bounded queue, so the band never runs far ahead of it.
    A failed writer surfaces its error in :meth:`put` or :meth:`close`; :meth:`abort`
    stops it, and hs2p then removes its temporary file and leaves ``path`` as it was.
    """

    def __init__(self, path: Path, *, frame: SlideFrame, metadata: Mapping[str, Any], tile_px: int) -> None:
        import hs2p

        self._queue: queue.Queue = queue.Queue(maxsize=2 * max(1, -(-frame.width // tile_px)))
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._write,
            kwargs={
                "write": hs2p.write_label_tiff,
                "path": Path(path),
                "dimensions": (frame.width, frame.height),
                "tile_size": tile_px,
                "spacing_um": frame.spacing_um,
                "metadata": dict(metadata),
            },
            name=f"label-tiff:{Path(path).name}",
            daemon=True,
        )
        self._thread.start()

    def _tiles(self):
        while True:
            item = self._queue.get()
            if item is _END:
                return
            if item is _ABORT:
                raise _Aborted
            yield item

    def _write(self, *, write, **kwargs) -> None:
        try:
            write(tiles=self._tiles(), **kwargs)
        except BaseException as error:  # noqa: BLE001 - handed to the feeding thread
            self._error = error

    def _send(self, item: object) -> None:
        while True:
            if not self._thread.is_alive():
                if self._error is not None:
                    raise self._error
                raise RuntimeError("the label-TIFF writer stopped before its map was complete")
            try:
                self._queue.put(item, timeout=_POLL_SECONDS)
                return
            except queue.Full:
                continue

    def put(self, tile: np.ndarray) -> None:
        self._send(tile)

    def close(self) -> None:
        self._send(_END)
        self._thread.join()
        if self._error is not None:
            raise self._error

    def abort(self) -> None:
        if self._thread.is_alive():
            try:
                self._send(_ABORT)
            except BaseException:  # noqa: BLE001 - the writer already stopped
                pass
        self._thread.join()


class _Band:
    """The map rows still being filled: full rows of storage tiles go to ``stream``.

    Tiles arrive in row order of the lattice, so the first row a tile writes never
    decreases: every map row above it is final. The band holds one row of storage tiles
    plus one model tile, enough for any tile's rows.
    """

    def __init__(self, frame: SlideFrame, stream: _LabelTiffStream, tile_px: int) -> None:
        self._frame = frame
        self._stream = stream
        self._tile_px = tile_px
        self._rows = np.zeros((tile_px + frame.tile_px, frame.width), dtype=np.uint8)
        self._top = 0
        self.not_predicted = 0

    def place(self, x: int, y: int, values: np.ndarray) -> None:
        (row_first, row_stop), (col_first, col_stop) = self._frame.contribution(x, y)
        if row_stop <= row_first or col_stop <= col_first:
            return
        while self._top + self._tile_px <= row_first:
            self._emit()
        column, row = self._frame.origin(x, y)
        self._rows[row_first - self._top : row_stop - self._top, col_first:col_stop] = values[
            row_first - row : row_stop - row, col_first - column : col_stop - column
        ]

    def finish(self) -> None:
        while self._top < self._frame.height:
            self._emit()

    def _emit(self) -> None:
        height = min(self._tile_px, self._frame.height - self._top)
        block = self._rows[:height]
        self.not_predicted += int(block.size - np.count_nonzero(block))
        for start in range(0, self._frame.width, self._tile_px):
            # A copy: the band is refilled before the writer takes the tile.
            self._stream.put(block[:, start : start + self._tile_px].copy())
        self._rows[: -self._tile_px] = self._rows[self._tile_px :]
        self._rows[-self._tile_px :] = 0
        self._top += self._tile_px


class _Tiles(Dataset):
    """The kept tiles of one slide, in order, as ``(3, tile, tile)`` uint8 pixels."""

    def __init__(self, source: RegionImageSource, ids: Sequence[str]) -> None:
        self._source = source
        self._ids = list(ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._source.load(self._ids[index])


def predict_slide_label_maps(
    models: Sequence[Any],
    *,
    image: ImageEntry,
    origins: Sequence[tuple[int, int]],
    frame: SlideFrame,
    paths: Sequence[str | Path],
    metadata: Sequence[Mapping[str, Any]],
    spacing_um: float,
    ensemble: bool = False,
    backend: str = "auto",
    tolerance: float = 0.05,
    batch_size: int = 8,
    num_workers: int = 0,
) -> list[float]:
    """Write one slide's label maps as label TIFFs at ``paths``, tile by tile.

    ``models`` are :class:`~soma.models.model.DenseModel` models with a frozen encoder
    stage, in eval mode on their device; models that share a stage encode each batch once.
    ``origins`` are the level-0 origins of the kept tiles on the slide grid ``frame`` was
    built for. Each tile is read at ``spacing_um`` (``tolerance``, ``backend``) from
    ``image``. Without ``ensemble`` there is one path (and one ``metadata`` mapping, stored
    in the file) per model, and each map holds that model's classes; with it there is one
    path, and the map holds the argmax of the models' mean softmax. A slide without a kept
    tile gets an all-zero map. Returns each map's not-predicted fraction.
    """
    models = list(models)
    paths = [Path(path) for path in paths]
    expected = 1 if ensemble else len(models)
    if not models:
        raise ValueError("predict_slide_label_maps needs at least one model")
    if len(paths) != expected or len(metadata) != expected:
        raise ValueError(f"{expected} map(s) to write, got {len(paths)} path(s) and {len(metadata)} metadata")
    for model in models:
        if int(model.head.num_classes) > MAX_CLASSES:
            raise ValueError(
                f"a label map holds at most {MAX_CLASSES} classes (class k is stored as k + 1 "
                f"in uint8), got a head with {model.head.num_classes}"
            )
    # Row order of the lattice: the band only moves down.
    ordered = sorted(((int(x), int(y)) for x, y in origins), key=lambda origin: (origin[1], origin[0]))
    streams = [
        _LabelTiffStream(path, frame=frame, metadata=meta, tile_px=WRITE_TILE_PX)
        for path, meta in zip(paths, metadata)
    ]
    bands = [_Band(frame, stream, WRITE_TILE_PX) for stream in streams]
    try:
        for batch_origins, values in _predicted_tiles(
            models,
            image=image,
            origins=ordered,
            ensemble=ensemble,
            spacing_um=spacing_um,
            backend=backend,
            tolerance=tolerance,
            batch_size=batch_size,
            num_workers=num_workers,
        ):
            for band, maps in zip(bands, values):
                for (x, y), labels in zip(batch_origins, maps):
                    band.place(x, y, labels)
        for band in bands:
            band.finish()
        for stream in streams:
            stream.close()
    except BaseException:
        for stream in streams:
            stream.abort()
        raise
    total = frame.width * frame.height
    return [band.not_predicted / total for band in bands]


def _predicted_tiles(
    models,
    *,
    image: ImageEntry,
    origins: Sequence[tuple[int, int]],
    ensemble: bool,
    spacing_um: float,
    backend: str,
    tolerance: float,
    batch_size: int,
    num_workers: int,
):
    """Yield ``(origins, values)`` per batch: ``values`` holds one ``(B, tile, tile)``
    uint8 array per map, class ``k`` as ``k + 1``."""
    if not origins:
        return
    ids = [f"{x}_{y}" for x, y in origins]
    source = RegionImageSource(
        {
            tid: ImageEntry(sample_id=tid, image_path=image.image_path, spacing_at_level_0=image.spacing_at_level_0)
            for tid in ids
        },
        regions=dict(zip(ids, origins)),
        size=models[0].head.geometry.target_size,
        spacing_um=spacing_um,
        backend=backend,
        tolerance=tolerance,
    )
    loader = DataLoader(_Tiles(source, ids), batch_size=int(batch_size), num_workers=int(num_workers), shuffle=False)
    encoder = models[0].encoder
    input_device = torch.device(getattr(encoder, "input_device", None) or _device_of(models[0]))
    start = 0
    with torch.inference_mode():
        for pixels in loader:
            count = int(pixels.shape[0])
            batch_origins = origins[start : start + count]
            start += count
            outputs = forward_sharing_encoder(models, pixels.to(input_device), pixels)
            if ensemble:
                maps = [mean_softmax(outputs).argmax(dim=1).to(torch.uint8).cpu().numpy()]
            else:
                maps = [
                    model.head.postprocess(output.logits.float())["prediction"]
                    for model, output in zip(models, outputs, strict=True)
                ]
            yield batch_origins, [np.asarray(m, dtype=np.uint8) + np.uint8(1) for m in maps]


def _device_of(model) -> torch.device:
    parameter = next(model.parameters(), None)
    return torch.device("cpu") if parameter is None else parameter.device
