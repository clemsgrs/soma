"""Segmentation label maps over whole slides, written while each slide streams tile by tile.

A slide's tiles lie on hs2p's slide grid (``TilingConfig(grid="slide")``). The bundles'
outputs come from one seam, :class:`TileOutputs`: given a slide's kept tiles in the order
the writer takes them (:class:`SlideTiles`) and a batch size, it yields each batch with
every bundle's output for it, in order. The label maps here and the detection points
(:mod:`soma.evaluation.whole_slide_points`) read nothing else. Behind it,
:class:`DeviceTileOutputs` reads each kept tile through a
:class:`~soma.data.images.RegionImageSource`, encodes it once by the frozen encoder stage
its models share, and decodes it by every model
(:func:`~soma.models.model.forward_sharing_encoder`), on one device of this process.

The class of each pixel goes into a rolling band of map rows, and full rows of the band go,
in order, to hs2p's label-TIFF writer (:func:`hs2p.write_label_tiff`). Neither a slide's
map nor all of its tiles' outputs are ever held in memory: only the band (and, to average,
the float sums of one tile's rows), a few rows of tiles in flight to the writer, and the
writer's own reduced level.

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
``stitch`` names how overlapping tiles are combined (:data:`STITCHES`).

``pick_one``, the default: each map pixel takes the class of exactly one tile. A tile
contributes ``[origin + margin, origin + tile - margin)`` on each axis, and from ``0`` on
an axis where its lattice index is 0, so the map starts at pixel 0. With ``overlap: 0`` (no
margin) each tile contributes its whole output. Map pixels no kept tile contributes are 0,
"not predicted"; so are the margins a tile gives up next to a tile that was not kept.

``average``: each map pixel is the argmax of the Hann-weighted mean softmax of every kept
tile that covers it; with ``ensemble``, each tile's softmax is first the mean over the
models. The window is :func:`_hann2d`, strictly positive, the one
:class:`~soma.evaluation.sliding_window.SlidingWindowSegmentationPredictor` blends with.
A tile adds its whole output, so next to a tile that was not kept, averaging labels the
margin that pick one gives up, from the kept tile alone. Map pixels no kept tile covers
are 0. The sums are float32, one plane per class, over one tile's height of rows and the
map's width; a row goes to the band once no later tile row covers it. With ``overlap: 0``
each pixel has one tile, and averaging writes the map picking does.

Restricting to the mask
-----------------------
With a :class:`MaskScope`, each row of map pixels becomes final in the band after both
stitches (and after an ensemble's mean), and is then set to 0 wherever the slide's mask
is out of scope, before it is written and counted. The mask is read for those rows only,
on the map's grid, so one band-sized array is added. ``not_predicted_fraction`` is then
the share of the in-scope pixels that are 0, and 0.0 when none is in scope.

Values: class ``k`` is written as ``k + 1``.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from soma.data.images import RegionImageSource
from soma.data.manifests import ImageEntry
from soma.models.model import DenseModelOutput, forward_sharing_encoder, mean_softmax

__all__ = [
    "DealtTileOutputs",
    "DeviceTileOutputs",
    "SlideFrame",
    "SlideTiles",
    "STITCHES",
    "TileBatch",
    "TileOutputs",
    "predict_slide_label_maps",
    "stitch_margin_px",
    "WRITE_TILE_PX",
]

#: Side of the square tiles a label TIFF is stored in (hs2p needs a multiple of 16).
WRITE_TILE_PX = 256
#: The largest class count a uint8 map holds once class ``k`` is written as ``k + 1``.
MAX_CLASSES = 255
#: How often a blocked hand-over to a writer checks that the writer is still running.
_POLL_SECONDS = 0.5
#: How the overlapping tiles of a map are combined (``predict.stitch``); the first is the default.
STITCHES = ("pick_one", "average")
#: The rows of averaged sums turned into classes at a time: it bounds the temporaries.
_RELEASE_ROWS = 32
#: The widest mask window a band is read in, in map pixels: ``soma evaluate``'s side. It
#: keeps each read under hs2p's cap on a mask without a level near the map's spacing.
_MASK_WINDOW_PX = 1024


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
        # The nearest whole margin below half a tile (overlap 1), printed in full so the
        # value proposed passes this check.
        nearest = 2 * min(round(margin), (tile_px - 1) // 2) / tile_px
        raise ValueError(
            f"extraction.tiling.overlap {overlap} gives {tile_px} px tiles a margin of "
            f"{margin:g} px (tile x overlap / 2), not a whole number of pixels. Pick an "
            f"overlap whose margin is whole, such as {nearest!r}."
        )
    return int(round(margin))


def _hann2d(height: int, width: int) -> np.ndarray:
    """Separable raised-cosine window, **strictly positive** everywhere (float32).

    Pixel ``i`` of ``n`` on each axis weighs ``0.5 - 0.5 * cos(2 * pi * (i + 1) / (n + 1))``:
    no zero endpoints, so overlapping tiles blend smoothly and the summed weight is never
    zero where a tile covers. A degenerate dim (length 1) is uniform.
    """

    def _w(n: int) -> np.ndarray:
        if n <= 1:
            return np.ones(n, dtype=np.float32)
        i = np.arange(1, n + 1, dtype=np.float32)
        return (0.5 - 0.5 * np.cos(2.0 * np.pi * i / (n + 1))).astype(np.float32)

    return np.outer(_w(height), _w(width)).astype(np.float32)


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


class MaskScope:
    """The map pixels a slide's mask keeps in scope: those whose mask value is one of
    ``values``.

    ``mask`` is an open hs2p mask aligned to the slide's level 0
    (:meth:`hs2p.Mask.align_to`). It is read on the map's grid, as ``soma evaluate``
    reads an annotation raster, in windows no wider than its: each map pixel takes the
    mask value under its top-left corner, and a map pixel that lies partly off the slide is
    out of scope. The last rows read are kept, so the maps of a slide's bundles share each
    read.
    """

    def __init__(self, mask, values, frame: SlideFrame) -> None:
        self._mask = mask
        self._frame = frame
        self._lut = np.zeros(256, dtype=bool)
        self._lut[sorted(int(value) for value in values)] = True
        self._last: tuple[int, int, np.ndarray] | None = None

    def rows(self, top: int, height: int) -> np.ndarray:
        """Whether each pixel of map rows ``[top, top + height)`` is in scope,
        ``(height, width)`` bool."""
        if self._last is not None and self._last[:2] == (top, height):
            return self._last[2]
        from soma.evaluation.slide_label_maps import _annotation_window

        frame = self._frame
        # A read starts on a lattice row and column, whose level-0 row and column are whole.
        first = top // frame.stride_px * frame.stride_px
        side = frame.stride_px * max(1, _MASK_WINDOW_PX // frame.stride_px)
        inside = np.zeros((height, frame.width), dtype=bool)
        for left in range(0, frame.width, side):
            size = (min(side, frame.width - left), top - first + height)
            raw = _annotation_window(self._mask, frame, (left, first), size)
            if raw is not None:
                raw = raw[top - first :]
                inside[: raw.shape[0], left : left + raw.shape[1]] = self._lut[raw]
        self._last = (top, height, inside)
        return inside


class _Band:
    """The map rows still being filled: full rows of storage tiles go to ``stream``.

    Writes arrive in row order, so the first row a write touches never decreases: every
    map row above it is final. The band holds one row of storage tiles plus one model
    tile, enough for any write (at most a tile's rows). With a ``scope``, a final row is 0
    wherever the scope is out.
    """

    def __init__(
        self, frame: SlideFrame, stream: _LabelTiffStream, tile_px: int, scope: MaskScope | None = None
    ) -> None:
        self._frame = frame
        self._stream = stream
        self._tile_px = tile_px
        self._scope = scope
        self._rows = np.zeros((tile_px + frame.tile_px, frame.width), dtype=np.uint8)
        self._top = 0
        self.in_scope = 0
        self.not_predicted = 0

    @property
    def not_predicted_fraction(self) -> float:
        """The share of the in-scope pixels emitted so far that hold 0; 0.0 without one."""
        return self.not_predicted / self.in_scope if self.in_scope else 0.0

    def place(self, x: int, y: int, values: np.ndarray) -> None:
        """Pick one: write the part of the tile at level-0 ``(x, y)`` that it contributes."""
        (row_first, row_stop), (col_first, col_stop) = self._frame.contribution(x, y)
        if row_stop <= row_first or col_stop <= col_first:
            return
        column, row = self._frame.origin(x, y)
        self.write(row_first, col_first, values[row_first - row : row_stop - row, col_first - column : col_stop - column])

    def write(self, row: int, column: int, values: np.ndarray) -> None:
        """Write ``values`` from map ``(column, row)``: every row above ``row`` is final."""
        while self._top + self._tile_px <= row:
            self._emit()
        rows, cols = values.shape
        self._rows[row - self._top : row - self._top + rows, column : column + cols] = values

    def finish(self) -> None:
        while self._top < self._frame.height:
            self._emit()

    def _emit(self) -> None:
        height = min(self._tile_px, self._frame.height - self._top)
        block = self._rows[:height]
        if self._scope is None:
            in_scope = block.size
        else:
            inside = self._scope.rows(self._top, height)
            np.multiply(block, inside, out=block)
            in_scope = int(np.count_nonzero(inside))
        # Every pixel out of scope is 0 now, so the nonzero pixels are all in scope.
        self.in_scope += in_scope
        self.not_predicted += in_scope - int(np.count_nonzero(block))
        for start in range(0, self._frame.width, self._tile_px):
            # A copy: the band is refilled before the writer takes the tile.
            self._stream.put(block[:, start : start + self._tile_px].copy())
        self._rows[: -self._tile_px] = self._rows[self._tile_px :]
        self._rows[-self._tile_px :] = 0
        self._top += self._tile_px


class _AveragingBand:
    """Average: the map rows tiles still add to, as Hann-weighted sums of softmax. Once no
    later tile covers a row, its classes go to ``labels``.

    Tiles arrive in row order of the lattice, so a tile at map row ``row`` covers no row
    above it: those rows are final. The sums hold one tile's height of rows, one float32
    plane per class. A pixel's weight is the same for every class, so the argmax of its
    weighted sum is that of its weighted mean, and no plane of weights is kept. A pixel no
    tile covered has a zero sum, and is 0, not predicted.
    """

    def __init__(self, frame: SlideFrame, labels: _Band, num_classes: int) -> None:
        self._frame = frame
        self._labels = labels
        self._window = _hann2d(frame.tile_px, frame.tile_px)
        self._sums = np.zeros((num_classes, frame.tile_px, frame.width), dtype=np.float32)
        self._top = 0

    @property
    def not_predicted_fraction(self) -> float:
        return self._labels.not_predicted_fraction

    def place(self, x: int, y: int, probabilities: np.ndarray) -> None:
        """Add the ``(classes, tile, tile)`` softmax of the tile at level-0 ``(x, y)``."""
        column, row = self._frame.origin(x, y)
        self._release(row)
        rows = min(self._frame.tile_px, self._frame.height - row)
        cols = min(self._frame.tile_px, self._frame.width - column)
        if rows <= 0 or cols <= 0:
            return
        self._sums[:, :rows, column : column + cols] += probabilities[:, :rows, :cols] * self._window[:rows, :cols]

    def finish(self) -> None:
        self._release(self._frame.height)
        self._labels.finish()

    def _release(self, row: int) -> None:
        """Hand the rows above map row ``row`` to the labels, and move the sums up."""
        count = min(row, self._frame.height) - self._top
        if count <= 0:
            return
        # Rows below the sums were covered by no tile: the labels already hold them as 0.
        final = min(count, self._frame.tile_px)
        for start in range(0, final, _RELEASE_ROWS):
            stop = min(start + _RELEASE_ROWS, final)
            self._labels.write(self._top + start, 0, _classes(self._sums[:, start:stop]))
        keep = self._frame.tile_px - final
        # Move the rows up one class plane and one chunk at a time: the source and the
        # target of each copy then do not overlap in memory, and NumPy copies no temporary.
        for plane in self._sums:
            for start in range(0, keep, final):
                stop = min(start + final, keep)
                plane[start:stop] = plane[start + final : stop + final]
        self._sums[:, keep:] = 0
        self._top += count


def _classes(sums: np.ndarray) -> np.ndarray:
    """The class + 1 of the largest of ``(classes, rows, width)`` sums, ``(rows, width)``
    uint8: the first class on a tie, as ``argmax``, and 0 where every sum is 0 (no tile
    covered the pixel). One class at a time, as ``argmax`` over the first axis would copy
    the sums and return int64."""
    best = sums[0].copy()
    labels = np.ones(best.shape, dtype=np.uint8)
    larger = np.empty(best.shape, dtype=bool)
    for k in range(1, len(sums)):
        np.greater(sums[k], best, out=larger)
        labels[larger] = k + 1
        np.maximum(best, sums[k], out=best)
    labels[best <= 0] = 0
    return labels


# --- the seam: an ordered stream of batch results ---------------------------------------- #


def tile_id(origin: tuple[int, int]) -> str:
    """The id of the tile at level-0 ``origin`` in :meth:`SlideTiles.source`."""
    return f"{int(origin[0])}_{int(origin[1])}"


@dataclass(frozen=True)
class SlideTiles:
    """A slide's kept tiles, in the order the map writer takes them, and how they are read.

    ``origins`` are the tiles' level-0 origins on the slide grid, put in row order of the
    lattice (by row, then by column): the bands only move down. Each tile is read from
    ``image`` at ``spacing_um`` (``backend``, ``tolerance``).
    """

    image: ImageEntry
    origins: tuple[tuple[int, int], ...]
    spacing_um: float
    backend: str = "auto"
    tolerance: float = 0.05

    def __post_init__(self) -> None:
        ordered = sorted(((int(x), int(y)) for x, y in self.origins), key=lambda origin: (origin[1], origin[0]))
        object.__setattr__(self, "origins", tuple(ordered))

    def source(self, size: int | tuple[int, int]) -> RegionImageSource:
        """The tiles as ``size`` px windows at ``spacing_um``, one id per origin
        (:func:`tile_id`)."""
        ids = [tile_id(origin) for origin in self.origins]
        image = self.image
        return RegionImageSource(
            {
                tid: ImageEntry(sample_id=tid, image_path=image.image_path, spacing_at_level_0=image.spacing_at_level_0)
                for tid in ids
            },
            regions=dict(zip(ids, self.origins)),
            size=size,
            spacing_um=self.spacing_um,
            backend=self.backend,
            tolerance=self.tolerance,
        )


class TileBatch(NamedTuple):
    """One batch's results: the origins of its tiles, in order, and every bundle's output
    for the batch, in the bundles' order. ``outputs`` may compute each output when it is
    read: read it to its end before the next batch."""

    origins: tuple[tuple[int, int], ...]
    outputs: Iterable[DenseModelOutput]


class TileOutputs(Protocol):
    """Where a whole-slide prediction takes its per-tile results from.

    An implementation owns the bundles' models and the device or devices they run on.
    ``heads`` are the bundles' task heads, in order. :meth:`batches` cuts a slide's kept
    tiles, in their order, into batches of ``batch_size`` tiles (the last one shorter) and
    yields each batch's results, in order. :meth:`subset` gives the same results for the
    bundles at ``indices`` only. The stitching and the point collection read nothing else.
    """

    @property
    def heads(self) -> Sequence[Any]: ...

    def subset(self, indices: Sequence[int]) -> "TileOutputs": ...

    def batches(self, tiles: SlideTiles, *, batch_size: int) -> Iterator[TileBatch]: ...


class DeviceTileOutputs:
    """The bundles' models on one device of this process.

    ``models`` are :class:`~soma.models.model.DenseModel` models with a frozen encoder
    stage, in eval mode on their device. Each batch is read by ``num_workers`` reader
    processes (``0``: in this process), encoded once by each encoder stage the models
    share, and decoded by every model (:func:`~soma.models.model.forward_sharing_encoder`),
    each output computed when it is read.
    """

    def __init__(self, models: Sequence[Any], *, num_workers: int = 0) -> None:
        self._models = list(models)
        self._num_workers = int(num_workers)

    @classmethod
    def load(cls, bundle_paths: Sequence[str | Path], *, num_workers: int = 0) -> "DeviceTileOutputs":
        """The bundles at ``bundle_paths`` on this process's device: CUDA when there is
        one, else the CPU. A bundle trained on cached grids runs behind the frozen encoder
        stage of the encoder recipe it records
        (:func:`soma.pipeline.build.bundle_encoder_stage`), one stage for every bundle with
        the same recipe."""
        from soma.learn import load as load_bundle
        from soma.models.model import DenseModel
        from soma.pipeline.build import bundle_encoder_stage

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = []
        for path in bundle_paths:
            bundle = load_bundle(path)
            stage = bundle_encoder_stage(bundle)
            model = bundle.model
            if model.encoder is None:
                model = DenseModel(model.decoder, model.head, encoder=stage, adaptor=model.adaptor)
            models.append(model.to(device).eval())
        return cls(models, num_workers=num_workers)

    @property
    def heads(self) -> list[Any]:
        return [model.head for model in self._models]

    def subset(self, indices: Sequence[int]) -> "DeviceTileOutputs":
        return DeviceTileOutputs([self._models[index] for index in indices], num_workers=self._num_workers)

    def batches(self, tiles: SlideTiles, *, batch_size: int) -> Iterator[TileBatch]:
        if not tiles.origins:
            return
        source = tiles.source(self._models[0].head.geometry.target_size)
        ids = [tile_id(origin) for origin in tiles.origins]
        loader = DataLoader(
            _Tiles(source, ids), batch_size=int(batch_size), num_workers=self._num_workers, shuffle=False
        )
        encoder = self._models[0].encoder
        input_device = torch.device(getattr(encoder, "input_device", None) or _device_of(self._models[0]))
        start = 0
        with torch.inference_mode():
            for pixels in loader:
                count = int(pixels.shape[0])
                origins = tiles.origins[start : start + count]
                start += count
                yield TileBatch(origins, forward_sharing_encoder(self._models, pixels.to(input_device), pixels))


class DealtTileOutputs:
    """The bundles on several devices, each with its own :class:`TileOutputs`.

    :meth:`batches` cuts a slide's kept tiles, in their order, into chunks of
    ``batch_size`` tiles: the batches one device would form. Chunk ``i`` goes to device
    ``i mod N``, which gets its chunks as a slide of its own tiles, in order, and cuts
    them into the same batches. A device with no chunk is given nothing. The chunks'
    results are collected in order, and chunk ``i`` is asked of its device only once
    chunk ``i - 1`` was read, so this object holds one chunk's results at a time.
    ``heads`` are the first device's, and :meth:`subset` takes the same bundles on every
    device.
    """

    def __init__(self, devices: Sequence[TileOutputs]) -> None:
        if not devices:
            raise ValueError("DealtTileOutputs needs at least one device")
        self._devices = list(devices)

    @property
    def heads(self) -> Sequence[Any]:
        return self._devices[0].heads

    def subset(self, indices: Sequence[int]) -> "DealtTileOutputs":
        return DealtTileOutputs([device.subset(indices) for device in self._devices])

    def batches(self, tiles: SlideTiles, *, batch_size: int) -> Iterator[TileBatch]:
        size = int(batch_size)
        chunks = [tiles.origins[start : start + size] for start in range(0, len(tiles.origins), size)]
        count = len(self._devices)
        # Every device is asked for its chunks before the first is read, so all start at once.
        streams = [
            iter(
                device.batches(
                    replace(tiles, origins=tuple(origin for chunk in chunks[k::count] for origin in chunk)),
                    batch_size=size,
                )
            )
            for k, device in enumerate(self._devices[: len(chunks)])
        ]
        try:
            for index, chunk in enumerate(chunks):
                batch = next(streams[index % count], None)
                if batch is None or tuple(batch.origins) != chunk:
                    raise RuntimeError(
                        f"device {index % count} did not give chunk {index} of slide "
                        f"'{tiles.image.sample_id}' (tiles {chunk[0]} to {chunk[-1]})"
                    )
                yield batch
            for k, stream in enumerate(streams):
                if next(stream, None) is not None:
                    raise RuntimeError(f"device {k} gave more batches than slide '{tiles.image.sample_id}' has chunks")
        finally:
            for stream in streams:
                close = getattr(stream, "close", None)
                if close is not None:
                    close()


class _Tiles(Dataset):
    """The kept tiles of one slide, in order, as ``(3, tile, tile)`` uint8 pixels."""

    def __init__(self, source: RegionImageSource, ids: Sequence[str]) -> None:
        self._source = source
        self._ids = list(ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._source.load(self._ids[index])


def _device_of(model) -> torch.device:
    parameter = next(model.parameters(), None)
    return torch.device("cpu") if parameter is None else parameter.device


# --- the label maps ------------------------------------------------------------------------ #


def predict_slide_label_maps(
    outputs: TileOutputs,
    *,
    tiles: SlideTiles,
    frame: SlideFrame,
    paths: Sequence[str | Path],
    metadata: Sequence[Mapping[str, Any]],
    ensemble: bool = False,
    stitch: str = "pick_one",
    batch_size: int = 8,
    scope: MaskScope | None = None,
) -> list[float]:
    """Write one slide's label maps as label TIFFs at ``paths``, tile by tile.

    ``outputs`` gives each batch's results (:class:`TileOutputs`) for the segmentation
    bundles it runs. ``tiles`` are the slide's kept tiles, on the slide grid ``frame`` was
    built for. Without ``ensemble`` there is one path (and one ``metadata`` mapping, stored
    in the file) per bundle, and each map holds that bundle's classes; with it there is one
    path, and the map holds the argmax of the bundles' mean softmax. ``stitch`` combines
    overlapping tiles (see the module docstring): ``pick_one`` or ``average``. With a
    ``scope``, every map is 0 where the scope is out. A slide without a kept tile gets an
    all-zero map. Returns each map's not-predicted fraction: over all its pixels, or over
    the in-scope pixels with a ``scope``.
    """
    heads = list(outputs.heads)
    paths = [Path(path) for path in paths]
    expected = 1 if ensemble else len(heads)
    if not heads:
        raise ValueError("predict_slide_label_maps needs at least one bundle")
    if stitch not in STITCHES:
        raise ValueError(f"stitch must be {' or '.join(map(repr, STITCHES))}, got {stitch!r}")
    # With no margin every pixel has one tile, and averaging writes the map picking does.
    average = stitch == "average" and frame.margin_px > 0
    if len(paths) != expected or len(metadata) != expected:
        raise ValueError(f"{expected} map(s) to write, got {len(paths)} path(s) and {len(metadata)} metadata")
    for head in heads:
        if int(head.num_classes) > MAX_CLASSES:
            raise ValueError(
                f"a label map holds at most {MAX_CLASSES} classes (class k is stored as k + 1 "
                f"in uint8), got a head with {head.num_classes}"
            )
    streams = [
        _LabelTiffStream(path, frame=frame, metadata=meta, tile_px=WRITE_TILE_PX)
        for path, meta in zip(paths, metadata)
    ]
    try:
        bands = [_Band(frame, stream, WRITE_TILE_PX, scope) for stream in streams]
        if average:
            classes = [heads[0].num_classes] if ensemble else [head.num_classes for head in heads]
            bands = [_AveragingBand(frame, band, int(count)) for band, count in zip(bands, classes)]
        for batch_origins, values in _predicted_tiles(
            outputs, tiles=tiles, ensemble=ensemble, average=average, batch_size=batch_size
        ):
            # Strict: draining ``values`` to its end also frees the batch's encoder grid.
            for band, maps in zip(bands, values, strict=True):
                for (x, y), labels in zip(batch_origins, maps):
                    band.place(x, y, labels)
            del values, maps, labels  # not held while the next batch is decoded
        for band in bands:
            band.finish()
        for stream in streams:
            stream.close()
    except BaseException:
        for stream in streams:
            stream.abort()
        raise
    return [band.not_predicted_fraction for band in bands]


def _predicted_tiles(outputs: TileOutputs, *, tiles: SlideTiles, ensemble: bool, average: bool, batch_size: int):
    """Yield ``(origins, values)`` per batch: ``values`` gives one array per map, the
    ``(B, tile, tile)`` uint8 classes, class ``k`` as ``k + 1``, or with ``average`` the
    ``(B, classes, tile, tile)`` float32 softmax. With ``average`` and without ``ensemble``,
    each array is computed when it is asked for: consume ``values`` to its end before the
    next batch."""
    heads = list(outputs.heads)
    for batch_origins, results in outputs.batches(tiles, batch_size=batch_size):
        if average:
            if ensemble:
                yield batch_origins, [mean_softmax(results).float().cpu().numpy()]
            else:
                # Each bundle's softmax goes to the host, and into its band, before the
                # next bundle is decoded.
                yield batch_origins, (torch.softmax(o.logits.float(), dim=1).cpu().numpy() for o in results)
            continue
        if ensemble:
            maps = [mean_softmax(results).argmax(dim=1).to(torch.uint8).cpu().numpy()]
        else:
            maps = [
                head.postprocess(output.logits.float())["prediction"]
                for head, output in zip(heads, results, strict=True)
            ]
        yield batch_origins, [np.asarray(m, dtype=np.uint8) + np.uint8(1) for m in maps]
