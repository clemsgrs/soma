"""Segmentation label maps over whole slides, written while each slide streams tile by tile.

A slide's tiles lie on hs2p's slide grid (``TilingConfig(grid="slide")``). Each kept tile
is read through a :class:`~soma.data.images.RegionImageSource`, encoded once by the frozen
encoder stage its models share, and decoded by every model
(:func:`~soma.models.model.forward_sharing_encoder`). The class of each pixel goes into a
rolling band of map rows, and full rows of the band go, in order, to hs2p's label-TIFF
writer (:func:`hs2p.write_label_tiff`). Neither a slide's map nor all of its tiles' outputs
are ever held in memory: only the band (and, to average, the float sums of one tile's
rows), a few rows of tiles in flight to the writer, and the writer's own reduced level.

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

__all__ = ["SlideFrame", "STITCHES", "predict_slide_label_maps", "stitch_margin_px", "WRITE_TILE_PX"]

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


class _Band:
    """The map rows still being filled: full rows of storage tiles go to ``stream``.

    Writes arrive in row order, so the first row a write touches never decreases: every
    map row above it is final. The band holds one row of storage tiles plus one model
    tile, enough for any write (at most a tile's rows).
    """

    def __init__(self, frame: SlideFrame, stream: _LabelTiffStream, tile_px: int) -> None:
        self._frame = frame
        self._stream = stream
        self._tile_px = tile_px
        self._rows = np.zeros((tile_px + frame.tile_px, frame.width), dtype=np.uint8)
        self._top = 0
        self.not_predicted = 0

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
        self.not_predicted += int(block.size - np.count_nonzero(block))
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
    def not_predicted(self) -> int:
        return self._labels.not_predicted

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
    stitch: str = "pick_one",
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
    path, and the map holds the argmax of the models' mean softmax. ``stitch`` combines
    overlapping tiles (see the module docstring): ``pick_one`` or ``average``. A slide
    without a kept tile gets an all-zero map. Returns each map's not-predicted fraction.
    """
    models = list(models)
    paths = [Path(path) for path in paths]
    expected = 1 if ensemble else len(models)
    if not models:
        raise ValueError("predict_slide_label_maps needs at least one model")
    if stitch not in STITCHES:
        raise ValueError(f"stitch must be {' or '.join(map(repr, STITCHES))}, got {stitch!r}")
    # With no margin every pixel has one tile, and averaging writes the map picking does.
    average = stitch == "average" and frame.margin_px > 0
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
    if average:
        classes = [models[0].head.num_classes] if ensemble else [model.head.num_classes for model in models]
        bands = [_AveragingBand(frame, band, int(count)) for band, count in zip(bands, classes)]
    try:
        for batch_origins, values in _predicted_tiles(
            models,
            image=image,
            origins=ordered,
            ensemble=ensemble,
            average=average,
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
    average: bool,
    spacing_um: float,
    backend: str,
    tolerance: float,
    batch_size: int,
    num_workers: int,
):
    """Yield ``(origins, values)`` per batch: ``values`` holds one array per map, the
    ``(B, tile, tile)`` uint8 classes, class ``k`` as ``k + 1``, or with ``average`` the
    ``(B, classes, tile, tile)`` float32 softmax."""
    if not origins:
        return
    source = tile_source(
        models, image=image, origins=origins, spacing_um=spacing_um, backend=backend, tolerance=tolerance
    )
    for batch_origins, outputs in tile_outputs(
        models, source=source, origins=origins, batch_size=batch_size, num_workers=num_workers
    ):
        if average:
            if ensemble:
                softmax = [mean_softmax(outputs).float()]
            else:
                softmax = [torch.softmax(output.logits.float(), dim=1) for output in outputs]
            yield batch_origins, [probabilities.cpu().numpy() for probabilities in softmax]
            continue
        if ensemble:
            maps = [mean_softmax(outputs).argmax(dim=1).to(torch.uint8).cpu().numpy()]
        else:
            maps = [
                model.head.postprocess(output.logits.float())["prediction"]
                for model, output in zip(models, outputs, strict=True)
            ]
        yield batch_origins, [np.asarray(m, dtype=np.uint8) + np.uint8(1) for m in maps]


def tile_id(origin: tuple[int, int]) -> str:
    """The id of the tile at level-0 ``origin`` in :func:`tile_source`."""
    return f"{int(origin[0])}_{int(origin[1])}"


def tile_source(
    models,
    *,
    image: ImageEntry,
    origins: Sequence[tuple[int, int]],
    spacing_um: float,
    backend: str,
    tolerance: float,
) -> RegionImageSource:
    """The kept tiles of one slide, one id per origin (:func:`tile_id`), at the models'
    tile size and ``spacing_um``."""
    ids = [tile_id(origin) for origin in origins]
    return RegionImageSource(
        {
            tid: ImageEntry(sample_id=tid, image_path=image.image_path, spacing_at_level_0=image.spacing_at_level_0)
            for tid in ids
        },
        regions=dict(zip(ids, ((int(x), int(y)) for x, y in origins))),
        size=models[0].head.geometry.target_size,
        spacing_um=spacing_um,
        backend=backend,
        tolerance=tolerance,
    )


def tile_outputs(
    models,
    *,
    source: RegionImageSource,
    origins: Sequence[tuple[int, int]],
    batch_size: int,
    num_workers: int,
):
    """Yield ``(origins, outputs)`` per batch of the tiles at ``origins``, in order:
    ``outputs`` lazily yields each model's output, behind one encoder forward for the
    models that share a stage (:func:`~soma.models.model.forward_sharing_encoder`).
    Consume ``outputs`` before the next batch."""
    ids = [tile_id(origin) for origin in origins]
    loader = DataLoader(_Tiles(source, ids), batch_size=int(batch_size), num_workers=int(num_workers), shuffle=False)
    encoder = models[0].encoder
    input_device = torch.device(getattr(encoder, "input_device", None) or _device_of(models[0]))
    start = 0
    with torch.inference_mode():
        for pixels in loader:
            count = int(pixels.shape[0])
            batch_origins = origins[start : start + count]
            start += count
            yield batch_origins, forward_sharing_encoder(models, pixels.to(input_device), pixels)


def _device_of(model) -> torch.device:
    parameter = next(model.parameters(), None)
    return torch.device("cpu") if parameter is None else parameter.device
