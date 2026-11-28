"""``soma.predict_slides``: one label TIFF per slide, streamed tile by tile.

The slides are synthetic TIFFs, the bundles carry the weight-free ``fake`` encoder's
recipe, and everything runs on the CPU. A map is read back through hs2p; its value 0 means
"not predicted" and class ``k`` is stored as ``k + 1``.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import soma
from soma.data import build_label_remap
from soma.learn.grid import GridBatch
from soma.models.decoders.base import Decoder
from tests.whole_slides import (
    MEMORY_PRELUDE,
    RECIPE,
    SPACING_UM,
    TIFFFILE,
    TILE,
    blocky,
    fold_provenance,
    model_of,
    peak_rss_of,
    read_map,
    save_bundle,
    tile_pixels,
    write_large_slide,
    write_mask,
    write_pyramid,
    write_records,
    write_slide,
)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def test_overlap_zero_writes_what_soma_predict_returns_for_each_tile(tmp_path):
    """With no overlap, each kept tile's whole output is written: the map read back
    through hs2p, minus 1, is ``soma.predict`` over the same tiles, pixel for pixel, and
    every pixel outside the kept tiles is 0."""
    height, width = 120, 150
    slide = write_slide(tmp_path / "s1.tif", blocky(height, width))
    tissue = np.zeros((height, width), dtype=np.uint8)
    tissue[0:96, 32:128] = 1  # exactly the tiles x in {32, 64, 96}, y in {0, 32, 64}
    mask = write_mask(tmp_path / "s1_mask.tif", tissue)
    records = write_records(
        tmp_path / "dataset.csv", [{"sample_id": "s1", "image_path": str(slide), "mask_path": str(mask)}]
    )
    bundle = save_bundle(tmp_path / "run" / "bundle")

    index = soma.predict_slides(
        output_dir=tmp_path / "predictions",
        bundles=[bundle],
        records=records,
        extraction={"tiling": {"overlap": 0.0, **TIFFFILE}, "masks": {"min_coverage": {"tissue": 0.5}}},
        batch_size=4,
    )

    (row,) = index.itertuples()
    label_map = read_map(tmp_path / "predictions" / row.path)
    assert label_map.shape == (height, width)
    regions = {f"{x}_{y}": (x, y) for x in (32, 64, 96) for y in (0, 32, 64)}
    pixels = tile_pixels(slide, regions)
    ids = list(regions)
    batches = [
        GridBatch(
            features=None,
            pixels=torch.stack([pixels.load(tid) for tid in ids[start : start + 4]]),
            targets={},
            sample_ids=tuple(ids[start : start + 4]),
        )
        for start in range(0, len(ids), 4)
    ]
    expected = soma.predict(model_of(bundle), batches, device="cpu")
    predicted = np.zeros_like(label_map, dtype=bool)
    for tid, (x, y) in regions.items():
        np.testing.assert_array_equal(label_map[y : y + TILE, x : x + TILE].astype(int) - 1, expected[tid])
        predicted[y : y + TILE, x : x + TILE] = True
    assert not label_map[~predicted].any()
    assert row.num_tiles == 9
    assert row.not_predicted_fraction == pytest.approx(1 - 9 * TILE * TILE / (height * width))


# --- pick one: which tile each pixel comes from --------------------------------------- #

#: Classes of the provenance decoder: its value at a pixel names the tile and the patch.
PROVENANCE_CLASSES = 250


def provenance_class(origin_block_x, origin_block_y, patch_x, patch_y):
    """The class the provenance decoder gives patch ``(patch_x, patch_y)`` of the tile whose
    first patch is block ``(origin_block_x, origin_block_y)`` of the read level."""
    return (13 * origin_block_x + 29 * origin_block_y + 3 * patch_x + 5 * patch_y) % PROVENANCE_CLASSES


@dataclass
class ProvenanceParams:
    """The provenance decoder has no params."""


class ProvenanceDecoder(Decoder):
    """Reads where its tile is from the pixels and paints each 8 px patch with
    :func:`provenance_class`.

    The provenance slide is one colour per 8 px block of the read level, red the block's
    column and green its row, so the ``fake`` encoder's patch means are those colours
    exactly: patch ``(0, 0)`` names the tile's origin, and a patch's index its place in
    the tile.
    """

    def __init__(self, *, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self._num_classes = int(num_classes)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def forward(self, X):
        colours = torch.round((X * 0.25 + 0.5) * 255).long()  # the fake encoder's normalisation
        batch, _, rows, cols = X.shape
        patch_y, patch_x = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
        classes = provenance_class(
            colours[:, 0, :1, :1], colours[:, 1, :1, :1], patch_x[None], patch_y[None]
        )
        logits = torch.nn.functional.one_hot(classes, self._num_classes).permute(0, 3, 1, 2).float()
        return logits.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)


@pytest.fixture
def provenance_decoder(monkeypatch):
    """Register :class:`ProvenanceDecoder` for one test, then restore the registry."""
    from soma.models import decoder_registry

    monkeypatch.setattr(decoder_registry, "_entries", dict(decoder_registry._entries))
    soma.register("decoders", "provenance", ProvenanceDecoder, params=ProvenanceParams)
    return {"name": "provenance", "params": {}}


def block_coordinates(height: int, width: int) -> np.ndarray:
    """RGB: red the column and green the row of each 8 px block."""
    rows, cols = np.mgrid[0:height, 0:width]
    return np.stack([cols // 8, rows // 8, np.full_like(rows, 128)], axis=-1).astype(np.uint8)


def test_each_pixel_comes_from_the_tile_the_pick_one_rule_names(tmp_path, provenance_decoder):
    """Overlap 0.5 on 32 px tiles: a 16 px stride and an 8 px margin. The slide is read at
    a level of downsample 3, so the level-0 stride is 48 px and no tile origin divides
    into map pixels by a power of two. Every map pixel, including the first row and
    column and the right and bottom edges, holds the class of the tile the rule names:
    tile 0 for the first 24 px of an axis, then tile ``(p - 8) // 16``."""
    read_h, read_w = 104, 134  # 6.5 by 8.4 strides: the last tiles overhang the edges
    slide, (height, width), base_spacing = write_pyramid(tmp_path / "s1.tif", block_coordinates(read_h, read_w), downsample=3)
    mask = write_mask(tmp_path / "s1_mask.tif", np.ones((height, width)), spacing_um=base_spacing)
    records = write_records(
        tmp_path / "dataset.csv", [{"sample_id": "s1", "image_path": str(slide), "mask_path": str(mask)}]
    )
    bundle = save_bundle(tmp_path / "run" / "bundle", num_classes=PROVENANCE_CLASSES, decoder=provenance_decoder)

    index = soma.predict_slides(
        output_dir=tmp_path / "predictions",
        bundles=[bundle],
        records=records,
        extraction={"tiling": {"overlap": 0.5, **TIFFFILE}},
    )

    (row,) = index.itertuples()
    # The lattice: origins every 48 level-0 px (16 px at the read level, downsample 3),
    # so the map is the read level's pixel grid.
    assert (row.width, row.height) == (read_w, read_h)
    assert row.spacing_um == pytest.approx(SPACING_UM)
    assert row.num_tiles == len(range(0, width, 48)) * len(range(0, height, 48))
    label_map = read_map(tmp_path / "predictions" / row.path)
    stride, margin = 16, 8

    def tile_of(position: np.ndarray) -> np.ndarray:
        return np.where(position < stride + margin, 0, (position - margin) // stride)

    rows, cols = np.mgrid[0:read_h, 0:read_w]
    tile_x, tile_y = tile_of(cols), tile_of(rows)
    expected = provenance_class(
        tile_x * stride // 8, tile_y * stride // 8, (cols - tile_x * stride) // 8, (rows - tile_y * stride) // 8
    )
    np.testing.assert_array_equal(label_map.astype(int) - 1, expected)


# --- average: the Hann-weighted mean softmax ------------------------------------------ #

#: Each tile column's softmax for the column-softmax decoder: the tile at ``x = 16 k``
#: (overlap 0.5 on 32 px tiles) gives every pixel ``TABLE[k]``.
TABLE = [[0.2, 0.5, 0.3], [0.05, 0.05, 0.9], [0.6, 0.2, 0.2]]


@dataclass
class ColumnSoftmaxParams:
    """``table[k]`` is the softmax of the tiles whose first patch has red ``k``."""

    table: list | None = None


class ColumnSoftmaxDecoder(Decoder):
    """One softmax over a whole tile, picked by the red of its first 8 px patch: the
    logits are ``log(table[red])``, so the softmax is ``table[red]`` exactly."""

    def __init__(self, *, input_dim: int, num_classes: int, table=None) -> None:
        super().__init__()
        self._num_classes = int(num_classes)
        self._table = torch.tensor(table or TABLE, dtype=torch.float32)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def forward(self, X):
        red = torch.round((X[:, 0, 0, 0] * 0.25 + 0.5) * 255).long()  # the fake encoder's normalisation
        batch, _, rows, cols = X.shape
        logits = torch.log(self._table[red])[:, :, None, None]
        return logits.expand(batch, self._num_classes, rows * 8, cols * 8).contiguous()


@pytest.fixture
def column_softmax(monkeypatch):
    """Register :class:`ColumnSoftmaxDecoder` for one test, then restore the registry."""
    from soma.models import decoder_registry

    monkeypatch.setattr(decoder_registry, "_entries", dict(decoder_registry._entries))
    soma.register("decoders", "column-softmax", ColumnSoftmaxDecoder, params=ColumnSoftmaxParams)
    return lambda table=None: {"name": "column-softmax", "params": {"table": table}}


def column_slide(tmp_path, *, tissue_columns: int | None = None) -> str:
    """A 32 x 48 px slide whose red is ``column // 16``: with overlap 0.5 on 32 px tiles,
    the tile at ``x = 16 k`` reads red ``k`` in its first patch. Its mask is tissue in the
    first ``tissue_columns`` columns, or everywhere."""
    height, width = 32, 48
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.arange(width)[None] // 16
    image[..., 2] = 128
    tissue = np.zeros((height, width), dtype=np.uint8)
    tissue[:, : width if tissue_columns is None else tissue_columns] = 1
    slide = write_slide(tmp_path / "s1.tif", image)
    mask = write_mask(tmp_path / "s1_mask.tif", tissue)
    return write_records(tmp_path / "dataset.csv", [{"sample_id": "s1", "image_path": str(slide), "mask_path": str(mask)}])


def hann(i: int, n: int = TILE) -> float:
    """The raised-cosine weight of pixel ``i`` of an ``n`` px tile, on one axis."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * (i + 1) / (n + 1))


#: Overlap 0.5; ``a_t: 0`` keeps the tissue of a slide smaller than four tiles.
HALF = {"tiling": {"overlap": 0.5, **TIFFFILE}, "filter": {"a_t": 0}}


def test_average_takes_the_argmax_of_the_hann_weighted_mean_softmax(tmp_path, column_softmax):
    """Map column 20 lies in tile 0's middle, 20 px into it, and 4 px into tile 1: pick one
    takes tile 0's class 1. The tiles of one column share their softmax, so the mean over
    the rows' weights drops out, and the mean softmax there is
    ``(w(20) * TABLE[0] + w(4) * TABLE[1]) / (w(20) + w(4))``, with ``w(20) = 0.827`` and
    ``w(4) = 0.210``: ``[0.170, 0.409, 0.421]``, class 2."""
    records = column_slide(tmp_path)
    bundle = save_bundle(tmp_path / "run" / "bundle", decoder=column_softmax())
    maps = {}
    for stitch in ("pick_one", "average"):
        index = soma.predict_slides(
            output_dir=tmp_path / stitch, bundles=bundle, records=records, extraction=HALF, stitch=stitch
        )
        (row,) = index.itertuples()
        maps[stitch] = read_map(tmp_path / stitch / row.path).astype(int) - 1

    w0, w1 = hann(20), hann(4)
    mean = (w0 * np.array(TABLE[0]) + w1 * np.array(TABLE[1])) / (w0 + w1)
    assert (round(w0, 3), round(w1, 3)) == (0.827, 0.210)
    np.testing.assert_allclose(mean, [0.170, 0.409, 0.421], atol=5e-4)
    assert (maps["pick_one"][:, 20] == 1).all()
    assert (maps["average"][:, 20] == 2).all()
    # Every column, worked out the same way over the tiles that cover it (tile ``k`` at
    # ``x = 16 k``); the narrowest top-two margin, 0.0125, is at column 20.
    np.testing.assert_array_equal(maps["average"], np.tile([1] * 20 + [2] * 22 + [0] * 6, (32, 1)))


#: A second bundle's table: tile 0 prefers class 0 and tile 1 class 1.
OTHER_TABLE = [[0.6, 0.2, 0.2], [0.05, 0.9, 0.05], [0.6, 0.2, 0.2]]


def test_average_writes_each_bundle_s_map_or_averages_the_ensemble_s_mean_softmax(tmp_path, column_softmax):
    """At column 20 (``w(20) = 0.827``, ``w(4) = 0.210``): bundle a alone gives
    ``[0.170, 0.409, 0.421]``, class 2, and bundle b alone ``[0.489, 0.342, 0.170]``,
    class 0. The ensemble averages each tile's softmax over the two, tile 0's
    ``[0.4, 0.35, 0.25]`` and tile 1's ``[0.05, 0.475, 0.475]``: ``[0.329, 0.375, 0.296]``,
    class 1. Pick one would take tile 0's class 0."""
    records = column_slide(tmp_path)
    bundles = [
        save_bundle(tmp_path / name / "bundle", decoder=column_softmax(table))
        for name, table in (("a", TABLE), ("b", OTHER_TABLE))
    ]
    w0, w1 = hann(20), hann(4)
    ensemble_tile_0, ensemble_tile_1 = (np.array(TABLE[:2]) + np.array(OTHER_TABLE[:2])) / 2
    np.testing.assert_allclose((w0 * np.array(OTHER_TABLE[0]) + w1 * np.array(OTHER_TABLE[1])) / (w0 + w1), [0.489, 0.342, 0.170], atol=5e-4)
    np.testing.assert_allclose(ensemble_tile_0, [0.4, 0.35, 0.25])
    np.testing.assert_allclose((w0 * ensemble_tile_0 + w1 * ensemble_tile_1) / (w0 + w1), [0.329, 0.375, 0.296], atol=5e-4)

    def column_20(output_dir, **kwargs) -> dict[str, np.ndarray]:
        index = soma.predict_slides(
            output_dir=output_dir, bundles=bundles, records=records, extraction=HALF, stitch="average", **kwargs
        )
        return {row.bundle: read_map(output_dir / row.path)[:, 20].astype(int) - 1 for row in index.itertuples()}

    each = column_20(tmp_path / "each")
    ensemble = column_20(tmp_path / "ensemble", ensemble=True)

    assert sorted(each) == ["a", "b"]
    assert (each["a"] == 2).all() and (each["b"] == 0).all()
    assert list(ensemble) == ["ensemble"] and (ensemble["ensemble"] == 1).all()


def test_next_to_a_tile_that_was_not_kept_average_labels_the_margin_pick_one_gives_up(tmp_path, column_softmax):
    """Tissue in the first 8 columns keeps the tiles of column 0 only (``x = 0``). Pick one
    writes their middle and their outer margin, columns 0 to 23, and gives up their inner
    margin, columns 24 to 31, which tile 1 would have written: those are 0. Averaging
    labels columns 0 to 31 from tile 0 alone, class 1. Columns 32 on are covered by no kept
    tile: 0 in both."""
    records = column_slide(tmp_path, tissue_columns=8)
    bundle = save_bundle(tmp_path / "run" / "bundle", decoder=column_softmax())
    maps, rows = {}, {}
    for stitch in ("pick_one", "average"):
        index = soma.predict_slides(
            output_dir=tmp_path / stitch,
            bundles=bundle,
            records=records,
            extraction={**HALF, "masks": {"min_coverage": {"tissue": 0.1}}},
            stitch=stitch,
        )
        (rows[stitch],) = index.itertuples()
        maps[stitch] = read_map(tmp_path / stitch / rows[stitch].path)

    assert rows["pick_one"].num_tiles == rows["average"].num_tiles == 2
    assert (maps["pick_one"][:, :24] == 2).all() and not maps["pick_one"][:, 24:].any()
    assert (maps["average"][:, :32] == 2).all() and not maps["average"][:, 32:].any()
    assert rows["pick_one"].not_predicted_fraction == pytest.approx(24 / 48)
    assert rows["average"].not_predicted_fraction == pytest.approx(16 / 48)


def test_with_overlap_zero_average_writes_the_pick_one_map(tmp_path):
    """With no overlap each pixel has one tile: both stitches write the same map."""
    _, records, _ = one_slide_cohort(tmp_path)
    bundle = save_bundle(tmp_path / "run" / "bundle")
    rows, maps = {}, {}
    for stitch in ("pick_one", "average"):
        index = soma.predict_slides(
            output_dir=tmp_path / stitch, bundles=bundle, records=records, extraction=OVERLAP_ZERO, stitch=stitch
        )
        (rows[stitch],) = index.itertuples()
        maps[stitch] = read_map(tmp_path / stitch / rows[stitch].path)

    np.testing.assert_array_equal(maps["average"], maps["pick_one"])
    assert maps["average"].any()
    assert rows["average"].not_predicted_fraction == rows["pick_one"].not_predicted_fraction


# --- restricted to the mask ----------------------------------------------------------- #

#: A mask vocabulary where values 1 and 2 select tiles: they are in scope. 0 and 3 are
#: declared, but select no tile, so they are out of scope.
SCOPED_MASKS = {"pixel_mapping": {"background": [0], "tissue": [1, 2], "other": [3]}, "min_coverage": {"tissue": 0}}
#: Overlap 0.5 with the scoped vocabulary.
SCOPED = {**HALF, "masks": SCOPED_MASKS}


def scoped_slide(tmp_path, name: str = "s1", *, height: int = 96, width: int = 128, seed: int = 0):
    """A slide and its mask at the slide's resolution, the map's grid: values 0 to 3 in
    4 px squares over the left half, 0 over the right half. Returns the slide's row and
    where the mask is in scope (values 1 and 2)."""
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 4, size=(height // 4, width // 4), dtype=np.uint8).repeat(4, 0).repeat(4, 1)
    values[:, width // 2 :] = 0
    slide = write_slide(tmp_path / f"{name}.tif", blocky(height, width, seed=seed))
    mask = write_mask(tmp_path / f"{name}_mask.tif", values)
    return {"sample_id": name, "image_path": str(slide), "mask_path": str(mask)}, (values == 1) | (values == 2)


def test_restrict_to_mask_writes_0_out_of_scope_and_the_unrestricted_labels_in_scope(tmp_path):
    row, in_scope = scoped_slide(tmp_path)
    records = write_records(tmp_path / "dataset.csv", [row])
    bundle = save_bundle(tmp_path / "run" / "bundle")
    maps = {}
    for restrict in (False, True):
        index = soma.predict_slides(
            output_dir=tmp_path / f"restrict_{restrict}",
            bundles=bundle,
            records=records,
            extraction=SCOPED,
            restrict_to_mask=restrict,
        )
        (map_row,) = index.itertuples()
        maps[restrict] = read_map(tmp_path / f"restrict_{restrict}" / map_row.path)

    # Unrestricted, the kept tiles are predicted in full, also out of scope.
    assert maps[False][~in_scope].any()
    np.testing.assert_array_equal(maps[True], np.where(in_scope, maps[False], 0))
    assert maps[True][in_scope].all()


def scoped_cohort(tmp_path, count: int) -> tuple[str, dict[str, np.ndarray]]:
    """The records of ``count`` scoped slides ``s0``, ``s1``, ... and where each is in scope."""
    rows, in_scope = [], {}
    for k in range(count):
        row, in_scope[f"s{k}"] = scoped_slide(tmp_path, f"s{k}", seed=k)
        rows.append(row)
    return write_records(tmp_path / "dataset.csv", rows), in_scope


def restricted_and_not(tmp_path, records, bundles, **kwargs) -> dict[bool, dict[tuple[str, str], np.ndarray]]:
    """Each map of a call with and without ``restrict_to_mask``, by ``(bundle, sample_id)``."""
    maps = {}
    for restrict in (False, True):
        output_dir = tmp_path / f"restrict_{restrict}"
        index = soma.predict_slides(
            output_dir=output_dir, bundles=bundles, records=records, restrict_to_mask=restrict, **kwargs
        )
        maps[restrict] = {(row.bundle, row.sample_id): read_map(output_dir / row.path) for row in index.itertuples()}
    return maps


@pytest.mark.parametrize(
    "how",
    [{}, {"stitch": "average"}, {"ensemble": True}, {"stitch": "average", "ensemble": True}],
    ids=["pick_one", "average", "pick_one-ensemble", "average-ensemble"],
)
def test_restrict_to_mask_applies_to_each_bundle_s_map_and_to_the_ensemble_s(tmp_path, how):
    """Two bundles over two slides: the restriction applies after stitching and after the
    ensemble's averaging, so each restricted map is its unrestricted map, 0 out of scope."""
    records, in_scope = scoped_cohort(tmp_path, 2)
    bundles = [save_bundle(tmp_path / name / "bundle", seed=seed) for seed, name in enumerate(("a", "b"))]

    maps = restricted_and_not(tmp_path, records, bundles, extraction=SCOPED, **how)

    assert len(maps[True]) == (1 if how.get("ensemble") else 2) * 2
    assert maps[True].keys() == maps[False].keys()
    for (bundle, sample_id), restricted in maps[True].items():
        unrestricted = maps[False][bundle, sample_id]
        assert unrestricted[~in_scope[sample_id]].any()
        np.testing.assert_array_equal(restricted, np.where(in_scope[sample_id], unrestricted, 0))


def test_restrict_to_mask_applies_to_the_maps_of_each_bundle_s_own_fold(tmp_path):
    records, in_scope = scoped_cohort(tmp_path, 4)
    splits = tmp_path / "splits.csv"
    pd.DataFrame(ROTATING).to_csv(splits, index=False)
    bundles = fold_bundles(tmp_path, records, splits)

    maps = restricted_and_not(
        tmp_path, records, bundles, splits=splits, fold="bundle", stitch="average", extraction=SCOPED
    )

    assert list(maps[True]) == [("fold_0", "s0"), ("fold_0", "s1"), ("fold_1", "s2"), ("fold_1", "s3")]
    for (bundle, sample_id), restricted in maps[True].items():
        np.testing.assert_array_equal(restricted, np.where(in_scope[sample_id], maps[False][bundle, sample_id], 0))


def test_restricted_not_predicted_fraction_is_the_share_of_the_in_scope_pixels(tmp_path):
    """Overlap 0 on 32 px tiles, kept at a tissue coverage of 0.5. In ``s1``, tile (0, 0) is
    all tissue and kept; an 8 x 8 px patch of tissue in the next tile covers 1/16 of it, so
    that tile is not kept. Restricted, 64 of the 1,088 tissue pixels are not predicted.
    Unrestricted, every map pixel outside the kept tile is: 8,192 of 9,216. ``s2``'s mask
    is empty: no pixel is in scope, so its fraction is 0.0 restricted and 1.0 unrestricted."""
    tissue = np.zeros((96, 96), dtype=np.uint8)
    tissue[:32, :32] = 1
    tissue[8:16, 48:56] = 1
    rows = []
    for name, mask in (("s1", tissue), ("s2", np.zeros_like(tissue))):
        slide = write_slide(tmp_path / f"{name}.tif", blocky(96, 96))
        mask_path = write_mask(tmp_path / f"{name}_mask.tif", mask)
        rows.append({"sample_id": name, "image_path": str(slide), "mask_path": str(mask_path)})
    records = write_records(tmp_path / "dataset.csv", rows)
    bundle = save_bundle(tmp_path / "run" / "bundle")
    fractions = {}
    for restrict in (False, True):
        with pytest.warns(UserWarning, match="slide 's2' has no kept tile"):
            index = soma.predict_slides(
                output_dir=tmp_path / f"restrict_{restrict}",
                bundles=bundle,
                records=records,
                extraction={**OVERLAP_ZERO, "filter": {"a_t": 0}},
                restrict_to_mask=restrict,
            )
        assert list(index.num_tiles) == [1, 0]
        fractions[restrict] = dict(zip(index.sample_id, index.not_predicted_fraction))

    assert fractions[True] == {"s1": pytest.approx(64 / 1088), "s2": 0.0}
    assert fractions[False] == {"s1": pytest.approx(8192 / 9216), "s2": 1.0}


def test_a_coarser_mask_is_read_on_the_map_s_grid_and_a_pixel_beyond_its_canvas_is_out_of_scope(tmp_path):
    """A 193 x 160 px slide at 0.25 um/px, read at 0.5 um/px: each map pixel is 2 x 2
    slide pixels, and the map is 97 px wide, so its last column lies half off the slide.
    The tissue mask is 25 x 20 px, about 8 slide pixels per mask pixel: tissue in its
    columns from 12 and its rows below 10. hs2p aligns it to the slide, and each map pixel
    takes the mask value under its top-left corner. The last map column is predicted
    without the restriction and is 0 with it, because the mask's canvas (the slide) does
    not cover it."""
    slide = write_slide(tmp_path / "s1.tif", blocky(160, 193), spacing_um=0.25)
    tissue = np.zeros((20, 25), dtype=np.uint8)
    tissue[:10, 12:] = 1
    mask = write_mask(tmp_path / "s1_mask.tif", tissue, spacing_um=0.25 * 193 / 25)
    records = write_records(
        tmp_path / "dataset.csv", [{"sample_id": "s1", "image_path": str(slide), "mask_path": str(mask)}]
    )
    bundle = save_bundle(tmp_path / "run" / "bundle")
    maps = {}
    for restrict in (False, True):
        index = soma.predict_slides(
            output_dir=tmp_path / f"restrict_{restrict}",
            bundles=bundle,
            records=records,
            extraction=HALF,
            restrict_to_mask=restrict,
        )
        (row,) = index.itertuples()
        maps[restrict] = read_map(tmp_path / f"restrict_{restrict}" / row.path)

    assert maps[True].shape == (80, 97)
    columns, rows = np.arange(97), np.arange(80)
    under = tissue[2 * rows * 20 // 160][:, 2 * columns * 25 // 193]  # the mask pixel under each corner
    in_scope = (under == 1) & (columns < 96)[None]
    np.testing.assert_array_equal(maps[True], np.where(in_scope, maps[False], 0))
    assert maps[False][:40, 96].all() and not maps[True][:, 96].any()
    assert maps[True][:40, 50:96].all()


# --- not predicted -------------------------------------------------------------------- #


def test_a_slide_without_tissue_gets_an_all_zero_map_and_a_warning(tmp_path):
    """One slide's mask is empty and another has no mask over a blank (white) slide, so
    hs2p's tissue segmentation finds nothing: each gets a map of zeros at the slide's
    size, a record with no tiles, and a warning naming it."""
    height, width = 120, 150
    empty_mask = write_mask(tmp_path / "s1_mask.tif", np.zeros((height, width)))
    s1 = write_slide(tmp_path / "s1.tif", blocky(height, width))
    s2 = write_slide(tmp_path / "s2.tif", np.full((height, width, 3), 255, dtype=np.uint8))
    records = write_records(
        tmp_path / "dataset.csv",
        [
            {"sample_id": "s1", "image_path": str(s1), "mask_path": str(empty_mask)},
            {"sample_id": "s2", "image_path": str(s2), "mask_path": None},
        ],
    )
    bundle = save_bundle(tmp_path / "run" / "bundle")

    with pytest.warns(UserWarning, match="no kept tile") as caught:
        index = soma.predict_slides(
            output_dir=tmp_path / "predictions",
            bundles=bundle,
            records=records,
            extraction={"tiling": dict(TIFFFILE)},
        )

    assert sorted(str(w.message).split("'")[1] for w in caught if "no kept tile" in str(w.message)) == ["s1", "s2"]
    assert list(index.sample_id) == ["s1", "s2"]
    for row in index.itertuples():
        label_map = read_map(tmp_path / "predictions" / row.path)
        assert label_map.shape == (height, width)
        assert not label_map.any()
        assert row.num_tiles == 0
        assert row.not_predicted_fraction == 1.0


# --- several bundles ------------------------------------------------------------------ #


def one_slide_cohort(tmp_path):
    """The overlap-0 slide: 9 kept tiles at x in {32, 64, 96}, y in {0, 32, 64}."""
    height, width = 120, 150
    slide = write_slide(tmp_path / "s1.tif", blocky(height, width))
    tissue = np.zeros((height, width), dtype=np.uint8)
    tissue[0:96, 32:128] = 1
    mask = write_mask(tmp_path / "s1_mask.tif", tissue)
    records = write_records(
        tmp_path / "dataset.csv", [{"sample_id": "s1", "image_path": str(slide), "mask_path": str(mask)}]
    )
    regions = {f"t{k}": (x, y) for k, (y, x) in enumerate((y, x) for y in (0, 32, 64) for x in (32, 64, 96))}
    return slide, records, regions


OVERLAP_ZERO = {"tiling": {"overlap": 0.0, **TIFFFILE}, "masks": {"min_coverage": {"tissue": 0.5}}}


def tile_logits(bundle_path, slide, regions) -> dict[str, torch.Tensor]:
    """Each tile's logits ``(C, H, W)`` from the bundle's model."""
    model = model_of(bundle_path)
    source = tile_pixels(slide, regions)
    with torch.no_grad():
        return {tid: model(pixels=source.load(tid)[None]).logits[0] for tid in regions}


@pytest.fixture
def encoder_forwards(monkeypatch):
    """Counts the frozen encoder stage's forwards (one per batch it encodes)."""
    calls = []
    forward = soma.extract.FrozenEncoder.forward

    def counted(self, pixels):
        calls.append(int(pixels.shape[0]))
        return forward(self, pixels)

    monkeypatch.setattr(soma.extract.FrozenEncoder, "forward", counted)
    return calls


def test_two_bundles_write_two_maps_behind_one_encoder_forward_per_batch(tmp_path, encoder_forwards):
    slide, records, regions = one_slide_cohort(tmp_path)
    bundles = [save_bundle(tmp_path / name / "bundle", seed=seed) for name, seed in (("a", 0), ("b", 1))]
    expected = {name: tile_logits(bundle, slide, regions) for name, bundle in zip("ab", bundles)}
    encoder_forwards.clear()

    index = soma.predict_slides(
        output_dir=tmp_path / "predictions", bundles=bundles, records=records, extraction=OVERLAP_ZERO, batch_size=4
    )

    assert encoder_forwards == [4, 4, 1]  # 9 tiles in batches of 4, encoded once for both bundles
    assert sorted(index.path) == ["a/s1.tif", "b/s1.tif"]
    maps = {row.bundle: read_map(tmp_path / "predictions" / row.path) for row in index.itertuples()}
    assert not np.array_equal(maps["a"], maps["b"])  # two models, two maps
    for name, logits in expected.items():
        for tid, (x, y) in regions.items():
            np.testing.assert_array_equal(maps[name][y : y + TILE, x : x + TILE] - 1, logits[tid].argmax(0).numpy())


def test_ensemble_writes_the_argmax_of_the_mean_softmax(tmp_path):
    slide, records, regions = one_slide_cohort(tmp_path)
    bundles = [save_bundle(tmp_path / name / "bundle", seed=seed) for name, seed in (("a", 0), ("b", 1))]
    logits = [tile_logits(bundle, slide, regions) for bundle in bundles]

    index = soma.predict_slides(
        output_dir=tmp_path / "predictions", bundles=bundles, records=records, extraction=OVERLAP_ZERO, ensemble=True
    )

    (row,) = index.itertuples()
    assert row.path == "ensemble/s1.tif"
    label_map = read_map(tmp_path / "predictions" / row.path)
    for tid, (x, y) in regions.items():
        mean = (torch.softmax(logits[0][tid], 0) + torch.softmax(logits[1][tid], 0)) / 2
        np.testing.assert_array_equal(label_map[y : y + TILE, x : x + TILE] - 1, mean.argmax(0).numpy())


# --- refusals ------------------------------------------------------------------------- #


@pytest.fixture
def cohort(tmp_path):
    """One small slide with tissue everywhere, and its records."""
    slide = write_slide(tmp_path / "s1.tif", blocky(96, 96))
    mask = write_mask(tmp_path / "s1_mask.tif", np.ones((96, 96)))
    return write_records(
        tmp_path / "dataset.csv", [{"sample_id": "s1", "image_path": str(slide), "mask_path": str(mask)}]
    )


def predict(tmp_path, records, bundles, **kwargs):
    kwargs.setdefault("extraction", {"tiling": dict(TIFFFILE)})
    return soma.predict_slides(output_dir=tmp_path / "predictions", bundles=bundles, records=records, **kwargs)


@pytest.mark.parametrize(
    ("tiling", "match"),
    [
        ({"requested_tile_size_px": 64}, "tiling.requested_tile_size_px is 64.*come from the bundles"),
        ({"requested_spacing_um": 1.0}, "tiling.requested_spacing_um is 1.0.*come from the bundles"),
        ({"overlap": 0.3}, "margin of 4.8 px.*not a whole number"),
    ],
    ids=["tile-size", "spacing", "non-integer-margin"],
)
def test_the_tile_is_the_bundles_and_its_margin_whole(tmp_path, cohort, tiling, match):
    bundle = save_bundle(tmp_path / "run" / "bundle")

    with pytest.raises(ValueError, match=match):
        predict(tmp_path, cohort, bundle, extraction={"tiling": {**TIFFFILE, **tiling}})
    assert not (tmp_path / "predictions").exists()


@pytest.mark.parametrize("tile_px", [224, 384, 448, 512])
@pytest.mark.parametrize("overlap", [0.3, 0.2, 0.1, 0.999])
def test_the_overlap_a_fractional_margin_refusal_proposes_is_accepted(tmp_path, cohort, tile_px, overlap):
    """The refusal of an overlap whose margin is not whole proposes one that is: fed back
    in as printed, it is accepted, also when the nearest whole margin would need overlap 1."""
    from soma.pipeline.slide_prediction import resolve_slide_prediction

    bundle = save_bundle(tmp_path / "run" / "bundle", tile_size=[tile_px, tile_px])
    where = {"output_dir": tmp_path / "predictions", "bundles": bundle, "records": cohort}

    with pytest.raises(ValueError, match="not a whole number of pixels") as refusal:
        resolve_slide_prediction(**where, extraction={"tiling": {**TIFFFILE, "overlap": overlap}})
    proposed = float(re.search(r"such as (\S+)\.$", str(refusal.value)).group(1))

    plan = resolve_slide_prediction(**where, extraction={"tiling": {**TIFFFILE, "overlap": proposed}})

    assert plan["extraction"]["tiling"]["overlap"] == proposed
    assert abs(tile_px * proposed / 2 - tile_px * overlap / 2) < 1


@pytest.mark.parametrize("stitch", ["mean", None, "Average"])
def test_another_stitch_is_refused_naming_the_two_allowed(tmp_path, cohort, stitch):
    bundle = save_bundle(tmp_path / "run" / "bundle")

    with pytest.raises(ValueError, match=r"predict\.stitch must be 'pick_one' or 'average', got"):
        predict(tmp_path, cohort, bundle, stitch=stitch)
    assert not (tmp_path / "predictions").exists()


@pytest.mark.parametrize("restrict", [False, True])
def test_the_record_and_the_map_s_metadata_hold_restrict_to_mask(tmp_path, cohort, restrict):
    import tifffile

    index = predict(tmp_path, cohort, save_bundle(tmp_path / "run" / "bundle"), restrict_to_mask=restrict)

    (row,) = index.itertuples()
    path = tmp_path / "predictions" / row.path
    with tifffile.TiffFile(path) as tiff:
        metadata = json.loads(tiff.pages[0].description)
    assert metadata["restrict_to_mask"] is restrict
    assert json.loads(path.with_suffix(".json").read_text())["restrict_to_mask"] is restrict


def test_restrict_to_mask_refuses_slides_without_a_mask_path_before_any_is_predicted(tmp_path):
    """Without a mask, hs2p's tissue segmentation keeps the tiles, and there is no mask to
    restrict the map to. The refusal names every such slide; nothing is written."""
    slides = [write_slide(tmp_path / f"s{k}.tif", blocky(96, 96)) for k in range(3)]
    rows = [{"sample_id": f"s{k}", "image_path": str(slide)} for k, slide in enumerate(slides)]
    rows[0]["mask_path"] = str(write_mask(tmp_path / "s0_mask.tif", np.ones((96, 96))))
    records = write_records(tmp_path / "dataset.csv", rows)

    with pytest.raises(
        ValueError,
        match=r"predict.restrict_to_mask needs a mask_path for every slide, and 2 slide\(s\) have none: \['s1', 's2'\]",
    ):
        predict(tmp_path, records, save_bundle(tmp_path / "run" / "bundle"), restrict_to_mask=True)
    assert not (tmp_path / "predictions").exists()


@pytest.mark.parametrize("restrict", ["yes", 1, None])
def test_restrict_to_mask_is_true_or_false(tmp_path, cohort, restrict):
    bundle = save_bundle(tmp_path / "run" / "bundle")

    with pytest.raises(ValueError, match=f"predict.restrict_to_mask must be true or false, got {restrict!r}"):
        predict(tmp_path, cohort, bundle, restrict_to_mask=restrict)
    assert not (tmp_path / "predictions").exists()


def test_a_plan_without_the_stitch_or_restrict_to_mask_key_takes_the_defaults_and_resumes(
    tmp_path, two_slides, encoder_forwards
):
    """A ``plan.yaml`` written before ``predict.stitch`` and ``predict.restrict_to_mask``
    replays as ``pick_one``, unrestricted, and a relaunch into its directory with the
    defaults is not refused."""
    from soma.pipeline.slide_prediction import slide_prediction_kwargs

    bundle = save_bundle(tmp_path / "run" / "bundle")
    output_dir = tmp_path / "predictions"
    first = soma.predict_slides(output_dir=output_dir, bundles=bundle, records=two_slides, **RESUME)
    saved = yaml.safe_load((output_dir / "plan.yaml").read_text())
    del saved["predict"]["stitch"], saved["predict"]["restrict_to_mask"]
    (output_dir / "plan.yaml").write_text(yaml.safe_dump(saved, sort_keys=False))
    encoder_forwards.clear()

    replayed = soma.predict_slides(**slide_prediction_kwargs(saved))

    assert encoder_forwards == []
    assert replayed.equals(first)
    predict_section = yaml.safe_load((output_dir / "plan.yaml").read_text())["predict"]
    assert (predict_section["stitch"], predict_section["restrict_to_mask"]) == ("pick_one", False)


def test_a_plan_s_own_tile_size_and_spacing_are_accepted(tmp_path, cohort):
    """``plan.yaml`` spells out the bundle's tile size and spacing, so setting them to
    the bundle's own values is what a replay does, and is accepted."""
    bundle = save_bundle(tmp_path / "run" / "bundle")
    tiling = {**TIFFFILE, "requested_tile_size_px": TILE, "requested_spacing_um": SPACING_UM}

    index = predict(tmp_path, cohort, bundle, extraction={"tiling": tiling})

    assert list(index.sample_id) == ["s1"]


# --- the slide reader ----------------------------------------------------------------- #

#: An extraction that names the mask reader only: the slide reader is left to the bundles.
MASK_READER_ONLY = {"tiling": {"mask_backend": "tifffile"}}


def reader_recipe(backend: str) -> dict:
    """What a bundle records when its slides were read with ``backend``."""
    return {"encoder": {**RECIPE, "backend": backend}}


def test_a_bundle_that_records_no_reader_is_read_with_auto(tmp_path, cohort):
    """With no ``extraction.tiling.backend`` and a bundle that records no slide reader,
    the plan reads the slides with ``auto``."""
    from soma.pipeline.slide_prediction import resolve_slide_prediction

    bundle = save_bundle(tmp_path / "run" / "bundle")

    plan = resolve_slide_prediction(
        output_dir=tmp_path / "predictions", bundles=bundle, records=cohort, extraction=MASK_READER_ONLY
    )

    assert plan["extraction"]["tiling"]["backend"] == "auto"


@pytest.mark.filterwarnings("ignore:extraction.tiling.backend is 'tifffile'")
@pytest.mark.parametrize(
    ("saved", "edited", "reader"),
    [(None, "tifffile", "tifffile"), ("tifffile", None, "openslide")],
    ids=["edited", "unset"],
)
def test_a_saved_plan_tiles_with_the_readers_it_names(tmp_path, cohort, saved, edited, reader):
    """``backend`` and ``mask_backend`` name the readers. A saved ``plan.yaml`` also holds
    hs2p's ``requested_backend`` and ``requested_mask_backend``, which hs2p tiles with:
    when the readers are changed, the old values must not stay behind."""
    from soma.pipeline.slide_prediction import resolve_slide_prediction

    bundle = save_bundle(tmp_path / "run" / "bundle", **reader_recipe("openslide"))
    where = {"output_dir": tmp_path / "predictions", "bundles": bundle, "records": cohort}
    given = {"tiling": {"backend": saved, "mask_backend": "tifffile"}}
    extraction = resolve_slide_prediction(**where, extraction=given)["extraction"]
    extraction["tiling"].update(backend=edited, mask_backend="pil")

    tiling = resolve_slide_prediction(**where, extraction=extraction)["extraction"]["tiling"]

    assert tiling["requested_backend"] == tiling["backend"] == reader
    assert tiling["requested_mask_backend"] == tiling["mask_backend"] == "pil"


def test_a_config_reader_other_than_the_bundles_is_used_with_a_warning_naming_both(tmp_path, cohort):
    bundle = save_bundle(tmp_path / "run" / "bundle", **reader_recipe("openslide"))

    with pytest.warns(UserWarning, match=r"extraction\.tiling\.backend is 'tifffile'.*'openslide'"):
        predict(tmp_path, cohort, bundle, extraction={"tiling": dict(TIFFFILE)})

    plan = yaml.safe_load((tmp_path / "predictions" / "plan.yaml").read_text())
    assert plan["extraction"]["tiling"]["backend"] == "tifffile"


def test_bundles_trained_with_different_readers_need_the_config_to_name_one(tmp_path, cohort):
    bundles = [
        save_bundle(tmp_path / "a" / "bundle", **reader_recipe("tifffile")),
        save_bundle(tmp_path / "b" / "bundle", **reader_recipe("openslide")),
    ]

    with pytest.raises(ValueError, match=r"different readers: .*/a/bundle with 'tifffile'; .*/b/bundle with 'openslide'"):
        predict(tmp_path, cohort, bundles, extraction=MASK_READER_ONLY)
    assert not (tmp_path / "predictions").exists()

    with pytest.warns(UserWarning, match=r"/b/bundle with 'openslide'"):
        index = predict(tmp_path, cohort, bundles, extraction={"tiling": dict(TIFFFILE)})
    assert sorted(index.bundle) == ["a", "b"]


def test_relative_sam2_paths_are_stored_absolute_and_replay_from_anywhere(tmp_path, cohort, monkeypatch):
    """A relative SAM2 checkpoint or config path resolves against the working directory,
    so ``plan.yaml`` replays to the same plan from another one."""
    from soma.pipeline.slide_prediction import resolve_slide_prediction, slide_prediction_kwargs

    bundle = save_bundle(tmp_path / "run" / "bundle")
    segmentation = {"method": "sam2", "sam2_checkpoint_path": "sam2/model.pt", "sam2_config_path": "sam2/model.yaml"}
    monkeypatch.chdir(tmp_path)

    plan = resolve_slide_prediction(
        output_dir="predictions",
        bundles=bundle,
        records=cohort,
        extraction={"tiling": dict(TIFFFILE), "segmentation": segmentation},
    )

    resolved = plan["extraction"]["segmentation"]
    assert resolved["sam2_checkpoint_path"] == str(tmp_path.resolve() / "sam2" / "model.pt")
    assert resolved["sam2_config_path"] == str(tmp_path.resolve() / "sam2" / "model.yaml")
    saved = yaml.safe_load(yaml.safe_dump(plan, sort_keys=False))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert resolve_slide_prediction(**slide_prediction_kwargs(saved)) == plan


@pytest.mark.parametrize(
    "given", [{"subset": "test"}, {"fold": 0}, {"fold": "bundle"}], ids=["subset", "fold", "fold-bundle"]
)
def test_subset_and_fold_need_splits(tmp_path, cohort, given):
    bundle = save_bundle(tmp_path / "run" / "bundle")

    with pytest.raises(ValueError, match="select from cohort.splits; set cohort.splits"):
        predict(tmp_path, cohort, bundle, **given)


@pytest.mark.parametrize(
    "other",
    [{"tile_size": [64, 64]}, {"spacing_um": 1.0}],
    ids=["tile-size", "spacing"],
)
def test_bundles_must_agree_on_tile_size_and_spacing(tmp_path, cohort, other):
    bundles = [save_bundle(tmp_path / "a" / "bundle"), save_bundle(tmp_path / "b" / "bundle", **other)]

    with pytest.raises(ValueError, match="must agree on tile size, read spacing"):
        predict(tmp_path, cohort, bundles)


@pytest.mark.parametrize(
    "other",
    [
        {"num_classes": 4},
        {"head": {"label_remap": build_label_remap({"class_0": [1], "class_1": [2], "class_2": [3]}, ignore=[0])}},
        {"head": {"ignore_index": 254}},
    ],
    ids=["classes", "raw-values", "ignore"],
)
def test_an_ensemble_needs_one_class_scheme(tmp_path, cohort, other):
    """The one map an ensemble writes records one class scheme: the bundles must agree on
    their classes, the raw annotation values that form them and the values ignored."""
    bundles = [save_bundle(tmp_path / "a" / "bundle"), save_bundle(tmp_path / "b" / "bundle", **other)]

    with pytest.raises(ValueError, match="must share one class scheme"):
        predict(tmp_path, cohort, bundles, ensemble=True)


@pytest.mark.parametrize(
    ("recorded", "match"),
    [
        ({"encoder": {"composite": True, "encoders": [RECIPE, RECIPE]}}, "composite of several encoders"),
        ({"task_family": "regression"}, "is a regression bundle; soma predict needs a segmentation or detection bundle"),
    ],
    ids=["composite", "other-family"],
)
def test_a_composite_or_other_family_bundle_is_refused(tmp_path, cohort, recorded, match):
    bundle = save_bundle(tmp_path / "run" / "bundle", **recorded)

    with pytest.raises(ValueError, match=match):
        predict(tmp_path, cohort, bundle)


def test_a_slide_coarser_than_the_read_spacing_is_refused(tmp_path):
    slide = write_slide(tmp_path / "s1.tif", blocky(96, 96), spacing_um=1.0)
    records = write_records(tmp_path / "dataset.csv", [{"sample_id": "s1", "image_path": str(slide)}])
    bundle = save_bundle(tmp_path / "run" / "bundle")

    with pytest.raises(ValueError, match="slide 's1' is at 1 um/px, coarser than the bundles' read spacing of 0.5"):
        predict(tmp_path, records, bundle)


# --- resume and drift ----------------------------------------------------------------- #


def sha256(path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def two_slides(tmp_path):
    rows = []
    for k, sample_id in enumerate(("s1", "s2")):
        slide = write_slide(tmp_path / f"{sample_id}.tif", blocky(96, 96, seed=k))
        mask = write_mask(tmp_path / f"{sample_id}_mask.tif", np.ones((96, 96)))
        rows.append({"sample_id": sample_id, "image_path": str(slide), "mask_path": str(mask)})
    return write_records(tmp_path / "dataset.csv", rows)


RESUME = {"extraction": {"tiling": {"overlap": 0.5, **TIFFFILE}}, "batch_size": 4}


def maps_of(output_dir) -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted((output_dir / "run").glob("*.tif"))}


def test_a_stopped_run_resumes_to_the_bytes_of_an_uninterrupted_one(tmp_path, two_slides, encoder_forwards, monkeypatch):
    bundle = save_bundle(tmp_path / "run" / "bundle")
    soma.predict_slides(output_dir=tmp_path / "whole", bundles=bundle, records=two_slides, **RESUME)
    per_slide = len(encoder_forwards) // 2
    encoder_forwards.clear()

    forward = soma.extract.FrozenEncoder.forward

    def stops_in_the_second_slide(self, pixels):
        if len(encoder_forwards) > per_slide:
            raise KeyboardInterrupt
        return forward(self, pixels)

    with monkeypatch.context() as patched:
        patched.setattr(soma.extract.FrozenEncoder, "forward", stops_in_the_second_slide)
        with pytest.raises(KeyboardInterrupt):
            soma.predict_slides(output_dir=tmp_path / "stopped", bundles=bundle, records=two_slides, **RESUME)
    stopped = tmp_path / "stopped"
    assert sorted(path.name for path in (stopped / "run").iterdir()) == ["s1.json", "s1.tif"]
    encoder_forwards.clear()

    index = soma.predict_slides(output_dir=stopped, bundles=bundle, records=two_slides, **RESUME)

    assert len(encoder_forwards) == per_slide  # only s2 was predicted again
    assert maps_of(stopped) == maps_of(tmp_path / "whole")
    assert list(index.sample_id) == ["s1", "s2"]
    assert pd.read_parquet(stopped / "index.parquet").equals(index)


@pytest.mark.parametrize("damage", ["missing", "truncated"])
def test_a_map_without_a_whole_record_is_predicted_again(tmp_path, two_slides, encoder_forwards, damage):
    bundle = save_bundle(tmp_path / "run" / "bundle")
    output_dir = tmp_path / "predictions"
    soma.predict_slides(output_dir=output_dir, bundles=bundle, records=two_slides, **RESUME)
    before = maps_of(output_dir)
    per_slide = len(encoder_forwards) // 2
    record = output_dir / "run" / "s1.json"
    if damage == "missing":
        record.unlink()
    else:
        record.write_text(record.read_text()[: len(record.read_text()) // 2])
    encoder_forwards.clear()

    soma.predict_slides(output_dir=output_dir, bundles=bundle, records=two_slides, **RESUME)

    assert len(encoder_forwards) == per_slide  # s1 again, not s2
    assert maps_of(output_dir) == before
    assert json.loads(record.read_text())["sample_id"] == "s1"


def test_a_relaunch_of_a_finished_plan_predicts_nothing(tmp_path, two_slides, encoder_forwards):
    bundle = save_bundle(tmp_path / "run" / "bundle")
    output_dir = tmp_path / "predictions"
    first = soma.predict_slides(output_dir=output_dir, bundles=bundle, records=two_slides, **RESUME)
    encoder_forwards.clear()

    replayed = soma.predict_slides(
        **soma.pipeline.slide_prediction.slide_prediction_kwargs(yaml.safe_load((output_dir / "plan.yaml").read_text()))
    )

    assert encoder_forwards == []
    assert replayed.equals(first)


def test_a_changed_overlap_is_refused_in_the_same_directory(tmp_path, two_slides):
    bundle = save_bundle(tmp_path / "run" / "bundle")
    output_dir = tmp_path / "predictions"
    soma.predict_slides(output_dir=output_dir, bundles=bundle, records=two_slides, **RESUME)
    plan = (output_dir / "plan.yaml").read_text()

    with pytest.raises(ValueError, match=r"differs from the current plan \(differing keys: \['extraction.tiling.overlap'\]\)"):
        soma.predict_slides(
            output_dir=output_dir,
            bundles=bundle,
            records=two_slides,
            extraction={"tiling": {"overlap": 0.25, **TIFFFILE}},
            batch_size=4,
        )
    assert (output_dir / "plan.yaml").read_text() == plan


@pytest.mark.parametrize("first", [False, True])
def test_a_changed_restrict_to_mask_is_refused_in_the_same_directory(tmp_path, two_slides, first):
    """The maps of either value stay in the directory as written; a relaunch with the other
    value would mix restricted and unrestricted maps."""
    bundle = save_bundle(tmp_path / "run" / "bundle")
    output_dir = tmp_path / "predictions"
    soma.predict_slides(output_dir=output_dir, bundles=bundle, records=two_slides, restrict_to_mask=first, **RESUME)
    plan = (output_dir / "plan.yaml").read_text()
    assert yaml.safe_load(plan)["predict"]["restrict_to_mask"] is first

    with pytest.raises(ValueError, match=r"differs from the current plan \(differing keys: \['predict.restrict_to_mask'\]\)"):
        soma.predict_slides(output_dir=output_dir, bundles=bundle, records=two_slides, restrict_to_mask=not first, **RESUME)
    assert (output_dir / "plan.yaml").read_text() == plan


def test_a_directory_predicted_with_another_reader_is_refused_with_how_to_keep_it(tmp_path, cohort):
    """A directory predicted when the slide reader did not come from the bundles has
    ``auto`` in its ``plan.yaml``. The same config now plans the bundle's reader, so the
    relaunch is refused, and the message says how to keep the directory's reader."""
    from soma.pipeline.slide_prediction import resolve_slide_prediction

    bundle = save_bundle(tmp_path / "run" / "bundle", **reader_recipe("tifffile"))
    output_dir = tmp_path / "predictions"
    before = resolve_slide_prediction(output_dir=output_dir, bundles=bundle, records=cohort, extraction=MASK_READER_ONLY)
    before["extraction"]["tiling"].update(backend="auto", requested_backend="auto")
    output_dir.mkdir()
    (output_dir / "plan.yaml").write_text(yaml.safe_dump(before, sort_keys=False))

    with pytest.raises(
        ValueError,
        match=r"read with 'auto'.*set extraction\.tiling\.backend: auto to keep it\. Use another predict\.output_dir",
    ):
        predict(tmp_path, cohort, bundle, extraction=MASK_READER_ONLY)


# --- memory --------------------------------------------------------------------------- #

#: The child process of the memory test: it predicts a small slide first (imports, the
#: bundle, the reader), resets the peak RSS, predicts the large slide and prints the RSS
#: before it and the peak during it.
MEMORY_CHILD = MEMORY_PRELUDE + r"""
tmp, stitch = Path(sys.argv[1]), sys.argv[2]
extraction = {"tiling": {"backend": "tifffile", "mask_backend": "tifffile", "overlap": 0.5}}
bundle = str(tmp / "run" / "bundle")
predict = dict(bundles=bundle, extraction=extraction, stitch=stitch)
soma.predict_slides(output_dir=tmp / "warm", records=str(tmp / "small.csv"), **predict)
reset_peak()
before = status("VmRSS")
soma.predict_slides(output_dir=tmp / "large", records=str(tmp / "large.csv"), batch_size=64, **predict)
print(json.dumps({"before": before, "peak": status("VmHWM")}))
"""


@pytest.mark.parametrize("stitch", ["pick_one", "average"])
def test_peak_memory_stays_far_below_the_raw_map(tmp_path, stitch):
    """A 32768 x 32768 px map (1 GiB as raw uint8, 12 GiB as float32 for the 3 classes)
    with three tissue patches in rows far apart, at overlap 0.5. The band holds a strip of
    rows (and, to average, the float sums of one tile's rows); hs2p's writer holds the
    reduced levels, about a third of the map. The peak RSS while predicting stays below
    half the raw map."""
    side = 32768
    write_large_slide(tmp_path / "large.tif", side)
    coarse = 32  # the mask is at 16 um/px
    tissue = np.zeros((side // coarse, side // coarse), dtype=np.uint8)
    for row, col in ((128, 128), (512, 512), (896, 768)):
        tissue[row : row + 4, col : col + 4] = 1
    write_mask(tmp_path / "large_mask.tif", tissue, spacing_um=SPACING_UM * coarse)
    write_slide(tmp_path / "small.tif", blocky(96, 96))
    write_mask(tmp_path / "small_mask.tif", np.ones((96, 96)))
    for name in ("large", "small"):
        write_records(
            tmp_path / f"{name}.csv",
            [{"sample_id": name, "image_path": str(tmp_path / f"{name}.tif"), "mask_path": str(tmp_path / f"{name}_mask.tif")}],
        )
    save_bundle(tmp_path / "run" / "bundle")

    rss = peak_rss_of(MEMORY_CHILD, str(tmp_path), stitch)

    (row,) = pd.read_parquet(tmp_path / "large" / "index.parquet").itertuples()
    assert (row.width, row.height, row.num_tiles) == (side, side, 3 * 8 * 8)
    raw = side * side
    assert rss["peak"] - rss["before"] < raw / 2


# --- devices -------------------------------------------------------------------------- #


def test_devices_split_the_slides_across_one_process_each(tmp_path):
    rows = []
    for k in range(3):
        slide = write_slide(tmp_path / f"s{k}.tif", blocky(96, 96, seed=k))
        mask = write_mask(tmp_path / f"s{k}_mask.tif", np.ones((96, 96)))
        rows.append({"sample_id": f"s{k}", "image_path": str(slide), "mask_path": str(mask)})
    records = write_records(tmp_path / "dataset.csv", rows)
    bundle = save_bundle(tmp_path / "run" / "bundle")
    in_turn = soma.predict_slides(output_dir=tmp_path / "in_turn", bundles=bundle, records=records, **RESUME)

    parallel = soma.predict_slides(
        output_dir=tmp_path / "parallel", bundles=bundle, records=records, devices=["cpu", "cpu"], **RESUME
    )

    assert parallel.equals(in_turn)
    assert maps_of(tmp_path / "parallel") == maps_of(tmp_path / "in_turn")
    logs = sorted(path.name for path in (tmp_path / "parallel" / "logs").iterdir())
    assert logs == ["worker_0.log", "worker_1.log"]


def test_a_bundle_changed_in_place_is_refused_in_the_same_directory(tmp_path, two_slides):
    """The plan names bundles by path; ``provenance.json`` records their digests, so a
    bundle retrained into the same path cannot add its maps to the old ones."""
    bundle = save_bundle(tmp_path / "run" / "bundle")
    output_dir = tmp_path / "predictions"
    soma.predict_slides(output_dir=output_dir, bundles=bundle, records=two_slides, **RESUME)
    shutil.rmtree(bundle)
    save_bundle(tmp_path / "run" / "bundle", seed=1)

    with pytest.raises(ValueError, match=r"bundle .*run/bundle changed since its maps were predicted into it"):
        soma.predict_slides(output_dir=output_dir, bundles=bundle, records=two_slides, **RESUME)


# --- the cohort ----------------------------------------------------------------------- #


def test_splits_select_one_subset_of_one_fold(tmp_path):
    """No target column is needed. With ``splits``, only ``subset`` of ``fold`` is
    predicted: here the test slide of fold 1."""
    rows = []
    for k in range(3):
        slide = write_slide(tmp_path / f"s{k}.tif", blocky(96, 96, seed=k))
        mask = write_mask(tmp_path / f"s{k}_mask.tif", np.ones((96, 96)))
        rows.append({"sample_id": f"s{k}", "image_path": str(slide), "mask_path": str(mask)})
    records = write_records(tmp_path / "dataset.csv", rows)
    splits = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2"] * 2,
            "split": ["test", "train", "tune", "train", "tune", "test"],
            "fold": [0, 0, 0, 1, 1, 1],
        }
    ).to_csv(splits, index=False)
    bundle = save_bundle(tmp_path / "run" / "bundle")

    index = predict(tmp_path, records, bundle, splits=splits, subset="test", fold=1)

    assert list(index.sample_id) == ["s2"]


# --- each bundle's own fold ------------------------------------------------------------- #

#: Two folds over four slides, each with its own test slides.
ROTATING = {
    "sample_id": ["s0", "s1", "s2", "s3"] * 2,
    "split": ["test", "test", "tune", "train", "tune", "train", "test", "test"],
    "fold": [0, 0, 0, 0, 1, 1, 1, 1],
}
OWN_FOLD = {"extraction": {"tiling": {"overlap": 0.0, **TIFFFILE}}, "batch_size": 4, "fold": "bundle"}


@pytest.fixture
def rotating(tmp_path):
    """Four slides with tissue everywhere, and two folds that test two slides each."""
    rows = []
    for k in range(4):
        slide = write_slide(tmp_path / f"s{k}.tif", blocky(96, 96, seed=k))
        mask = write_mask(tmp_path / f"s{k}_mask.tif", np.ones((96, 96)))
        rows.append({"sample_id": f"s{k}", "image_path": str(slide), "mask_path": str(mask)})
    records = write_records(tmp_path / "dataset.csv", rows)
    splits = tmp_path / "splits.csv"
    pd.DataFrame(ROTATING).to_csv(splits, index=False)
    return records, splits


def fold_bundles(tmp_path, records, splits, run="run", seed=0):
    """A run's two fold bundles, each recording its fold as ``soma run`` does."""
    return [
        save_bundle(tmp_path / run / f"fold_{k}" / "bundle", seed=seed + k, provenance=fold_provenance(records, splits, k))
        for k in (0, 1)
    ]


def files_of(output_dir) -> dict[str, str]:
    return {str(path.relative_to(output_dir)): sha256(path) for path in sorted(output_dir.glob("*/s*"))}


def test_each_bundle_predicts_its_own_fold_and_bundles_of_one_fold_share_its_forwards(
    tmp_path, rotating, encoder_forwards
):
    """``fold="bundle"``: each bundle predicts the test slides of the fold it records. Two
    runs over the same splits give two bundles of fold 0; both predict its slides behind
    one encoder forward per batch."""
    records, splits = rotating
    bundles = [*fold_bundles(tmp_path, records, splits, run="a"), fold_bundles(tmp_path, records, splits, run="b")[0]]
    encoder_forwards.clear()

    index = soma.predict_slides(output_dir=tmp_path / "predictions", bundles=bundles, records=records, splits=splits, **OWN_FOLD)

    assert list(zip(index.bundle, index.sample_id)) == [
        ("a-fold_0", "s0"), ("a-fold_0", "s1"), ("a-fold_1", "s2"), ("a-fold_1", "s3"), ("b-fold_0", "s0"), ("b-fold_0", "s1"),
    ]
    assert encoder_forwards == [4, 4, 1] * 4  # 9 tiles per slide, encoded once per slide
    provenance = json.loads((tmp_path / "predictions" / "provenance.json").read_text())
    assert [(entry["name"], entry["fold"]) for entry in provenance["bundles"]] == [("a-fold_0", 0), ("a-fold_1", 1), ("b-fold_0", 0)]
    assert [entry["fold_assignments"] for entry in provenance["bundles"]] == [
        json.loads((Path(bundle) / "provenance.json").read_text())["fold_assignments"] for bundle in bundles
    ]


def test_fold_bundle_averages_each_fold_s_slides_as_an_explicit_fold_call_does(tmp_path, rotating):
    """``fold="bundle"`` with ``stitch="average"`` and an overlap: each slide's maps are
    averaged over its own bundles only, so each fold's directory holds the files and index
    rows of a call with that bundle alone, an explicit fold and the same stitch."""
    records, splits = rotating
    bundles = fold_bundles(tmp_path, records, splits)
    averaged = {**OWN_FOLD, "extraction": HALF, "stitch": "average"}

    index = soma.predict_slides(output_dir=tmp_path / "own", bundles=bundles, records=records, splits=splits, **averaged)

    assert yaml.safe_load((tmp_path / "own" / "plan.yaml").read_text())["predict"]["stitch"] == "average"
    assert list(zip(index.bundle, index.sample_id)) == [("fold_0", "s0"), ("fold_0", "s1"), ("fold_1", "s2"), ("fold_1", "s3")]
    own = files_of(tmp_path / "own")
    explicit = []
    for k, bundle in enumerate(bundles):
        alone = tmp_path / f"fold_{k}"
        explicit.append(
            soma.predict_slides(output_dir=alone, bundles=bundle, records=records, splits=splits, **{**averaged, "fold": k})
        )
        assert {path: digest for path, digest in own.items() if path.startswith(f"fold_{k}/")} == files_of(alone)
    assert index.equals(pd.concat(explicit, ignore_index=True))


def test_fold_bundle_refuses_an_ensemble(tmp_path, rotating):
    records, splits = rotating
    bundles = fold_bundles(tmp_path, records, splits)

    with pytest.raises(ValueError, match="cohort.fold: bundle gives each bundle the slides of its own fold"):
        predict(tmp_path, records, bundles, splits=splits, fold="bundle", ensemble=True)


def test_fold_bundle_refuses_a_bundle_that_records_no_fold_assignments(tmp_path, rotating):
    """A bundle saved before soma recorded them, or by ``soma.save`` with no fold."""
    records, splits = rotating
    bundle = save_bundle(tmp_path / "run" / "bundle")

    with pytest.raises(ValueError, match=r"run/bundle records no fold_assignments.*each with an explicit cohort\.fold"):
        predict(tmp_path, records, bundle, splits=splits, fold="bundle")


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        ("reassigned", r"fold_0/bundle records fold 0 with fold_assignments \w{12}, but fold 0 of .*splits.csv has fold_assignments \w{12}"),
        ("fewer-folds", r"fold_1/bundle records fold 1 with fold_assignments \w{12}, but .*splits.csv has 1 fold\(s\)"),
    ],
)
def test_fold_bundle_refuses_a_bundle_trained_on_other_splits(tmp_path, rotating, edit, match):
    records, splits = rotating
    bundles = fold_bundles(tmp_path, records, splits)
    edited = pd.DataFrame(ROTATING)
    if edit == "reassigned":  # fold 0 tunes on s3 and trains on s2
        edited.loc[[2, 3], "split"] = ["train", "tune"]
    else:
        edited = edited[edited.fold == 0]
    edited.to_csv(splits, index=False)

    with pytest.raises(ValueError, match=match + ": these splits differ from the ones the bundle was trained on"):
        predict(tmp_path, records, bundles, splits=splits, fold="bundle")


def test_fold_bundle_refuses_a_subset_that_is_not_a_split_of_a_bundle_s_fold(tmp_path, rotating):
    records, splits = rotating
    bundles = fold_bundles(tmp_path, records, splits)

    with pytest.raises(
        ValueError, match=r"cohort.subset 'test_external' is not a split of fold 0 of bundle .*fold_0/bundle; its splits are \['test', 'train', 'tune'\]"
    ):
        predict(tmp_path, records, bundles, splits=splits, fold="bundle", subset="test_external")


@pytest.mark.parametrize("subset", ["train", Path("train")], ids=["str", "path"])
def test_fold_bundle_refuses_the_train_subset(tmp_path, rotating, subset):
    """Each bundle was trained on its fold's train split, so it held none of it out. The
    refusal reads the subset the plan stores, so a non-``str`` spelling of it is refused too."""
    records, splits = rotating
    bundles = fold_bundles(tmp_path, records, splits)

    with pytest.raises(
        ValueError, match=r"cohort.fold: bundle predicts only the slides each bundle held out.*explicit cohort\.fold"
    ):
        predict(tmp_path, records, bundles, splits=splits, fold="bundle", subset=subset)


def test_fold_bundle_refuses_a_bundle_whose_fold_has_no_slides_in_the_subset(tmp_path, rotating):
    """A fold with no tune rows still has a ``tune`` split, an empty one: its bundle would
    predict nothing and be missing from the output without a word."""
    records, splits = rotating
    edited = pd.DataFrame(ROTATING)
    edited.loc[4, "split"] = "train"  # fold 1 tunes on nothing
    edited.to_csv(splits, index=False)
    bundles = fold_bundles(tmp_path, records, splits)

    with pytest.raises(ValueError, match=r"cohort.subset 'tune' of fold 1 of bundle .*fold_1/bundle has no slides in it"):
        predict(tmp_path, records, bundles, splits=splits, fold="bundle", subset="tune")


def test_a_stopped_fold_bundle_prediction_resumes_to_the_bytes_of_an_uninterrupted_one(
    tmp_path, rotating, encoder_forwards, monkeypatch
):
    """Stopped in fold 1's first slide, then replayed from its ``plan.yaml``, which keeps
    ``cohort.fold: bundle``: only fold 1's slides are predicted again."""
    from soma.pipeline.slide_prediction import slide_prediction_kwargs

    records, splits = rotating
    bundles = fold_bundles(tmp_path, records, splits)
    whole = tmp_path / "whole"
    soma.predict_slides(output_dir=whole, bundles=bundles, records=records, splits=splits, **OWN_FOLD)
    per_slide = len(encoder_forwards) // 4
    encoder_forwards.clear()
    forward = soma.extract.FrozenEncoder.forward

    def stops_in_the_third_slide(self, pixels):
        if len(encoder_forwards) > 2 * per_slide:
            raise KeyboardInterrupt
        return forward(self, pixels)

    stopped = tmp_path / "stopped"
    with monkeypatch.context() as patched:
        patched.setattr(soma.extract.FrozenEncoder, "forward", stops_in_the_third_slide)
        with pytest.raises(KeyboardInterrupt):
            soma.predict_slides(output_dir=stopped, bundles=bundles, records=records, splits=splits, **OWN_FOLD)
    assert sorted(files_of(stopped)) == ["fold_0/s0.json", "fold_0/s0.tif", "fold_0/s1.json", "fold_0/s1.tif"]
    plan = yaml.safe_load((stopped / "plan.yaml").read_text())
    assert plan["cohort"]["fold"] == "bundle"
    encoder_forwards.clear()

    index = soma.predict_slides(**slide_prediction_kwargs(plan))

    assert len(encoder_forwards) == 2 * per_slide  # s2 and s3 only
    assert files_of(stopped) == files_of(whole)
    assert index.equals(pd.read_parquet(whole / "index.parquet"))


@pytest.mark.parametrize("change", ["fold", "fold_assignments"])
def test_a_relaunch_is_refused_after_a_bundle_s_fold_changed(tmp_path, rotating, change):
    """The plan keeps ``cohort.fold: bundle`` and the bundles' paths, and a bundle's digest
    covers its model, not its ``provenance.json``: the fold and ``fold_assignments`` that
    ``provenance.json`` records of each bundle guard the directory."""
    records, splits = rotating
    bundles = fold_bundles(tmp_path, records, splits)
    output_dir = tmp_path / "predictions"
    soma.predict_slides(output_dir=output_dir, bundles=bundles, records=records, splits=splits, **OWN_FOLD)
    recorded = [Path(bundle) / "provenance.json" for bundle in bundles]
    if change == "fold":  # the two bundles now claim each other's fold
        texts = [path.read_text() for path in recorded]
        for path, text in zip(recorded, reversed(texts)):
            path.write_text(text)
    else:  # new splits in the same file, and bundles that record them
        edited = pd.DataFrame(ROTATING)
        edited.loc[[2, 3], "split"] = ["train", "tune"]
        edited.to_csv(splits, index=False)
        for k, path in enumerate(recorded):
            path.write_text(json.dumps({**json.loads(path.read_text()), **fold_provenance(records, splits, k)}))

    with pytest.raises(ValueError, match=r"fold_0/bundle records another fold than when its predictions were written into it"):
        soma.predict_slides(output_dir=output_dir, bundles=bundles, records=records, splits=splits, **OWN_FOLD)
