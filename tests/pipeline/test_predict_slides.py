"""``soma.predict_slides``: one label TIFF per slide, streamed tile by tile.

The slides are synthetic TIFFs, the bundles carry the weight-free ``fake`` encoder's
recipe, and everything runs on the CPU. A map is read back through hs2p; its value 0 means
"not predicted" and class ``k`` is stored as ``k + 1``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import torch
import yaml

import soma
from soma.data.images import RegionImageSource
from soma.data.manifests import ImageEntry
from soma.learn import DenseLearner, FitResult
from soma.learn.grid import GridBatch
from soma.models import DenseModel
from soma.models.decoders.base import Decoder
from soma.pipeline.build import build_grid_model, bundle_encoder_stage

SPACING_UM = 0.5
TILE = 32
#: hs2p's readers for the slides and the masks: never cuCIM, which crashes on some
#: synthetic TIFFs.
TIFFFILE = {"backend": "tifffile", "mask_backend": "tifffile"}
#: The recipe of the encoder a bundle's grids came from: ``fake`` pools 8 px patches.
RECIPE = {"encoder": "fake", "target_size": TILE, "spacing_um": SPACING_UM, "precision": "fp32"}


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def _px_per_cm(spacing_um: float) -> float:
    return 1e4 / spacing_um


def write_slide(path, image: np.ndarray, *, spacing_um: float = SPACING_UM):
    """A single-level tiled RGB TIFF at ``spacing_um``."""
    tifffile.imwrite(
        path,
        image,
        photometric="rgb",
        tile=(16, 16),
        resolution=(_px_per_cm(spacing_um), _px_per_cm(spacing_um)),
        resolutionunit="CENTIMETER",
    )
    return path


def write_mask(path, mask: np.ndarray, *, spacing_um: float = SPACING_UM):
    """A single-level tiled tissue mask (1 = tissue) at ``spacing_um``."""
    tifffile.imwrite(
        path,
        mask.astype(np.uint8),
        tile=(16, 16),
        resolution=(_px_per_cm(spacing_um), _px_per_cm(spacing_um)),
        resolutionunit="CENTIMETER",
    )
    return path


def blocky(height: int, width: int, *, seed: int = 0, block: int = 8) -> np.ndarray:
    """An RGB image of random ``block`` px squares: every patch the encoder pools differs."""
    rng = np.random.default_rng(seed)
    cells = rng.integers(0, 256, size=(-(-height // block), -(-width // block), 3), dtype=np.uint8)
    return np.ascontiguousarray(cells.repeat(block, 0).repeat(block, 1)[:height, :width])


def write_records(path, rows: list[dict]) -> str:
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def save_bundle(path, *, seed: int = 0, num_classes: int = 3, decoder=None, **preprocessing):
    """A segmentation bundle over the ``fake`` encoder's grids, with random weights."""
    stage = soma.extract.live_encoder(**RECIPE)
    torch.manual_seed(seed)
    model = build_grid_model(
        feature_dim=stage.feature_dim,
        decoder=decoder or {"name": "linear", "params": {}},
        head={"name": "segmentation", "params": {"num_classes": num_classes, "geometry": stage.geometry}},
    )
    fit = FitResult(DenseLearner(model), pd.DataFrame(), 0, ("loss", 0.0), None, 0)
    recorded = {
        "shape": "grid",
        "task_family": "segmentation",
        "num_classes": num_classes,
        "class_names": [f"class_{k}" for k in range(num_classes)],
        "tile_size": [TILE, TILE],
        "spacing_um": SPACING_UM,
        "encoder": dict(RECIPE),
        **preprocessing,
    }
    return soma.save(fit, path, preprocessing=recorded)


def read_map(path) -> np.ndarray:
    """The base level of a written label TIFF, read through hs2p."""
    from hs2p.wsi.wsi import WSI

    wsi = WSI(path=path, backend="tifffile")
    width, height = wsi.level_dimensions[0]
    region = np.asarray(wsi.read_region((0, 0), 0, (width, height)))
    return region[..., 0] if region.ndim == 3 else region


def model_of(bundle_path) -> DenseModel:
    """The bundle's model behind its frozen encoder stage, as ``predict_slides`` runs it."""
    bundle = soma.load(bundle_path)
    stage = bundle_encoder_stage(bundle)
    model = bundle.model
    return DenseModel(model.decoder, model.head, encoder=stage, adaptor=model.adaptor).eval()


def tile_pixels(slide_path, regions: dict[str, tuple[int, int]]) -> RegionImageSource:
    entries = {tid: ImageEntry(sample_id=tid, image_path=slide_path) for tid in regions}
    return RegionImageSource(entries, regions=regions, size=TILE, spacing_um=SPACING_UM, backend="tifffile")


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


def write_pyramid(path, read_level: np.ndarray, *, downsample: int, read_spacing_um: float = SPACING_UM):
    """A two-level tiled RGB TIFF: ``read_level`` at ``read_spacing_um`` is level 1, and
    level 0 repeats each of its pixels ``downsample`` times on each axis."""
    level0 = read_level.repeat(downsample, 0).repeat(downsample, 1)
    base_spacing = read_spacing_um / downsample
    with tifffile.TiffWriter(path) as tiff:
        options = {"photometric": "rgb", "tile": (16, 16), "resolutionunit": "CENTIMETER"}
        resolution = _px_per_cm(base_spacing)
        tiff.write(level0, subifds=1, resolution=(resolution, resolution), **options)
        resolution = _px_per_cm(read_spacing_um)
        tiff.write(read_level, subfiletype=1, resolution=(resolution, resolution), **options)
    return path, level0.shape[:2], base_spacing


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


def test_a_plan_s_own_tile_size_and_spacing_are_accepted(tmp_path, cohort):
    """``plan.yaml`` spells out the bundle's tile size and spacing, so setting them to
    the bundle's own values is what a replay does, and is accepted."""
    bundle = save_bundle(tmp_path / "run" / "bundle")
    tiling = {**TIFFFILE, "requested_tile_size_px": TILE, "requested_spacing_um": SPACING_UM}

    index = predict(tmp_path, cohort, bundle, extraction={"tiling": tiling})

    assert list(index.sample_id) == ["s1"]


@pytest.mark.parametrize("given", [{"subset": "test"}, {"fold": 0}], ids=["subset", "fold"])
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


def test_an_ensemble_needs_one_class_scheme(tmp_path, cohort):
    bundles = [save_bundle(tmp_path / "a" / "bundle"), save_bundle(tmp_path / "b" / "bundle", num_classes=4)]

    with pytest.raises(ValueError, match="must share one class scheme"):
        predict(tmp_path, cohort, bundles, ensemble=True)


@pytest.mark.parametrize(
    ("recorded", "match"),
    [
        ({"encoder": {"composite": True, "encoders": [RECIPE, RECIPE]}}, "composite of several encoders"),
        ({"task_family": "detection"}, "is a detection bundle.*segmentation label maps"),
    ],
    ids=["composite", "detection"],
)
def test_a_composite_or_detection_bundle_is_refused(tmp_path, cohort, recorded, match):
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


# --- memory --------------------------------------------------------------------------- #

#: The child process of the memory test: it predicts a small slide first (imports, the
#: bundle, the reader), resets the peak RSS, predicts the large slide and prints the RSS
#: before it and the peak during it.
MEMORY_CHILD = r"""
import json, re, sys
from pathlib import Path

import soma

def status(key):
    text = Path("/proc/self/status").read_text()
    return int(re.search(rf"^{key}:\s+(\d+) kB", text, re.M).group(1)) * 1024

tmp = Path(sys.argv[1])
extraction = {"tiling": {"backend": "tifffile", "mask_backend": "tifffile"}}
bundle = str(tmp / "run" / "bundle")
soma.predict_slides(output_dir=tmp / "warm", bundles=bundle, records=str(tmp / "small.csv"), extraction=extraction)
try:
    Path("/proc/self/clear_refs").write_text("5")  # the peak restarts from here
except OSError:
    pass  # the peak then includes the warm-up's, which only makes the bound harder
before = status("VmRSS")
soma.predict_slides(output_dir=tmp / "large", bundles=bundle, records=str(tmp / "large.csv"), extraction=extraction, batch_size=64)
print(json.dumps({"before": before, "peak": status("VmHWM")}))
"""


def write_large_slide(path, side: int, *, tile: int = 512):
    """A ``side`` x ``side`` px uniform RGB pyramid at 0.5 um/px, written tile by tile,
    with levels down to below 512 px so hs2p reads the mask on a small grid."""
    downsamples = [4**k for k in range(8) if side // 4**k >= 512]
    block = np.full((tile, tile, 3), 200, dtype=np.uint8)
    with tifffile.TiffWriter(path, bigtiff=True) as tiff:
        for k, downsample in enumerate(downsamples):
            size = side // downsample
            resolution = _px_per_cm(SPACING_UM * downsample)
            tiles = (block for _ in range((-(-size // tile)) ** 2))
            options = {"subifds": len(downsamples) - 1} if k == 0 else {"subfiletype": 1}
            tiff.write(
                tiles,
                shape=(size, size, 3),
                dtype=np.uint8,
                tile=(tile, tile),
                photometric="rgb",
                compression="zlib",
                resolution=(resolution, resolution),
                resolutionunit="CENTIMETER",
                **options,
            )
    return path


def test_peak_memory_stays_far_below_the_raw_map(tmp_path):
    """A 32768 x 32768 px map (1 GiB as raw uint8) with three tissue patches in rows far
    apart. The band holds a strip of rows; hs2p's writer holds the reduced levels, about
    a third of the map. The peak RSS while predicting stays below half the raw map."""
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
    checkout = str(Path(soma.__file__).resolve().parents[1])
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [checkout, os.environ.get("PYTHONPATH")])),
        "CUDA_VISIBLE_DEVICES": "",
    }

    done = subprocess.run(
        [sys.executable, "-c", MEMORY_CHILD, str(tmp_path)], capture_output=True, text=True, env=env, timeout=300
    )

    assert done.returncode == 0, done.stderr[-4000:]
    rss = json.loads(done.stdout.strip().splitlines()[-1])
    (row,) = pd.read_parquet(tmp_path / "large" / "index.parquet").itertuples()
    assert (row.width, row.height, row.num_tiles) == (side, side, 3 * 16)
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
