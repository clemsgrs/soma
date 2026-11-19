"""``soma.predict_slides`` over detection bundles: one points parquet per slide.

The slides are synthetic TIFFs, the bundles carry the weight-free ``fake`` encoder's
recipe and a weight-free decoder, and everything runs on the CPU. Each parquet holds the
slide's peaks in level-0 pixels: ``x``, ``y``, ``class`` and ``score``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch

import soma
from soma.learn.grid import GridBatch
from soma.models.decoders.base import Decoder
from tests.whole_slides import (
    SPACING_UM,
    TIFFFILE,
    TILE,
    model_of,
    tile_pixels,
    save_detection_bundle,
    write_mask,
    write_pyramid,
    write_records,
    write_slide,
)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


# --- weight-free decoders --------------------------------------------------------------- #

#: The logit of a pixel without a peak: its score is far below any ``min_score``.
BACKGROUND_LOGIT = -20.0
#: The colour of a slide block that holds no marker (red is never 255).
BACKGROUND_RGB = (40, 40, 40)


@dataclass
class MarkerParams:
    """``gain`` scales the logit of every peak the marker decoder puts."""

    gain: float = 1.0


class MarkerDecoder(Decoder):
    """Puts one peak where the slide says, read from the pixels.

    A slide block of 8 px is a marker when its red is 255 and its green below 64 (the white
    a read past the slide's edge returns is not): its green is ``8 * dy + dx``, the peak's
    pixel in the block, and its blue the peak's logit, ``gain * blue / 20``. The ``fake``
    encoder's tokens are those block colours exactly, so the peaks are known level-0
    positions. Every other pixel has the background logit.
    """

    def __init__(self, *, input_dim: int, num_classes: int, gain: float = 1.0) -> None:
        super().__init__()
        self._num_classes = int(num_classes)
        self._gain = float(gain)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def forward(self, X):
        colours = torch.round((X * 0.25 + 0.5) * 255).long()  # the fake encoder's normalisation
        batch, _, rows, cols = X.shape
        logits = torch.full((batch, self._num_classes, rows * 8, cols * 8), BACKGROUND_LOGIT)
        b, r, c = torch.nonzero((colours[:, 0] == 255) & (colours[:, 1] < 64), as_tuple=True)
        green, blue = colours[b, 1, r, c], colours[b, 2, r, c]
        logits[b, 0, r * 8 + green // 8, c * 8 + green % 8] = self._gain * blue.float() / 20
        return logits


@dataclass
class FixedPeaksParams:
    """The fixed-peaks decoder has no params."""


#: The tile-frame pixels ``(x, y)`` where the fixed-peaks decoder puts a peak in every tile.
FIXED_PEAKS = {
    "middle": (16, 16),
    "near-left": (3, 16),
    "near-top": (16, 3),
    "near-corner": (3, 3),
    "far-right": (29, 16),
    "far-bottom": (16, 29),
}


class FixedPeaksDecoder(Decoder):
    """The same peaks in every tile, whatever its pixels: :data:`FIXED_PEAKS`."""

    def __init__(self, *, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self._num_classes = int(num_classes)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def forward(self, X):
        batch, _, rows, cols = X.shape
        logits = torch.full((batch, self._num_classes, rows * 8, cols * 8), BACKGROUND_LOGIT)
        for x, y in FIXED_PEAKS.values():
            logits[:, 0, y, x] = 4.0
        return logits


@pytest.fixture
def decoders(monkeypatch):
    """Register the two decoders for one test, then restore the registry."""
    from soma.models import decoder_registry

    monkeypatch.setattr(decoder_registry, "_entries", dict(decoder_registry._entries))
    soma.register("decoders", "marker", MarkerDecoder, params=MarkerParams)
    soma.register("decoders", "fixed-peaks", FixedPeaksDecoder, params=FixedPeaksParams)
    return {"marker": {"name": "marker", "params": {}}, "fixed-peaks": {"name": "fixed-peaks", "params": {}}}


def marker_slide(height: int, width: int, markers) -> np.ndarray:
    """An RGB image of 8 px blocks: one marker block per ``(x, y, blue)``, its peak at
    pixel ``(x, y)`` with logit ``blue / 20``; every other block is the background."""
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[...] = BACKGROUND_RGB
    for x, y, blue in markers:
        block_x, block_y = x // 8 * 8, y // 8 * 8
        image[block_y : block_y + 8, block_x : block_x + 8] = (255, 8 * (y % 8) + x % 8, blue)
    return image


def score(blue: float, gain: float = 1.0) -> float:
    return float(torch.sigmoid(torch.tensor(gain * blue / 20, dtype=torch.float32)))


def one_slide(tmp_path, image: np.ndarray, *, spacing_um: float = SPACING_UM, mask=None) -> str:
    """The records of one slide ``s1``, with a mask of tissue everywhere unless given."""
    slide = write_slide(tmp_path / "s1.tif", image, spacing_um=spacing_um)
    tissue = np.ones(image.shape[:2]) if mask is None else mask
    mask_path = write_mask(tmp_path / "s1_mask.tif", tissue, spacing_um=spacing_um)
    return write_records(
        tmp_path / "dataset.csv", [{"sample_id": "s1", "image_path": str(slide), "mask_path": str(mask_path)}]
    )


def points_of(path) -> list[tuple[float, float, int, float]]:
    """The rows of a points parquet, sorted by position."""
    frame = pd.read_parquet(path)
    assert list(frame.columns) == ["x", "y", "class", "score"]
    return sorted((float(r.x), float(r.y), int(r[2]), float(r.score)) for r in frame.itertuples(index=False))


def tile_points(bundle, slide, regions: dict[str, tuple[int, int]]) -> list[tuple[float, float, int, float]]:
    """``soma.predict`` over the tiles at ``regions``, each tile's points moved by its
    level-0 origin (the bundle's head is bound at origin 0, one level-0 pixel per pixel)."""
    pixels = tile_pixels(slide, regions)
    ids = list(regions)
    batch = GridBatch(
        features=None, pixels=torch.stack([pixels.load(tid) for tid in ids]), targets={}, sample_ids=tuple(ids)
    )
    frame = soma.predict(model_of(bundle), [batch], device="cpu").frame
    return sorted(
        (float(r.x) + regions[r.sample_id][0], float(r.y) + regions[r.sample_id][1], int(r[3]), float(r.score))
        for r in frame[["sample_id", "x", "y", "class", "score"]].itertuples(index=False)
    )


OVERLAP_ZERO = {"tiling": {"overlap": 0.0, **TIFFFILE}}


# --- overlap 0 ---------------------------------------------------------------------------- #


def test_overlap_zero_points_are_soma_predict_s_points_moved_to_level_0(tmp_path, decoders):
    """With no overlap each tile keeps all its peaks, and with every object away from a
    seam no two are merged: the slide's points are ``soma.predict``'s points over the
    same tiles, each moved by its tile's level-0 origin."""
    markers = [(10, 12, 120), (45, 20, 150), (70, 50, 90), (100, 75, 180), (20, 80, 60), (60, 85, 200)]
    records = one_slide(tmp_path, marker_slide(96, 128, markers))
    bundle = save_detection_bundle(tmp_path / "run" / "bundle", decoder=decoders["marker"])

    index = soma.predict_slides(
        output_dir=tmp_path / "predictions", bundles=bundle, records=records, extraction=OVERLAP_ZERO, batch_size=5
    )

    (row,) = index.itertuples()
    assert row.path == "run/s1.parquet"
    assert row.num_tiles == 12
    points = points_of(tmp_path / "predictions" / row.path)
    regions = {f"{x}_{y}": (x, y) for x in (0, 32, 64, 96) for y in (0, 32, 64)}
    assert points == pytest.approx(tile_points(bundle, tmp_path / "s1.tif", regions))
    assert points == pytest.approx(sorted((x, y, 0, score(blue)) for x, y, blue in markers))
    assert row.num_points == len(markers)


def test_an_object_on_a_seam_found_by_both_tiles_appears_once(tmp_path, decoders):
    """One object across the seam at x = 32: the left tile finds it at its last column and
    the right tile at its first, 1.4 px (0.7 um) apart, closer than the NMS distance of
    3 um. The slide keeps the higher-scoring one; an object away from the seam stays."""
    markers = [(31, 10, 200), (32, 11, 100), (50, 25, 150)]
    records = one_slide(tmp_path, marker_slide(96, 96, markers))
    bundle = save_detection_bundle(tmp_path / "run" / "bundle", decoder=decoders["marker"], match_distance=3.0)
    by_tile = tile_points(bundle, tmp_path / "s1.tif", {"left": (0, 0), "right": (32, 0)})
    assert [(x, y) for x, y, _, _ in by_tile] == [(31, 10), (32, 11), (50, 25)]  # both tiles found it

    index = soma.predict_slides(
        output_dir=tmp_path / "predictions", bundles=bundle, records=records, extraction=OVERLAP_ZERO
    )

    (row,) = index.itertuples()
    assert points_of(tmp_path / "predictions" / row.path) == pytest.approx(
        [(31, 10, 0, score(200)), (50, 25, 0, score(150))]
    )


# --- overlap above 0: pick one by location ----------------------------------------------- #


def test_a_tile_keeps_the_peaks_in_the_region_it_contributes(tmp_path, decoders):
    """Overlap 0.5 on 32 px tiles: a 16 px stride and an 8 px margin, so a tile
    contributes [8, 24) on each axis, and from 0 on an axis where it is first. Every tile
    puts the same six peaks. The middle one is kept everywhere; the ones 3 px from the
    left or top are kept only by the first column or row; the ones in the far margin
    (29 px) never reach the output."""
    records = one_slide(tmp_path, np.full((96, 96, 3), 120, dtype=np.uint8))
    bundle = save_detection_bundle(tmp_path / "run" / "bundle", decoder=decoders["fixed-peaks"], match_distance=3.0)

    index = soma.predict_slides(
        output_dir=tmp_path / "predictions",
        bundles=bundle,
        records=records,
        extraction={"tiling": {"overlap": 0.5, **TIFFFILE}, "masks": {"min_coverage": {"tissue": 0.9}}},
    )

    (row,) = index.itertuples()
    origins = (0, 16, 32, 48, 64)  # the tiles the mask covers whole
    assert row.num_tiles == len(origins) ** 2
    middle = [(16 + x, 16 + y) for x in origins for y in origins]
    first_column = [(3, 16 + y) for y in origins]
    first_row = [(16 + x, 3) for x in origins]
    corner = [(3, 3)]
    expected = sorted((float(x), float(y)) for x, y in middle + first_column + first_row + corner)
    points = points_of(tmp_path / "predictions" / row.path)
    assert [(x, y) for x, y, _, _ in points] == expected


# --- level 0 ------------------------------------------------------------------------------- #


def test_points_land_on_level_0_for_a_slide_read_at_a_coarser_level(tmp_path, decoders):
    """The slide's level 0 is at 0.5 / 3 um/px and the tiles are read from its level 1, at
    0.5 um/px: a peak at read-level pixel p lies at level-0 pixel 3 p, in a tile whose
    level-0 origin is a multiple of 96."""
    markers = [(5, 9, 160), (40, 12, 140), (20, 50, 120), (85, 70, 100)]
    slide, (height, width), base_spacing = write_pyramid(
        tmp_path / "s1.tif", marker_slide(96, 96, markers), downsample=3
    )
    mask = write_mask(tmp_path / "s1_mask.tif", np.ones((height, width)), spacing_um=base_spacing)
    records = write_records(
        tmp_path / "dataset.csv", [{"sample_id": "s1", "image_path": str(slide), "mask_path": str(mask)}]
    )
    bundle = save_detection_bundle(tmp_path / "run" / "bundle", decoder=decoders["marker"])

    index = soma.predict_slides(
        output_dir=tmp_path / "predictions", bundles=bundle, records=records, extraction=OVERLAP_ZERO
    )

    (row,) = index.itertuples()
    assert (row.width, row.height) == (288, 288)
    assert row.spacing_um == pytest.approx(0.5 / 3)
    assert row.read_spacing_um == pytest.approx(SPACING_UM)
    points = points_of(tmp_path / "predictions" / row.path)
    assert [(x, y) for x, y, _, _ in points] == [(15.0, 27.0), (60.0, 150.0), (120.0, 36.0), (255.0, 210.0)]


# --- refusals ------------------------------------------------------------------------------ #


@pytest.fixture
def cohort(tmp_path):
    """One small slide with tissue everywhere, and its records."""
    return one_slide(tmp_path, np.full((96, 96, 3), 120, dtype=np.uint8))


def test_a_margin_below_the_nms_distance_is_refused_before_anything_is_written(tmp_path, cohort, decoders):
    """Overlap 0.5 on 32 px tiles leaves an 8 px (4 um) margin, less than the 5 um NMS
    distance. At overlap 0 no tile drops a margin, so the same bundle predicts."""
    bundle = save_detection_bundle(tmp_path / "run" / "bundle", decoder=decoders["marker"], nms_distance=5.0)
    output_dir = tmp_path / "predictions"

    with pytest.raises(ValueError, match=r"margin of 8 px \(4 um\).*less than the NMS distance .*\(5 um\).*at least 0\.625"):
        soma.predict_slides(
            output_dir=output_dir, bundles=bundle, records=cohort, extraction={"tiling": {"overlap": 0.5, **TIFFFILE}}
        )
    assert not output_dir.exists()

    index = soma.predict_slides(output_dir=output_dir, bundles=bundle, records=cohort, extraction=OVERLAP_ZERO)
    assert list(index.sample_id) == ["s1"]


def test_a_margin_as_wide_as_the_nms_distance_is_accepted(tmp_path, cohort, decoders):
    bundle = save_detection_bundle(tmp_path / "run" / "bundle", decoder=decoders["marker"], nms_distance=4.0)

    index = soma.predict_slides(
        output_dir=tmp_path / "predictions",
        bundles=bundle,
        records=cohort,
        extraction={"tiling": {"overlap": 0.5, **TIFFFILE}},
    )

    assert list(index.sample_id) == ["s1"]


def test_segmentation_and_detection_bundles_are_not_mixed(tmp_path, cohort, decoders):
    from tests.whole_slides import save_bundle

    bundles = [
        save_bundle(tmp_path / "a" / "bundle"),
        save_detection_bundle(tmp_path / "b" / "bundle", decoder=decoders["marker"]),
    ]

    with pytest.raises(ValueError, match="must be of one task family"):
        soma.predict_slides(output_dir=tmp_path / "predictions", bundles=bundles, records=cohort, extraction=OVERLAP_ZERO)


@pytest.mark.parametrize(
    "other",
    [{"num_classes": 2}, {"match_distance": 4.0}, {"nms_distance": 2.0}, {"min_score": 0.05}],
    ids=["classes", "match-distance", "nms-distance", "min-score"],
)
def test_a_detection_ensemble_needs_one_set_of_peak_settings(tmp_path, cohort, decoders, other):
    """An ensemble takes one set of peaks from the mean heatmap and records one set of
    distances: the bundles must agree on the classes, the distances and ``min_score``."""
    bundles = [
        save_detection_bundle(tmp_path / "a" / "bundle", decoder=decoders["marker"]),
        save_detection_bundle(tmp_path / "b" / "bundle", decoder=decoders["marker"], **other),
    ]

    with pytest.raises(ValueError, match="must share the classes, the matching and NMS distances and min_score"):
        soma.predict_slides(
            output_dir=tmp_path / "predictions", bundles=bundles, records=cohort, extraction=OVERLAP_ZERO, ensemble=True
        )


# --- the artefact -------------------------------------------------------------------------- #


def metadata_of(path) -> dict:
    return json.loads(pq.read_schema(path).metadata[b"soma"])


def test_the_parquet_records_what_scoring_needs_and_applies_no_threshold(tmp_path, cohort, decoders):
    """Every peak above ``min_score`` is a row, the one below the bundle's tune threshold
    too; the threshold, the classes, the distances in um, the spacings and the bundle's
    identity are in the schema metadata."""
    records = one_slide(tmp_path, marker_slide(96, 96, [(20, 20, 10), (60, 60, 100)]))
    bundle = save_detection_bundle(
        tmp_path / "run" / "bundle", decoder=decoders["marker"], match_distance=3.0, score_threshold={"class_0": 0.7}
    )

    index = soma.predict_slides(
        output_dir=tmp_path / "predictions", bundles=bundle, records=records, extraction=OVERLAP_ZERO
    )

    path = tmp_path / "predictions" / "run" / "s1.parquet"
    assert points_of(path) == pytest.approx([(20, 20, 0, score(10)), (60, 60, 0, score(100))])
    assert score(10) < 0.7
    schema = pq.read_schema(path)
    assert [str(schema.field(name).type) for name in ("x", "y", "class", "score")] == ["double", "double", "int64", "double"]
    (identity,) = json.loads((tmp_path / "predictions" / "provenance.json").read_text())["bundles"]
    assert identity["path"] == str(bundle)
    meta = metadata_of(path)
    assert meta.pop("soma_version") == soma.__version__
    assert meta == {
        "kind": "slide_points",
        "task_family": "detection",
        "classes": ["class_0"],
        "match_distance_um": 3.0,
        "nms_distance_um": 3.0,
        "min_score": 0.01,
        "level0_spacing_um": SPACING_UM,
        "sample_id": "s1",
        "ensemble": False,
        "read_spacing_um": SPACING_UM,
        "tile_size_px": TILE,
        "overlap": 0.0,
        "bundles": [identity],
        "threshold": {"class_0": 0.7},
    }
    (row,) = index.itertuples()
    assert (row.width, row.height, row.spacing_um, row.num_tiles, row.num_points) == (96, 96, SPACING_UM, 9, 2)


def test_a_slide_without_a_kept_tile_or_a_peak_gets_a_parquet_without_rows(tmp_path, decoders):
    """``s1``'s mask is empty: a warning, no tiles, no rows. ``s2`` has tissue but no
    object: its tiles are predicted and give no peak, so no rows and no warning."""
    s1 = write_slide(tmp_path / "s1.tif", marker_slide(96, 96, [(20, 20, 100)]))
    s1_mask = write_mask(tmp_path / "s1_mask.tif", np.zeros((96, 96)))
    s2 = write_slide(tmp_path / "s2.tif", marker_slide(96, 96, []))
    s2_mask = write_mask(tmp_path / "s2_mask.tif", np.ones((96, 96)))
    records = write_records(
        tmp_path / "dataset.csv",
        [
            {"sample_id": "s1", "image_path": str(s1), "mask_path": str(s1_mask)},
            {"sample_id": "s2", "image_path": str(s2), "mask_path": str(s2_mask)},
        ],
    )
    bundle = save_detection_bundle(tmp_path / "run" / "bundle", decoder=decoders["marker"])

    with pytest.warns(UserWarning, match="no kept tile") as caught:
        index = soma.predict_slides(
            output_dir=tmp_path / "predictions", bundles=bundle, records=records, extraction=OVERLAP_ZERO
        )

    (warned,) = [str(w.message) for w in caught if "no kept tile" in str(w.message)]
    assert warned.startswith("slide 's1' has no kept tile") and warned.endswith("its points file has no rows")
    assert list(index.num_tiles) == [0, 9]
    assert list(index.num_points) == [0, 0]
    for row in index.itertuples():
        path = tmp_path / "predictions" / row.path
        assert points_of(path) == []
        assert metadata_of(path)["sample_id"] == row.sample_id


# --- several bundles ----------------------------------------------------------------------- #


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


def two_gains(tmp_path, decoders):
    """Two bundles whose peaks differ only in their logit's gain: 1 and 0.2."""
    marker = decoders["marker"]["name"]
    return [
        save_detection_bundle(tmp_path / name / "bundle", decoder={"name": marker, "params": {"gain": gain}})
        for name, gain in (("a", 1.0), ("b", 0.2))
    ]


def test_two_bundles_write_two_parquets_behind_one_encoder_forward_per_batch(tmp_path, decoders, encoder_forwards):
    records = one_slide(tmp_path, marker_slide(96, 96, [(40, 50, 200)]))
    bundles = two_gains(tmp_path, decoders)
    encoder_forwards.clear()

    index = soma.predict_slides(
        output_dir=tmp_path / "predictions", bundles=bundles, records=records, extraction=OVERLAP_ZERO, batch_size=4
    )

    assert encoder_forwards == [4, 4, 1]  # 9 tiles in batches of 4, encoded once for both bundles
    assert list(index.path) == ["a/s1.parquet", "b/s1.parquet"]
    assert points_of(tmp_path / "predictions" / "a" / "s1.parquet") == pytest.approx([(40, 50, 0, score(200))])
    assert points_of(tmp_path / "predictions" / "b" / "s1.parquet") == pytest.approx([(40, 50, 0, score(200, 0.2))])


def test_an_ensemble_takes_the_peaks_of_the_mean_heatmap_and_records_no_threshold(tmp_path, decoders):
    """The peak's score is the mean of the two heatmaps there, not the higher of the two
    bundles' peaks; the ensemble records both bundles and no threshold."""
    records = one_slide(tmp_path, marker_slide(96, 96, [(40, 50, 200)]))
    bundles = two_gains(tmp_path, decoders)

    index = soma.predict_slides(
        output_dir=tmp_path / "predictions", bundles=bundles, records=records, extraction=OVERLAP_ZERO, ensemble=True
    )

    (row,) = index.itertuples()
    assert row.path == "ensemble/s1.parquet"
    path = tmp_path / "predictions" / row.path
    assert points_of(path) == pytest.approx([(40, 50, 0, (score(200) + score(200, 0.2)) / 2)])
    meta = metadata_of(path)
    assert meta["ensemble"] is True
    assert meta["threshold"] is None
    assert [identity["path"] for identity in meta["bundles"]] == [str(bundle) for bundle in bundles]


# --- resume -------------------------------------------------------------------------------- #


def test_a_parquet_without_its_record_is_predicted_again_to_the_same_bytes(tmp_path, decoders, encoder_forwards):
    rows = []
    for k, sample_id in enumerate(("s1", "s2")):
        slide = write_slide(tmp_path / f"{sample_id}.tif", marker_slide(96, 96, [(20 + 30 * k, 40, 120)]))
        mask = write_mask(tmp_path / f"{sample_id}_mask.tif", np.ones((96, 96)))
        rows.append({"sample_id": sample_id, "image_path": str(slide), "mask_path": str(mask)})
    records = write_records(tmp_path / "dataset.csv", rows)
    bundle = save_detection_bundle(tmp_path / "run" / "bundle", decoder=decoders["marker"])
    output_dir = tmp_path / "predictions"
    run = {"output_dir": output_dir, "bundles": bundle, "records": records, "extraction": OVERLAP_ZERO, "batch_size": 4}
    first = soma.predict_slides(**run)
    per_slide = len(encoder_forwards) // 2
    before = {name: (output_dir / "run" / name).read_bytes() for name in ("s1.parquet", "s2.parquet")}
    (output_dir / "run" / "s1.json").unlink()
    encoder_forwards.clear()

    again = soma.predict_slides(**run)

    assert len(encoder_forwards) == per_slide  # s1 again, not s2
    assert {name: (output_dir / "run" / name).read_bytes() for name in before} == before
    assert again.equals(first)
    assert pd.read_parquet(output_dir / "index.parquet").equals(first)
