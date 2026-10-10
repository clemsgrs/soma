"""The cache-backed stores conform to the source protocols in place (design §4.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from soma.data import (
    AnnotationManifest,
    CachedLabelMapSource,
    CachedSetSource,
    GridSource,
    ImageManifest,
    LabelMapSource,
    PointSource,
    Points,
    SetSource,
    TargetSource,
    build_label_remap,
    covers,
    require_coverage,
)
from soma.dense import CachedGridSource, DenseSourceProvenance, dense_grid_metadata, write_dense_grid
from soma.dense.geometry import GridGeometry, compute_dense_geometry
from soma.dense.reader import write_mask_crop
from soma.testing import check_grid_source, check_set_source, check_target_source


def test_cached_set_source_conforms_for_bags(tmp_path: Path) -> None:
    for i in range(3):
        torch.save(torch.randn(4 + i, 16), tmp_path / f"s{i}.pt")
    source = CachedSetSource(tmp_path)
    assert isinstance(source, SetSource)
    assert source.rank == 2 and source.feature_dim == 16
    assert source.sample_ids == ["s0", "s1", "s2"]
    assert source.coords("s0") is None
    assert not hasattr(source, "is_slide_level")
    check_set_source(source)


def test_cached_set_source_conforms_for_vectors_and_hierarchies(tmp_path: Path) -> None:
    vectors = tmp_path / "vectors"
    vectors.mkdir()
    for i in range(3):
        torch.save(torch.randn(16).half(), vectors / f"s{i}.pt")
    source = CachedSetSource(vectors)
    assert source.rank == 1 and source.load("s1").dtype == torch.float32
    check_set_source(source)

    hipt = tmp_path / "hipt"
    hipt.mkdir()
    torch.save(torch.randn(2, 5, 16), hipt / "a.pt")
    assert CachedSetSource(hipt).rank == 3
    check_set_source(CachedSetSource(hipt))


def test_cached_set_source_disowns_payloads_the_manifest_marks_unsuccessful(
    tmp_path: Path,
) -> None:
    # A stale payload beside an ``empty`` / ``error`` manifest row (a re-extraction in
    # the same dir) must not pass coverage or load; a payload the manifest does not
    # list (an incremental cache) is still served.
    for sample_id in ("s0", "s1", "s2", "s3"):
        torch.save(torch.randn(4, 16), tmp_path / f"{sample_id}.pt")
    pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2"],
            "feature_status": ["success", "empty", "error"],
            "feature_path": ["", "", ""],
        }
    ).to_csv(tmp_path / "process_list.csv", index=False)

    source = CachedSetSource(tmp_path)
    assert source.sample_ids == ["s0", "s3"] and len(source) == 2
    assert source.empty_feature_samples == ["s1"]
    assert covers(source, ["s0", "s3"]) and not covers(source, ["s1"])
    with pytest.raises(ValueError, match=r"\['s1', 's2'\]"):
        require_coverage(source, ["s0", "s1", "s2"])
    for disowned in ("s1", "s2"):
        with pytest.raises(KeyError):
            source.load(disowned)
    check_set_source(source)


def test_cached_set_source_serves_sibling_coords(tmp_path: Path) -> None:
    torch.save(torch.randn(3, 8), tmp_path / "s0.pt")
    np.save(tmp_path / "s0.coords.npy", np.arange(6).reshape(3, 2))
    source = CachedSetSource(tmp_path)
    assert torch.equal(source.coords("s0"), torch.arange(6).reshape(3, 2))
    check_set_source(source)


def _write_grid(root: Path, sample_id: str, *, feature_dim: int = 6) -> None:
    geometry = compute_dense_geometry(target_size=(30, 20), patch_size=14)
    metadata = dense_grid_metadata(
        geometry, feature_dim=feature_dim, pad_mode="constant", spacing_um=0.5
    )
    metadata.update({"source_spacing_um": 0.25, "effective_spacing_um": 0.5})
    grid = torch.randn(feature_dim, *geometry.grid_shape)
    write_dense_grid(root, sample_id, grid, metadata)


def test_cached_grid_source_conforms_and_carries_provenance(tmp_path: Path) -> None:
    for i in range(2):
        _write_grid(tmp_path, f"g{i}")
    source = CachedGridSource(
        tmp_path, provenance=DenseSourceProvenance(kind="dense_cache", feature_dir=tmp_path)
    )
    assert isinstance(source, GridSource)
    assert source.sample_ids == ["g0", "g1"]
    assert source.feature_dim == 6
    assert source.provenance.to_dict()["kind"] == "dense_cache"
    geometry = source.geometry("g0")
    assert isinstance(geometry, GridGeometry)
    assert geometry.grid_shape == (3, 2)
    assert geometry.level0_px_per_token_px == 2.0
    assert source.spacing("g0") == 0.5
    assert not hasattr(source, "validate_coverage")
    check_grid_source(source)


def test_cached_grid_source_with_payload_stems_conforms(tmp_path: Path) -> None:
    _write_grid(tmp_path / "slide", "0_0")
    source = CachedGridSource(tmp_path, payload_stems={"slide__x0_y0": "slide/0_0"})
    assert source.sample_ids == ["slide__x0_y0"]
    check_grid_source(source)


def test_empty_grid_source_reports_zero_feature_dim(tmp_path: Path) -> None:
    assert CachedGridSource(tmp_path).feature_dim == 0


def test_cached_label_map_source_remaps_and_fills_outside_with_ignore(tmp_path: Path) -> None:
    remap = build_label_remap({"tumor": [2], "stroma": [3]}, ignore=[0], ignore_index=255)
    full = np.array([[2, 3], [0, 2]], dtype=np.uint8)
    write_mask_crop(tmp_path / "full.png", full)
    write_mask_crop(tmp_path / "edge.png", full[:1])  # bottom row overhangs the slide
    source = CachedLabelMapSource(
        {"full": tmp_path / "full.png", "edge": tmp_path / "edge.png"},
        size=(2, 2),
        label_remap=remap,
        ignore_index=255,
    )
    assert isinstance(source, TargetSource)
    assert source.load("full").tolist() == [[0, 1], [255, 0]]
    assert source.load("edge").tolist() == [[0, 1], [255, 255]]
    assert source.load("full").dtype == torch.long
    check_target_source(source)


def test_label_map_source_reads_flat_masks_from_manifests(tmp_path: Path) -> None:
    mask = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.uint8)
    Image.fromarray(mask).save(tmp_path / "a.png")
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(tmp_path / "a_img.png")
    frame = pd.DataFrame(
        {"sample_id": ["a"], "image_path": [tmp_path / "a_img.png"], "label_mask_path": [tmp_path / "a.png"]}
    )
    source = LabelMapSource.from_manifests(
        AnnotationManifest.from_frame(frame),
        ImageManifest.from_frame(frame),
        size=(2, 3),
        mask_vocabulary={"value_0": 0, "value_1": 1, "value_2": 2},
    )
    assert source.load("a").tolist() == mask.tolist()
    check_target_source(source)


def test_point_source_reads_points_and_ignore_masks(tmp_path: Path) -> None:
    pd.DataFrame({"x": [10, 20], "y": [5, 6], "class": [0, 1]}).to_csv(
        tmp_path / "a.csv", index=False
    )
    ignore = np.zeros((8, 32), dtype=np.uint8)
    ignore[:, 16:] = 255
    Image.fromarray(ignore).save(tmp_path / "a_ignore.png")
    frame = pd.DataFrame(
        {
            "sample_id": ["a"],
            "points_path": [tmp_path / "a.csv"],
            "ignore_mask_path": [tmp_path / "a_ignore.png"],
        }
    )
    source = PointSource.from_manifest(AnnotationManifest.from_frame(frame))
    points = source.load("a")
    assert isinstance(points, Points) and len(points) == 2
    assert points.xy.tolist() == [[10.0, 5.0], [20.0, 6.0]]
    assert points.classes.tolist() == [0, 1]
    assert points.ignore_mask.shape == (8, 32)
    check_target_source(source)
    with pytest.raises(KeyError):
        source.load("ghost")


def test_cached_set_source_reads_hs2p_coordinate_archives(tmp_path: Path) -> None:
    torch.save(torch.randn(2, 8), tmp_path / "s0.pt")
    np.savez_compressed(
        tmp_path / "s0.coordinates.npz",
        tile_index=np.array([0, 1], dtype=np.int32),
        x=np.array([32, 64], dtype=np.int64),
        y=np.array([0, 96], dtype=np.int64),
        tissue_fractions=np.array([1.0, 0.5], dtype=np.float32),
    )
    source = CachedSetSource(tmp_path)
    assert torch.equal(source.coords("s0"), torch.tensor([[32, 0], [64, 96]]))
    check_set_source(source)


def test_cached_grid_source_anchors_rois_at_the_sidecar_origin(tmp_path: Path) -> None:
    """slide2vec's region writer records the ROI's level-0 origin as ``x`` / ``y``; the
    geometry carries it, and two ROIs read at different source spacings share a layout."""
    geometry = compute_dense_geometry(target_size=32, patch_size=16)
    for sample_id, (x, y), source_spacing in (("r0", (1000, 2000), 0.25), ("r1", (0, 64), 0.4)):
        metadata = dense_grid_metadata(
            geometry, feature_dim=4, pad_mode="constant", spacing_um=0.5
        )
        metadata.update(
            {"x": x, "y": y, "source_spacing_um": source_spacing, "effective_spacing_um": 0.5}
        )
        write_dense_grid(tmp_path, sample_id, torch.randn(4, *geometry.grid_shape), metadata)
    source = CachedGridSource(tmp_path)

    r0, r1 = source.geometry("r0"), source.geometry("r1")
    assert r0.origin_level0 == (1000.0, 2000.0) and r0.level0_px_per_token_px == 2.0
    assert r0.token_to_level0((0, 0)) == (1016.0, 2016.0)
    assert r1.origin_level0 == (0.0, 64.0) and r1.level0_px_per_token_px == 1.25
    assert r0 != r1
    assert r0.layout == r1.layout
    check_grid_source(source)
