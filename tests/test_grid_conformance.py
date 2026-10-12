"""Conformance of the grid shape's protocols: grid heads, the live path's image source and
encoder stage, and the composite grid source."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from soma.data import ImageManifest, TileImageSource
from soma.data.manifests import ImageEntry
from soma.dense import CachedGridSource
from soma.dense.composite import CompositeDenseFeatureStore
from soma.dense.geometry import compute_dense_geometry
from soma.dense.store import dense_grid_metadata, write_dense_grid
from soma.learn import GridBatch
from soma.tasks.detection import DetectionHead
from soma.tasks.segmentation import SegmentationHead
from soma.testing import (
    ConformanceError,
    FakeEncoder,
    check_encoder_stage,
    check_grid_source,
    check_image_source,
    check_task_head,
)


def _grid_batch(targets: dict[str, torch.Tensor], num_classes: int) -> GridBatch:
    torch.manual_seed(0)
    # The head's input is the decoder's output: logits on an 8x8 grid.
    return GridBatch(
        features=torch.randn(3, num_classes, 8, 8) * 3,
        pixels=None,
        targets=targets,
        sample_ids=("a", "b", "c"),
    )


# --- check_task_head, grid branch ----------------------------------------------------------


def test_segmentation_head_conforms_on_its_label_maps():
    head = SegmentationHead(
        num_classes=3,
        geometry=compute_dense_geometry(target_size=30, patch_size=4),
        metrics=["mean_dice", "mean_iou", "dataset_global_mean_dice", "foreground_mean_dice", "dice_per_class"],
    )
    mask = torch.randint(0, 3, (3, 30, 30), generator=torch.Generator().manual_seed(2))
    mask[0, :5] = 255  # ignored pixels count nowhere
    check_task_head(head, _grid_batch({"mask": mask}, 3))


def test_detection_head_conforms_on_its_points():
    head = DetectionHead(
        num_classes=2,
        geometry=compute_dense_geometry(target_size=32, patch_size=4),
        spacing_um=0.5,
        match_distance=2.0,
        validation_threshold=0.3,
        metrics=["mean_f1", "f1_per_class", "precision", "recall", "mean_f1_per_image"],
    )
    points = torch.full((3, 4, 3), float("nan"))
    points[0, :3] = torch.tensor([[5.0, 6.0, 0.0], [20.0, 21.0, 1.0], [12.0, 28.0, 0.0]])
    points[1, :1] = torch.tensor([[30.0, 2.0, 1.0]])
    ignore = torch.zeros(3, 32, 32, dtype=torch.bool)
    ignore[2, 16:] = True
    check_task_head(head, _grid_batch({"points": points, "ignore": ignore}, 2))


def test_a_grid_head_whose_metrics_disagree_with_evaluate_is_refused():
    class Lenient(SegmentationHead):
        def finalize_eval_metrics(self, counts):
            return {"mean_dice": 1.0}

    head = Lenient(num_classes=2, geometry=compute_dense_geometry(target_size=16, patch_size=4))
    mask = torch.randint(0, 2, (3, 16, 16), generator=torch.Generator().manual_seed(3))
    with pytest.raises(ConformanceError, match="compute_metrics"):
        check_task_head(head, _grid_batch({"mask": mask}, 2))


# --- live path: encoder stage and image source ---------------------------------------------


def test_fake_encoder_conforms_to_the_encoder_stage_contract():
    check_encoder_stage(FakeEncoder(target_size=30, patch_size=8))


def test_a_stage_with_parameters_is_refused():
    class Trainable(FakeEncoder):
        def __init__(self) -> None:
            super().__init__(target_size=16, patch_size=8)
            self.scale = torch.nn.Parameter(torch.ones(1))

    with pytest.raises(ConformanceError, match="parameters"):
        check_encoder_stage(Trainable())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU and a gated slide2vec encoder")
def test_frozen_encoder_conforms_to_the_encoder_stage_contract():
    from soma.extract import live_encoder

    try:
        stage = live_encoder("phikon", target_size=224, spacing_um=0.5)
    except Exception as error:  # noqa: BLE001 - weights unavailable offline
        pytest.skip(f"encoder unavailable: {error}")
    check_encoder_stage(stage)


def _tiles(root: Path, ids: list[str], size: int = 16) -> ImageManifest:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    entries = []
    for sample_id in ids:
        path = root / f"{sample_id}.png"
        Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8)).save(path)
        entries.append(ImageEntry(sample_id=sample_id, image_path=path, spacing_at_level_0=0.5))
    return ImageManifest(entries)


def test_tile_image_source_conforms(tmp_path):
    source = TileImageSource.from_manifest(_tiles(tmp_path, ["t0", "t1"]), size=16)
    check_image_source(source)
    assert source.spacing("t0") == 0.5
    assert source.geometry("t0").level0_px_per_token_px == 1.0


def test_a_flat_tile_declaring_a_finer_spacing_is_resampled_to_the_requested_one(tmp_path):
    # A 100 px PNG declared at 0.2 µm/px read at 0.5 µm/px covers 40 px, as slide2vec's
    # cached extraction area-resamples it; the frame keeps 2.5 level-0 px per px.
    path = tmp_path / "tile.png"
    Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8)).save(path)
    entry = ImageEntry(sample_id="t", image_path=path, spacing_at_level_0=0.2)
    source = TileImageSource([entry], size=40, spacing_um=0.5)

    assert tuple(source.load("t").shape) == (3, 40, 40)
    assert source.geometry("t").level0_px_per_token_px == pytest.approx(2.5)
    assert source.spacing("t") == pytest.approx(0.5)


def _pyramid(path: Path, *, level0_spacing_um: float, size: int = 128) -> Path:
    """A two-level tiled TIFF (consecutive IFDs, as every hs2p backend reads them): level 1
    is level 0 downsampled 2x."""
    tifffile = pytest.importorskip("tifffile")
    image = np.random.default_rng(0).integers(0, 256, (size, size, 3), dtype=np.uint8)
    resolution = 1e4 / level0_spacing_um  # pixels per cm
    with tifffile.TiffWriter(path) as writer:
        for level in (0, 1):
            factor = 2**level
            writer.write(
                np.ascontiguousarray(image[::factor, ::factor]),
                photometric="rgb",
                tile=(32, 32),
                resolution=(resolution / factor, resolution / factor),
                resolutionunit="CENTIMETER",
                subfiletype=level,
            )
    return path


def test_a_pyramidal_tile_reports_the_spacing_of_the_level_hs2p_serves(tmp_path):
    # Level 1 (0.48 µm/px) is within 5% of the requested 0.5: hs2p serves it unresized,
    # so the frame is 0.48 µm/px with 2 level-0 px per px, not the requested 0.5.
    path = _pyramid(tmp_path / "slide.tif", level0_spacing_um=0.24)
    source = TileImageSource([ImageEntry(sample_id="s", image_path=path)], size=64, spacing_um=0.5)

    assert tuple(source.load("s").shape) == (3, 64, 64)
    assert source.spacing("s") == pytest.approx(0.48, rel=1e-3)
    assert source.geometry("s").level0_px_per_token_px == pytest.approx(2.0, rel=1e-3)


def test_a_pyramidal_tile_is_read_through_the_declared_level0_spacing(tmp_path):
    # The file says 0.5 µm/px at level 0; the manifest declares 0.25, making level 1 the
    # 0.5 µm/px level the read must serve.
    path = _pyramid(tmp_path / "slide.tif", level0_spacing_um=0.5)
    entry = ImageEntry(sample_id="s", image_path=path, spacing_at_level_0=0.25)
    source = TileImageSource([entry], size=64, spacing_um=0.5)

    assert tuple(source.load("s").shape) == (3, 64, 64)
    assert source.spacing("s") == pytest.approx(0.5)
    assert source.geometry("s").level0_px_per_token_px == pytest.approx(2.0)


def test_a_region_image_source_serves_each_roi_window_at_its_origin(tmp_path):
    from soma.data.images import RegionImageSource

    path = _pyramid(tmp_path / "slide.tif", level0_spacing_um=0.5)
    entries = [ImageEntry(sample_id=sid, image_path=path) for sid in ("r0", "r1")]
    source = RegionImageSource(entries, regions={"r0": (0, 0), "r1": (64, 32)}, size=32, spacing_um=0.5)

    check_image_source(source)
    whole = TileImageSource(entries[:1], size=128, spacing_um=0.5).load("r0")
    assert torch.equal(source.load("r1"), whole[:, 32:64, 64:96])
    assert source.geometry("r1").origin_level0 == (64.0, 32.0)
    assert source.subset(["r1"]).geometry("r1").origin_level0 == (64.0, 32.0)


def test_an_image_source_serving_floats_is_refused(tmp_path):
    class Floats(TileImageSource):
        def load(self, sample_id):
            return super().load(sample_id).float()

    source = Floats.from_manifest(_tiles(tmp_path, ["t0"]), size=16)
    with pytest.raises(ConformanceError, match="uint8"):
        check_image_source(source)


# --- composite grid source ----------------------------------------------------------------


def _member(root: Path, name: str, ids: list[str], *, patch: int, k: int) -> CachedGridSource:
    geometry = compute_dense_geometry(target_size=8, patch_size=patch)
    meta = dense_grid_metadata(geometry, feature_dim=k, pad_mode="reflect", spacing_um=0.5)
    meta["source_spacing_um"] = meta["effective_spacing_um"] = 0.5
    for sample_id in ids:
        write_dense_grid(root / name, sample_id, torch.rand(k, *geometry.grid_shape), meta)
    return CachedGridSource(root / name)


@pytest.mark.parametrize("resolution", ["grid", "target"])
def test_composite_grid_source_conforms(tmp_path, resolution):
    ids = ["s0", "s1"]
    composite = CompositeDenseFeatureStore(
        [_member(tmp_path, "a", ids, patch=4, k=3), _member(tmp_path, "b", ids, patch=2, k=4)],
        concat_resolution=resolution,
    )
    check_grid_source(composite)
