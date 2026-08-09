"""Offline tests for the multi-encoder composite (load-time channel concat, §8).

Builds two member dense stores with *different* patch sizes / token grids but a shared
supervision ``target_size``, wraps them in a ``CompositeDenseFeatureStore``, and checks
the concat shape + that the decoder-free fold runs end-to-end through the composite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from soma.dense import DenseFeatureStore, DenseSampleSpacing
from soma.dense.composite import CompositeDenseFeatureStore, resample_grid_to_target
from soma.dense.geometry import compute_dense_geometry
from soma.dense.store import dense_grid_metadata, write_dense_grid

TARGET = 8


def _write_member(
    root: Path,
    name: str,
    sample_ids,
    *,
    patch: int,
    k: int,
    spacing=None,
    source_spacing=None,
    effective_spacing=None,
):
    out = root / name
    geom = compute_dense_geometry(target_size=TARGET, patch_size=patch)
    meta = dense_grid_metadata(
        geom, feature_dim=k, pad_mode="reflect", spacing_um=spacing,
        feature_kind="cls_attention", attention_blocks=(-1,),
    )
    if source_spacing is not None or spacing is not None:
        meta["source_spacing_um"] = source_spacing if source_spacing is not None else spacing
    if effective_spacing is not None or spacing is not None:
        meta["effective_spacing_um"] = (
            effective_spacing if effective_spacing is not None else spacing
        )
    for sid in sample_ids:
        write_dense_grid(out, sid, torch.rand(k, *geom.grid_shape), meta)
    return DenseFeatureStore(out)


def test_resample_grid_to_target_reaches_target_resolution():
    geom = compute_dense_geometry(target_size=TARGET, patch_size=4)  # 2x2 grid -> 8x8
    grid = torch.rand(3, *geom.grid_shape)
    out = resample_grid_to_target(grid, geom)
    assert out.shape == (3, TARGET, TARGET)


def test_composite_concats_members_at_target_resolution(tmp_path: Path):
    ids = ["s0", "s1", "s2", "s3"]
    a = _write_member(tmp_path, "a", ids, patch=4, k=3)  # 2x2 grid
    b = _write_member(tmp_path, "b", ids, patch=2, k=5)  # 4x4 grid
    comp = CompositeDenseFeatureStore([a, b])

    assert comp.feature_dim == 8  # 3 + 5
    assert sorted(comp.available_samples) == ids
    assert comp.grid_shape == (TARGET, TARGET)
    grid = comp.load("s0")
    assert grid.shape == (8, TARGET, TARGET)  # concat at shared target pixel grid
    geom = comp.geometry("s0")
    assert geom.grid_shape == (TARGET, TARGET) and geom.patch_size == (1, 1)
    meta = comp.metadata("s0")
    assert meta["feature_kind"] == "composite" and meta["feature_dim"] == 8
    assert meta["concat_resolution"] == "target"
    assert len(meta["members"]) == 2


def test_composite_rejects_disjoint_samples(tmp_path: Path):
    a = _write_member(tmp_path, "a", ["s0", "s1"], patch=4, k=3)
    b = _write_member(tmp_path, "b", ["s2", "s3"], patch=4, k=3)
    with pytest.raises(ValueError, match="no common samples"):
        CompositeDenseFeatureStore([a, b])


def test_composite_rejects_target_size_mismatch(tmp_path: Path):
    a = _write_member(tmp_path, "a", ["s0"], patch=4, k=3)  # target 8
    # member b at target 16 (different supervision size) -> mismatch on load.
    out = tmp_path / "b"
    geom = compute_dense_geometry(target_size=16, patch_size=4)
    meta = dense_grid_metadata(geom, feature_dim=3, pad_mode="reflect")
    write_dense_grid(out, "s0", torch.rand(3, *geom.grid_shape), meta)
    comp = CompositeDenseFeatureStore([a, DenseFeatureStore(out)])
    with pytest.raises(ValueError, match="disagree on target_size"):
        comp.geometry("s0")


def test_composite_exposes_one_agreed_resolved_spacing(tmp_path: Path):
    a = _write_member(tmp_path, "a", ["s0"], patch=4, k=3, spacing=0.5)
    b = _write_member(tmp_path, "b", ["s0"], patch=2, k=5, spacing=0.5)

    assert CompositeDenseFeatureStore([a, b]).spacing("s0") == DenseSampleSpacing(
        source_spacing_um=0.5,
        effective_spacing_um=0.5,
    )


def test_composite_rejects_member_source_spacing_disagreement(tmp_path: Path):
    a = _write_member(
        tmp_path,
        "a",
        ["s0"],
        patch=4,
        k=3,
        spacing=0.5,
        source_spacing=0.25,
    )
    b = _write_member(
        tmp_path,
        "b",
        ["s0"],
        patch=2,
        k=5,
        spacing=0.5,
        source_spacing=0.3,
    )

    with pytest.raises(ValueError, match=r"source_spacing_um.*s0.*0.25.*0.3"):
        CompositeDenseFeatureStore([a, b]).spacing("s0")


def test_composite_grid_mode_concats_at_largest_member_grid(tmp_path: Path):
    ids = ["s0", "s1"]
    a = _write_member(tmp_path, "a", ids, patch=4, k=3)  # 2x2 grid
    b = _write_member(tmp_path, "b", ids, patch=2, k=5)  # 4x4 grid (the largest)
    comp = CompositeDenseFeatureStore([a, b], concat_resolution="grid")

    grid = comp.load("s0")
    assert grid.shape == (8, 4, 4)  # concat at the largest member token grid, not target
    geom = comp.geometry("s0")
    # grid_shape = the real (h, w) decoder input; encoded_size = target; crop = full frame.
    assert geom.grid_shape == (4, 4)
    assert geom.encoded_size == (TARGET, TARGET)
    assert geom.crop_box == (0, 0, TARGET, TARGET)
    # ratio target/grid = 8/4 = 2 → the decoder learns one 2x upsample (not a single jump).
    assert geom.encoded_size[0] / geom.grid_shape[0] == 2.0
    meta = comp.metadata("s0")
    assert meta["concat_resolution"] == "grid" and meta["grid_shape"] == [4, 4]


def test_composite_grid_mode_explicit_grid_size(tmp_path: Path):
    ids = ["s0"]
    a = _write_member(tmp_path, "a", ids, patch=4, k=3)
    b = _write_member(tmp_path, "b", ids, patch=2, k=5)
    comp = CompositeDenseFeatureStore([a, b], concat_resolution="grid", concat_grid_size=(3, 3))
    assert comp.load("s0").shape == (8, 3, 3)
    assert comp.geometry("s0").grid_shape == (3, 3)


def test_composite_member_norm_l2_makes_per_member_slices_unit_norm(tmp_path: Path):
    ids = ["s0"]
    a = _write_member(tmp_path, "a", ids, patch=4, k=3)
    b = _write_member(tmp_path, "b", ids, patch=2, k=5)
    comp = CompositeDenseFeatureStore(
        [a, b], concat_resolution="grid", member_norms=["l2", "l2"]
    )
    grid = comp.load("s0")  # (8, 4, 4)
    # Each member's channel slice is independently unit-L2 per pixel.
    assert torch.allclose(grid[:3].norm(dim=0), torch.ones(4, 4), atol=1e-5)
    assert torch.allclose(grid[3:].norm(dim=0), torch.ones(4, 4), atol=1e-5)
    meta = comp.metadata("s0")
    assert [m["member_norm"] for m in meta["members"]] == ["l2", "l2"]


def test_composite_target_mode_is_byte_identical_without_norm(tmp_path: Path):
    # Default (no member_norm) target mode must reproduce a plain per-member resample+concat.
    ids = ["s0"]
    a = _write_member(tmp_path, "a", ids, patch=4, k=3)
    b = _write_member(tmp_path, "b", ids, patch=2, k=5)
    comp = CompositeDenseFeatureStore([a, b])  # defaults: target, no norm
    expected = torch.cat(
        [
            resample_grid_to_target(a.load("s0"), a.geometry("s0")),
            resample_grid_to_target(b.load("s0"), b.geometry("s0")),
        ],
        dim=0,
    )
    assert torch.allclose(comp.load("s0"), expected, atol=0.0)


def test_decoder_fold_runs_through_composite_grid_mode(tmp_path: Path):
    from soma.config import DecoderConfig, EvalConfig, TaskConfig, TrainingConfig
    from soma.dataset import SegmentationManifest, Splits
    from soma.pipeline import train_one_segmentation_fold

    ids = ["s0", "s1", "s2", "s3"]
    a = _write_member(tmp_path, "a", ids, patch=4, k=3)  # 2x2 grid
    b = _write_member(tmp_path, "b", ids, patch=2, k=4)  # 4x4 grid
    comp = CompositeDenseFeatureStore([a, b], concat_resolution="grid", member_norms=["l2", "l2"])

    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for sid in ids:
        mask = rng.integers(0, 2, size=(TARGET, TARGET), dtype=np.uint8)
        Image.fromarray(mask).save(masks_dir / f"{sid}.png")
        rows.append((sid, f"{sid}.jpg", str(masks_dir / f"{sid}.png")))
    manifest_csv = tmp_path / "manifest.csv"
    manifest_csv.write_text(
        "sample_id,image_path,mask_path\n" + "\n".join(f"{s},{i},{m}" for s, i, m in rows) + "\n"
    )
    splits_csv = tmp_path / "splits.csv"
    assign = {ids[0]: "train", ids[1]: "train", ids[2]: "tune", ids[3]: "test"}
    splits_csv.write_text(
        "sample_id,split,fold\n" + "\n".join(f"{s},{v},0" for s, v in assign.items()) + "\n"
    )
    manifest = SegmentationManifest(manifest_csv)
    splits = Splits(splits_csv, manifest)

    result = train_one_segmentation_fold(
        feature_store=comp,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(name="segmentation", params={"num_classes": 2}),
        training=TrainingConfig(epochs=1, batch_size=2),
        fold_dir=tmp_path / "fold",
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_dice", "mean_iou"]),
    )
    assert "test" in result.test_reports
    assert 0.0 <= result.test_reports["test"].metrics["mean_dice"] <= 1.0


def test_composite_spacing_auto_defaults_from_shared_member_spacing():
    # conch + h0-mini both advertise 0.5 µm/px → auto-default resolves to 0.5 (no explicit
    # preprocessing.requested_spacing_um needed). Mismatched/multi-spacing members must pin.
    from soma.config import CompositeConfig, EncoderMemberConfig
    from soma.pipeline import Pipeline

    comp = CompositeConfig(
        encoders=[EncoderMemberConfig(name="conch"), EncoderMemberConfig(name="h0-mini")]
    )
    assert Pipeline._resolve_composite_spacing(comp) == 0.5


def test_pipeline_resolve_preprocessing_propagates_composite_spacing():
    from soma.config import (
        CompositeConfig,
        DecoderConfig,
        EncoderMemberConfig,
        PipelineConfig,
        PreprocessingConfig,
        TaskConfig,
    )
    from soma.pipeline import Pipeline

    cfg = PipelineConfig(
        dataset_csv="data.csv",
        splits_csv="splits.csv",
        output_root="out",
        dataset_type="segmentation",
        preprocessing=PreprocessingConfig(requested_tile_size_px=TARGET),
        decoder=DecoderConfig(name="lightweight_conv"),
        composite=CompositeConfig(
            encoders=[EncoderMemberConfig(name="conch"), EncoderMemberConfig(name="h0-mini")]
        ),
        task=TaskConfig(name="segmentation", params={"num_classes": 2}),
    )
    pipeline = object.__new__(Pipeline)
    pipeline._config = cfg

    assert pipeline._resolve_preprocessing().requested_spacing_um == 0.5


def test_pixel_classifier_fold_runs_through_composite(tmp_path: Path):
    pytest.importorskip("xgboost")
    from soma.config import EvalConfig, PixelClassifierConfig, TaskConfig, TrainingConfig
    from soma.dataset import SegmentationManifest, Splits
    from soma.pipeline import train_one_pixel_classifier_fold

    ids = ["s0", "s1", "s2", "s3"]
    a = _write_member(tmp_path, "a", ids, patch=4, k=3)
    b = _write_member(tmp_path, "b", ids, patch=2, k=4)
    comp = CompositeDenseFeatureStore([a, b])

    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for sid in ids:
        mask = rng.integers(0, 2, size=(TARGET, TARGET), dtype=np.uint8)
        Image.fromarray(mask).save(masks_dir / f"{sid}.png")
        rows.append((sid, f"{sid}.jpg", str(masks_dir / f"{sid}.png")))
    manifest_csv = tmp_path / "manifest.csv"
    manifest_csv.write_text(
        "sample_id,image_path,mask_path\n" + "\n".join(f"{s},{i},{m}" for s, i, m in rows) + "\n"
    )
    splits_csv = tmp_path / "splits.csv"
    assign = {ids[0]: "train", ids[1]: "train", ids[2]: "tune", ids[3]: "test"}
    splits_csv.write_text(
        "sample_id,split,fold\n" + "\n".join(f"{s},{v},0" for s, v in assign.items()) + "\n"
    )
    manifest = SegmentationManifest(manifest_csv)
    splits = Splits(splits_csv, manifest)

    result = train_one_pixel_classifier_fold(
        feature_store=comp,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(name="segmentation", params={"num_classes": 2}),
        training=TrainingConfig(epochs=1, batch_size=1, max_train_pixels=500),
        fold_dir=tmp_path / "fold",
        pixel_classifier=PixelClassifierConfig(
            name="xgboost", params={"n_estimators": 8, "max_depth": 2, "early_stopping_rounds": None}
        ),
        evaluation=EvalConfig(metrics=["mean_dice", "mean_iou"]),
    )
    assert "test" in result.test_reports
    assert 0.0 <= result.test_reports["test"].metrics["mean_dice"] <= 1.0
    assert (tmp_path / "fold" / "preds" / "test" / "s3.png").is_file()
