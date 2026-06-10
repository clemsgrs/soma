"""End-to-end (offline) routing smoke test for dataset_type='segmentation'.

Builds a pre-extracted dense feature store + mask manifest + splits on disk, then
runs ``train_one_segmentation_fold`` — exercising geometry-from-sidecar, decoder +
head construction, the SegmentationDataset loaders, ``Trainer.fit`` (with the
streaming ``_tune``), checkpoint reload, and ``_evaluate_segmentation``. No encoder
/ GPU needed: the grids are written directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from soma.config import DecoderConfig, EvalConfig, TaskConfig, TrainingConfig
from soma.dataset import SegmentationManifest, Splits
from soma.dense import DenseFeatureStore
from soma.dense.geometry import compute_dense_geometry
from soma.dense.store import dense_grid_metadata, write_dense_grid
from soma.pipeline import train_one_segmentation_fold

NUM_CLASSES = 2
TARGET = 8
PATCH = 4
FEATURE_DIM = 4


def _build_dense_run(root: Path, sample_ids: list[str]) -> tuple[SegmentationManifest, Splits, DenseFeatureStore]:
    dense_dir = root / "dense"
    masks_dir = root / "masks"
    dense_dir.mkdir()
    masks_dir.mkdir()

    geom = compute_dense_geometry(target_size=TARGET, patch_size=PATCH)  # grid 2x2, encoded 8
    meta = dense_grid_metadata(geom, feature_dim=FEATURE_DIM, pad_mode="reflect")

    rows = []
    rng = np.random.default_rng(0)
    for sid in sample_ids:
        write_dense_grid(dense_dir, sid, torch.randn(FEATURE_DIM, *geom.grid_shape), meta)
        mask = rng.integers(0, NUM_CLASSES, size=(TARGET, TARGET), dtype=np.uint8)
        mask_path = masks_dir / f"{sid}.png"
        Image.fromarray(mask).save(mask_path)
        rows.append((sid, f"{sid}.jpg", str(mask_path)))

    manifest_csv = root / "manifest.csv"
    manifest_csv.write_text(
        "sample_id,image_path,mask_path\n"
        + "\n".join(f"{sid},{img},{mask}" for sid, img, mask in rows)
        + "\n"
    )
    splits_csv = root / "splits.csv"
    # 4 samples: 2 train, 1 tune, 1 test, single fold.
    split_assign = {sample_ids[0]: "train", sample_ids[1]: "train", sample_ids[2]: "tune", sample_ids[3]: "test"}
    splits_csv.write_text(
        "sample_id,split,fold\n"
        + "\n".join(f"{sid},{split},0" for sid, split in split_assign.items())
        + "\n"
    )

    manifest = SegmentationManifest(manifest_csv)
    splits = Splits(splits_csv, manifest)
    store = DenseFeatureStore(dense_dir)
    return manifest, splits, store


def test_train_one_segmentation_fold_end_to_end(tmp_path: Path):
    sample_ids = ["s0", "s1", "s2", "s3"]
    manifest, splits, store = _build_dense_run(tmp_path, sample_ids)

    result = train_one_segmentation_fold(
        feature_store=store,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(name="segmentation", params={"num_classes": NUM_CLASSES}),
        training=TrainingConfig(epochs=1, batch_size=2),
        fold_dir=tmp_path / "fold",
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_dice", "mean_iou"]),
    )

    # tune + test reports carry dense metrics (no scalar predictions in 1f-b).
    assert set(result.tune_report.metrics) >= {"mean_dice", "mean_iou"}
    assert "test" in result.test_reports
    assert set(result.test_reports["test"].metrics) >= {"mean_dice", "mean_iou"}
    assert result.tune_report.predictions == []
    assert 0.0 <= result.tune_report.metrics["mean_dice"] <= 1.0
    # metrics.json + checkpoint were written.
    assert (tmp_path / "fold" / "metrics.json").is_file()
    assert (tmp_path / "fold" / "best_model.pt").is_file()

    # 1g dense artifacts: per-tile rasters + predictions CSV land for each split.
    fold = tmp_path / "fold"
    assert (fold / "predictions_test.csv").is_file()
    test_raster = fold / "preds" / "test" / "s3.png"
    assert test_raster.is_file()
    raster = np.asarray(Image.open(test_raster))
    assert raster.shape == (TARGET, TARGET)  # head cropped logits to target_size
    assert raster.dtype == np.uint8
    assert raster.max() < NUM_CLASSES
    # Overlays are fail-soft: the cached-feature fixture has no real source tiles.
    assert not (fold / "pred_overlays" / "test" / "s3.png").exists()


def test_segmentation_fold_requires_num_classes(tmp_path: Path):
    manifest, splits, store = _build_dense_run(tmp_path, ["s0", "s1", "s2", "s3"])
    with pytest.raises(ValueError, match="num_classes"):
        train_one_segmentation_fold(
            feature_store=store,
            dataset=manifest,
            fold_split=splits.folds[0],
            task=TaskConfig(name="segmentation"),  # no num_classes
            training=TrainingConfig(epochs=1, batch_size=2),
            fold_dir=tmp_path / "fold",
            decoder=DecoderConfig(name="lightweight_conv"),
        )


def _seg_pipeline_config(tmp_path: Path):
    from soma.config import PipelineConfig

    return PipelineConfig(
        dataset_csv=tmp_path / "manifest.csv",
        splits_csv=tmp_path / "splits.csv",
        output_root=tmp_path / "out",
        dataset_type="segmentation",
        decoder=DecoderConfig(name="lightweight_conv"),
        task=TaskConfig(name="segmentation", params={"num_classes": NUM_CLASSES}),
        training=TrainingConfig(epochs=1, batch_size=2),
        evaluation=EvalConfig(metrics=["mean_dice", "mean_iou"]),
    )


def test_pipeline_uses_segmentation_manifest(tmp_path: Path):
    from soma.pipeline import Pipeline

    _build_dense_run(tmp_path, ["s0", "s1", "s2", "s3"])
    pipeline = Pipeline(_seg_pipeline_config(tmp_path), feature_dir=tmp_path / "dense")
    assert isinstance(pipeline.dataset, SegmentationManifest)
    assert pipeline._get_feature_store(run_dir=tmp_path / "out").feature_dim == FEATURE_DIM


def test_pipeline_run_end_to_end_segmentation(tmp_path: Path):
    """The real entry point: Pipeline.run() (run-summary panel, train, summary)."""
    from soma.pipeline import Pipeline

    _build_dense_run(tmp_path, ["s0", "s1", "s2", "s3"])
    pipeline = Pipeline(_seg_pipeline_config(tmp_path), feature_dir=tmp_path / "dense")
    result = pipeline.run()
    assert "test/mean_dice" in result.summary
    assert result.fold_results[0].test_reports["test"].metrics["mean_dice"] >= 0.0
    # Dense artifacts land via the real entry point too (run -> train -> eval).
    # run_dir is a structured layout path under output_root, so glob for them.
    out = tmp_path / "out"
    assert list(out.rglob("preds/test/s3.png")), "pred raster missing under run dir"
    assert list(out.rglob("predictions_test.csv")), "predictions CSV missing under run dir"
