"""End-to-end (offline) routing test for dataset_type='detection'.

Builds a pre-extracted dense feature store + point-annotation manifest + splits on
disk, then runs ``train_one_detection_fold`` — exercising geometry-from-sidecar,
decoder + DetectionHead construction, the DetectionDataset loaders, ``Trainer.fit``
(streaming F1@δ tune metric), checkpoint reload, the tune-split threshold sweep, and
``_evaluate_detection`` (metrics + level-0 predictions CSV). No encoder / GPU needed:
the grids are written directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from soma.config import (
    DecoderConfig,
    EvalConfig,
    PreprocessingConfig,
    TaskConfig,
    TrainingConfig,
)
from soma.dataset import DetectionManifest, Splits
from soma.dense import DenseFeatureStore
from soma.dense.geometry import compute_dense_geometry
from soma.dense.store import dense_grid_metadata, write_dense_grid
from soma.pipeline import train_one_detection_fold

NUM_CLASSES = 2
TARGET = 16
PATCH = 4
FEATURE_DIM = 4
# Detection distances are µm-only; grids are read flat at this bookkeeping spacing, so
# level0_spacing == run_spacing == SPACING makes the point transform the identity, and
# match_distance/sigma in µm divide by SPACING to recover target-frame pixels.
SPACING = 0.2  # µm/px; 0.6 µm -> 3 px (δ), 0.3 µm -> 1.5 px (σ)


def _build_detection_run(root: Path, sample_ids: list[str]):
    dense_dir = root / "dense"
    points_dir = root / "points"
    dense_dir.mkdir()
    points_dir.mkdir()

    geom = compute_dense_geometry(target_size=TARGET, patch_size=PATCH)
    meta = dense_grid_metadata(geom, feature_dim=FEATURE_DIM, pad_mode="reflect", spacing_um=SPACING)

    rng = np.random.default_rng(0)
    rows = []
    for sid in sample_ids:
        write_dense_grid(dense_dir, sid, torch.randn(FEATURE_DIM, *geom.grid_shape), meta)
        # A couple of points per tile, well inside the frame, classes 0/1.
        pts = points_dir / f"{sid}.csv"
        pts.write_text("x,y,class\n4,4,0\n11,11,1\n")
        rows.append((sid, f"{sid}.jpg", str(pts)))

    manifest_csv = root / "manifest.csv"
    manifest_csv.write_text(
        "sample_id,image_path,points_path\n"
        + "\n".join(f"{sid},{img},{pp}" for sid, img, pp in rows)
        + "\n"
    )
    splits_csv = root / "splits.csv"
    assign = {sample_ids[0]: "train", sample_ids[1]: "train", sample_ids[2]: "tune", sample_ids[3]: "test"}
    splits_csv.write_text(
        "sample_id,split,fold\n" + "\n".join(f"{sid},{s},0" for sid, s in assign.items()) + "\n"
    )

    manifest = DetectionManifest(manifest_csv)
    splits = Splits(splits_csv, manifest)
    store = DenseFeatureStore(dense_dir)
    return manifest, splits, store


def test_train_one_detection_fold_end_to_end(tmp_path: Path):
    sample_ids = ["s0", "s1", "s2", "s3"]
    manifest, splits, store = _build_detection_run(tmp_path, sample_ids)

    result = train_one_detection_fold(
        feature_store=store,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(
            name="detection",
            params={
                "num_classes": NUM_CLASSES,
                "match_distance": 0.6,  # µm -> 3 px at SPACING
                "sigma": 0.3,  # µm -> 1.5 px
                "level0_spacing": SPACING,
            },
        ),
        training=TrainingConfig(epochs=2, batch_size=2),
        fold_dir=tmp_path / "fold",
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_f1", "f1_per_class"]),
        preprocessing=PreprocessingConfig(requested_spacing_um=SPACING, requested_tile_size_px=TARGET),
    )

    assert "mean_f1" in result.tune_report.metrics
    assert "test" in result.test_reports
    assert "mean_f1" in result.test_reports["test"].metrics

    # Frozen thresholds + per-split prediction CSVs are written.
    thr = json.loads((tmp_path / "fold" / "detection_thresholds.json").read_text())
    assert len(thr["score_threshold_per_class"]) == NUM_CLASSES
    assert (tmp_path / "fold" / "predictions_test.csv").exists()
    header = (tmp_path / "fold" / "predictions_test.csv").read_text().splitlines()[0]
    assert header == "sample_id,x,y,class,score"


def test_train_one_detection_fold_on_attention_grids(tmp_path: Path):
    """Detection on cls_attention grids is a pure feature_kind flip — the head/decoder
    are input_dim-agnostic, so a K-channel attention grid trains the same fold."""
    dense_dir = tmp_path / "dense"
    points_dir = tmp_path / "points"
    dense_dir.mkdir()
    points_dir.mkdir()

    K = 6  # attention channels (blocks × heads), not a patch-feature dim
    geom = compute_dense_geometry(target_size=TARGET, patch_size=PATCH)
    meta = dense_grid_metadata(
        geom, feature_dim=K, pad_mode="reflect", spacing_um=SPACING,
        feature_kind="cls_attention", attention_blocks=(-1,),
    )
    sample_ids = ["s0", "s1", "s2", "s3"]
    rows = []
    for sid in sample_ids:
        write_dense_grid(dense_dir, sid, torch.rand(K, *geom.grid_shape), meta)
        pts = points_dir / f"{sid}.csv"
        pts.write_text("x,y,class\n4,4,0\n11,11,1\n")
        rows.append((sid, f"{sid}.jpg", str(pts)))
    (tmp_path / "manifest.csv").write_text(
        "sample_id,image_path,points_path\n"
        + "\n".join(f"{s},{i},{p}" for s, i, p in rows) + "\n"
    )
    assign = {sample_ids[0]: "train", sample_ids[1]: "train", sample_ids[2]: "tune", sample_ids[3]: "test"}
    (tmp_path / "splits.csv").write_text(
        "sample_id,split,fold\n" + "\n".join(f"{s},{v},0" for s, v in assign.items()) + "\n"
    )
    manifest = DetectionManifest(tmp_path / "manifest.csv")
    splits = Splits(tmp_path / "splits.csv", manifest)
    store = DenseFeatureStore(dense_dir)
    assert store.metadata("s0")["feature_kind"] == "cls_attention"
    assert store.feature_dim == K

    result = train_one_detection_fold(
        feature_store=store,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(
            name="detection",
            params={
                "num_classes": NUM_CLASSES,
                "match_distance": 0.6,
                "sigma": 0.3,
                "level0_spacing": SPACING,
            },
        ),
        training=TrainingConfig(epochs=1, batch_size=2),
        fold_dir=tmp_path / "fold",
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_f1"]),
        preprocessing=PreprocessingConfig(requested_spacing_um=SPACING, requested_tile_size_px=TARGET),
    )
    assert "mean_f1" in result.test_reports["test"].metrics


def test_train_one_detection_fold_requires_num_classes(tmp_path: Path):
    import pytest

    manifest, splits, store = _build_detection_run(tmp_path, ["s0", "s1", "s2", "s3"])
    with pytest.raises(ValueError, match="num_classes"):
        train_one_detection_fold(
            feature_store=store,
            dataset=manifest,
            fold_split=splits.folds[0],
            task=TaskConfig(name="detection", params={}),
            training=TrainingConfig(epochs=1, batch_size=2),
            fold_dir=tmp_path / "fold",
            decoder=DecoderConfig(name="lightweight_conv"),
        )


def test_train_one_detection_fold_holdout_test_skips_test(tmp_path: Path):
    """evaluation.holdout_test: no test inference, no test artifacts, tune-only metrics.

    The tune-split threshold sweep and tune evaluation still run, so the run is a
    valid model-selection candidate while never touching the declared test split.
    """
    sample_ids = ["s0", "s1", "s2", "s3"]
    manifest, splits, store = _build_detection_run(tmp_path, sample_ids)

    result = train_one_detection_fold(
        feature_store=store,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(
            name="detection",
            params={
                "num_classes": NUM_CLASSES,
                "match_distance": 0.6,
                "sigma": 0.3,
                "level0_spacing": SPACING,
            },
        ),
        training=TrainingConfig(epochs=2, batch_size=2),
        fold_dir=tmp_path / "fold",
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_f1", "f1_per_class"], holdout_test=True),
        preprocessing=PreprocessingConfig(requested_spacing_um=SPACING, requested_tile_size_px=TARGET),
    )

    # Tune still evaluated; no test split touched.
    assert "mean_f1" in result.tune_report.metrics
    assert result.test_reports == {}

    # Threshold sweep (tune-only) still ran and was frozen.
    thr = json.loads((tmp_path / "fold" / "detection_thresholds.json").read_text())
    assert len(thr["score_threshold_per_class"]) == NUM_CLASSES

    # No test prediction CSV; metrics.json carries tune only.
    assert not (tmp_path / "fold" / "predictions_test.csv").exists()
    metrics = json.loads((tmp_path / "fold" / "metrics.json").read_text())
    assert list(metrics.keys()) == ["tune"]
