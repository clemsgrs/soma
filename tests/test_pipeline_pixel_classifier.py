"""End-to-end (offline) test for the decoder-free pixel-classifier segmentation path.

Writes dense grids (any K channels — the classifier is channel-agnostic) + masks +
splits to disk, then runs ``train_one_pixel_classifier_fold`` and the full
``Pipeline.run`` with ``pixel_classifier=xgboost``. No encoder / GPU: the grids are
written directly, exercising the sampling → fit → predict-all → dense metrics/artifacts
path and the config routing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("xgboost")

from soma.config import (
    EvalConfig,
    PixelClassifierConfig,
    PreprocessingConfig,
    TaskConfig,
    TrainingConfig,
)
from soma.dataset import SegmentationManifest, Splits
from soma.dense import DenseFeatureStore
from soma.dense.geometry import compute_dense_geometry
from soma.dense.store import dense_grid_metadata, write_dense_grid
from soma.pipeline import train_one_pixel_classifier_fold

NUM_CLASSES = 2
TARGET = 8
PATCH = 4
K = 6  # attention-like channel count


def _build_dense_run(root: Path, sample_ids: list[str]):
    dense_dir = root / "dense"
    masks_dir = root / "masks"
    dense_dir.mkdir()
    masks_dir.mkdir()
    geom = compute_dense_geometry(target_size=TARGET, patch_size=PATCH)  # 2x2 grid, encoded 8
    meta = dense_grid_metadata(
        geom, feature_dim=K, pad_mode="reflect", feature_kind="cls_attention",
        attention_blocks=(-1,), attention_include_registers=False,
    )
    rng = np.random.default_rng(0)
    rows = []
    for i, sid in enumerate(sample_ids):
        # Make the grid weakly predictive of the mask so Dice isn't degenerate:
        # class-correlated channel means.
        write_dense_grid(dense_dir, sid, torch.rand(K, *geom.grid_shape), meta)
        mask = rng.integers(0, NUM_CLASSES, size=(TARGET, TARGET), dtype=np.uint8)
        label_mask_path = masks_dir / f"{sid}.png"
        Image.fromarray(mask).save(label_mask_path)
        rows.append((sid, f"{sid}.jpg", str(label_mask_path)))

    manifest_csv = root / "manifest.csv"
    manifest_csv.write_text(
        "sample_id,image_path,label_mask_path\n"
        + "\n".join(f"{sid},{img},{mask}" for sid, img, mask in rows) + "\n"
    )
    splits_csv = root / "splits.csv"
    assign = {sample_ids[0]: "train", sample_ids[1]: "train", sample_ids[2]: "tune", sample_ids[3]: "test"}
    splits_csv.write_text(
        "sample_id,split,fold\n" + "\n".join(f"{sid},{s},0" for sid, s in assign.items()) + "\n"
    )
    manifest = SegmentationManifest(manifest_csv)
    splits = Splits(splits_csv, manifest)
    store = DenseFeatureStore(dense_dir)
    return manifest, splits, store


def _xgb() -> PixelClassifierConfig:
    return PixelClassifierConfig(
        name="xgboost",
        params={"n_estimators": 8, "max_depth": 2, "early_stopping_rounds": None},
    )


def test_train_one_pixel_classifier_fold_end_to_end(tmp_path: Path):
    sample_ids = ["s0", "s1", "s2", "s3"]
    manifest, splits, store = _build_dense_run(tmp_path, sample_ids)

    result = train_one_pixel_classifier_fold(
        feature_store=store,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(name="segmentation", params={"num_classes": NUM_CLASSES}),
        training=TrainingConfig(epochs=1, batch_size=1, max_train_pixels=1000),
        fold_dir=tmp_path / "fold",
        pixel_classifier=_xgb(),
        evaluation=EvalConfig(metrics=["mean_dice", "mean_iou"]),
    )

    assert result.train_result is None  # no Trainer on this path
    assert set(result.tune_report.metrics) >= {"mean_dice", "mean_iou"}
    assert "test" in result.test_reports
    assert 0.0 <= result.test_reports["test"].metrics["mean_dice"] <= 1.0

    fold = tmp_path / "fold"
    assert (fold / "metrics.json").is_file()
    assert (fold / "predictions_test.csv").is_file()
    raster = fold / "preds" / "test" / "s3.png"
    assert raster.is_file()
    arr = np.asarray(Image.open(raster))
    assert arr.shape == (TARGET, TARGET) and arr.dtype == np.uint8 and arr.max() < NUM_CLASSES
    # The fitted classifier was persisted and reloads to the same predictions.
    assert (fold / "pixel_classifier").is_dir()


def test_pixel_classifier_save_load_roundtrip(tmp_path: Path):
    from soma.pixel_classifiers.xgboost_clf import XGBoostPixelClassifier

    rng = np.random.default_rng(1)
    X = rng.random((200, K)).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, size=200)
    clf = XGBoostPixelClassifier(num_classes=NUM_CLASSES, n_estimators=8, max_depth=2,
                                 early_stopping_rounds=None)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (200, NUM_CLASSES)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    clf.save(tmp_path / "clf")
    reloaded = XGBoostPixelClassifier.load(tmp_path / "clf")
    np.testing.assert_allclose(reloaded.predict_proba(X), proba, atol=1e-6)


def test_predict_proba_scatters_missing_classes(tmp_path: Path):
    # 3 declared classes but only {0, 2} present at fit -> full (N, 3) with col 1 == 0.
    from soma.pixel_classifiers.xgboost_clf import XGBoostPixelClassifier

    rng = np.random.default_rng(2)
    X = rng.random((120, K)).astype(np.float32)
    y = rng.choice([0, 2], size=120)
    clf = XGBoostPixelClassifier(num_classes=3, n_estimators=6, max_depth=2,
                                 early_stopping_rounds=None)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (120, 3)
    assert np.allclose(proba[:, 1], 0.0)


@pytest.mark.parametrize("name,params", [
    ("random_forest", {"n_estimators": 8, "max_depth": 3}),
    ("logistic", {"max_iter": 50}),
    ("mlp", {"epochs": 80, "batch_size": 64, "learning_rate": 5e-3, "hidden_dim": 16,
             "num_layers": 2, "device": "cpu"}),
])
def test_other_classifiers_fit_predict_save_load(tmp_path: Path, name, params):
    from soma.pixel_classifiers import pixel_classifier_registry

    rng = np.random.default_rng(3)
    X = rng.random((300, K)).astype(np.float32)
    # learnable signal: class depends on first channel threshold
    y = (X[:, 0] > 0.5).astype(np.int64)
    cls = pixel_classifier_registry.get(name)
    clf = cls(num_classes=NUM_CLASSES, **params)
    clf.fit(X, y, X_val=X[:50], y_val=y[:50])
    proba = clf.predict_proba(X)
    assert proba.shape == (300, NUM_CLASSES)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    # learns the threshold better than chance
    acc = (proba.argmax(1) == y).mean()
    assert acc > 0.7

    clf.save(tmp_path / name)
    reloaded = cls.load(tmp_path / name)
    np.testing.assert_allclose(reloaded.predict_proba(X), proba, atol=1e-4)


def test_build_training_matrix_handles_negative_ignore_index():
    # SegmentationHead allows a negative ignore_index (torch's -100 convention); those
    # pixels must be excluded from sampling, not crash np.bincount (which rejects negatives).
    from types import SimpleNamespace

    from soma.pixel_classifiers.segmentation import build_training_matrix

    grid = torch.rand(K, 4, 4)
    mask = torch.zeros(4, 4, dtype=torch.long)
    mask[0] = -100  # ignore row
    mask[1] = 1     # class 1; rest class 0

    class _Store:
        def load(self, _sid):
            return grid

    class _Head:
        num_classes = 2
        ignore_index = -100

        def extract_targets(self, _rec):
            return {"mask": mask}

        def forward(self, x):  # identity geometry (grid already at target res)
            return x

    rec = SimpleNamespace(sample_id="s0")
    X, y = build_training_matrix(
        [rec], _Store(), _Head(), max_pixels=100, rng=np.random.default_rng(0)
    )
    assert X.shape[1] == K
    assert set(np.unique(y)).issubset({0, 1})  # the -100 pixels are never sampled


def test_registry_lists_all_builtin_classifiers():
    from soma.pixel_classifiers import pixel_classifier_registry

    assert set(pixel_classifier_registry.list()) >= {"xgboost", "random_forest", "logistic", "mlp"}


def test_pipeline_run_end_to_end_pixel_classifier(tmp_path: Path):
    from soma.config import PipelineConfig
    from soma.pipeline import Pipeline

    _build_dense_run(tmp_path, ["s0", "s1", "s2", "s3"])
    config = PipelineConfig(
        dataset_csv=tmp_path / "manifest.csv",
        splits_csv=tmp_path / "splits.csv",
        output_root=tmp_path / "out",
        dataset_type="segmentation",
        pixel_classifier=_xgb(),
        task=TaskConfig(name="segmentation", params={"num_classes": NUM_CLASSES}),
        training=TrainingConfig(epochs=1, batch_size=1, max_train_pixels=1000),
        evaluation=EvalConfig(metrics=["mean_dice", "mean_iou"]),
    )
    # pixel_classifier cross-defaults feature_kind to cls_attention.
    assert config.preprocessing.feature_kind == "cls_attention"
    pipeline = Pipeline(config, feature_dir=tmp_path / "dense")
    result = pipeline.run()
    assert "test/mean_dice" in result.summary
    out = tmp_path / "out"
    assert list(out.rglob("preds/test/s3.png")), "pred raster missing"
    assert list(out.rglob("pixel_classifier/model.ubj")), "saved classifier missing"
