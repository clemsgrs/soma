"""The grid fold on detection, offline: pre-extracted dense grids + point files on disk.

``train_one_grid_fold`` builds the PointSource, the head (µm distances resolved at the
grids' spacing), the decoder and ``DenseModel``, fits, predicts every peak above
``min_score``, sweeps per-class thresholds on tune (or scores at ``validation_threshold``
without a tune split) and evaluates. No encoder / GPU needed: the grids are written
directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from soma.config import DecoderConfig, EvalConfig, PreprocessingConfig, TaskConfig, TrainingConfig
from soma.data._legacy import legacy_folds_from_csv, legacy_samples_from_csv
from soma.dense import CachedGridSource
from soma.dense.geometry import compute_dense_geometry
from soma.dense.store import dense_grid_metadata, write_dense_grid
from soma.evaluation import PointPredictions
from soma.pipeline import train_one_grid_fold

NUM_CLASSES = 2
TARGET = 16
PATCH = 4
FEATURE_DIM = 4
# Grids are read flat at this spacing: level-0 == frame, and µm distances divide by it.
SPACING = 0.2  # µm/px; 0.6 µm -> 3 px (δ), 0.3 µm -> 1.5 px (σ)
PARAMS = {"num_classes": NUM_CLASSES, "match_distance": 0.6, "sigma": 0.3}


def _build(root: Path, sample_ids: list[str], *, spacings: dict[str, float] | None = None):
    dense_dir = root / "dense"
    points_dir = root / "points"
    dense_dir.mkdir(parents=True)
    points_dir.mkdir()
    geometry = compute_dense_geometry(target_size=TARGET, patch_size=PATCH)
    rows = []
    for sid in sample_ids:
        spacing = (spacings or {}).get(sid, SPACING)
        meta = dense_grid_metadata(geometry, feature_dim=FEATURE_DIM, pad_mode="reflect", spacing_um=spacing)
        meta.update(source_spacing_um=spacing, effective_spacing_um=spacing)
        write_dense_grid(dense_dir, sid, torch.randn(FEATURE_DIM, *geometry.grid_shape), meta)
        points = points_dir / f"{sid}.csv"
        points.write_text("x,y,class\n4,4,0\n11,11,1\n")
        rows.append(f"{sid},{sid}.jpg,{points},{spacing}")
    manifest_csv = root / "manifest.csv"
    manifest_csv.write_text("sample_id,image_path,points_path,spacing_at_level_0\n" + "\n".join(rows) + "\n")
    manifest = legacy_samples_from_csv(manifest_csv)
    return manifest, CachedGridSource(dense_dir)


def _splits(root: Path, manifest, assign: dict[str, str]):
    path = root / "splits.csv"
    path.write_text("sample_id,split,fold\n" + "\n".join(f"{s},{v},0" for s, v in assign.items()) + "\n")
    return legacy_folds_from_csv(path, manifest).folds[0]


def _run(tmp_path, manifest, store, fold_split, *, params=PARAMS, training=None, evaluation=None):
    return train_one_grid_fold(
        feature_store=store,
        dataset=manifest,
        fold_split=fold_split,
        task=TaskConfig(name="detection", params=dict(params)),
        training=training or TrainingConfig(epochs=2, batch_size=2),
        fold_dir=tmp_path / "fold",
        decoder=DecoderConfig(name="lightweight_conv"),
        dataset_type="detection",
        evaluation=evaluation or EvalConfig(metrics=["mean_f1", "f1_per_class"]),
        preprocessing=PreprocessingConfig(requested_spacing_um=SPACING, requested_tile_size_px=TARGET),
    )


IDS = ["s0", "s1", "s2", "s3"]
ASSIGN = {"s0": "train", "s1": "train", "s2": "tune", "s3": "test"}


def test_grid_fold_scores_detection_from_persisted_point_predictions(tmp_path: Path):
    manifest, store = _build(tmp_path / "data", IDS)
    result = _run(tmp_path, manifest, store, _splits(tmp_path, manifest, ASSIGN))

    fold = tmp_path / "fold"
    assert "mean_f1" in result.tune_evaluation.metrics
    assert "mean_f1" in result.test_evaluations["test"].metrics
    assert not (fold / "detection_thresholds.json").exists()
    test = PointPredictions.from_parquet(fold / "predictions_test.parquet")
    tune = PointPredictions.from_parquet(fold / "predictions_tune.parquet")
    assert test.sample_ids == ["s3"] and tune.sample_ids == ["s2"]
    assert test.match_distance_px == pytest.approx(3.0)
    assert json.loads((fold / "metrics.json").read_text()).keys() == {"tune", "test"}


def test_grid_fold_without_a_tune_split_scores_at_the_validation_threshold(tmp_path: Path, monkeypatch):
    """Train stands in for tune, so a sweep would pick an in-sample cut: no sweep runs."""
    import soma.evaluation

    manifest, store = _build(tmp_path / "data", IDS)
    fold_split = _splits(tmp_path, manifest, {"s0": "train", "s1": "train", "s2": "train", "s3": "test"})

    def no_sweep(*args, **kwargs):  # pragma: no cover - the behaviour under test
        raise AssertionError("swept a threshold without a tune split")

    monkeypatch.setattr(soma.evaluation, "sweep_threshold", no_sweep)
    result = _run(
        tmp_path, manifest, store, fold_split,
        params={**PARAMS, "validation_threshold": 0.7},
        training=TrainingConfig(epochs=1, batch_size=2, allow_missing_tune=True),
    )
    assert "mean_f1" in result.test_evaluations["test"].metrics


def test_grid_fold_folds_the_class_scheme_into_its_point_source(tmp_path: Path):
    """``classes`` / ``drop``: the fixture's two ids become one named class."""
    manifest, store = _build(tmp_path / "data", IDS)
    result = _run(
        tmp_path, manifest, store, _splits(tmp_path, manifest, ASSIGN),
        params={"classes": {"cell": [0]}, "drop": [1], "match_distance": 0.6, "sigma": 0.3},
        training=TrainingConfig(epochs=1, batch_size=2),
    )
    assert set(result.test_evaluations["test"].metrics) == {"mean_f1", "f1_class_0"}
    predictions = PointPredictions.from_parquet(tmp_path / "fold" / "predictions_test.parquet")
    assert predictions.classes == ("cell",)


def test_grid_fold_holdout_test_never_touches_the_test_split(tmp_path: Path):
    manifest, store = _build(tmp_path / "data", IDS)
    result = _run(
        tmp_path, manifest, store, _splits(tmp_path, manifest, ASSIGN),
        evaluation=EvalConfig(metrics=["mean_f1"], holdout_test=True),
        training=TrainingConfig(epochs=1, batch_size=2),
    )
    assert result.test_evaluations == {}
    assert not (tmp_path / "fold" / "predictions_test.parquet").exists()


def test_grid_fold_refuses_detection_over_several_effective_spacings(tmp_path: Path):
    manifest, store = _build(tmp_path / "data", IDS, spacings={"s3": 0.4})
    with pytest.raises(ValueError, match="one effective spacing"):
        _run(tmp_path, manifest, store, _splits(tmp_path, manifest, ASSIGN))


def test_grid_fold_requires_num_classes(tmp_path: Path):
    manifest, store = _build(tmp_path / "data", IDS)
    with pytest.raises(ValueError, match="num_classes"):
        _run(tmp_path, manifest, store, _splits(tmp_path, manifest, ASSIGN), params={"match_distance": 0.6})


def test_grid_fold_writes_opt_in_heatmaps_and_point_overlays(tmp_path: Path):
    from PIL import Image

    _, store = _build(tmp_path / "data", IDS)
    rows = []
    for sid in IDS:  # real tiles, so the overlays can render
        Image.fromarray(np.full((TARGET, TARGET, 3), 127, dtype=np.uint8)).save(tmp_path / f"{sid}.png")
        rows.append(f"{sid},{tmp_path / f'{sid}.png'},{tmp_path / 'data' / 'points' / f'{sid}.csv'},{SPACING}")
    manifest_csv = tmp_path / "manifest_with_tiles.csv"
    manifest_csv.write_text("sample_id,image_path,points_path,spacing_at_level_0\n" + "\n".join(rows) + "\n")
    manifest = legacy_samples_from_csv(manifest_csv)
    _run(
        tmp_path, manifest, store, _splits(tmp_path, manifest, ASSIGN),
        evaluation=EvalConfig(metrics=["mean_f1"], save_detection_heatmaps=True),
        training=TrainingConfig(epochs=1, batch_size=2),
    )
    fold = tmp_path / "fold"
    heatmap = np.load(fold / "heatmaps" / "test" / "s3.npz")["heatmap"]
    assert heatmap.shape == (NUM_CLASSES, TARGET, TARGET) and heatmap.dtype == np.float16
    assert (fold / "overlays" / "test" / "pred" / "s3.png").is_file()
    assert (fold / "overlays" / "test" / "gt" / "s3.png").is_file()
