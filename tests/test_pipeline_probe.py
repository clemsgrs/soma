"""End-to-end wiring of the closed-form probe through soma's shared training entry (#259).

No GPU, no encoder weights, no HEST download: a curated ``spatial_expression`` Manifest +
per-spot feature ``.pt`` files stand in for the real extraction, so this drives the REAL
multi-fold ``train()`` loop → ``train_one_probe_fold`` → the shared ``summary.json`` writer
and asserts a finite fold-averaged mean-Pearson plus per-gene detail. The real encoder run
(slide2vec weights + the scoped HEST download) is a manual vertical-slice reproduction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from soma.config import EvalConfig, TaskConfig, TrainingConfig
from soma.curation.manifest import write_manifest
from soma.dataset import Splits, load_manifest
from soma.features import FeatureStore
from soma.pipeline import train

FEATURE_DIM = 8
N_GENES = 4
PCA_COMPONENTS = 4
GENES = ["GENEA", "GENEB", "GENEC", "GENED"]


def _build_manifest_and_features(tmp_path: Path):
    """Curate a 16-spot, 2-fold spatial_expression Manifest + correlated feature files."""
    rng = np.random.default_rng(0)
    n_spots = 16
    weight = rng.standard_normal((FEATURE_DIM, N_GENES))
    features = rng.standard_normal((n_spots, FEATURE_DIM))
    targets = np.log1p(
        np.abs(features @ weight + 0.1 * rng.standard_normal((n_spots, N_GENES)))
    )

    manifest_dir = tmp_path / "curated"
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True)

    dataset_rows = []
    sample_ids = []
    for i in range(n_spots):
        sid = f"spot{i:02d}"
        sample_ids.append(sid)
        dataset_rows.append(
            {
                "sample_id": sid,
                "image_path": str(manifest_dir / "tiles" / f"{sid}.png"),
                "target_index": i,
            }
        )
        torch.save(torch.tensor(features[i], dtype=torch.float32), feature_dir / f"{sid}.pt")

    # Two folds that need not partition: each fold has train>=10, test>=6 (>= PCA dim).
    split_rows = []
    for sid in sample_ids[:10]:
        split_rows.append({"sample_id": sid, "split": "train", "fold": 0})
    for sid in sample_ids[10:]:
        split_rows.append({"sample_id": sid, "split": "test", "fold": 0})
    for sid in sample_ids[6:]:
        split_rows.append({"sample_id": sid, "split": "train", "fold": 1})
    for sid in sample_ids[:6]:
        split_rows.append({"sample_id": sid, "split": "test", "fold": 1})

    manifest = write_manifest(
        manifest_dir,
        dataset_type="spatial_expression",
        dataset_rows=dataset_rows,
        split_rows=split_rows,
        summary={"dataset": "toy"},
        target_matrix=targets,
        genes=GENES,
    )
    return manifest, feature_dir


def _run_probe(tmp_path: Path):
    manifest, feature_dir = _build_manifest_and_features(tmp_path)
    dataset = load_manifest(manifest.dataset_csv, "spatial_expression")
    splits = Splits(manifest.splits_csv, dataset)
    store = FeatureStore(feature_dir)
    run_dir = tmp_path / "run"
    result = train(
        feature_store=store,
        dataset=dataset,
        splits=splits,
        task=TaskConfig(name="regression", params={"pca_components": PCA_COMPONENTS}),
        training=TrainingConfig(method="ridge_pca_probe", seed=0),
        evaluation=EvalConfig(metrics=["pearson"]),
        dataset_type="spatial_expression",
        run_dir=run_dir,
    )
    return result, run_dir


def test_probe_run_writes_finite_fold_averaged_headline(tmp_path):
    result, run_dir = _run_probe(tmp_path)
    summary = json.loads((run_dir / "summary.json").read_text())
    # Multi-fold headline: the fold-averaged mean-Pearson.
    assert "test/mean_pearson_mean" in summary
    assert np.isfinite(summary["test/mean_pearson_mean"])
    assert -1.0 <= summary["test/mean_pearson_mean"] <= 1.0
    assert result.summary["test/mean_pearson_mean"] == summary["test/mean_pearson_mean"]


def test_probe_headline_is_mean_over_folds_of_per_fold_mean_pearson(tmp_path):
    _, run_dir = _run_probe(tmp_path)
    summary = json.loads((run_dir / "summary.json").read_text())
    fold_scores = []
    for fold in range(2):
        metrics = json.loads((run_dir / f"fold_{fold}" / "metrics.json").read_text())
        fold_scores.append(metrics["test"]["mean_pearson"])
    assert summary["test/mean_pearson_mean"] == float(np.mean(fold_scores))


def test_probe_retains_per_gene_mean_std_detail(tmp_path):
    _, run_dir = _run_probe(tmp_path)
    summary = json.loads((run_dir / "summary.json").read_text())
    for index, gene in enumerate(GENES):
        key = f"test/pearson_gene{index:02d}_{gene}"
        assert f"{key}_mean" in summary, f"missing per-gene detail {key}_mean"
        assert f"{key}_std" in summary
        assert np.isfinite(summary[f"{key}_mean"])


def test_probe_writes_per_spot_predictions(tmp_path):
    _, run_dir = _run_probe(tmp_path)
    preds = pd.read_csv(run_dir / "fold_0" / "predictions_test.csv")
    assert list(preds.columns) == ["sample_id", *GENES]
    assert len(preds) == 6  # fold 0 test spots
    assert preds[GENES].to_numpy().dtype.kind == "f"


def test_probe_reuses_one_feature_store_across_folds(tmp_path):
    # Embeddings are extracted once and reused across folds: a single FeatureStore instance
    # serves every fold (the shared cache), never re-extracting per fold.
    manifest, feature_dir = _build_manifest_and_features(tmp_path)
    dataset = load_manifest(manifest.dataset_csv, "spatial_expression")
    splits = Splits(manifest.splits_csv, dataset)
    assert splits.num_folds == 2

    store = FeatureStore(feature_dir)
    loaded_ids: list[str] = []
    original_load = store.load

    def _tracking_load(sample_id):
        loaded_ids.append(sample_id)
        return original_load(sample_id)

    store.load = _tracking_load  # type: ignore[method-assign]
    train(
        feature_store=store,
        dataset=dataset,
        splits=splits,
        task=TaskConfig(name="regression", params={"pca_components": PCA_COMPONENTS}),
        training=TrainingConfig(method="ridge_pca_probe", seed=0),
        evaluation=EvalConfig(metrics=["pearson"]),
        dataset_type="spatial_expression",
        run_dir=tmp_path / "run",
    )
    # Every spot is loaded (once per fold membership); the same store served both folds.
    assert set(loaded_ids) == set(dataset.sample_ids)
