"""The HEST Ridge+PCA probe as a benchmark-side predictor: numerical oracles.

``RidgePCAProbe.fit(...).predict(...)`` must reproduce an independent, inline
transcription of HEST's ``StandardScaler -> PCA -> Ridge`` fit, and ``soma.evaluate``
with ``mean_pearson`` plus the per-gene metrics must reproduce scipy's per-gene Pearson
pooled over the test spots (aggregation order: per gene, then the mean over genes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import soma
from soma.benchmarks.hest.probe import DEFAULT_PCA_COMPONENTS, RidgePCAProbe, gene_pearson, ridge_alpha


def _toy_fold(seed: int, *, n_train: int, n_test: int, feature_dim: int, n_genes: int):
    """Targets are a noisy, non-negative function of the features (so correlated)."""
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((feature_dim, n_genes))
    x_train = rng.standard_normal((n_train, feature_dim))
    x_test = rng.standard_normal((n_test, feature_dim))
    y_train = np.log1p(np.abs(x_train @ weight + 0.1 * rng.standard_normal((n_train, n_genes))))
    y_test = np.log1p(np.abs(x_test @ weight + 0.1 * rng.standard_normal((n_test, n_genes))))
    return x_train, y_train, x_test, y_test


def _reference(x_train, y_train, x_test, *, pca_components, seed, alpha):
    reducer = Pipeline(
        [("scaler", StandardScaler()), ("pca", PCA(n_components=pca_components, random_state=seed))]
    )
    ridge = Ridge(solver="lsqr", alpha=alpha, fit_intercept=False, max_iter=1000)
    ridge.fit(reducer.fit_transform(x_train), y_train)
    return ridge.predict(reducer.transform(x_test))


def test_ridge_alpha_is_the_hest_rule():
    assert ridge_alpha(50) == pytest.approx(0.0078125)  # 100 / (256 * 50)
    assert DEFAULT_PCA_COMPONENTS == 256
    with pytest.raises(ValueError):
        ridge_alpha(0)


def test_probe_predictions_match_a_direct_sklearn_reference():
    x_train, y_train, x_test, _ = _toy_fold(0, n_train=40, n_test=13, feature_dim=16, n_genes=5)
    ids = [f"spot{i}" for i in range(13)]
    predictions = RidgePCAProbe(8, seed=0).fit(x_train, y_train).predict(x_test, ids)
    reference = _reference(x_train, y_train, x_test, pca_components=8, seed=0, alpha=ridge_alpha(5))
    assert predictions.task_family == "regression"
    assert predictions.prediction_columns == tuple(f"value_{k}" for k in range(5))
    assert predictions.sample_ids == ids
    np.testing.assert_allclose(
        predictions.frame[list(predictions.prediction_columns)].to_numpy(), reference, rtol=1e-10, atol=1e-10
    )


def test_evaluate_reports_mean_and_per_gene_pearson():
    genes = ["ERBB2", "ESR1", "PGR", "MKI67"]
    x_train, y_train, x_test, y_test = _toy_fold(1, n_train=40, n_test=17, feature_dim=16, n_genes=4)
    ids = [f"spot{i}" for i in range(17)]
    predictions = RidgePCAProbe(8, seed=1).fit(x_train, y_train).predict(x_test, ids)
    targets = pd.DataFrame(y_test, columns=[f"value_{k}" for k in range(4)], index=pd.Index(ids, name="sample_id"))

    evaluation = soma.evaluate(
        predictions, targets, metrics=["mean_pearson", *(gene_pearson(k, g) for k, g in enumerate(genes))]
    )

    reference = _reference(x_train, y_train, x_test, pca_components=8, seed=1, alpha=ridge_alpha(4))
    per_gene = [pearsonr(y_test[:, k], reference[:, k]).statistic for k in range(4)]
    for k, gene in enumerate(genes):
        assert evaluation.metrics[f"pearson_gene{k:02d}_{gene}"] == pytest.approx(per_gene[k], abs=1e-9)
    assert evaluation.metrics["mean_pearson"] == pytest.approx(float(np.mean(per_gene)), abs=1e-9)


def test_a_constant_gene_scores_zero():
    x_train, y_train, x_test, y_test = _toy_fold(2, n_train=30, n_test=10, feature_dim=8, n_genes=2)
    y_test[:, 1] = 3.0
    ids = [f"s{i}" for i in range(10)]
    predictions = RidgePCAProbe(4).fit(x_train, y_train).predict(x_test, ids)
    targets = pd.DataFrame(y_test, columns=["value_0", "value_1"], index=pd.Index(ids, name="sample_id"))
    evaluation = soma.evaluate(predictions, targets, metrics=[gene_pearson(1, "flat")])
    assert evaluation.metrics["pearson_gene01_flat"] == 0.0


def test_the_probe_refuses_more_components_than_the_fold_rank():
    x_train, y_train, _, _ = _toy_fold(3, n_train=6, n_test=2, feature_dim=8, n_genes=2)
    with pytest.raises(ValueError, match="exceeds the fold's rank"):
        RidgePCAProbe(8).fit(x_train, y_train)
