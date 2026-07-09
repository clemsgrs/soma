"""Closed-form Ridge+PCA probe — reference-equivalence + aggregation-order tests (#259).

The core CI test for the HEST-benchmark probe: on a small in-memory feature store +
target matrix + fold split it reproduces the per-gene Pearson and the fold-averaged
headline of a **direct sklearn reference** (StandardScaler → PCA → Ridge with the HEST
alpha rule). No GPU, no encoder weights, no HEST download — pure numerics.

PCA-256 needs >=256 samples/features, which a tiny fixture cannot supply, so the
reference-equivalence tests run at a smaller PCA dim the fixture supports. The Ridge alpha
rule stays the shipped ``100 / (256 * n_genes)`` (a fixed 256 constant, independent of the
PCA dim) so equivalence holds exactly — this is transcribed verbatim from HEST.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from soma.training.probe import (
    DEFAULT_PCA_COMPONENTS,
    PROBE_METHOD,
    fit_predict_probe,
    per_gene_pearson,
    ridge_alpha,
    score_probe_fold,
)


def _toy_fold(seed: int, *, n_train: int, n_test: int, feature_dim: int, n_genes: int):
    """A toy fold whose targets are a noisy linear map of the features (correlated)."""
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((feature_dim, n_genes))
    x_train = rng.standard_normal((n_train, feature_dim))
    x_test = rng.standard_normal((n_test, feature_dim))
    # log1p-like non-negative targets, correlated with X plus noise (never scaled).
    y_train = np.log1p(np.abs(x_train @ weight + 0.1 * rng.standard_normal((n_train, n_genes))))
    y_test = np.log1p(np.abs(x_test @ weight + 0.1 * rng.standard_normal((n_test, n_genes))))
    return x_train, y_train, x_test, y_test


def _reference_predictions(x_train, y_train, x_test, *, pca_components, seed, alpha):
    """An independent, inline transcription of HEST's StandardScaler→PCA→Ridge fit."""
    reducer = Pipeline(
        [("scaler", StandardScaler()), ("pca", PCA(n_components=pca_components, random_state=seed))]
    )
    xr_train = reducer.fit_transform(np.asarray(x_train, dtype=np.float64))
    xr_test = reducer.transform(np.asarray(x_test, dtype=np.float64))
    ridge = Ridge(solver="lsqr", alpha=alpha, fit_intercept=False, max_iter=1000)
    ridge.fit(xr_train, np.asarray(y_train, dtype=np.float64))
    return np.asarray(ridge.predict(xr_test), dtype=np.float64)


# --- the alpha rule (transcribed verbatim) --------------------------------------------


def test_ridge_alpha_matches_hest_rule():
    assert ridge_alpha(50) == pytest.approx(0.0078125)  # 100 / (256 * 50)
    assert ridge_alpha(3) == pytest.approx(100.0 / (256 * 3))
    with pytest.raises(ValueError):
        ridge_alpha(0)


def test_default_pca_components_is_256():
    assert DEFAULT_PCA_COMPONENTS == 256
    assert PROBE_METHOD == "ridge_pca_probe"


# --- reference equivalence (the core AC) ----------------------------------------------


def test_fit_predict_probe_matches_direct_sklearn_reference():
    pca_components, seed, n_genes = 8, 0, 5
    x_train, y_train, x_test, _ = _toy_fold(
        seed, n_train=40, n_test=13, feature_dim=16, n_genes=n_genes
    )
    got = fit_predict_probe(
        x_train, y_train, x_test,
        pca_components=pca_components, seed=seed, alpha=ridge_alpha(n_genes),
    )
    ref = _reference_predictions(
        x_train, y_train, x_test,
        pca_components=pca_components, seed=seed, alpha=ridge_alpha(n_genes),
    )
    assert got.shape == (13, n_genes)
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)


def test_score_probe_fold_reproduces_per_gene_pearson_and_headline():
    pca_components, seed, n_genes = 8, 1, 5
    x_train, y_train, x_test, y_test = _toy_fold(
        seed, n_train=40, n_test=17, feature_dim=16, n_genes=n_genes
    )
    result = score_probe_fold(
        x_train, y_train, x_test, y_test, pca_components=pca_components, seed=seed
    )

    # Independent reference: same fit, then per-gene scipy pearson pooled over test spots.
    ref_pred = _reference_predictions(
        x_train, y_train, x_test,
        pca_components=pca_components, seed=seed, alpha=ridge_alpha(n_genes),
    )
    ref_per_gene = np.array(
        [pearsonr(y_test[:, g], ref_pred[:, g]).statistic for g in range(n_genes)]
    )

    assert result.per_gene_pearson.shape == (n_genes,)
    np.testing.assert_allclose(result.per_gene_pearson, ref_per_gene, rtol=1e-9, atol=1e-9)
    # Headline = mean over genes of the per-gene Pearson (the fold score).
    assert result.mean_pearson == pytest.approx(float(np.mean(ref_per_gene)))


def test_fold_averaged_headline_is_mean_over_folds_of_mean_over_genes():
    # Aggregation order: per-gene → mean over genes (fold score) → mean over folds.
    pca_components, n_genes = 8, 4
    fold_scores = []
    for fold in range(3):
        x_train, y_train, x_test, y_test = _toy_fold(
            fold + 10, n_train=36, n_test=15, feature_dim=16, n_genes=n_genes
        )
        result = score_probe_fold(
            x_train, y_train, x_test, y_test, pca_components=pca_components, seed=0
        )
        # The per-fold score is the mean over genes.
        assert result.mean_pearson == pytest.approx(float(np.mean(result.per_gene_pearson)))
        fold_scores.append(result.mean_pearson)
    headline = float(np.mean(fold_scores))
    assert np.isfinite(headline)


def test_per_gene_pearson_zero_for_constant_column():
    # A constant predicted column has undefined correlation -> 0.0 (soma metric convention).
    y_true = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
    y_pred = np.array([[5.0, 0.0], [5.0, 9.0], [5.0, 1.0], [5.0, 7.0]])  # col 0 constant
    result = per_gene_pearson(y_true, y_pred)
    assert result[0] == 0.0
    assert np.isfinite(result[1])


def test_score_probe_fold_is_deterministic():
    x_train, y_train, x_test, y_test = _toy_fold(
        7, n_train=32, n_test=11, feature_dim=12, n_genes=4
    )
    a = score_probe_fold(x_train, y_train, x_test, y_test, pca_components=6, seed=3)
    b = score_probe_fold(x_train, y_train, x_test, y_test, pca_components=6, seed=3)
    np.testing.assert_array_equal(a.per_gene_pearson, b.per_gene_pearson)
    assert a.mean_pearson == b.mean_pearson
