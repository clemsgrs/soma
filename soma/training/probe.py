"""Closed-form Ridge+PCA probe — the HEST-benchmark spatial_expression trainer.

Transcribed from HEST-bench's ``hest/bench/trainer.py`` + ``benchmark.py``
(Jaume et al., NeurIPS 2024): a frozen patch encoder predicts a highly-variable-gene
expression vector from a tile, scored by Pearson correlation. The fit is **closed-form**
(no trained head, no tune split): per fold, fit ``StandardScaler`` → PCA on the train
embeddings (X only; the log1p targets are never scaled), solve a multi-output Ridge, then
predict the fold's test spots and score each gene by Pearson correlation pooled over all
test spots.

This is the ``regression`` family's non-gradient sibling trainer, selected from soma's
shared training entry by :data:`PROBE_METHOD` on ``TrainingConfig.method``. The per-fold
fit + score lives here as a pure, GPU-free numeric core so it is verifiable against a
direct sklearn reference; the fold-loop / cache / summary wiring lives in
``soma.pipeline.train_one_probe_fold`` and reuses the shared multi-fold machinery.

Fidelity notes (see soma issue #259 and ``design/hest-benchmark-design.md`` §6):

* PCA is pinned to :data:`DEFAULT_PCA_COMPONENTS` (256) by the shipped benchmark;
  ``random_state=seed`` makes it deterministic.
* Ridge uses ``solver="lsqr"``, ``fit_intercept=False``, ``max_iter=1000`` and the HEST
  penalty rule :func:`ridge_alpha` — ``alpha = 100 / (256 * n_genes)`` with 256 a **fixed
  constant transcribed verbatim**, independent of the PCA ``n_components`` actually used
  (so a small-fixture reference-equivalence test at a smaller PCA dim still matches).
* Aggregation order is per-gene Pearson (pooled over the fold's test spots) → mean over
  genes = the fold score; the mean over folds is done by the shared summary writer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# TrainingConfig.method value that selects this closed-form trainer.
PROBE_METHOD = "ridge_pca_probe"

# HEST pins the PCA latent dim to 256; the HestBenchmark ships this as its latent dim.
DEFAULT_PCA_COMPONENTS = 256

# HEST's Ridge penalty rule uses a PCA-256 constant in the denominator, transcribed
# verbatim from hest/bench/trainer.py. It is deliberately independent of the PCA
# n_components actually used at fit time (issue #259): the shipped benchmark runs at
# 256, and a reference-equivalence test at a smaller PCA dim keeps this constant so the
# probe and its direct-sklearn reference match exactly.
_ALPHA_PCA_CONSTANT = 256
_RIDGE_MAX_ITER = 1000


def ridge_alpha(n_genes: int) -> float:
    """HEST Ridge penalty ``alpha = 100 / (256 * n_genes)`` (``0.0078125`` for 50 genes)."""
    if n_genes < 1:
        raise ValueError(f"n_genes must be >= 1, got {n_genes}")
    return 100.0 / (_ALPHA_PCA_CONSTANT * n_genes)


@dataclass(frozen=True)
class ProbeFoldScore:
    """One fold's probe result.

    ``predictions`` are the per-spot predicted gene vectors ``[n_test, n_genes]``;
    ``per_gene_pearson`` is the ``[n_genes]`` Pearson correlation of each gene pooled over
    all test spots; ``mean_pearson`` is the mean over genes — the fold score the shared
    summary writer then averages over folds into the headline.
    """

    predictions: np.ndarray
    per_gene_pearson: np.ndarray
    mean_pearson: float


def per_gene_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-gene Pearson correlation over the pooled test spots (soma metric convention).

    Returns a ``[n_genes]`` array; a gene whose true or predicted column is constant (an
    undefined correlation) contributes ``0.0`` — the same non-finite → 0.0 rule as
    ``soma.evaluation.metrics._pearson``, so the probe headline is consistent with the
    regression metric elsewhere.
    """
    import warnings

    from scipy.stats import ConstantInputWarning, pearsonr

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"per_gene_pearson shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    n_spots, n_genes = y_true.shape
    out = np.zeros(n_genes, dtype=np.float64)
    if n_spots < 2:
        return out
    with warnings.catch_warnings():
        # A constant gene column (undefined correlation) is expected for sparsely
        # expressed genes; we map it to 0.0 rather than surfacing a warning per gene.
        warnings.simplefilter("ignore", ConstantInputWarning)
        for gene in range(n_genes):
            statistic = pearsonr(y_true[:, gene], y_pred[:, gene]).statistic
            out[gene] = float(statistic) if np.isfinite(statistic) else 0.0
    return out


def fit_predict_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    pca_components: int,
    seed: int,
    alpha: float,
    max_iter: int = _RIDGE_MAX_ITER,
) -> np.ndarray:
    """Fit ``StandardScaler`` → PCA → multi-output Ridge on the train fold, predict test.

    The scaler + PCA are fit on ``x_train`` only and applied verbatim to ``x_test``; the
    targets ``y_train`` are used raw (never scaled). Returns the ``[n_test, n_genes]``
    predictions. This is the exact fit HEST's ``train_test_reg`` performs.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)

    reducer = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_components, random_state=seed)),
        ]
    )
    xr_train = reducer.fit_transform(x_train)
    xr_test = reducer.transform(x_test)
    ridge = Ridge(solver="lsqr", alpha=alpha, fit_intercept=False, max_iter=max_iter)
    ridge.fit(xr_train, y_train)
    return np.asarray(ridge.predict(xr_test), dtype=np.float64)


def score_probe_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    pca_components: int,
    seed: int,
) -> ProbeFoldScore:
    """Fit the closed-form probe on one fold and score it (per-gene → mean over genes).

    ``pca_components`` is the PCA latent dim (256 in the shipped benchmark); the Ridge
    penalty is :func:`ridge_alpha` computed from the gene count (a fixed-256 rule,
    independent of ``pca_components``). Returns the per-spot predictions, the per-gene
    Pearson array, and the mean-over-genes fold score.
    """
    y_train = np.asarray(y_train, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    n_genes = y_train.shape[1]
    predictions = fit_predict_probe(
        x_train,
        y_train,
        x_test,
        pca_components=pca_components,
        seed=seed,
        alpha=ridge_alpha(n_genes),
    )
    gene_pearson = per_gene_pearson(y_test, predictions)
    return ProbeFoldScore(
        predictions=predictions,
        per_gene_pearson=gene_pearson,
        mean_pearson=float(np.mean(gene_pearson)),
    )
