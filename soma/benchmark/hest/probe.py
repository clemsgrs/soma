"""The HEST-Benchmark closed-form Ridge+PCA probe, as a benchmark-side predictor.

Transcribed from HEST-bench's ``hest/bench/trainer.py`` + ``benchmark.py`` (Jaume et
al., NeurIPS 2024): a frozen patch encoder predicts a highly-variable-gene expression
vector from a tile, scored by Pearson correlation. The fit is closed-form (no trained
head, no tune split): ``StandardScaler`` -> PCA on the train embeddings (X only; the
log1p targets are never scaled), then a multi-output Ridge.

:class:`RidgePCAProbe` speaks soma's verbs: ``fit(features, targets)`` then
``predict(features, sample_ids)`` returns :class:`~soma.evaluation.Predictions` with one
``value_<k>`` column per gene, which :func:`soma.evaluate` scores with ``mean_pearson``
and the per-gene :func:`gene_pearson` metrics.

Fidelity notes (soma issue #259):

* PCA is pinned to :data:`DEFAULT_PCA_COMPONENTS` (256) by the shipped benchmark;
  ``random_state=seed`` makes it deterministic.
* Ridge uses ``solver="lsqr"``, ``fit_intercept=False``, ``max_iter=1000`` and the HEST
  penalty rule :func:`ridge_alpha` — ``alpha = 100 / (256 * n_genes)`` with 256 a fixed
  constant transcribed verbatim, independent of the PCA dim actually used.
* Aggregation order is per-gene Pearson (pooled over the fold's test spots) -> mean over
  genes = the fold score; the mean over folds is the summary's.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = ["DEFAULT_PCA_COMPONENTS", "RidgePCAProbe", "gene_pearson", "ridge_alpha"]

#: HEST pins the PCA latent dim to 256.
DEFAULT_PCA_COMPONENTS = 256
# HEST's Ridge penalty rule uses a PCA-256 constant in the denominator, transcribed
# verbatim; deliberately independent of the PCA n_components actually used.
_ALPHA_PCA_CONSTANT = 256
_RIDGE_MAX_ITER = 1000


def ridge_alpha(n_genes: int) -> float:
    """HEST Ridge penalty ``alpha = 100 / (256 * n_genes)`` (``0.0078125`` for 50 genes)."""
    if n_genes < 1:
        raise ValueError(f"n_genes must be >= 1, got {n_genes}")
    return 100.0 / (_ALPHA_PCA_CONSTANT * n_genes)


class RidgePCAProbe:
    """``StandardScaler`` -> PCA -> multi-output Ridge, fit on the train fold only.

    Args:
        pca_components: PCA latent dim (256 in the shipped benchmark).
        alpha: Ridge penalty; ``None`` applies :func:`ridge_alpha` to the gene count.
        seed: PCA ``random_state``.
    """

    def __init__(
        self, pca_components: int = DEFAULT_PCA_COMPONENTS, alpha: float | None = None, *, seed: int = 0
    ) -> None:
        self.pca_components = int(pca_components)
        self.alpha = None if alpha is None else float(alpha)
        self.seed = int(seed)
        self._reducer = None
        self._ridge = None
        self._num_outputs: int | None = None

    def fit(self, features, targets) -> "RidgePCAProbe":
        """Fit on ``features`` ``(n, D)`` and raw ``targets`` ``(n, G)`` (array or frame)."""
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if y.ndim != 2 or y.shape[0] != x.shape[0]:
            raise ValueError(f"targets must be (n, G) aligned with features (n={x.shape[0]}), got {y.shape}")
        rank = min(x.shape)
        if self.pca_components > rank:
            raise ValueError(
                f"PCA n_components={self.pca_components} exceeds the fold's rank "
                f"min(n_train={x.shape[0]}, feature_dim={x.shape[1]})={rank}; lower "
                "pca_components or provide more train spots."
            )
        self._reducer = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=self.pca_components, random_state=self.seed)),
            ]
        )
        reduced = self._reducer.fit_transform(x)
        alpha = ridge_alpha(y.shape[1]) if self.alpha is None else self.alpha
        self._ridge = Ridge(solver="lsqr", alpha=alpha, fit_intercept=False, max_iter=_RIDGE_MAX_ITER)
        self._ridge.fit(reduced, y)
        self._num_outputs = int(y.shape[1])
        return self

    def predict_array(self, features) -> np.ndarray:
        if self._ridge is None:
            raise RuntimeError("fit the probe before predicting")
        reduced = self._reducer.transform(np.asarray(features, dtype=np.float64))
        return np.asarray(self._ridge.predict(reduced), dtype=np.float64).reshape(len(reduced), -1)

    def predict(self, features, sample_ids: Sequence[str]):
        """``Predictions`` (regression family) with one ``value_<k>`` column per output."""
        from soma.evaluation import Predictions

        values = self.predict_array(features)
        ids = [str(sid) for sid in sample_ids]
        if len(ids) != values.shape[0]:
            raise ValueError(f"{len(ids)} sample ids for {values.shape[0]} feature rows")
        columns = [f"value_{k}" for k in range(values.shape[1])]
        frame = pd.DataFrame(values, columns=columns)
        frame.insert(0, "sample_id", ids)
        return Predictions(frame, task_family="regression", prediction_columns=columns)


def gene_pearson(index: int, name: str):
    """A frame metric: Pearson of output ``value_<index>`` pooled over the samples, named
    ``pearson_gene<index:02d>_<name>`` (a constant column scores ``0.0``)."""
    from soma.evaluation.metrics import pearson

    column = f"value_{index}"

    def metric(predictions: pd.DataFrame, targets: pd.DataFrame) -> float:
        return pearson(targets[column].to_numpy(dtype=float), predictions[column].to_numpy(dtype=float))

    metric.__name__ = f"pearson_gene{index:02d}_{name}"
    return metric
