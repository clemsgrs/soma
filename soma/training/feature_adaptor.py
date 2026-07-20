"""The feature adaptor — a buffer-carrying front module over frozen encoder features.

Frozen encoders span 768→4608 dimensions with very different activation scales, so a
shared aggregator/head and its single externally-calibrated learning rate do not see
comparable inputs across encoders. The adaptor closes that gap: it sits at the head of
the model, ahead of the aggregator/head, and applies a composition of independently
optional stages to the features before anything trainable sees them.

Two stages exist, composed in a fixed order — **normalize → project** — and each is
independently optional:

* **normalize** (``normalization``, issue #283) puts every encoder's activations on a
  comparable scale.
* **project** (``projection``, issue #284) maps every encoder to a common *width*. A
  wider embedding means a larger aggregator, so "smaller encoders are more
  label-efficient" can be manufactured from dimensionality alone; projecting to a shared
  ``target_dim`` by a label-free map is the dim-matched ablation that removes that
  confound.

Three invariants define this module:

* **Fitted state lives in buffers, not parameters.** ``zscore``'s center/scale and the
  projection's mean/matrix are registered buffers, so the optimizer never sees them (they
  are not weights to learn — they are an estimate of the Support distribution) while they
  still ride in the checkpoint and are re-applied verbatim by the final-checkpoint test
  pass. The projection in particular must stay frozen: a *learned* projection would
  relocate the very capacity confound it exists to remove.
* **Fitting is leak-free.** :meth:`FeatureAdaptor.fit` is the only thing that moves the
  buffers, and the pipeline calls it with the Support (train) split's features alone.
  Running tune/test features through :meth:`FeatureAdaptor.forward` never updates them.
* **The output width is the adaptor's to declare.** With a projection active
  :attr:`FeatureAdaptor.output_dim` is ``target_dim`` rather than the encoder's native
  dim, and the aggregator/head are constructed against it — the *dim rewire* that
  equalizes downstream trainable capacity across a roster.

The adaptor is constructed **only when at least one stage is active**. With every section
``none`` the module is absent entirely and the model is structurally identical to a run
that predates this module — soma's "omitted section = absent" convention, and the anchor
for byte-identical legacy runs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from soma.config import NormalizationConfig, ProjectionConfig

__all__ = [
    "FEATURE_ADAPTER_FILENAME",
    "FeatureAdaptor",
    "build_feature_adaptor",
    "feature_adaptor_output_dim",
    "write_feature_adapter_sidecar",
]

#: QC sidecar filename written next to a fold's checkpoint.
FEATURE_ADAPTER_FILENAME = "feature_adapter.json"

#: Normalization methods whose transform needs statistics estimated from the Support set.
_FITTED_METHODS = frozenset({"zscore"})

#: Projection methods whose map needs to be estimated from the Support set.
_FITTED_PROJECTIONS = frozenset({"pca"})


def _random_projection_matrix(
    *, num_features: int, target_dim: int, seed: int, encoder_identity: str
) -> Tensor:
    """The fixed Gaussian map for ``projection: random``.

    Seeded from ``projection.seed`` combined with the **encoder identity and the in/out
    dims**, so two encoders in one roster never share a matrix while each one's matrix is
    reproducible from the config alone. Drawn from a private
    :class:`torch.Generator` — never the global RNG — so the map is identical across
    training trajectories no matter what seeding the training loop does around it.

    Entries are ``N(0, 1/target_dim)``, the scaling under which ``E[<Rx, Ry>] = <x, y>``:
    inner products (and so distances) are approximately preserved.
    """
    material = f"{seed}|{encoder_identity}|{num_features}|{target_dim}".encode("utf-8")
    derived = int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big")
    generator = torch.Generator().manual_seed(derived % (2**63 - 1))
    draw = torch.randn(
        (target_dim, num_features), generator=generator, dtype=torch.float64
    )
    return draw / math.sqrt(target_dim)


class FeatureAdaptor(nn.Module):
    """Applies the active adaptor stages to features before the aggregator/head.

    Accepts any tensor whose **last** axis is the feature dimension — ``(N, D)`` tiles,
    ``(B, N, D)`` MIL bags, ``(B, D)`` single embeddings — and returns the same shape with
    that last axis replaced by :attr:`output_dim`, so the same module serves every path.
    Without a projection ``output_dim == D`` and the shape is unchanged.

    The stages compose in a fixed order — **normalize → project** — and are independently
    optional: standardize without projecting, project without standardizing, or both.

    Args:
        method: Normalization method (``none`` | ``zscore`` | ``l2`` | ``layernorm``).
            ``none`` means the normalize stage is absent.
        eps: Floor on the divisor, so a constant or near-constant channel cannot blow up.
        num_features: Feature dimension ``D``, fixing the buffer shapes up front so a
            checkpoint loads into an unfitted module.
        projection: Label-free projection to a common width, or ``None``/``none`` for no
            projection stage.
        encoder_identity: The encoder this adaptor sits behind. Only ``random`` uses it —
            it seeds that encoder's matrix, so two encoders in a roster never share one.
    """

    def __init__(
        self,
        *,
        method: str = "none",
        eps: float = 1e-6,
        num_features: int,
        projection: ProjectionConfig | None = None,
        encoder_identity: str = "",
    ) -> None:
        super().__init__()
        projection = projection or ProjectionConfig()
        if method == "none" and projection.method == "none":
            raise ValueError(
                "FeatureAdaptor is never constructed with every stage off — with no "
                "active stage the module must be absent. Use build_feature_adaptor()."
            )
        if num_features < 1:
            raise ValueError(
                f"FeatureAdaptor requires a positive feature dimension, got {num_features}."
            )
        self.method = str(method)
        self.eps = float(eps)
        self.num_features = int(num_features)
        self.projection_method = str(projection.method)
        self.projection_seed = int(projection.seed)
        self.encoder_identity = str(encoder_identity)
        self.target_dim = (
            int(projection.target_dim)
            if projection.target_dim is not None
            else self.num_features
        )
        # PCA preflight, half 1: `target_dim <= D` is knowable before a single row is
        # read, so refuse here rather than after a full pass over the Support set. Random
        # projection is unconstrained — it may expand as well as reduce.
        if self.projection_method == "pca" and self.target_dim > self.num_features:
            raise ValueError(
                f"projection.target_dim={self.target_dim} exceeds the encoder's feature "
                f"dimension D={self.num_features}: PCA has at most D components. Lower "
                "target_dim to at most D, or use projection.method='random' to expand."
            )
        # Buffers, not parameters: fitted state the optimizer must never touch, but which
        # must ride in the checkpoint. Identity-initialized so an unfitted adaptor is a
        # well-defined no-op and a state_dict loads into it.
        self.register_buffer("center", torch.zeros(self.num_features))
        self.register_buffer("scale", torch.ones(self.num_features))
        self.register_buffer("eps_floored", torch.zeros((), dtype=torch.long))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))
        # The projection is frozen too — a buffer, not a learned layer, so it cannot
        # reintroduce or relocate the capacity confound it exists to remove.
        self.register_buffer("projection_mean", torch.zeros(self.num_features))
        self.register_buffer(
            "projection_matrix", torch.zeros(self.target_dim, self.num_features)
        )
        self.register_buffer(
            "explained_variance_ratio", torch.zeros(self.target_dim)
        )
        self.register_buffer(
            "projection_n_fit_samples", torch.zeros((), dtype=torch.long)
        )
        self.register_buffer("projection_fitted", torch.zeros((), dtype=torch.bool))
        if self.projection_method == "random":
            # Label-free *and* data-free: the matrix is fully determined by the config,
            # so it is drawn at construction and never touches the Support set at all.
            self.projection_matrix.copy_(
                _random_projection_matrix(
                    num_features=self.num_features,
                    target_dim=self.target_dim,
                    seed=self.projection_seed,
                    encoder_identity=self.encoder_identity,
                ).to(self.projection_matrix.dtype)
            )
            self.projection_fitted.fill_(True)

    @property
    def output_dim(self) -> int:
        """The width the aggregator/head must be built against — the dim rewire.

        With a projection active this is ``target_dim``, *not* the encoder's native dim,
        which is what equalizes downstream trainable capacity across a roster of encoders
        of differing width.
        """
        return self.target_dim if self.projects else self.num_features

    @property
    def projects(self) -> bool:
        return self.projection_method != "none"

    @property
    def requires_fit(self) -> bool:
        """Whether this adaptor needs :meth:`fit` before it can transform."""
        return (
            self.method in _FITTED_METHODS
            or self.projection_method in _FITTED_PROJECTIONS
        )

    @property
    def is_fitted(self) -> bool:
        normalization_ready = self.method not in _FITTED_METHODS or bool(
            self.fitted.item()
        )
        projection_ready = self.projection_method not in _FITTED_PROJECTIONS or bool(
            self.projection_fitted.item()
        )
        return normalization_ready and projection_ready

    @property
    def n_fit_samples(self) -> int:
        """Feature rows the fitted projection was estimated from (QC signal)."""
        return int(self.projection_n_fit_samples.item())

    @property
    def num_eps_floored(self) -> int:
        """How many channels had their scale clamped up to ``eps`` (QC signal)."""
        return int(self.eps_floored.item())

    def fit(self, batches: Iterable[Tensor]) -> "FeatureAdaptor":
        """Estimate the fitted state from ``batches`` of Support-set features.

        ``batches`` is any iterable of tensors whose last axis is the feature dimension —
        one per Support sample for the MIL path, so a whole cohort's tiles never has to be
        materialized at once. Statistics are accumulated in float64 for numerical stability
        across the millions of tiles a real cohort has.

        Stages are fit **in pipeline order**: the projection is estimated from features
        that have already been normalized, matching what ``forward`` will feed it. That
        costs a second pass, so ``batches`` must be **re-iterable** (a sequence, or an
        object whose ``__iter__`` restarts the stream) whenever both a fitted
        normalization and a fitted projection are active — a one-shot generator would
        arrive at the projection pass already exhausted.

        A stage with nothing to fit (``l2``/``layernorm``/``random``) is skipped; calling
        this is a harmless no-op so callers need not special-case the method.
        """
        if not self.requires_fit:
            return self
        needs_two_passes = (
            self.method in _FITTED_METHODS
            and self.projection_method in _FITTED_PROJECTIONS
        )
        if needs_two_passes and iter(batches) is batches:
            # A one-shot iterator would arrive at the projection pass exhausted, and the
            # PCA preflight would then blame the Support set size for what is really an
            # exhausted stream. Name the real cause instead.
            raise TypeError(
                "Fitting both a normalization and a projection needs two passes over the "
                "Support features, so `batches` must be re-iterable (a sequence, or an "
                "object whose __iter__ restarts the stream) — a one-shot iterator or "
                "generator would arrive at the projection pass already exhausted."
            )
        if self.method in _FITTED_METHODS:
            self._fit_normalization(batches)
        if self.projection_method in _FITTED_PROJECTIONS:
            self._fit_projection(batches)
        return self

    def _fit_normalization(self, batches: Iterable[Tensor]) -> None:
        count = 0
        total = torch.zeros(self.num_features, dtype=torch.float64)
        total_sq = torch.zeros(self.num_features, dtype=torch.float64)
        for batch in batches:
            rows = torch.as_tensor(batch).reshape(-1, self.num_features).to(torch.float64)
            count += rows.shape[0]
            total += rows.sum(dim=0)
            total_sq += (rows * rows).sum(dim=0)
        if count == 0:
            raise ValueError(
                f"Cannot fit a '{self.method}' feature adaptor: the Support set yielded no "
                "feature rows. Check that the train split has samples with features."
            )

        mean = total / count
        # E[x^2] - E[x]^2, clamped at 0 against float round-off on constant channels.
        variance = torch.clamp(total_sq / count - mean * mean, min=0.0)
        std = torch.sqrt(variance)
        floored = std < self.eps
        scale = torch.where(floored, torch.full_like(std, self.eps), std)

        self.center.copy_(mean.to(self.center.dtype))
        self.scale.copy_(scale.to(self.scale.dtype))
        self.eps_floored.fill_(int(floored.sum().item()))
        self.fitted.fill_(True)

    def _fit_projection(self, batches: Iterable[Tensor]) -> None:
        """Fit the PCA basis on **normalized** Support features (order: normalize → project).

        Accumulates the first two moments in one streaming pass (float64), so a whole
        cohort's tiles never has to be materialized: the ``D x D`` Gram matrix is the only
        thing that grows with the feature dimension.
        """
        count = 0
        total = torch.zeros(self.num_features, dtype=torch.float64)
        gram = torch.zeros(self.num_features, self.num_features, dtype=torch.float64)
        for batch in batches:
            rows = torch.as_tensor(batch).reshape(-1, self.num_features)
            rows = self._normalize(rows).to(torch.float64)
            count += rows.shape[0]
            total += rows.sum(dim=0)
            gram += rows.T @ rows
        # PCA preflight, half 2: a basis of `target_dim` components is only defined once
        # at least that many rows have been seen. Name the shortfall — with a small
        # Support set ("K means K") this is the constraint an operator actually hits.
        if count < self.target_dim:
            raise ValueError(
                f"Cannot fit a PCA projection to target_dim={self.target_dim}: the "
                f"Support set yielded only {count} feature row(s). PCA needs at least "
                f"target_dim rows — lower projection.target_dim to at most {count}, "
                "enlarge the Support set, or use projection.method='random', which has "
                "no such constraint."
            )

        mean = total / count
        # E[xx^T] - mean mean^T, on the (count - 1) denominator so the eigenvalues are the
        # usual unbiased component variances that explained_variance_ratio reports.
        covariance = (gram - count * torch.outer(mean, mean)) / max(count - 1, 1)
        # Symmetrize against float round-off, so eigh sees an exactly symmetric matrix and
        # repeated fits on identical data are bit-for-bit identical.
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)  # ascending
        order = torch.argsort(eigenvalues, descending=True)[: self.target_dim]
        components = eigenvectors[:, order].T.contiguous()  # (target_dim, D)
        top_eigenvalues = torch.clamp(eigenvalues[order], min=0.0)

        # Pinned sign convention. An eigenvector is only defined up to sign, and which
        # sign LAPACK hands back is not contractual — so fix it ourselves: make the
        # largest-magnitude entry of every component positive. Without this, two fits on
        # identical data could differ by a sign flip and the run would not be reproducible.
        pivots = components.abs().argmax(dim=1)
        pivot_signs = torch.sign(components.gather(1, pivots.unsqueeze(1)))
        pivot_signs = torch.where(
            pivot_signs == 0, torch.ones_like(pivot_signs), pivot_signs
        )
        components = components * pivot_signs

        total_variance = torch.clamp(eigenvalues, min=0.0).sum()
        ratio = (
            top_eigenvalues / total_variance
            if total_variance > 0
            else torch.zeros_like(top_eigenvalues)
        )

        self.projection_mean.copy_(mean.to(self.projection_mean.dtype))
        self.projection_matrix.copy_(components.to(self.projection_matrix.dtype))
        self.explained_variance_ratio.copy_(
            ratio.to(self.explained_variance_ratio.dtype)
        )
        self.projection_n_fit_samples.fill_(count)
        self.projection_fitted.fill_(True)

    def _normalize(self, X: Tensor) -> Tensor:
        if self.method == "none":
            return X
        if self.method == "zscore":
            # Deliberately the normalize stage's *own* flag, not the whole-adaptor
            # `is_fitted`: `_fit_projection` normalizes rows in between the two fits,
            # when the projection is by definition not yet fitted.
            if not bool(self.fitted.item()):
                raise RuntimeError(
                    "FeatureAdaptor(method='zscore') was used before it was fit. Fit it on "
                    "the Support (train) split, or load a checkpoint that carries its state."
                )
            return (X - self.center) / self.scale
        if self.method == "l2":
            return X / torch.clamp(
                X.norm(p=2, dim=-1, keepdim=True), min=self.eps
            )
        if self.method == "layernorm":
            return torch.nn.functional.layer_norm(
                X, (X.shape[-1],), eps=self.eps
            )
        raise ValueError(f"Unhandled feature-adaptor method {self.method!r}.")

    def _project(self, X: Tensor) -> Tensor:
        if not self.projects:
            return X
        if not bool(self.projection_fitted.item()):
            raise RuntimeError(
                f"FeatureAdaptor(projection='{self.projection_method}') was used before "
                "it was fit. Fit it on the Support (train) split, or load a checkpoint "
                "that carries its state."
            )
        # `random` leaves projection_mean at zeros, so the same expression serves both:
        # PCA centers intrinsically on its own fitted mean, random does not center.
        return (X - self.projection_mean) @ self.projection_matrix.T

    def forward(self, X: Tensor) -> Tensor:
        """Apply the active stages in order: **normalize → project**."""
        return self._project(self._normalize(X))

    def forward_grid(self, grid: Tensor) -> Tensor:
        """Apply the adaptor **channel-axis** to a dense grid ``(B, d, h, w)``.

        The dense path's features are not rows with the feature dim last — they are grids
        whose *channel* axis is the feature dim. Every spatial position is one feature
        vector, so the transform is the same one :meth:`forward` applies; only the axis it
        lives on differs. Moving ``d`` to the last axis, transforming, and moving it back
        keeps a single implementation of the stages for every path, and makes "channel-axis
        on ``(B, d, h, w)``" a property of the adaptor rather than of each caller.

        Returns ``(B, output_dim, h, w)`` — under a projection the channel count is
        ``target_dim``, which is the width the decoder must have been built against.
        """
        if grid.ndim != 4:
            raise ValueError(
                f"forward_grid expects a dense grid (B, d, h, w); got shape "
                f"{tuple(grid.shape)}."
            )
        return self.forward(grid.movedim(1, -1)).movedim(-1, 1).contiguous()

    def qc_summary(self) -> dict[str, Any]:
        """The `feature_adapter` QC sidecar payload."""
        projection: dict[str, Any] = {
            "method": self.projection_method,
            "target_dim": self.target_dim if self.projects else None,
            "seed": self.projection_seed,
            "n_fit_samples": self.n_fit_samples if self.projects else None,
            "input_dim": self.num_features,
            "output_dim": self.output_dim,
        }
        if self.projection_method == "pca":
            # The signal that says whether target_dim was wide enough to keep the
            # encoder's structure, or narrow enough to have thrown information away.
            projection["explained_variance_ratio"] = [
                float(value) for value in self.explained_variance_ratio
            ]
        return {
            "normalization": {
                "method": self.method,
                "eps": self.eps,
                "eps_floored_channels": self.num_eps_floored,
            },
            "projection": projection,
            "num_features": self.num_features,
            "output_dim": self.output_dim,
            "fitted": self.is_fitted,
        }

    def extra_repr(self) -> str:
        return (
            f"method={self.method!r}, eps={self.eps}, "
            f"num_features={self.num_features}, "
            f"projection={self.projection_method!r}, output_dim={self.output_dim}"
        )


def build_feature_adaptor(
    normalization: NormalizationConfig | None,
    projection: ProjectionConfig | None = None,
    *,
    num_features: int,
    encoder_identity: str = "",
) -> FeatureAdaptor | None:
    """Build the adaptor for a config, or ``None`` when no stage is active.

    ``None`` is the load-bearing return value: it is what keeps a run with every section
    off structurally identical to a model built before this module existed. The two
    sections are independently optional — either one alone brings the adaptor into being.
    """
    normalization = normalization or NormalizationConfig()
    projection = projection or ProjectionConfig()
    if normalization.method == "none" and projection.method == "none":
        return None
    return FeatureAdaptor(
        method=normalization.method,
        eps=normalization.eps,
        num_features=num_features,
        projection=projection,
        encoder_identity=encoder_identity,
    )


def feature_adaptor_output_dim(
    adaptor: FeatureAdaptor | None, *, num_features: int
) -> int:
    """The width the aggregator/head must be constructed against.

    The **dim rewire**: with a projection active this is ``target_dim`` rather than the
    encoder's native dim, so the downstream trainable parameter count is equal across a
    roster of encoders of differing width — which is the entire point of the dim-matched
    ablation (issue #284). Without an adaptor it is just the native dim.
    """
    return num_features if adaptor is None else adaptor.output_dim


def write_feature_adapter_sidecar(adaptor: FeatureAdaptor | None, fold_dir: Path) -> Path | None:
    """Write the ``feature_adapter.json`` QC sidecar, or nothing when no adaptor exists.

    Reports the normalization method, the ``eps`` floor, and how many channels that floor
    actually caught — a near-zero-variance channel is the failure mode worth noticing, so
    a non-zero count is the signal an operator wants surfaced next to the checkpoint.
    """
    if adaptor is None:
        return None
    path = Path(fold_dir) / FEATURE_ADAPTER_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(adaptor.qc_summary(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return path
