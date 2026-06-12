"""PixelClassifier base class — a per-pixel ``(K,) → class`` classifier.

The decoder-free segmentation component (design — attention-pixel segmentation §6),
parallel to :class:`soma.decoders.base.Decoder` but **context-free**: it maps a single
pixel's ``K``-vector (per-head CLS-attention, or any dense channel) to a class, with no
spatial receptive field. That hard line is what keeps the family swappable and uniform —
spatial-context models are the *neural decoder* path, never a ``PixelClassifier``.

Each subclass owns its framework, training loop, and serialization entirely behind these
methods: XGBoost runs its boosting rounds, the MLP runs its mini-batch SGD epochs —
*internally*, behind the same ``fit``. The torch ``Trainer``, per-epoch dataloaders, and
``.pt`` checkpoints do **not** apply here.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

__all__ = ["PixelClassifier", "scatter_proba_to_full"]

_META_FILENAME = "pixel_classifier.json"


def scatter_proba_to_full(
    proba: np.ndarray, fit_classes: np.ndarray, num_classes: int
) -> np.ndarray:
    """Scatter a ``(N, len(fit_classes))`` proba matrix into ``(N, num_classes)``.

    Classifiers that only saw a subset of classes at fit time emit fewer columns;
    this places each column at its true class index and leaves absent classes zero,
    so the downstream argmax / metrics never misalign columns to classes. No-op fast
    path when the fit already covered every class in order.
    """
    fit_classes = np.asarray(fit_classes)
    if fit_classes.shape[0] == num_classes and np.array_equal(fit_classes, np.arange(num_classes)):
        return proba
    full = np.zeros((proba.shape[0], num_classes), dtype=proba.dtype)
    full[:, fit_classes] = proba
    return full


class PixelClassifier(ABC):
    """Abstract base for per-pixel classifiers.

    Constructed with ``num_classes`` plus subclass-specific hyperparameters; the
    pipeline injects ``num_classes`` from the task config (as it does for decoders).

    The pixel matrices are plain NumPy: ``X`` is ``(N, K)`` float32 (N sampled or full
    pixels, K channels), ``y`` is ``(N,)`` int. ``predict_proba`` always returns
    ``(N, num_classes)`` — even if some classes were absent at ``fit`` time — so the
    downstream argmax / metrics never silently misalign columns to classes.
    """

    def __init__(self, *, num_classes: int) -> None:
        if int(num_classes) < 2:
            raise ValueError(f"PixelClassifier needs num_classes >= 2, got {num_classes}.")
        self._num_classes = int(num_classes)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Fit on sampled pixels ``X (N, K)`` / labels ``y (N,)``.

        ``X_val`` / ``y_val`` are sampled pixels from the *tune* split; early-stopping
        fitters (XGBoost, MLP) use them, one-shot fitters ignore them. ``sample_weight``
        is an optional per-pixel weight (e.g. inverse class frequency).
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict ``(N, num_classes)`` class probabilities for pixels ``X (N, K)``."""
        ...

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Persist the fitted classifier into directory ``path`` (created if needed)."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str) -> "PixelClassifier":
        """Reconstruct a fitted classifier previously written by :meth:`save`."""
        ...

    # --- shared serialization helpers (subclasses persist their payload alongside) ---

    @staticmethod
    def _meta_path(path: Path | str) -> Path:
        return Path(path) / _META_FILENAME

    def _write_meta(self, path: Path | str, extra: dict | None = None) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        meta = {"num_classes": self._num_classes, "type": type(self).__name__}
        if extra:
            meta.update(extra)
        self._meta_path(out).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _read_meta(path: Path | str) -> dict:
        return json.loads(PixelClassifier._meta_path(path).read_text(encoding="utf-8"))
