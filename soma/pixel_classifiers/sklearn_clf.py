"""scikit-learn per-pixel classifiers: random forest and (multinomial) logistic.

Both are one-shot fitters — they ignore ``X_val``/``y_val`` (no early stopping) — and
serialize with joblib. sklearn handles non-contiguous / missing classes via its own
``classes_`` encoding; ``predict_proba`` columns are scattered back to the full label
space so an absent class reads as probability 0.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import numpy as np

from soma.pixel_classifiers.base import PixelClassifier, scatter_proba_to_full
from soma.pixel_classifiers.registry import pixel_classifier_registry

_MODEL_FILENAME = "model.joblib"


class _SklearnPixelClassifier(PixelClassifier):
    """Shared fit/predict/save/load for sklearn estimators with ``predict_proba``."""

    def __init__(self, *, num_classes: int) -> None:
        super().__init__(num_classes=num_classes)
        self._model = None
        self._fit_classes: np.ndarray | None = None

    @abstractmethod
    def _build_estimator(self):
        """Return an unfitted sklearn estimator (subclass-specific hyperparameters)."""
        ...

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,  # ignored (one-shot fitter)
        sample_weight: np.ndarray | None = None,
    ) -> None:
        self._fit_classes = np.unique(y)
        if self._fit_classes.size < 2:
            self._model = None  # single class: constant predictor (sklearn can't fit it)
            return
        estimator = self._build_estimator()
        if sample_weight is not None:
            estimator.fit(X, y, sample_weight=sample_weight)
        else:
            estimator.fit(X, y)
        self._model = estimator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._fit_classes is None:
            raise RuntimeError("predict_proba called before fit/load.")
        if self._model is None:
            full = np.zeros((X.shape[0], self._num_classes), dtype=np.float32)
            full[:, int(self._fit_classes[0])] = 1.0
            return full
        raw = self._model.predict_proba(X).astype(np.float32)  # columns == model.classes_
        return scatter_proba_to_full(raw, np.asarray(self._model.classes_), self._num_classes)

    def save(self, path: Path | str) -> None:
        import joblib

        if self._fit_classes is None:
            raise RuntimeError("save called before fit.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            joblib.dump(self._model, out / _MODEL_FILENAME)
        self._write_meta(out, extra={"fit_classes": [int(c) for c in self._fit_classes]})

    @classmethod
    def load(cls, path: Path | str) -> "_SklearnPixelClassifier":
        import joblib

        meta = cls._read_meta(path)
        obj = cls(num_classes=int(meta["num_classes"]))
        obj._fit_classes = np.asarray(meta.get("fit_classes", list(range(obj._num_classes))))
        model_path = Path(path) / _MODEL_FILENAME
        obj._model = joblib.load(model_path) if model_path.is_file() else None
        return obj


class RandomForestPixelClassifier(_SklearnPixelClassifier):
    """Random forest per-pixel classifier (scikit-learn)."""

    def __init__(
        self,
        *,
        num_classes: int,
        n_estimators: int = 200,
        max_depth: int | None = None,
        max_features: str | int | float = "sqrt",
        n_jobs: int = -1,
        random_state: int = 0,
        **extra,
    ) -> None:
        super().__init__(num_classes=num_classes)
        self._kwargs = dict(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            max_features=max_features,
            n_jobs=int(n_jobs),
            random_state=int(random_state),
            **extra,
        )

    def _build_estimator(self):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**self._kwargs)


class LogisticPixelClassifier(_SklearnPixelClassifier):
    """Multinomial logistic-regression per-pixel classifier (scikit-learn)."""

    def __init__(
        self,
        *,
        num_classes: int,
        C: float = 1.0,
        max_iter: int = 200,
        n_jobs: int = -1,
        random_state: int = 0,
        **extra,
    ) -> None:
        super().__init__(num_classes=num_classes)
        self._kwargs = dict(
            C=float(C),
            max_iter=int(max_iter),
            n_jobs=int(n_jobs),
            random_state=int(random_state),
            **extra,
        )

    def _build_estimator(self):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(**self._kwargs)


pixel_classifier_registry.register("random_forest", RandomForestPixelClassifier)
pixel_classifier_registry.register("logistic", LogisticPixelClassifier)
