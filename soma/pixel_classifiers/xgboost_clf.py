"""XGBoost per-pixel classifier — the paper's headline classifier (arXiv:2602.18747).

Gradient-boosted trees on the per-pixel ``K``-vector. XGBoost is an optional
dependency, imported lazily so the rest of soma runs without it; a clear error fires
only if this classifier is actually selected without the package installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from soma.pixel_classifiers.base import PixelClassifier, scatter_proba_to_full
from soma.pixel_classifiers.registry import pixel_classifier_registry

_MODEL_FILENAME = "model.ubj"


def _require_xgboost():
    try:
        import xgboost  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The 'xgboost' pixel classifier requires the xgboost package "
            "(`pip install xgboost`)."
        ) from exc
    return xgboost


class XGBoostPixelClassifier(PixelClassifier):
    """Per-pixel multiclass XGBoost classifier.

    Defaults follow the paper (``tree_method='hist'``, 100 rounds). ``X_val``/``y_val``
    drive ``early_stopping_rounds`` when given. ``predict_proba`` always returns a full
    ``(N, num_classes)`` matrix: XGBoost's internal label encoder only knows the classes
    present at fit time, so columns are scattered back to their true class indices (zeros
    for any class absent from the training sample).
    """

    def __init__(
        self,
        *,
        num_classes: int,
        n_estimators: int = 100,
        tree_method: str = "hist",
        max_depth: int = 6,
        learning_rate: float = 0.3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        early_stopping_rounds: int | None = 10,
        n_jobs: int = -1,
        random_state: int = 0,
        **extra,
    ) -> None:
        super().__init__(num_classes=num_classes)
        self._hparams = dict(
            n_estimators=int(n_estimators),
            tree_method=str(tree_method),
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            n_jobs=int(n_jobs),
            random_state=int(random_state),
            **extra,
        )
        self._early_stopping_rounds = early_stopping_rounds
        self._model = None
        # classes seen at fit time, in the model's column order (for proba scatter).
        self._fit_classes: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        xgboost = _require_xgboost()
        # XGBoost requires contiguous 0..m-1 labels, so remap the (possibly sparse) true
        # class ids present in the train sample to a dense encoding; ``_fit_classes`` maps
        # the model's output columns back to true class indices for the proba scatter.
        classes, y_enc = np.unique(y, return_inverse=True)
        self._fit_classes = classes
        if classes.size < 2:
            # Degenerate: only one class sampled — a tree ensemble can't train on it.
            # Record a constant predictor (predict_proba returns one-hot at that class).
            self._model = None
            return
        use_es = (
            X_val is not None and y_val is not None and bool(self._early_stopping_rounds)
        )
        eval_set = None
        if use_es:
            keep = np.isin(y_val, classes)  # val rows in unseen classes can't be encoded
            if keep.any():
                eval_set = [(X_val[keep], np.searchsorted(classes, y_val[keep]))]
            else:
                use_es = False
        model = xgboost.XGBClassifier(
            early_stopping_rounds=self._early_stopping_rounds if use_es else None,
            **self._hparams,
        )
        fit_kwargs: dict = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if use_es:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = False
        model.fit(X, y_enc, **fit_kwargs)
        self._model = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._fit_classes is None:
            raise RuntimeError("XGBoostPixelClassifier.predict_proba called before fit/load.")
        if self._model is None:  # single-class constant predictor
            full = np.zeros((X.shape[0], self._num_classes), dtype=np.float32)
            full[:, int(self._fit_classes[0])] = 1.0
            return full
        raw = self._model.predict_proba(X)  # (N, m) aligned to 0..m-1 == self._fit_classes
        return scatter_proba_to_full(raw, self._fit_classes, self._num_classes)

    def save(self, path: Path | str) -> None:
        if self._fit_classes is None:
            raise RuntimeError("XGBoostPixelClassifier.save called before fit.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        # A single-class fit has no booster — persist only the constant-predictor meta.
        if self._model is not None:
            self._model.save_model(out / _MODEL_FILENAME)
        self._write_meta(out, extra={"fit_classes": [int(c) for c in self._fit_classes]})

    @classmethod
    def load(cls, path: Path | str) -> "XGBoostPixelClassifier":
        xgboost = _require_xgboost()
        meta = cls._read_meta(path)
        obj = cls(num_classes=int(meta["num_classes"]))
        obj._fit_classes = np.asarray(meta.get("fit_classes", list(range(obj._num_classes))))
        model_path = Path(path) / _MODEL_FILENAME
        if model_path.is_file():
            model = xgboost.XGBClassifier()
            model.load_model(model_path)
            obj._model = model
        else:
            obj._model = None  # constant predictor
        return obj


pixel_classifier_registry.register("xgboost", XGBoostPixelClassifier)
