"""``Baseline``: a predictor that ignores the features.

The pipeline fills samples the model cannot score (no tiles, no embedding) with it so
:func:`soma.evaluate` still scores the whole split; the rows carry ``is_placeholder``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from soma.evaluation.predictions import (
    PLACEHOLDER_COLUMN,
    Predictions,
    encode_labels,
    family_prediction_columns,
)

__all__ = ["Baseline"]

_CLASSIFICATION = frozenset({"binary_classification", "multiclass_classification"})


class Baseline:
    """Per-family constant predictions fitted on training targets.

    Classification predicts the training class prior (and its argmax), ordinal the mean
    training label (rounded and clipped for the label), regression the mean training
    value, survival a constant risk of ``0.0`` (placeholders tie among themselves).
    Build one with :meth:`fit`.
    """

    def __init__(
        self,
        task_family: str,
        values: dict[str, Any],
        *,
        prediction_columns: Sequence[str],
        classes: Sequence[Any] | None = None,
    ) -> None:
        self.task_family = task_family
        self.values = dict(values)
        self.prediction_columns = tuple(prediction_columns)
        self.classes = None if classes is None else tuple(classes)

    @classmethod
    def fit(
        cls,
        train_targets: pd.DataFrame,
        *,
        task_family: str,
        classes: Sequence[Any] | None = None,
    ) -> "Baseline":
        """Fit on the training split's targets (e.g. ``cohort.targets("train")``).

        ``classes`` names the raw label of each class index, as on
        :class:`~soma.evaluation.Predictions`; without it labels must be class indices.
        """
        if len(train_targets) == 0:
            raise ValueError("Baseline.fit needs at least one training target")
        if task_family in _CLASSIFICATION or task_family == "ordinal_classification":
            labels = encode_labels(train_targets["label"], classes).to_numpy()
            num_classes = len(classes) if classes is not None else int(labels.max()) + 1
            if task_family in _CLASSIFICATION:
                counts = np.bincount(labels, minlength=num_classes).astype(float)
                prior = counts / counts.sum()
                values: dict[str, Any] = {f"prob_{i}": float(p) for i, p in enumerate(prior)}
                values["predicted_label"] = int(np.argmax(prior))
                columns = family_prediction_columns(task_family, num_classes)
            else:
                raw_score = float(np.mean(labels))
                values = {
                    "predicted_label": int(np.clip(np.rint(raw_score), 0, max(num_classes - 1, 0))),
                    "raw_score": raw_score,
                }
                columns = family_prediction_columns(task_family)
        elif task_family == "regression":
            values = {"predicted_value": float(np.mean(train_targets["value"].to_numpy(dtype=float)))}
            columns = family_prediction_columns(task_family)
        elif task_family == "survival":
            values = {"risk_score": 0.0}
            columns = family_prediction_columns(task_family)
        else:
            raise ValueError(f"no baseline for task family {task_family!r}")
        return cls(task_family, values, prediction_columns=columns, classes=classes)

    def predict(self, sample_ids: Sequence[str]) -> Predictions:
        """The constant prediction for each id, flagged ``is_placeholder=True``."""
        ids = [str(sid) for sid in sample_ids]
        frame = pd.DataFrame({"sample_id": pd.Series(ids, dtype=object)})
        for column in self.prediction_columns:
            value = self.values[column]
            dtype = np.int64 if isinstance(value, (int, np.integer)) else np.float64
            frame[column] = np.full(len(ids), value, dtype=dtype)
        frame[PLACEHOLDER_COLUMN] = np.ones(len(ids), dtype=bool)
        return Predictions(
            frame,
            task_family=self.task_family,
            prediction_columns=self.prediction_columns,
            classes=self.classes,
        )
