"""``Predictions``: what a model said about each sample, and nothing about the truth.

A frame keyed by ``sample_id`` whose other columns the task head declares
(:attr:`soma.tasks.TaskHead.prediction_columns`). Targets never travel here; they come
from the cohort (:meth:`soma.data.Cohort.targets`) and meet the predictions only inside
:func:`soma.evaluate`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "PLACEHOLDER_COLUMN",
    "Predictions",
    "encode_labels",
    "family_prediction_columns",
    "family_target_keys",
    "infer_task_family",
]

#: Marks a row filled by a :class:`~soma.evaluation.Baseline` instead of the model.
PLACEHOLDER_COLUMN = "is_placeholder"
_METADATA_KEY = b"soma"
_CLASSIFICATION = frozenset({"binary_classification", "multiclass_classification"})


def family_prediction_columns(task_family: str, num_classes: int | None = None) -> tuple[str, ...]:
    """The prediction columns soma's heads declare for a set-shape ``task_family``.

    ``prob_0..prob_{k-1}`` + ``predicted_label`` (classification), ``predicted_label`` +
    ``raw_score`` (ordinal), ``predicted_value`` (regression), ``risk_score`` (survival).
    """
    if task_family in _CLASSIFICATION:
        if not num_classes:
            raise ValueError(f"{task_family} prediction columns need num_classes")
        return (*(f"prob_{i}" for i in range(int(num_classes))), "predicted_label")
    if task_family == "ordinal_classification":
        return ("predicted_label", "raw_score")
    if task_family == "regression":
        return ("predicted_value",)
    if task_family == "survival":
        return ("risk_score",)
    raise ValueError(
        f"task family {task_family!r} declares no set-shape prediction columns; a custom "
        "head overrides prediction_columns and to_columns"
    )


def family_target_keys(task_family: str) -> tuple[str, ...]:
    """The cohort target keys ``evaluate`` reads for a set-shape ``task_family``."""
    if task_family in _CLASSIFICATION or task_family == "ordinal_classification":
        return ("label",)
    if task_family == "regression":
        return ("value",)
    if task_family == "survival":
        return ("time", "event")
    raise ValueError(f"task family {task_family!r} has no set-shape targets")


def infer_task_family(columns: Iterable[str]) -> str:
    """Read the task family off a predictions table's columns (no soma metadata)."""
    names = set(columns)
    probabilities = [c for c in names if c.startswith("prob_")]
    if probabilities:
        return "binary_classification" if len(probabilities) == 2 else "multiclass_classification"
    if "raw_score" in names and "predicted_label" in names:
        return "ordinal_classification"
    if "predicted_value" in names:
        return "regression"
    if "risk_score" in names:
        return "survival"
    raise ValueError(
        f"cannot infer the task family from columns {sorted(names)}; pass task_family="
    )


def _label_key(value: Any) -> str:
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def encode_labels(labels: pd.Series, classes: Sequence[Any] | None) -> pd.Series:
    """Map raw cohort labels onto class indices (``classes[i]`` is index ``i``).

    Without ``classes`` the labels must already be integer class indices.
    """
    if classes is not None:
        index = {_label_key(c): i for i, c in enumerate(classes)}
        keys = labels.map(_label_key)
        unknown = sorted(set(keys) - set(index))
        if unknown:
            raise ValueError(
                f"target label(s) {unknown} are not among the predictions' classes {list(classes)}"
            )
        return keys.map(index).astype(np.int64)
    numeric = pd.to_numeric(labels, errors="coerce")
    if numeric.isna().any() or not np.all(np.mod(numeric.to_numpy(dtype=float), 1) == 0):
        bad = sorted({_label_key(v) for v, n in zip(labels, numeric) if pd.isna(n) or n % 1})
        raise ValueError(
            f"target label(s) {bad[:10]} are not class indices and the predictions declare no "
            "classes; pass Predictions(..., classes=[...]) naming the label of each class index"
        )
    return numeric.astype(np.int64)


def _probability_index(name: str) -> int:
    return int(name.removeprefix("prob_"))


def _jsonable(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


class Predictions:
    """A frame of per-sample predictions keyed by ``sample_id``.

    Args:
        frame: One row per sample: a ``sample_id`` column, the declared prediction
            columns, an optional boolean ``is_placeholder`` column, and any extra columns
            a custom learner adds.
        task_family: The head's ``task_family``; it selects the metrics that apply.
        prediction_columns: The columns the head declares. Defaults to every column of
            ``frame`` except ``sample_id`` and ``is_placeholder``.
        classes: The raw label of each class index, in ``prob_<i>`` order. With it,
            :func:`~soma.evaluate` maps the cohort's raw labels onto class indices;
            without it, labels must already be integer class indices.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        task_family: str,
        prediction_columns: Sequence[str] | None = None,
        classes: Sequence[Any] | None = None,
    ) -> None:
        if "sample_id" not in frame.columns:
            raise ValueError(f"predictions need a 'sample_id' column, got {list(frame.columns)}")
        frame = frame.reset_index(drop=True).copy()
        frame["sample_id"] = frame["sample_id"].astype(str)
        if prediction_columns is None:
            prediction_columns = [c for c in frame.columns if c not in {"sample_id", PLACEHOLDER_COLUMN}]
        missing = [c for c in prediction_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"predictions lack their declared column(s) {missing}")
        if PLACEHOLDER_COLUMN in frame.columns:
            frame[PLACEHOLDER_COLUMN] = frame[PLACEHOLDER_COLUMN].astype(bool)
        self._frame = frame
        self._task_family = str(task_family)
        self._prediction_columns = tuple(str(c) for c in prediction_columns)
        self._classes = None if classes is None else tuple(_jsonable(c) for c in classes)

    # --- access --------------------------------------------------------------------- #

    @property
    def frame(self) -> pd.DataFrame:
        """A copy of the underlying frame."""
        return self._frame.copy()

    @property
    def task_family(self) -> str:
        return self._task_family

    @property
    def prediction_columns(self) -> tuple[str, ...]:
        return self._prediction_columns

    @property
    def classes(self) -> tuple[Any, ...] | None:
        return self._classes

    @property
    def sample_ids(self) -> list[str]:
        return self._frame["sample_id"].tolist()

    @property
    def has_placeholders(self) -> bool:
        """Whether the frame carries the ``is_placeholder`` column (even all ``False``)."""
        return PLACEHOLDER_COLUMN in self._frame.columns

    @property
    def probability_columns(self) -> list[str]:
        """``prob_<i>`` prediction columns in class-index order."""
        probs = [c for c in self._prediction_columns if c.startswith("prob_")]
        return sorted(probs, key=_probability_index)

    @property
    def column_roles(self) -> dict[str, str]:
        """Each column's role: ``id``, ``prediction``, ``placeholder`` or ``extra``."""
        roles = {}
        for column in self._frame.columns:
            if column == "sample_id":
                roles[column] = "id"
            elif column in self._prediction_columns:
                roles[column] = "prediction"
            elif column == PLACEHOLDER_COLUMN:
                roles[column] = "placeholder"
            else:
                roles[column] = "extra"
        return roles

    def __len__(self) -> int:
        return len(self._frame)

    def __repr__(self) -> str:
        return (
            f"Predictions(task_family={self._task_family!r}, n={len(self)}, "
            f"columns={list(self._frame.columns)})"
        )

    # --- combination ---------------------------------------------------------------- #

    @classmethod
    def concat(cls, parts: Sequence["Predictions"]) -> "Predictions":
        """Stack predictions of one family; ``is_placeholder`` is ``False`` where absent."""
        parts = list(parts)
        if not parts:
            raise ValueError("Predictions.concat needs at least one Predictions")
        first = parts[0]
        for other in parts[1:]:
            if other.task_family != first.task_family:
                raise ValueError(
                    f"cannot concat {first.task_family!r} and {other.task_family!r} predictions"
                )
            if other.classes != first.classes:
                raise ValueError(f"cannot concat predictions over classes {first.classes} and {other.classes}")
            if other.prediction_columns != first.prediction_columns:
                raise ValueError(
                    f"cannot concat predictions declaring {first.prediction_columns} and "
                    f"{other.prediction_columns}"
                )
        frames = [p._frame for p in parts]
        if any(p.has_placeholders for p in parts):
            frames = [
                f if PLACEHOLDER_COLUMN in f.columns else f.assign(**{PLACEHOLDER_COLUMN: False})
                for f in frames
            ]
        columns = list(dict.fromkeys(c for f in frames for c in f.columns))
        # Empty parts only contribute their columns: concatenating an empty frame would
        # coerce the others' dtypes.
        non_empty = [f for f in frames if len(f)] or frames[:1]
        frame = pd.concat(non_empty, ignore_index=True, sort=False).reindex(columns=columns)
        return cls(
            frame,
            task_family=first.task_family,
            prediction_columns=first.prediction_columns,
            classes=first.classes,
        )

    # --- disk ----------------------------------------------------------------------- #

    def to_parquet(self, path: str | Path) -> Path:
        """Write the frame with ``task_family``, the soma version, the column roles and
        the classes in the parquet schema metadata."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        from soma import __version__

        path = Path(path)
        table = pa.Table.from_pandas(self._frame, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata[_METADATA_KEY] = json.dumps(
            {
                "task_family": self._task_family,
                "soma_version": __version__,
                "prediction_columns": list(self._prediction_columns),
                "columns": self.column_roles,
                "classes": None if self._classes is None else list(self._classes),
            }
        ).encode("utf-8")
        pq.write_table(table.replace_schema_metadata(metadata), path)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path, *, task_family: str | None = None) -> "Predictions":
        """Read predictions written by :meth:`to_parquet` or by any parquet writer.

        A file without soma metadata declares every column other than ``sample_id`` /
        ``is_placeholder`` a prediction column; its family is ``task_family`` or, when
        omitted, read off the column names.
        """
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        frame = table.to_pandas()
        raw = (table.schema.metadata or {}).get(_METADATA_KEY)
        if raw is None:
            return cls(frame, task_family=task_family or infer_task_family(frame.columns))
        meta = json.loads(raw)
        if task_family is not None and task_family != meta["task_family"]:
            raise ValueError(
                f"{path} holds {meta['task_family']!r} predictions, not {task_family!r}"
            )
        return cls(
            frame,
            task_family=meta["task_family"],
            prediction_columns=meta["prediction_columns"],
            classes=meta.get("classes"),
        )
