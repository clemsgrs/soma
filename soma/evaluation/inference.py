"""``predict``: run a learner over a loader and collect its declared prediction columns.

Side artefacts (attention maps, rasters) are predict callbacks, never keywords. A
callback implements any of ``on_predict_start(learner)``, ``on_predict_batch(batch,
output)`` and ``on_predict_end()``; ``output`` is the learner's
:class:`~soma.learn.PredictOutput` (``raw`` holds the full model output).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

from soma.evaluation.predictions import Predictions

__all__ = ["predict"]


def _call(callbacks: Sequence[Any], hook: str, *args: Any) -> None:
    for callback in callbacks:
        method = getattr(callback, hook, None)
        if method is not None:
            method(*args)


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _predict_grid(learner, loader, callbacks: Sequence[Any], device):
    """Grid heads: collect each sample's output and let the head assemble its noun."""
    head = learner.head
    learner.eval()
    sample_ids: list[str] = []
    collected: dict[str, list[Any]] = {}
    _call(callbacks, "on_predict_start", learner)
    with torch.inference_mode():
        for batch in loader:
            batch = learner.transfer_batch(batch, device)
            output = learner.predict_step(batch)
            ids = [str(sid) for sid in batch.sample_ids]
            sample_ids.extend(ids)
            for name, values in output.columns.items():
                if len(values) != len(ids):
                    raise ValueError(
                        f"predict_step column '{name}' holds {len(values)} values for a batch "
                        f"of {len(ids)} samples; every column holds one value per sample"
                    )
                collected.setdefault(name, []).extend(values[i] for i in range(len(ids)))
            _call(callbacks, "on_predict_batch", batch, output)
    _call(callbacks, "on_predict_end")
    return head.assemble_predictions(sample_ids, collected)


def predict(learner_or_model, loader, callbacks: Sequence[Any] = ()):
    """Predict every sample of ``loader``.

    A bare model is wrapped in its shape's learner (:func:`soma.learn.learner_for`).
    The learner runs in eval mode under ``torch.inference_mode`` on the device its
    parameters live on; each batch goes through ``transfer_batch`` and
    ``predict_step``. For the set shape the frame is built from the step's ``columns``
    only, in loader order, and the head's ``task_family``, ``prediction_columns`` and
    ``classes`` describe it: a :class:`Predictions`. A grid head (one declaring
    ``target_kind``) assembles its own noun from the per-sample outputs
    (``assemble_predictions``): :class:`~soma.evaluation.LabelMapPredictions` or
    :class:`~soma.evaluation.PointPredictions`, covering every sample of the loader.
    """
    from soma.learn import Learner, learner_for

    learner = learner_or_model if isinstance(learner_or_model, Learner) else learner_for(learner_or_model)
    parameter = next(learner.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    head = learner.head
    if getattr(head, "target_kind", None) is not None:
        return _predict_grid(learner, loader, callbacks, device)
    declared = tuple(head.prediction_columns)
    learner.eval()
    sample_ids: list[str] = []
    columns: dict[str, list[np.ndarray]] = {}
    _call(callbacks, "on_predict_start", learner)
    with torch.inference_mode():
        for batch in loader:
            batch = learner.transfer_batch(batch, device)
            output = learner.predict_step(batch)
            ids = [str(sid) for sid in batch.sample_ids]
            sample_ids.extend(ids)
            for name, values in output.columns.items():
                array = _as_numpy(values)
                if array.shape[:1] != (len(ids),):
                    raise ValueError(
                        f"predict_step column '{name}' has shape {array.shape} for a batch of "
                        f"{len(ids)} samples; every column holds one value per sample"
                    )
                columns.setdefault(name, []).append(array)
            _call(callbacks, "on_predict_batch", batch, output)
    _call(callbacks, "on_predict_end")
    missing = [name for name in declared if sample_ids and name not in columns]
    if missing:
        raise ValueError(f"predict_step returned no column(s) {missing} that the head declares")
    data: dict[str, Any] = {"sample_id": pd.Series(sample_ids, dtype=object)}
    for name in (*declared, *(c for c in columns if c not in declared)):
        data[name] = np.concatenate(columns[name]) if name in columns else np.empty(0, dtype=np.float64)
    return Predictions(
        pd.DataFrame(data),
        task_family=head.task_family,
        prediction_columns=declared,
        classes=head.classes,
    )
