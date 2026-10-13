"""``soma.fit``: sugar over :class:`Trainer` for a bare model or a ``Learner``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import nn

from soma.learn.callbacks import Checkpoint, EarlyStopping, Progress
from soma.learn.dense import DenseLearner
from soma.learn.learner import Learner
from soma.learn.mil import MILLearner
from soma.learn.trainer import FitResult, Trainer
from soma.models.model import DenseModel, EmbeddingModel, MILModel

__all__ = ["fit", "learner_for"]


def learner_for(model: nn.Module, *, loss=None, optimizer=None, schedulers: Sequence[Any] = ()) -> Learner:
    """Wrap a bare model in the learner for its shape."""
    if isinstance(model, (MILModel, EmbeddingModel)):
        return MILLearner(model, loss=loss, optimizer=optimizer, schedulers=schedulers)
    if isinstance(model, DenseModel):
        if loss is not None:
            raise ValueError("loss= applies to the set shape; subclass DenseLearner for a custom dense loss")
        return DenseLearner(model, optimizer=optimizer, schedulers=schedulers)
    raise TypeError(
        f"cannot pick a learner for {type(model).__name__}; pass a Learner or a MILModel / "
        "EmbeddingModel / DenseModel"
    )


def fit(
    learner_or_model: Learner | nn.Module,
    train,
    tune=None,
    *,
    loss=None,
    optimizer=None,
    schedulers: Sequence[Any] = (),
    monitor: str = "loss",
    mode: str = "min",
    patience: int | None = 10,
    callbacks: Sequence[Any] = (),
    progress: bool = True,
    **trainer_kwargs: Any,
) -> FitResult:
    """Fit a learner (or a bare model wrapped in its shape's learner).

    Builds ``Checkpoint(monitor, mode)`` and ``EarlyStopping(monitor, mode, patience)``
    (``patience=None`` installs no early stopping) and a ``Progress`` callback; the
    remaining keyword arguments go to :class:`Trainer`. A ``Checkpoint``,
    ``EarlyStopping`` or ``Progress`` passed in ``callbacks`` replaces the one ``fit``
    would build, so a run states its whole selection policy (``selection="last"``,
    ``hold_until_moves``, a progress label) with plain constructors.
    """
    if isinstance(learner_or_model, Learner):
        if loss is not None or optimizer is not None or schedulers:
            raise ValueError(
                "loss=, optimizer= and schedulers= configure a bare model; a Learner already "
                "owns them"
            )
        learner = learner_or_model
    else:
        learner = learner_for(learner_or_model, loss=loss, optimizer=optimizer, schedulers=schedulers)
    supplied = list(callbacks)
    defaults = (
        (Checkpoint, lambda: Checkpoint(monitor=monitor, mode=mode)),
        (
            EarlyStopping,
            lambda: None if patience is None else EarlyStopping(monitor=monitor, mode=mode, patience=patience),
        ),
        (Progress, lambda: Progress() if progress else None),
    )
    installed: list[Any] = []
    for kind, build in defaults:
        given = [callback for callback in supplied if isinstance(callback, kind)]
        installed.extend(given or [callback for callback in (build(),) if callback is not None])
    installed.extend(c for c in supplied if not isinstance(c, (Checkpoint, EarlyStopping, Progress)))
    return Trainer(callbacks=installed, **trainer_kwargs).fit(learner, train, tune)
