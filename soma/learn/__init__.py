"""Learning: ``Learner`` (subclass-first), the generic ``Trainer`` and its callbacks."""

from soma.learn.callbacks import Checkpoint, EarlyStopping, History, Progress
from soma.learn.dense import DenseLearner, accumulate_dense_stats
from soma.learn.fit import fit, learner_for
from soma.learn.learner import (
    Budget,
    Learner,
    SchedulerSpec,
    StepOutput,
    learner_registry,
    optimizer_registry,
    resolve_leaf,
    scheduler_registry,
)
from soma.learn.mil import MILLearner
from soma.learn.trainer import FitResult, Trainer

__all__ = [
    "Budget",
    "Checkpoint",
    "DenseLearner",
    "EarlyStopping",
    "FitResult",
    "History",
    "Learner",
    "MILLearner",
    "Progress",
    "SchedulerSpec",
    "StepOutput",
    "Trainer",
    "accumulate_dense_stats",
    "fit",
    "learner_for",
    "learner_registry",
    "optimizer_registry",
    "resolve_leaf",
    "scheduler_registry",
]
