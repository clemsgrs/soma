"""Learning: ``Learner`` (subclass-first), the generic ``Trainer`` and its callbacks."""

from soma.learn.augment import SegmentationAugment, augment_registry
from soma.learn.bundle import ModelBundle, load, record_builder, save
from soma.learn.callbacks import Checkpoint, EarlyStopping, History, Progress
from soma.learn.dense import DenseLearner
from soma.learn.fit import fit, learner_for
from soma.learn.grid import GridBatch
from soma.learn.learner import (
    Budget,
    Learner,
    PredictOutput,
    SchedulerSpec,
    StepOutput,
    learner_registry,
    optimizer_registry,
    resolve_leaf,
    scheduler_registry,
)
from soma.learn.loaders import Loaders, loaders
from soma.learn.mil import MILLearner
from soma.learn.roi_population import SegmentationRoiPopulation, resolve_segmentation_roi_population
from soma.learn.roi_sampler import SegmentationRoiBatchSampler
from soma.learn.trainer import FitResult, Trainer

__all__ = [
    "Budget",
    "Checkpoint",
    "DenseLearner",
    "EarlyStopping",
    "FitResult",
    "GridBatch",
    "History",
    "Learner",
    "Loaders",
    "MILLearner",
    "ModelBundle",
    "PredictOutput",
    "Progress",
    "SchedulerSpec",
    "SegmentationAugment",
    "SegmentationRoiBatchSampler",
    "SegmentationRoiPopulation",
    "StepOutput",
    "Trainer",
    "augment_registry",
    "fit",
    "learner_for",
    "learner_registry",
    "load",
    "loaders",
    "optimizer_registry",
    "record_builder",
    "resolve_leaf",
    "resolve_segmentation_roi_population",
    "save",
    "scheduler_registry",
]
