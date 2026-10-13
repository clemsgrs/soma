"""soma — Modular experimentation framework for computational pathology.

The root exports the verbs, the nouns they take or return, and the namespaces. Everything
else is reached through its namespace (``soma.models.ABMIL``, ``soma.learn.Trainer``,
``soma.evaluation.Baseline``, ``soma.data.from_arrays``); config sub-dataclasses live in
:mod:`soma.config`. The benchmark package is imported explicitly.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("soma-pathology")
except _PackageNotFoundError:  # not installed (e.g. running from a source checkout)
    __version__ = "0.0.0+unknown"

from soma.config import ExperimentConfig
from soma.discovery import components, register
from soma.plan import Plan, resolve
from soma import presets
from soma import data
from soma.data import (
    AnnotationManifest,
    Cohort,
    FoldSplit,
    GridSource,
    ImageManifest,
    ImageSource,
    SampleRecord,
    SetSource,
    TargetSource,
)
from soma import evaluation
from soma.evaluation import (
    Evaluation,
    LabelMapPredictions,
    PointPredictions,
    Predictions,
    evaluate,
    predict,
)
from soma import extract
from soma import models
from soma import learn
from soma.learn import FitResult, ModelBundle, fit, load, save
from soma.learn.seed import seed_everything
from soma import pipeline
from soma.pipeline import Pipeline, PipelineResult
from soma import testing

__all__ = [
    # verbs
    "extract",
    "fit",
    "predict",
    "evaluate",
    "save",
    "load",
    "seed_everything",
    "Pipeline",
    "resolve",
    "components",
    "register",
    # nouns
    "Cohort",
    "SampleRecord",
    "FoldSplit",
    "ImageManifest",
    "AnnotationManifest",
    "SetSource",
    "GridSource",
    "TargetSource",
    "ImageSource",
    "FitResult",
    "ModelBundle",
    "Predictions",
    "PointPredictions",
    "LabelMapPredictions",
    "Evaluation",
    "ExperimentConfig",
    "Plan",
    "PipelineResult",
    # namespaces (``extract`` is the verb above)
    "data",
    "models",
    "learn",
    "evaluation",
    "pipeline",
    "presets",
    "testing",
]
