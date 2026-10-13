"""soma — Modular experimentation framework for computational pathology."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("soma-pathology")
except PackageNotFoundError:  # not installed (e.g. running from a source checkout)
    __version__ = "0.0.0+unknown"

from soma.config import CacheOptions, Component, EncoderLeaf, ExperimentConfig
from soma.discovery import components, register
from soma.plan import Plan, resolve
from soma import presets
from soma.data import (
    AnnotationManifest,
    CachedLabelMapSource,
    CachedSetSource,
    Cohort,
    FoldSplit,
    GridGeometry,
    GridSource,
    ImageManifest,
    ImageSource,
    LabelMapSource,
    PointSource,
    Points,
    SampleRecord,
    SetSource,
    TargetSource,
    TileImageSource,
    covers,
    from_arrays,
    from_directory,
    group_by,
    require_coverage,
)
from soma.evaluation import (
    Baseline,
    Evaluation,
    LabelMapPredictions,
    PointPredictions,
    Predictions,
    evaluate,
    predict,
)
from soma.dense import CachedGridSource, DenseSourceProvenance
from soma.extraction import (
    ExtractionArtifacts,
    FeatureExtractionResult,
    FeatureExtractor,
    FeatureProvenance,
    FeatureSource,
    PooledFeatureSource,
)
from soma.learn import (
    Budget,
    Checkpoint,
    DenseLearner,
    EarlyStopping,
    FitResult,
    History,
    Learner,
    MILLearner,
    ModelBundle,
    Progress,
    SchedulerSpec,
    StepOutput,
    Trainer,
    fit,
    load,
    save,
)
from soma.pipeline import FoldResult, Pipeline, PipelineResult
from soma.pipeline.representation import evaluate_representation
from soma.preprocessing import (
    overlay_mask_on_slide,
    save_overlay_preview,
    write_coordinate_preview,
)
from soma.training.model import (
    DenseModel,
    DenseModelOutput,
    EmbeddingModel,
    EmbeddingModelOutput,
)
from soma.training.sample_dataset import SampleBatch, SampleDataset, sample_collate_fn
from soma.training.seed import seed_everything

__all__ = [
    "__version__",
    # Config
    "CacheOptions",
    "Component",
    "EncoderLeaf",
    "ExperimentConfig",
    "Plan",
    "resolve",
    # Discovery
    "components",
    "register",
    "presets",
    # Data
    "AnnotationManifest",
    "Cohort",
    "FoldSplit",
    "GridGeometry",
    "ImageManifest",
    "ImageSource",
    "LabelMapSource",
    "CachedLabelMapSource",
    "PointSource",
    "Points",
    "SampleRecord",
    "SetSource",
    "TargetSource",
    "TileImageSource",
    "covers",
    "from_arrays",
    "from_directory",
    "group_by",
    "require_coverage",
    "FeatureExtractor",
    "ExtractionArtifacts",
    "FeatureExtractionResult",
    "FeatureProvenance",
    "FeatureSource",
    "PooledFeatureSource",
    "CachedSetSource",
    "GridSource",
    "CachedGridSource",
    "DenseSourceProvenance",
    # Learning
    "Budget",
    "Checkpoint",
    "DenseLearner",
    "EarlyStopping",
    "FitResult",
    "History",
    "Learner",
    "MILLearner",
    "ModelBundle",
    "Progress",
    "SchedulerSpec",
    "StepOutput",
    "Trainer",
    "fit",
    "load",
    "save",
    "seed_everything",
    # Evaluation
    "Baseline",
    "Evaluation",
    "LabelMapPredictions",
    "PointPredictions",
    "Predictions",
    "evaluate",
    "predict",
    "FoldResult",
    "Pipeline",
    "PipelineResult",
    "evaluate_representation",
    "overlay_mask_on_slide",
    "save_overlay_preview",
    "write_coordinate_preview",
    "EmbeddingModel",
    "EmbeddingModelOutput",
    "DenseModel",
    "DenseModelOutput",
    "SampleBatch",
    "SampleDataset",
    "sample_collate_fn",
]
