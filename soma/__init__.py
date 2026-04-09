"""soma — Modular experimentation framework for computational pathology."""

from soma.config import (
    AggregatorConfig,
    CacheConfig,
    EncoderConfig,
    PipelineConfig,
    PreprocessingConfig,
    TaskConfig,
    TrainingConfig,
)
from soma.dataset import Dataset, FoldSplit, SampleRecord, Splits
from soma.extraction import FeatureExtractor
from soma.features import FeatureStore

try:
    from soma.pipeline import FoldResult, Pipeline, PipelineResult, train, train_one_fold
    from soma.training.slide_dataset import SlideBatch, SlideDataset, slide_collate_fn
    from soma.training.slide_model import SlideModel, SlideModelOutput
except ModuleNotFoundError:
    FoldResult = None
    Pipeline = None
    PipelineResult = None
    train = None
    train_one_fold = None
    SlideBatch = None
    SlideDataset = None
    SlideModel = None
    SlideModelOutput = None
    slide_collate_fn = None

__all__ = [
    # Config
    "AggregatorConfig",
    "CacheConfig",
    "EncoderConfig",
    "PipelineConfig",
    "PreprocessingConfig",
    "TaskConfig",
    "TrainingConfig",
    # Data
    "Dataset",
    "FoldSplit",
    "SampleRecord",
    "Splits",
    "FeatureExtractor",
    "FeatureStore",
]

if Pipeline is not None:
    __all__.extend(
        [
            "FoldResult",
            "Pipeline",
            "PipelineResult",
            "train",
            "train_one_fold",
            "SlideBatch",
            "SlideDataset",
            "SlideModel",
            "SlideModelOutput",
            "slide_collate_fn",
        ]
    )
