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
from soma.pipeline import FoldResult, Pipeline, PipelineResult, train, train_one_fold
from soma.training.slide_dataset import SlideBatch, SlideDataset, slide_collate_fn
from soma.training.slide_model import SlideModel, SlideModelOutput

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
    "FeatureExtractor",
    "FoldSplit",
    "SampleRecord",
    "Splits",
    "FeatureStore",
    # Pipeline
    "FoldResult",
    "Pipeline",
    "PipelineResult",
    "train",
    "train_one_fold",
    # Slide-level
    "SlideBatch",
    "SlideDataset",
    "SlideModel",
    "SlideModelOutput",
    "slide_collate_fn",
]
