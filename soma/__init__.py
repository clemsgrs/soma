"""soma — Modular experimentation framework for computational pathology."""

from soma.config import (
    AggregatorConfig,
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

__all__ = [
    # Config
    "AggregatorConfig",
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
]
