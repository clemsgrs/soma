"""soma — Modular experimentation framework for computational pathology."""

from soma.config import (
    AggregatorConfig,
    CacheConfig,
    EncoderConfig,
    EvalConfig,
    PipelineConfig,
    PreprocessingConfig,
    SubgroupConfig,
    TaskConfig,
    TrainingConfig,
    HeatmapConfig,
)
from soma.dataset import Dataset, FoldSplit, SampleRecord, Splits
from soma.extraction import FeatureExtractor
from soma.features import FeatureStore
from soma.pipeline import FoldResult, Pipeline, PipelineResult, train, train_one_fold
from soma.tile_extraction import TileFeatureExtractor
from soma.training.slide_dataset import SlideBatch, SlideDataset, slide_collate_fn
from soma.training.slide_model import SlideModel, SlideModelOutput
from soma.training.tile_dataset import TileBatch, TileDataset, tile_collate_fn
from soma.training.tile_model import TileClassifier, TileClassifierOutput

__all__ = [
    # Config
    "AggregatorConfig",
    "CacheConfig",
    "EncoderConfig",
    "EvalConfig",
    "PipelineConfig",
    "PreprocessingConfig",
    "SubgroupConfig",
    "TaskConfig",
    "TrainingConfig",
    "HeatmapConfig",
    # Data
    "Dataset",
    "FoldSplit",
    "SampleRecord",
    "Splits",
    "FeatureExtractor",
    "FeatureStore",
    "TileFeatureExtractor",
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
    "TileBatch",
    "TileClassifier",
    "TileClassifierOutput",
    "TileDataset",
    "tile_collate_fn",
]
