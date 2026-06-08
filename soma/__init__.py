"""soma — Modular experimentation framework for computational pathology."""

from soma.config import (
    AggregatorConfig,
    CacheConfig,
    EncoderConfig,
    EvalConfig,
    ExecutionConfig,
    PipelineConfig,
    PreprocessingConfig,
    PreviewConfig,
    SubgroupConfig,
    TaskConfig,
    TrainingConfig,
    HeatmapConfig,
)
from soma.encoders import list_models
from soma.aggregators import list_aggregators
from soma.dataset import Dataset, FoldSplit, SampleRecord, Splits
from soma.decoders import list_decoders
from soma.dense import DenseFeatureStore
from soma.dense_extraction import DenseTileFeatureExtractor
from soma.extraction import FeatureExtractor
from soma.features import FeatureStore
from soma.pipeline import FoldResult, Pipeline, PipelineResult, train, train_one_fold
from soma.preprocessing import (
    overlay_mask_on_slide,
    save_overlay_preview,
    write_coordinate_preview,
)
from soma.tile_extraction import TileFeatureExtractor
from soma.training.model import EmbeddingModel, EmbeddingModelOutput
from soma.training.sample_dataset import SampleBatch, SampleDataset, sample_collate_fn
from soma.tasks import list_task_heads

__all__ = [
    # Config
    "AggregatorConfig",
    "CacheConfig",
    "EncoderConfig",
    "EvalConfig",
    "ExecutionConfig",
    "PipelineConfig",
    "PreprocessingConfig",
    "PreviewConfig",
    "SubgroupConfig",
    "TaskConfig",
    "TrainingConfig",
    "HeatmapConfig",
    "list_models",
    "list_aggregators",
    "list_decoders",
    "list_task_heads",
    # Data
    "Dataset",
    "FoldSplit",
    "SampleRecord",
    "Splits",
    "FeatureExtractor",
    "FeatureStore",
    "TileFeatureExtractor",
    "DenseFeatureStore",
    "DenseTileFeatureExtractor",
    "FoldResult",
    "Pipeline",
    "PipelineResult",
    "train",
    "train_one_fold",
    "overlay_mask_on_slide",
    "save_overlay_preview",
    "write_coordinate_preview",
    "EmbeddingModel",
    "EmbeddingModelOutput",
    "SampleBatch",
    "SampleDataset",
    "sample_collate_fn",
]
