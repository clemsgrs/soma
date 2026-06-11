"""soma — Modular experimentation framework for computational pathology."""

from soma.config import (
    AggregatorConfig,
    AugmentationConfig,
    CacheConfig,
    DecoderConfig,
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
from soma.dataset import Dataset, FoldSplit, SampleRecord, SegmentationManifest, Splits
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
from soma.dense.live import LiveSegmentationSource
from soma.training.model import (
    EmbeddingModel,
    EmbeddingModelOutput,
    LiveSegmentationModel,
    SegmentationModel,
    SegmentationModelOutput,
)
from soma.training.sample_dataset import SampleBatch, SampleDataset, sample_collate_fn
from soma.training.segmentation_dataset import (
    LiveSegmentationDataset,
    SegmentationBatch,
    SegmentationDataset,
    segmentation_collate_fn,
)
from soma.tasks import list_task_heads

__all__ = [
    # Config
    "AggregatorConfig",
    "AugmentationConfig",
    "CacheConfig",
    "DecoderConfig",
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
    "SegmentationManifest",
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
    "LiveSegmentationModel",
    "LiveSegmentationSource",
    "SegmentationModel",
    "SegmentationModelOutput",
    "SampleBatch",
    "SampleDataset",
    "sample_collate_fn",
    "LiveSegmentationDataset",
    "SegmentationBatch",
    "SegmentationDataset",
    "segmentation_collate_fn",
]
