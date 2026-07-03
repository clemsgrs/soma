"""soma — Modular experimentation framework for computational pathology."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("soma-pathology")
except PackageNotFoundError:  # not installed (e.g. running from a source checkout)
    __version__ = "0.0.0+unknown"

from soma.config import (
    AggregatorConfig,
    AttentionConfig,
    AugmentationConfig,
    CacheConfig,
    CompositeConfig,
    DecoderConfig,
    EncoderConfig,
    EncoderMemberConfig,
    EvalConfig,
    ExecutionConfig,
    MasksConfig,
    PipelineConfig,
    PixelClassifierConfig,
    PreprocessingConfig,
    PreviewConfig,
    SamplingConfig,
    SubgroupConfig,
    TaskConfig,
    TrainingConfig,
    HeatmapConfig,
)
from soma.encoders import list_models
from soma.aggregators import list_aggregators
from soma.dataset import (
    Dataset,
    DetectionManifest,
    FoldSplit,
    SampleRecord,
    SegmentationManifest,
    Splits,
)
from soma.decoders import list_decoders
from soma.dense import (
    CacheBackedDenseSource,
    DenseFeatureSource,
    DenseFeatureStore,
    DenseSourceProvenance,
)
from soma.dense_extraction import DenseTileFeatureExtractor
from soma.extraction import FeatureExtractor
from soma.features import FeatureStore
from soma.pixel_classifiers import list_pixel_classifiers
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
    "__version__",
    # Config
    "AggregatorConfig",
    "AttentionConfig",
    "AugmentationConfig",
    "CacheConfig",
    "CompositeConfig",
    "DecoderConfig",
    "EncoderConfig",
    "EncoderMemberConfig",
    "EvalConfig",
    "ExecutionConfig",
    "MasksConfig",
    "PipelineConfig",
    "PixelClassifierConfig",
    "PreprocessingConfig",
    "PreviewConfig",
    "SamplingConfig",
    "SubgroupConfig",
    "TaskConfig",
    "TrainingConfig",
    "HeatmapConfig",
    "list_models",
    "list_aggregators",
    "list_decoders",
    "list_pixel_classifiers",
    "list_task_heads",
    # Data
    "Dataset",
    "FoldSplit",
    "SampleRecord",
    "SegmentationManifest",
    "DetectionManifest",
    "Splits",
    "FeatureExtractor",
    "FeatureStore",
    "CacheBackedDenseSource",
    "DenseFeatureSource",
    "TileFeatureExtractor",
    "DenseFeatureStore",
    "DenseSourceProvenance",
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
