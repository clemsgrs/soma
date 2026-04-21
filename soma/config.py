"""Frozen configuration dataclasses for soma experiments.

These dataclasses form the public configuration surface used by the pipeline,
the CLI, and the docs. Keep the field names stable and document any new field
here so the Sphinx reference stays accurate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
from hs2p import PreviewConfig

from soma.evaluation.metrics import resolve_metrics


def _default_preview_config() -> PreviewConfig:
    return PreviewConfig(
        save_mask_preview=True,
        save_tiling_preview=True,
        downsample=32,
        tissue_contour_color=(37, 94, 59),
        mask_overlay_alpha=0.5,
    )


@dataclass(frozen=True)
class PreprocessingConfig:
    """Whole-slide preprocessing, tiling, and geometry settings.

    The preprocessing backend controls tissue segmentation and tile
    extraction. ``requested_spacing_um`` and ``requested_tile_size_px`` are
    the primary scale-selection knobs. The hierarchical fields are used only
    when the aggregator requests HIPT-style region geometry. ``sam2_num_workers``
    caps concurrent SAM2 tissue-segmentation workers when the backend supports
    that execution path.
    """

    backend: str = "auto"
    requested_tile_size_px: int | None = None
    requested_spacing_um: float | None = None
    requested_region_size_px: int | None = None
    region_tile_multiple: int | None = None
    read_tile_size_px: int | None = None
    read_region_size_px: int | None = None
    tissue_method: str | None = None
    tissue_threshold: float = 0.1
    overlap: float = 0.0
    seg_downsample: int = 64
    sam2_num_workers: int | None = None
    tolerance: float = 0.05
    ref_tile_size_px: int | None = None
    a_t: int = 4
    tissue_mask_tissue_value: int = 1
    preview: PreviewConfig = field(default_factory=_default_preview_config)

    # Hierarchical (HIPT-style) fields — auto-derived from aggregator config
    hierarchical: bool = False
    npatch: int | None = None
    hierarchical_patch_size_px: int | None = None

    @property
    def requested_backend(self) -> str:
        """Backend requested by config before runtime auto-resolution."""
        return self.backend

    @property
    def has_hierarchical_geometry(self) -> bool:
        return self.region_tile_multiple is not None or self.requested_region_size_px is not None


@dataclass(frozen=True)
class ExecutionConfig:
    """Runtime execution settings for preprocessing and feature extraction."""

    num_gpus: int | None = None
    num_workers: int | None = None
    num_preprocessing_workers: int | None = None
    prefetch_factor: int | None = None
    persistent_workers: bool | None = None
    precision: str | None = None


@dataclass(frozen=True)
class EncoderConfig:
    """Foundation-model encoder selection and model-adjacent settings.

    ``name`` selects the encoder preset. ``spacing_um`` and ``input_size``
    describe the preset's native geometry and compatibility envelope. Other
    values may be accepted by a preset, but they should be treated as
    compatibility choices and may be suboptimal. ``output_variant`` exposes
    preset-specific feature variants when the encoder supports them.
    ``allow_non_recommended_settings`` opts into slide2vec's warning-only mode
    when intentionally sweeping non-default geometry or variant settings.
    """

    name: str
    precision: str | None = None
    batch_size: int = 32
    adaptive_batching: bool = False
    input_size: int | None = None
    spacing_um: float | None = None
    output_variant: str | None = None
    allow_non_recommended_settings: bool = False
    save_tile_features: bool = False


@dataclass(frozen=True)
class CacheConfig:
    """Shared cache policy for tiling and extracted features.

    Reusing the cache keeps repeated experiments from recomputing expensive
    tiling or embedding steps when the upstream configuration has not changed.
    """

    enabled: bool = True
    root_dir: str | Path | None = None
    reuse_policy: str = "strict"
    save_tile_features_for_slide: bool = True


@dataclass(frozen=True)
class AggregatorConfig:
    """MIL aggregator selection and constructor parameters.

    ``name`` selects the registered aggregator class. ``params`` are passed
    through to the aggregator constructor after the pipeline injects the
    feature dimension.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskConfig:
    """Task-head selection and constructor parameters.

    ``name`` selects the registered task head. ``params`` are merged with any
    dataset-derived auto parameters before instantiation.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubgroupConfig:
    """Columns used for subgroup metric breakdowns."""

    columns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation metrics and subgroup analysis configuration.

    Metrics are validated against the selected task family, and subgroup
    columns are used to break down the reported metrics in the run outputs.
    """

    metrics: list[str] = field(default_factory=list)
    subgroups: SubgroupConfig = field(default_factory=SubgroupConfig)


@dataclass(frozen=True)
class TrainingConfig:
    """Training-loop hyperparameters and optimizer settings.

    ``batch_size`` and ``gradient_accumulation`` control the effective batch
    size, while ``epochs``, ``learning_rate``, ``optimizer``, ``scheduler``,
    and ``patience`` define the optimization schedule. ``allow_missing_tune``
    enables a deliberate train-as-tune fallback when a fold has no tune split.
    """

    seed: int = 0
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    scheduler: str = "cosine"
    patience: int = 10
    batch_size: int = 1
    gradient_accumulation: int = 1
    allow_missing_tune: bool = False


@dataclass(frozen=True)
class HeatmapConfig:
    """Attention heatmap generation and rendering settings."""

    enabled: bool = False
    cmap: str = "coolwarm"
    alpha: float = 0.5
    blur_sigma: float = 0.0


@dataclass(frozen=True)
class PipelineConfig:
    """Complete specification for a pipeline run.

    Args:
        dataset_csv: Path to the dataset manifest.
        splits_csv: Path to the split manifest.
        output_root: Directory for the run outputs.
        dataset_type: Input mode for the pipeline. ``"slide"`` means whole
            slide bags with optional MIL aggregation, ``"tile"`` means
            patch-level classification, and ``"patient"`` means
            patient-level aggregation. ``aggregator`` must be ``None`` unless
            ``dataset_type`` is ``"slide"``.
        preprocessing: Whole-slide preprocessing and tiling settings.
        execution: Runtime execution settings for preprocessing and feature
            extraction.
        cache: Shared cache policy.
        encoder: Foundation-model encoder configuration, or ``None`` for
            workflows that do not need one.
        aggregator: MIL aggregator configuration for slide-level bag
            learning, or ``None`` for tile/patient pipelines.
        task: Task-head configuration. Required.
        evaluation: Metric and subgroup evaluation configuration.
        training: Training hyperparameters.
        heatmaps: Attention heatmap rendering settings.
        tags: Free-form labels attached to the experiment metadata.
    """

    dataset_csv: str | Path
    splits_csv: str | Path
    output_root: str | Path
    dataset_type: str
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    encoder: EncoderConfig | None = None
    aggregator: AggregatorConfig | None = None
    task: TaskConfig = field(default=None)  # type: ignore[assignment]
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    heatmaps: HeatmapConfig = field(default_factory=HeatmapConfig)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.task is None:
            raise TypeError("PipelineConfig requires a 'task' argument (e.g. TaskConfig(name='classification'))")
        _valid_dataset_types = {"slide", "tile", "patient"}
        if self.dataset_type not in _valid_dataset_types:
            raise ValueError(
                f"Invalid dataset_type {self.dataset_type!r}. "
                f"Must be one of: {sorted(_valid_dataset_types)}"
            )
        if self.dataset_type == "tile" and self.aggregator is not None:
            raise ValueError(
                "aggregator must be None for dataset_type='tile' — "
                "tile classifiers do not use MIL aggregation."
            )
        if self.dataset_type == "patient" and self.aggregator is not None:
            raise ValueError(
                "aggregator must be None for dataset_type='patient' — "
                "patient-level pipelines use a pretrained patient encoder, not a trainable aggregator."
            )
        # Validate that requested metrics are valid for the task family.
        resolve_metrics(self.task.name, self.evaluation.metrics)


# --- YAML serialization ---


def _config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    """Convert a PipelineConfig to a plain dict suitable for YAML."""
    return _normalize_yaml_value(asdict(config))


def _normalize_yaml_value(obj: Any) -> Any:
    """Recursively normalize dataclass output into YAML-safe primitives."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return [_normalize_yaml_value(value) for value in obj]
    if isinstance(obj, list):
        return [_normalize_yaml_value(value) for value in obj]
    if isinstance(obj, dict):
        return {key: _normalize_yaml_value(value) for key, value in obj.items()}
    return obj


def save_config(config: PipelineConfig, path: Path | str) -> None:
    """Serialize a PipelineConfig to a YAML file."""
    data = _config_to_dict(config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_config(path: Path | str) -> PipelineConfig:
    """Load a PipelineConfig from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return _dict_to_config(data)


def _load_task_config(data: dict[str, Any]) -> TaskConfig:
    task_data = data.get("task")
    if not task_data or "name" not in task_data:
        raise ValueError(
            "Config is missing required 'task.name' (e.g. task: {name: binary_classification})"
        )
    return TaskConfig(name=task_data["name"], params=task_data.get("params", {}))


def _load_evaluation_config(data: dict[str, Any]) -> EvalConfig:
    evaluation_data = data.get("evaluation", {})
    subgroup_data = evaluation_data.get("subgroups", {})
    columns = subgroup_data.get("columns", []) if subgroup_data else []
    return EvalConfig(
        metrics=evaluation_data.get("metrics", []),
        subgroups=SubgroupConfig(columns=columns),
    )


def _dict_to_config(data: dict[str, Any]) -> PipelineConfig:
    """Reconstruct a PipelineConfig from a plain dict."""
    encoder_data = data.get("encoder")
    heatmap_data = data.get("heatmaps")
    execution_data = data.get("execution") or {}
    preprocessing_data = dict(data.get("preprocessing", {}))
    preview_data = dict(preprocessing_data.pop("preview", {}))
    tissue_contour_color = preview_data.get("tissue_contour_color")
    if isinstance(tissue_contour_color, list):
        preview_data["tissue_contour_color"] = tuple(tissue_contour_color)
    return PipelineConfig(
        dataset_csv=data["dataset_csv"],
        splits_csv=data["splits_csv"],
        output_root=data["output_root"],
        dataset_type=data["dataset_type"],
        preprocessing=PreprocessingConfig(
            **preprocessing_data,
            preview=PreviewConfig(**preview_data),
        ),
        execution=ExecutionConfig(**execution_data),
        cache=CacheConfig(**data.get("cache", {})),
        encoder=EncoderConfig(**encoder_data) if encoder_data is not None else None,
        aggregator=AggregatorConfig(**data["aggregator"]) if data.get("aggregator") else None,
        task=_load_task_config(data),
        evaluation=_load_evaluation_config(data),
        training=TrainingConfig(**data.get("training", {})),
        heatmaps=HeatmapConfig(**heatmap_data) if heatmap_data is not None else HeatmapConfig(),
        tags=data.get("tags", []),
    )
