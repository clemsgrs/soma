"""Frozen configuration dataclasses for soma experiments.

These dataclasses form the public configuration surface used by the pipeline,
the CLI, and the docs. Keep the field names stable and document any new field
here so the Sphinx reference stays accurate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
import copy
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


def _config_resource_path() -> resources.abc.Traversable:
    return resources.files("soma.configs").joinpath("default.yaml")


@lru_cache(maxsize=1)
def _load_default_config_data() -> dict[str, Any]:
    with _config_resource_path().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Bundled default config must be a mapping")
    return data


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two plain dicts without mutating either input."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _layout_to_config_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize a user-supplied config into the canonical nested layout."""
    if not isinstance(data, dict):
        raise TypeError("Config file must contain a mapping at the top level")

    allowed_sections = {
        "run",
        "data",
        "preprocessing",
        "encoder",
        "aggregation",
        "task",
        "evaluation",
        "training",
        "execution",
        "cache",
        "reports",
    }
    unknown_keys = [key for key in data if key not in allowed_sections]
    if unknown_keys:
        raise ValueError(
            "Config file uses unsupported top-level keys: "
            + ", ".join(sorted(str(key) for key in unknown_keys))
            + ". Use the nested run/data/preprocessing/encoder/aggregation/task/"
            "evaluation/training/execution/cache/reports layout."
        )

    layout: dict[str, Any] = {}

    for section in ("run", "data", "reports"):
        if section in data:
            value = data[section]
            if not isinstance(value, dict):
                raise TypeError(f"Config section '{section}' must be a mapping")
            layout[section] = copy.deepcopy(value)

    for section in (
        "preprocessing",
        "encoder",
        "aggregation",
        "task",
        "evaluation",
        "training",
        "execution",
        "cache",
    ):
        if section in data:
            value = data[section]
            if section == "aggregation":
                if value is not None and not isinstance(value, dict):
                    raise TypeError("Config section 'aggregation' must be a mapping or null")
            elif value is not None and not isinstance(value, dict):
                raise TypeError(f"Config section '{section}' must be a mapping")
            layout[section] = copy.deepcopy(value)

    return layout


def _layout_to_pipeline_config(data: dict[str, Any]) -> PipelineConfig:
    run_data = data.get("run", {})
    data_data = data.get("data", {})
    preprocessing_data = dict(data.get("preprocessing", {}))
    preview_data = dict(preprocessing_data.pop("preview", {}))
    tissue_contour_color = preview_data.get("tissue_contour_color")
    if isinstance(tissue_contour_color, list):
        preview_data["tissue_contour_color"] = tuple(tissue_contour_color)

    reporting_data = data.get("reports", {})
    heatmap_data = reporting_data.get("heatmaps")
    training_data = dict(data.get("training", {}))
    if "seed" in run_data and "seed" not in training_data:
        training_data["seed"] = run_data["seed"]

    return PipelineConfig(
        dataset_csv=data_data["dataset_csv"],
        splits_csv=data_data["splits_csv"],
        output_root=run_data["output_root"],
        dataset_type=data_data["dataset_type"],
        preprocessing=PreprocessingConfig(
            **preprocessing_data,
            preview=PreviewConfig(**preview_data),
        ),
        execution=ExecutionConfig(**data.get("execution", {})),
        cache=CacheConfig(**data.get("cache", {})),
        encoder=EncoderConfig(**data["encoder"]) if data.get("encoder") is not None else None,
        aggregator=AggregatorConfig(**data["aggregation"]) if data.get("aggregation") else None,
        task=_load_task_config(data),
        evaluation=_load_evaluation_config(data),
        training=TrainingConfig(**training_data),
        heatmaps=HeatmapConfig(**heatmap_data) if heatmap_data is not None else HeatmapConfig(),
        tags=list(run_data.get("tags", [])),
    )


@dataclass(frozen=True)
class PreprocessingConfig:
    """Whole-slide preprocessing, tiling, and geometry settings.

    The preprocessing backend controls tissue segmentation and tile
    extraction. ``requested_spacing_um`` and ``requested_tile_size_px`` are
    the primary scale-selection knobs. The hierarchical fields are used only
    when the aggregator requests HIPT-style region geometry. ``sam2_device``
    and ``sam2_num_workers`` tune SAM2 tissue-segmentation execution when the
    backend supports that path.
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
    sam2_device: str = "cpu"
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
    """Runtime execution settings for preprocessing and feature extraction.

    ``num_workers_per_gpu`` is the CPU DataLoader budget for each GPU rank.
    ``None`` means auto-size the per-rank worker count from the available CPU
    budget and the resolved GPU count.
    """

    num_gpus: int | None = None
    num_workers_per_gpu: int | None = None
    num_preprocessing_workers: int | None = None
    prefetch_factor: int | None = None
    precision: str | None = None


@dataclass(frozen=True)
class EncoderConfig:
    """Foundation-model encoder selection and model-adjacent settings.

    ``name`` selects the encoder preset. ``output_variant`` exposes
    preset-specific feature variants when the encoder supports them.
    ``allow_non_recommended_settings`` opts into slide2vec's warning-only mode
    when intentionally sweeping non-default runtime settings.
    """

    name: str
    precision: str | None = None
    batch_size: int = 32
    adaptive_batching: bool = False
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


def _config_to_layout_dict(config: PipelineConfig) -> dict[str, Any]:
    """Convert a PipelineConfig into the canonical nested YAML layout."""
    data = {
        "run": {
            "output_root": _normalize_yaml_value(config.output_root),
            "seed": config.training.seed,
            "tags": _normalize_yaml_value(config.tags),
        },
        "data": {
            "dataset_csv": _normalize_yaml_value(config.dataset_csv),
            "splits_csv": _normalize_yaml_value(config.splits_csv),
            "dataset_type": config.dataset_type,
        },
        "preprocessing": _normalize_yaml_value(asdict(config.preprocessing)),
        "execution": _normalize_yaml_value(asdict(config.execution)),
        "cache": _normalize_yaml_value(asdict(config.cache)),
        "task": _normalize_yaml_value(asdict(config.task)),
        "evaluation": _normalize_yaml_value(asdict(config.evaluation)),
        # seed lives under run.seed in YAML; training.seed is excluded here to avoid
        # duplication — _layout_to_pipeline_config copies run.seed into TrainingConfig on load.
        "training": _normalize_yaml_value(
            {
                key: value
                for key, value in asdict(config.training).items()
                if key != "seed"
            }
        ),
        "reports": {
            "heatmaps": _normalize_yaml_value(asdict(config.heatmaps)),
        },
    }
    data["preprocessing"]["preview"] = _normalize_yaml_value(asdict(config.preprocessing.preview))
    data["encoder"] = (
        _normalize_yaml_value(asdict(config.encoder))
        if config.encoder is not None
        else None
    )
    data["aggregation"] = (
        _normalize_yaml_value(asdict(config.aggregator))
        if config.aggregator is not None
        else None
    )
    return data


def save_config(config: PipelineConfig, path: Path | str) -> None:
    """Serialize a PipelineConfig to a fully resolved nested YAML file."""
    data = _deep_merge_dicts(_load_default_config_data(), _config_to_layout_dict(config))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_config(path: Path | str) -> PipelineConfig:
    """Load a PipelineConfig from YAML, merging bundled defaults first."""
    with open(path) as f:
        raw_data = yaml.safe_load(f) or {}
    if not isinstance(raw_data, dict):
        raise TypeError("Config file must contain a top-level mapping")
    canonical = _deep_merge_dicts(_load_default_config_data(), _layout_to_config_dict(raw_data))
    return _layout_to_pipeline_config(canonical)


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
