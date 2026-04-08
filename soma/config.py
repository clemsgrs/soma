"""Frozen dataclass configurations for soma experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for WSI preprocessing (tissue segmentation + tiling)."""

    backend: str = "auto"
    target_tile_size_px: int | None = None
    target_spacing_um: float | None = None
    target_region_size_px: int | None = None
    region_tile_multiple: int | None = None
    effective_tile_size_px: int | None = None
    effective_region_size_px: int | None = None
    tissue_method: str = "hsv"
    tissue_threshold: float = 0.1
    overlap: float = 0.0
    seg_downsample: int = 64
    tolerance: float = 0.05
    ref_tile_size_px: int | None = None
    a_t: int = 4
    tissue_mask_tissue_value: int = 1

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
        return self.region_tile_multiple is not None or self.target_region_size_px is not None


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for foundation model encoding."""

    name: str = "uni2"
    precision: str = "fp16"
    batch_size: int = 32
    adaptive_batching: bool = False
    num_workers: int = 4
    input_size: int | None = None
    spacing_um: float | None = None
    output_variant: str | None = None
    save_tile_features: bool = False


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for the shared feature cache."""

    enabled: bool = True
    root_dir: str | Path | None = None
    reuse_policy: str = "strict"
    save_tile_features_for_slide: bool = True


@dataclass(frozen=True)
class AggregatorConfig:
    """Configuration for the MIL aggregator."""

    name: str = "abmil"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for the task head."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the training loop."""

    seed: int = 0
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    scheduler: str = "cosine"
    patience: int = 10
    batch_size: int = 1


@dataclass(frozen=True)
class PipelineConfig:
    """Complete specification for a pipeline run."""

    dataset_csv: str | Path
    splits_csv: str | Path
    output_dir: str | Path
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    aggregator: AggregatorConfig | None = field(default_factory=AggregatorConfig)
    task: TaskConfig = field(default=None)  # type: ignore[assignment]
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.task is None:
            raise TypeError("PipelineConfig requires a 'task' argument (e.g. TaskConfig(name='classification'))")


# --- YAML serialization ---


def _config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    """Convert a PipelineConfig to a plain dict suitable for YAML."""
    data = asdict(config)
    _convert_paths(data)
    return data


def _convert_paths(obj: Any) -> None:
    """Recursively convert Path objects to strings in a dict."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, Path):
                obj[key] = str(value)
            elif isinstance(value, (dict, list)):
                _convert_paths(value)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            if isinstance(value, Path):
                obj[i] = str(value)
            elif isinstance(value, (dict, list)):
                _convert_paths(value)


def save_config(config: PipelineConfig, path: Path | str) -> None:
    """Serialize a PipelineConfig to a YAML file."""
    data = _config_to_dict(config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_config(path: Path | str) -> PipelineConfig:
    """Load a PipelineConfig from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return _dict_to_config(data)


def _load_task_config(data: dict[str, Any]) -> TaskConfig:
    task_data = data.get("task")
    if not task_data or "name" not in task_data:
        raise ValueError(
            "Config is missing required 'task.name' (e.g. task: {name: classification})"
        )
    return TaskConfig(**task_data)


def _dict_to_config(data: dict[str, Any]) -> PipelineConfig:
    """Reconstruct a PipelineConfig from a plain dict."""
    return PipelineConfig(
        dataset_csv=data["dataset_csv"],
        splits_csv=data["splits_csv"],
        output_dir=data["output_dir"],
        preprocessing=PreprocessingConfig(**data.get("preprocessing", {})),
        cache=CacheConfig(**data.get("cache", {})),
        encoder=EncoderConfig(**data.get("encoder", {})),
        aggregator=AggregatorConfig(**data["aggregator"]) if data.get("aggregator") else None,
        task=_load_task_config(data),
        training=TrainingConfig(**data.get("training", {})),
        tags=data.get("tags", []),
    )
