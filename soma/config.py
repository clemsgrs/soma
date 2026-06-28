"""Frozen configuration dataclasses for soma experiments.

These dataclasses form the public configuration surface used by the pipeline,
the CLI, and the docs. Keep the field names stable and document any new field
here so the Sphinx reference stays accurate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
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
        "composite",
        "aggregation",
        "decoder",
        "pixel_classifier",
        "task",
        "evaluation",
        "training",
        "execution",
        "cache",
        "augmentation",
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
        "decoder",
        "pixel_classifier",
        "task",
        "evaluation",
        "training",
        "execution",
        "cache",
        "augmentation",
    ):
        if section in data:
            value = data[section]
            if section in ("aggregation", "decoder", "pixel_classifier"):
                if value is not None and not isinstance(value, dict):
                    raise TypeError(f"Config section '{section}' must be a mapping or null")
            elif value is not None and not isinstance(value, dict):
                raise TypeError(f"Config section '{section}' must be a mapping")
            layout[section] = copy.deepcopy(value)

    if "composite" in data:
        value = data["composite"]
        if value is not None and not isinstance(value, dict):
            raise TypeError(
                "Config section 'composite' must be a mapping "
                "({encoders: [...], concat_resolution, concat_grid_size}) or null"
            )
        layout["composite"] = copy.deepcopy(value)

    return layout


def _encoder_member_from_dict(member: dict[str, Any]) -> "EncoderMemberConfig":
    """Build one composite member, rebuilding its nested ``attention`` sub-config."""
    member = dict(member)
    attention_data = dict(member.pop("attention", {}))
    return EncoderMemberConfig(**member, attention=AttentionConfig(**attention_data))


def _composite_from_dict(data: dict[str, Any]) -> "CompositeConfig":
    """Build a :class:`CompositeConfig` from the ``composite:`` mapping."""
    data = dict(data)
    members = data.pop("encoders", None)
    if not members or not isinstance(members, list):
        raise TypeError("composite.encoders must be a non-empty list of encoder mappings")
    grid_size = data.pop("concat_grid_size", None)
    if isinstance(grid_size, list):
        grid_size = tuple(grid_size)
    concat_resolution = data.pop("concat_resolution", None)
    if data:
        raise ValueError(
            "Config section 'composite' uses unsupported keys: "
            + ", ".join(sorted(str(key) for key in data))
        )
    return CompositeConfig(
        encoders=[_encoder_member_from_dict(member) for member in members],
        concat_resolution=concat_resolution,
        concat_grid_size=grid_size,
    )


def _normalize_label_mapping(entries: Any, *, field_name: str) -> Any:
    """Normalize a class→value mapping, tolerating hs2p's list-of-single-entry-mapping YAML
    style (``[{background: 0}, {tumor: 1}]``) so an hs2p ``masks`` block pastes in unchanged."""
    if entries is None or isinstance(entries, dict):
        return entries
    if isinstance(entries, (list, tuple)):
        merged: dict[str, Any] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise TypeError(
                    f"masks.{field_name} must be a mapping or a list of single-entry mappings"
                )
            merged.update(entry)
        return merged
    raise TypeError(f"masks.{field_name} must be a mapping or a list of single-entry mappings")


def _masks_from_dict(data: dict[str, Any]) -> "MasksConfig":
    data = dict(data)
    pixel_mapping = _normalize_label_mapping(data.pop("pixel_mapping", None), field_name="pixel_mapping")
    min_coverage = _normalize_label_mapping(data.pop("min_coverage", None), field_name="min_coverage")
    colors = _normalize_label_mapping(data.pop("colors", None), field_name="colors")
    if data:
        raise ValueError(
            "Config section 'masks' uses unsupported keys: "
            + ", ".join(sorted(str(key) for key in data))
            + ". Supported: pixel_mapping, min_coverage, colors."
        )
    if pixel_mapping is None:
        raise ValueError("masks.pixel_mapping is required (class name → mask pixel value).")
    return MasksConfig(
        pixel_mapping={str(k): int(v) for k, v in pixel_mapping.items()},
        min_coverage={str(k): float(v) for k, v in (min_coverage or {}).items()},
        colors=(
            {str(k): (list(v) if v is not None else None) for k, v in colors.items()}
            if colors is not None
            else None
        ),
    )


def _layout_to_pipeline_config(data: dict[str, Any]) -> PipelineConfig:
    run_data = data.get("run", {})
    data_data = data.get("data", {})
    preprocessing_data = dict(data.get("preprocessing", {}))
    preview_data = dict(preprocessing_data.pop("preview", {}))
    attention_data = dict(preprocessing_data.pop("attention", {}))
    # masks/sampling are nested under preprocessing (#109).
    masks_data = preprocessing_data.pop("masks", None)
    sampling_data = preprocessing_data.pop("sampling", None)
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
        feature_mode=data_data.get("feature_mode", "cached"),
        preprocessing=PreprocessingConfig(
            **preprocessing_data,
            attention=AttentionConfig(**attention_data),
            preview=PreviewConfig(**preview_data),
            masks=_masks_from_dict(masks_data) if masks_data else None,
            sampling=SamplingConfig(**sampling_data) if sampling_data else None,
        ),
        execution=ExecutionConfig(**data.get("execution", {})),
        cache=CacheConfig(**data.get("cache", {})),
        # Truthiness (like aggregation/decoder below): a blank/omitted `encoder:` is no
        # encoder, not an empty EncoderConfig — and with the neutral default there is no
        # name to fall back on.
        encoder=EncoderConfig(**data["encoder"]) if data.get("encoder") else None,
        composite=_composite_from_dict(data["composite"]) if data.get("composite") else None,
        aggregator=AggregatorConfig(**data["aggregation"]) if data.get("aggregation") else None,
        decoder=DecoderConfig(**data["decoder"]) if data.get("decoder") else None,
        pixel_classifier=(
            PixelClassifierConfig(**data["pixel_classifier"])
            if data.get("pixel_classifier")
            else None
        ),
        task=_load_task_config(data),
        evaluation=_load_evaluation_config(data),
        training=TrainingConfig(**training_data),
        heatmaps=HeatmapConfig(**heatmap_data) if heatmap_data is not None else HeatmapConfig(),
        augmentation=AugmentationConfig(**data.get("augmentation", {})),
        tags=list(run_data.get("tags", [])),
    )


@dataclass(frozen=True)
class AttentionConfig:
    """Which frozen-encoder self-attention to extract (``feature_kind='cls_attention'``).

    The attention-map segmentation path (re-impl of arXiv:2602.18747) treats a ViT's
    per-head prefix-token self-attention as a dense ``(K, gh, gw)`` feature grid. These
    knobs select *which* attention:

    * ``blocks`` — which transformer blocks' attention to capture (negative indexes
      from the end; ``(-1,)`` = last block, the paper's choice). Multiple blocks are
      concatenated along the channel axis in the listed order.
    * ``include_registers`` — also keep the register-token query rows (Darcet et al.)
      as extra channels, not just the CLS row. No-op for models without registers.

    Per-head is always preserved (head specialization is the signal); channels are
    ordered ``[block][cls, reg…][head]`` and that order is recorded in the grid sidecar.
    """

    blocks: tuple[int, ...] = (-1,)
    include_registers: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", tuple(int(b) for b in self.blocks))
        if not self.blocks:
            raise ValueError("attention.blocks must list at least one block index")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Whole-slide preprocessing, tiling, and geometry settings.

    The preprocessing backend controls tissue segmentation and tile
    extraction. ``requested_spacing_um`` and ``requested_tile_size_px`` are
    the primary scale-selection knobs. ``requested_region_size_px`` and
    ``region_tile_multiple`` describe HIPT-style hierarchical region geometry.
    ``sam2_device`` and ``sam2_num_workers`` tune SAM2 tissue-segmentation
    execution when the backend supports that path.
    """

    backend: str = "auto"
    requested_tile_size_px: int | None = None
    requested_spacing_um: float | None = None
    requested_region_size_px: int | None = None
    region_tile_multiple: int | None = None
    read_tile_size_px: int | None = None
    read_region_size_px: int | None = None
    tissue_method: str | None = None
    # Tissue coverage threshold expressed as a masks-shaped map (mirrors hs2p's
    # ``TilingConfig.min_coverage``); ``min_coverage["tissue"]`` is the minimum tissue
    # fraction to keep a tile. The single source of truth for the threshold across both the
    # pooled and dense seams — there is no separate scalar. (The per-class segmentation
    # coverage map lives on the top-level ``MasksConfig.min_coverage``, a distinct block.)
    min_coverage: dict[str, float] = field(default_factory=lambda: {"tissue": 0.1})
    overlap: float = 0.0
    seg_downsample: int = 64
    sam2_device: str = "cpu"
    sam2_num_workers: int | None = None
    tolerance: float = 0.05
    ref_tile_size_px: int | None = None
    a_t: int = 4
    tissue_mask_tissue_value: int = 1
    # Dense (segmentation) encoder-window knobs (design §5, window-as-knob). These are
    # NOT the tiling ``overlap`` above: they control how the padded supervision tile
    # reaches the frozen encoder. ``dense_window_size=None`` ⇒ ``whole`` (one padded
    # forward); a smaller window slides the encoder over patch-aligned windows and
    # blends the token grids over ``dense_window_overlap``.
    dense_window_size: int | None = None
    dense_window_overlap: float = 0.0
    # What the dense (segmentation) encoder emits per tile (design — attention-pixel
    # segmentation §3/§7). ``patch_features`` = the ViT patch-token grid (the existing
    # decoder path); ``cls_attention`` = per-head prefix-token self-attention as a
    # ``(K, gh, gw)`` grid (the pixel-classifier path). ``None`` = auto: the pipeline
    # cross-defaults it from the trainable component (decoder ⇒ patch_features,
    # pixel_classifier ⇒ cls_attention) — both overridable by setting this explicitly.
    # Orthogonal to the component (either grid feeds a decoder or a pixel-classifier).
    # ``attention`` selects which blocks / whether to keep register rows for cls_attention.
    feature_kind: str | None = None
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    preview: PreviewConfig = field(default_factory=_default_preview_config)
    # Segmentation slide-manifest ingestion (design — segmentation ingestion §5/§8; #109).
    # ``masks``/``sampling`` are a tiling/preprocessing concern (annotation-based tile
    # selection mirrors slide2vec's own PreprocessingConfig), so they nest here rather than
    # at the top level. The presence of ``masks`` selects the slide-manifest input mode
    # (slides + annotation masks → soma-sampled ROIs). Cross-field rules (segmentation-only,
    # ``sampling`` requires ``masks``, ``per_annotation`` deferred) live on PipelineConfig.
    masks: MasksConfig | None = None
    sampling: SamplingConfig | None = None

    def __post_init__(self) -> None:
        _valid_feature_kinds = {None, "patch_features", "cls_attention"}
        if self.feature_kind not in _valid_feature_kinds:
            raise ValueError(
                f"Invalid feature_kind {self.feature_kind!r}; must be 'patch_features', "
                "'cls_attention', or None (auto)."
            )
        if self.dense_window_size is not None and int(self.dense_window_size) <= 0:
            raise ValueError(
                f"dense_window_size must be a positive int or None, got {self.dense_window_size!r}"
            )
        if not (0.0 <= float(self.dense_window_overlap) < 1.0):
            raise ValueError(
                f"dense_window_overlap must be in [0, 1), got {self.dense_window_overlap!r}"
            )
        if self.dense_window_size is None and float(self.dense_window_overlap) != 0.0:
            raise ValueError(
                "dense_window_overlap requires dense_window_size — overlap is meaningless "
                "for the 'whole' path (no windows). Set dense_window_size or clear the overlap."
            )
        coverage = {str(k): float(v) for k, v in self.min_coverage.items()}
        for label, frac in coverage.items():
            if not 0.0 <= frac <= 1.0:
                raise ValueError(
                    f"preprocessing.min_coverage['{label}'] must be in [0, 1], got {frac!r}."
                )
        object.__setattr__(self, "min_coverage", coverage)

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
class EncoderMemberConfig:
    """One member of a multi-encoder composite (``composite.encoders`` list, design §7).

    Each member carries its *own* extraction spec — ``name``, ``feature_kind``, and
    ``attention`` knobs — so the heterogeneous paper setup (different FMs, some attention,
    some embedding) needs no special-casing. Members are extracted into independent caches
    and concatenated at load time.

    ``feature_kind`` is ``None`` (auto) by default and **cross-defaults from the
    consumer** at config finalization — ``cls_attention`` when a ``pixel_classifier`` is
    set (the decoder-free path), ``patch_features`` for the trained decoder / detection —
    mirroring the single-encoder cross-default. An explicit value wins.

    ``member_norm`` (``{none, l2, layernorm}``) is the per-member normalization applied at
    load time **before concat** (after the resample), so one large-magnitude encoder does
    not dominate the decoder. ``None`` (auto) defaults by the resolved ``feature_kind``:
    ``l2`` for ``patch_features`` (raw tokens differ wildly in norm), ``none`` for
    ``cls_attention`` (already bounded/comparable — keeps the pixel-classifier path
    byte-identical).

    ``dense_window_size`` / ``dense_window_overlap`` are the **per-member** sliding-window
    knobs for dense extraction: encoders that lack ``dynamic_img_size`` (e.g. CONCH at its
    native 448, H0-mini at 224) must slide their native window over the supervision tile,
    and different members need different windows. ``None`` falls back to the run's shared
    ``preprocessing.dense_window_size`` / ``dense_window_overlap``.
    """

    name: str
    feature_kind: str | None = None
    member_norm: str | None = None
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    output_variant: str | None = None
    precision: str | None = None
    batch_size: int = 32
    adaptive_batching: bool = False
    allow_non_recommended_settings: bool = False
    dense_window_size: int | None = None
    dense_window_overlap: float | None = None

    def __post_init__(self) -> None:
        if self.feature_kind not in {None, "patch_features", "cls_attention"}:
            raise ValueError(
                f"EncoderMemberConfig.feature_kind must be 'patch_features', "
                f"'cls_attention', or None (auto), got {self.feature_kind!r}."
            )
        if self.member_norm not in {None, "none", "l2", "layernorm"}:
            raise ValueError(
                f"EncoderMemberConfig.member_norm must be 'none', 'l2', 'layernorm', or "
                f"None (auto), got {self.member_norm!r}."
            )


@dataclass(frozen=True)
class CompositeConfig:
    """Multi-encoder composite block (design §7 Angle 5).

    Concatenates several frozen encoders' dense grids into one ``(Σd_i, ·, ·)`` feature
    grid. ``encoders`` lists the members (extracted into independent caches); the two
    knobs control how their grids are combined at load time:

    - ``concat_resolution`` (``{grid, target}``): ``grid`` resamples each member's native
      token grid to a common ``(h, w)`` and concatenates there (the trained decoder /
      detection consume token grids); ``target`` upsamples every member to the mask
      ``target_size`` first (the decoder-free per-pixel classifier). ``None`` (auto)
      resolves by consumer: ``target`` when a ``pixel_classifier`` is set, else ``grid``.
    - ``concat_grid_size``: the common ``(h, w)`` for ``grid`` mode. ``None`` defaults to
      the largest member grid (finest available token resolution). Ignored in ``target``
      mode.
    """

    encoders: list[EncoderMemberConfig]
    concat_resolution: str | None = None
    concat_grid_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not self.encoders:
            raise ValueError("composite.encoders must list at least one member encoder.")
        if self.concat_resolution not in {None, "grid", "target"}:
            raise ValueError(
                f"composite.concat_resolution must be 'grid', 'target', or None (auto), "
                f"got {self.concat_resolution!r}."
            )
        if self.concat_grid_size is not None:
            h, w = self.concat_grid_size
            object.__setattr__(self, "concat_grid_size", (int(h), int(w)))
            if int(h) <= 0 or int(w) <= 0:
                raise ValueError(
                    f"composite.concat_grid_size must be positive, got {self.concat_grid_size}."
                )


@dataclass(frozen=True)
class CacheConfig:
    """Shared cache policy for tiling and extracted features.

    Reusing the cache keeps repeated experiments from recomputing expensive
    tiling or embedding steps when the upstream configuration has not changed.
    """

    enabled: bool = True
    root_dir: str | Path | None = None
    reuse_policy: str = "strict"
    fingerprint_files: bool = False
    validate_payloads: bool = False
    # On-disk storage dtype for ALL feature caches (pooled tile/bag/slide/patient/
    # hierarchical AND dense grids): None ⇒ follow the compute precision (fp16 run →
    # fp16 features, else fp32), 'fp16'/'fp32' force it. fp16 halves cache size + the
    # per-epoch read I/O. Folded into every cache key (guarded so legacy fp32 keys stay
    # byte-stable) so an fp16 cache and an fp32 cache never collide.
    dtype: str | None = None


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
class DecoderConfig:
    """Dense decoder selection and constructor parameters (segmentation).

    ``name`` selects the registered decoder class. ``params`` are passed through to
    the decoder constructor after the pipeline injects the feature dimension and
    ``num_classes``. The decoder is the segmentation counterpart of ``aggregator``.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PixelClassifierConfig:
    """Per-pixel classifier selection and constructor parameters (segmentation).

    The decoder-free analog of ``decoder`` (design — attention-pixel segmentation §6):
    instead of a neural decoder + Trainer, a per-pixel ``(K,) → class`` classifier
    (XGBoost / random forest / logistic / pointwise MLP) consumes the dense grid. ``name``
    selects the registered classifier; ``params`` pass through to its constructor after the
    pipeline injects ``num_classes``. Mutually exclusive with ``decoder`` under
    ``dataset_type='segmentation'``.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MasksConfig:
    """Annotation-mask → class scheme for the segmentation slide-manifest input mode.

    Mirrors hs2p's ``masks`` config 1:1 and is forwarded untouched into hs2p annotation
    sampling (design — segmentation ingestion §5/§8). Its presence selects the slide-manifest
    input mode: ``dataset.csv`` rows are ``(sample_id, image_path (WSI), mask_path (annotation
    WSI))`` and soma samples ROIs from each slide, instead of the pre-cropped tile manifest.

    * ``pixel_mapping`` — class name → mask pixel value; must be non-empty with unique pixel
      values. No reserved label name is required: a background-free vocabulary like
      ``{tumor: 2}`` is accepted (``background`` stays an opt-in name for the ignore-label
      remap mode — see :func:`soma.dense.reader.build_label_remap`).
    * ``min_coverage`` — per-class minimum tile coverage (in ``[0, 1]``) to sample a tile;
      keys must be a subset of ``pixel_mapping``.
    * ``colors`` — optional class → ``[r, g, b]`` (or ``None``) overlay color for mask previews;
      keys must be a subset of ``pixel_mapping``.
    """

    pixel_mapping: dict[str, int]
    min_coverage: dict[str, float] = field(default_factory=dict)
    colors: dict[str, list[int] | None] | None = None

    def __post_init__(self) -> None:
        if not self.pixel_mapping:
            raise ValueError("masks.pixel_mapping is required and must be non-empty.")
        values = list(self.pixel_mapping.values())
        if len(set(values)) != len(values):
            raise ValueError(
                "masks.pixel_mapping must use unique pixel values (no two labels may "
                f"share a raw value): {self.pixel_mapping!r}."
            )
        unknown = sorted(set(self.min_coverage) - set(self.pixel_mapping))
        if unknown:
            raise ValueError(
                "masks.min_coverage references labels absent from pixel_mapping: "
                + ", ".join(unknown)
            )
        for label, frac in self.min_coverage.items():
            if not 0.0 <= float(frac) <= 1.0:
                raise ValueError(
                    f"masks.min_coverage['{label}'] must be in [0, 1], got {frac!r}."
                )
        if self.colors is not None:
            unexpected = sorted(set(self.colors) - set(self.pixel_mapping))
            if unexpected:
                raise ValueError(
                    "masks.colors references labels absent from pixel_mapping: "
                    + ", ".join(unexpected)
                )
            for label, color in self.colors.items():
                if color is None:
                    continue
                if (
                    not isinstance(color, (list, tuple))
                    or len(color) != 3
                    or any(not isinstance(c, int) or isinstance(c, bool) or c < 0 or c > 255 for c in color)
                ):
                    raise ValueError(
                        f"masks.colors['{label}'] must be None or a length-3 RGB list of "
                        f"ints in [0, 255], got {color!r}."
                    )


@dataclass(frozen=True)
class SamplingConfig:
    """ROI sampling strategy for the segmentation slide-manifest mode (sibling of ``masks``).

    * ``strategy`` — ``joint`` (tile the union of all classes, then keep tiles passing any
      class's ``min_coverage``) or ``independent`` (one sampling pass per class).
    * ``output_mode`` — ``merged`` (one merged coordinate set per slide: the union of tiles
      passing any active class threshold, collapsed to a single per-slide result; each tile is
      encoded once with its full multi-class mask attached downstream — the dense-segmentation
      contract) or ``per_annotation`` (one set per ``(slide, class)``). Forwarded into hs2p
      ``CoordinateOutputMode`` by the segmentation pipeline. ``per_annotation`` feature
      extraction is deferred (soma issue #86).
    """

    strategy: str = "joint"
    output_mode: str = "merged"

    def __post_init__(self) -> None:
        if self.strategy not in {"joint", "independent"}:
            raise ValueError(
                f"sampling.strategy must be 'joint' or 'independent', got {self.strategy!r}."
            )
        if self.output_mode not in {"merged", "per_annotation"}:
            raise ValueError(
                f"sampling.output_mode must be 'merged' or 'per_annotation', got "
                f"{self.output_mode!r}."
            )


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
    ``save_probabilities`` (segmentation only) additionally writes a per-tile
    float16 ``(C, H, W)`` softmax sidecar under ``probs/`` — opt-in because it is
    ~C×/precision× larger than the always-written argmax raster, and unlocks
    post-hoc soft-Dice/calibration/entropy/ensembling without re-running inference.
    ``holdout_test`` skips *all* test-split work (no test inference, no
    ``predictions_test.csv``, no ``test`` entries in ``metrics.json``/``summary.json``)
    and reports tune only — the model-selection protocol for benchmark sweeps: rank
    every candidate by tune score, then re-run only the winner with the test held out
    in. The test split may still be declared in ``splits.csv``; it is simply not
    touched. Tune evaluation, threshold sweeps, checkpoint selection, and training
    are unaffected.
    """

    metrics: list[str] = field(default_factory=list)
    subgroups: SubgroupConfig = field(default_factory=SubgroupConfig)
    save_probabilities: bool = False
    holdout_test: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    """Training-loop hyperparameters and optimizer settings.

    ``batch_size`` and ``gradient_accumulation`` control the effective batch
    size, while ``epochs``, ``learning_rate``, ``optimizer``, ``scheduler``,
    and ``patience`` define the optimization schedule. ``monitor`` and
    ``monitor_mode`` choose the tune loss or metric used for selected-checkpoint
    selection and early stopping. ``tune_is_test`` ties the tune and test
    splits to the same samples for protocols with a single held-out set: a fold
    may provide either a tune split or a test split (not both), and that split
    is used for both checkpoint selection and test reporting. ``allow_missing_tune``
    enables a deliberate train-as-tune fallback when a fold has no tune split.
    """

    seed: int = 0
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    scheduler: str = "cosine"
    patience: int = 10
    monitor: str = "tune_loss"
    monitor_mode: str = "min"
    batch_size: int = 1
    gradient_accumulation: int = 1
    tune_is_test: bool = False
    allow_missing_tune: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    # Class-stratified pixel-sampling budget for the pixel-classifier segmentation path
    # (design §9): the total number of supervised pixels drawn across the train cohort to
    # fit a per-pixel classifier (XGBoost/RF/logreg/MLP). Eval still predicts *every*
    # pixel. Ignored by the neural decoder/Trainer path.
    max_train_pixels: int = 2_000_000

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("TrainingConfig.epochs must be >= 1")
        if self.max_train_pixels < 1:
            raise ValueError("TrainingConfig.max_train_pixels must be >= 1")
        if self.batch_size < 1:
            raise ValueError("TrainingConfig.batch_size must be >= 1")
        if self.gradient_accumulation < 1:
            raise ValueError("TrainingConfig.gradient_accumulation must be >= 1")
        if self.patience < 1:
            raise ValueError("TrainingConfig.patience must be >= 1")
        if not self.monitor:
            raise ValueError("TrainingConfig.monitor must be non-empty")
        if self.monitor_mode not in {"min", "max"}:
            raise ValueError("TrainingConfig.monitor_mode must be 'min' or 'max'")
        if self.num_workers < 0:
            raise ValueError("TrainingConfig.num_workers must be >= 0")


@dataclass(frozen=True)
class AugmentationConfig:
    """Image/mask augmentation for the live segmentation path (``feature_mode='live'``).

    Augmentation is only possible when tiles are re-encoded each step (live), never
    on cached grids (the encoder is frozen-but-applied per step — see design §4/§13.B).
    Geometric ops (``horizontal_flip``/``vertical_flip``/``rotation_degrees``/
    ``translate``/``scale``) apply jointly to the image **and** mask (mask resampled
    nearest-neighbor automatically); photometric ops (``brightness``/``contrast``/
    ``saturation``/``hue``) apply to the **image only**. ``rotation_degrees``,
    ``translate``, and ``scale`` together drive a single ``RandomAffine``; affine
    out-of-canvas pixels fill the image with 0 and the mask with ``ignore_index`` (so
    they are excluded from loss/metrics). All-default = no-op (legal: the live-no-aug
    parity / future ``sliding_window`` case).
    """

    horizontal_flip: float = 0.0  # probability in [0, 1]
    vertical_flip: float = 0.0  # probability in [0, 1]
    rotation_degrees: float = 0.0  # RandomAffine symmetric degree range (±value)
    translate: float = 0.0  # RandomAffine max translate fraction (both axes)
    scale: float = 0.0  # RandomAffine scale jitter: scale ∈ (1-value, 1+value)
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0

    def __post_init__(self) -> None:
        for name in ("horizontal_flip", "vertical_flip"):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"AugmentationConfig.{name} must be in [0, 1], got {value}")
        for name in ("rotation_degrees", "translate", "scale", "brightness", "contrast", "saturation", "hue"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"AugmentationConfig.{name} must be >= 0, got {getattr(self, name)}")
        if float(self.translate) > 1.0:
            raise ValueError(f"AugmentationConfig.translate must be in [0, 1], got {self.translate}")
        if float(self.scale) >= 1.0:
            raise ValueError(f"AugmentationConfig.scale must be in [0, 1), got {self.scale}")
        if float(self.hue) > 0.5:
            raise ValueError(f"AugmentationConfig.hue must be in [0, 0.5], got {self.hue}")

    def is_enabled(self) -> bool:
        """True if any field departs from its no-op default."""
        return any(
            float(v) != 0.0
            for v in (
                self.horizontal_flip,
                self.vertical_flip,
                self.rotation_degrees,
                self.translate,
                self.scale,
                self.brightness,
                self.contrast,
                self.saturation,
                self.hue,
            )
        )


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
    feature_mode: str = "cached"
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    encoder: EncoderConfig | None = None
    composite: CompositeConfig | None = None
    aggregator: AggregatorConfig | None = None
    decoder: DecoderConfig | None = None
    pixel_classifier: PixelClassifierConfig | None = None
    task: TaskConfig = field(default=None)  # type: ignore[assignment]
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    heatmaps: HeatmapConfig = field(default_factory=HeatmapConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.task is None:
            raise TypeError("PipelineConfig requires a 'task' argument (e.g. TaskConfig(name='classification'))")
        _valid_dataset_types = {"slide", "tile", "patient", "segmentation", "detection"}
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
        if self.dataset_type == "segmentation":
            # Exactly one trainable component: a neural decoder (Trainer path) XOR a
            # per-pixel classifier (decoder-free path). They are the two orthogonal
            # "axis 2" options (design §7); never both, never neither.
            if self.decoder is not None and self.pixel_classifier is not None:
                raise ValueError(
                    "dataset_type='segmentation' takes a decoder XOR a pixel_classifier, "
                    "not both — a decoder is the neural (Trainer) path, a pixel_classifier "
                    "is the decoder-free per-pixel path."
                )
            if self.decoder is None and self.pixel_classifier is None:
                raise ValueError(
                    "dataset_type='segmentation' requires either a decoder (neural path) "
                    "or a pixel_classifier (decoder-free attention-map path)."
                )
            if self.aggregator is not None:
                raise ValueError(
                    "aggregator must be None for dataset_type='segmentation' — "
                    "segmentation uses a decoder or pixel-classifier (dense per-pixel), "
                    "not MIL aggregation."
                )
            if self.task.name != "segmentation":
                raise ValueError(
                    "dataset_type='segmentation' requires task.name='segmentation', "
                    f"got {self.task.name!r}."
                )
            # Cross-default feature_kind from the component when the user left it auto
            # (None): a pixel_classifier wants per-head CLS-attention, a decoder wants the
            # patch-feature grid. Both remain overridable (an explicit feature_kind wins).
            if self.preprocessing.feature_kind is None:
                resolved_kind = (
                    "cls_attention" if self.pixel_classifier is not None else "patch_features"
                )
                object.__setattr__(
                    self,
                    "preprocessing",
                    replace(self.preprocessing, feature_kind=resolved_kind),
                )
            if self.pixel_classifier is not None and self.feature_mode != "cached":
                raise ValueError(
                    "pixel_classifier requires feature_mode='cached' — the decoder-free "
                    "path fits on cached dense grids (no live re-encode / augmentation; "
                    "those belong to the neural-decoder path)."
                )
        elif self.dataset_type == "detection":
            # Detection-v1 reuses the segmentation dense front half: a decoder regresses a
            # per-class peak heatmap that the DetectionHead turns into points (design §6).
            if self.decoder is None:
                raise ValueError(
                    "dataset_type='detection' requires a decoder (the heatmap regressor; "
                    "e.g. decoder: {name: lightweight_conv})."
                )
            if self.pixel_classifier is not None:
                raise ValueError(
                    "pixel_classifier must be None for dataset_type='detection' — detection "
                    "uses a neural decoder + DetectionHead, not the decoder-free path."
                )
            if self.aggregator is not None:
                raise ValueError(
                    "aggregator must be None for dataset_type='detection' — detection is a "
                    "dense per-pixel task, not MIL aggregation."
                )
            if self.task.name != "detection":
                raise ValueError(
                    "dataset_type='detection' requires task.name='detection', "
                    f"got {self.task.name!r}."
                )
            # Detection consumes the ViT patch-token grid (same as the seg decoder path).
            if self.preprocessing.feature_kind is None:
                object.__setattr__(
                    self,
                    "preprocessing",
                    replace(self.preprocessing, feature_kind="patch_features"),
                )
            if self.feature_mode != "cached":
                raise ValueError(
                    "dataset_type='detection' v1 is cached-only — live re-encode + "
                    "geometric point-target augmentation is a deferred increment."
                )
        else:
            if self.decoder is not None:
                raise ValueError(
                    f"decoder must be None for dataset_type={self.dataset_type!r} — "
                    "decoders are only used for dataset_type='segmentation' or 'detection'."
                )
            if self.pixel_classifier is not None:
                raise ValueError(
                    f"pixel_classifier must be None for dataset_type={self.dataset_type!r} — "
                    "pixel-classifiers are only used for dataset_type='segmentation'."
                )
        # preprocessing.masks/preprocessing.sampling are the segmentation slide-manifest
        # ingestion mode (design — segmentation ingestion §5/§8; relocated under
        # preprocessing in #109). The presence of masks selects slide-manifest input
        # (slides + annotation masks → soma-sampled ROIs); both only apply to segmentation.
        masks = self.preprocessing.masks
        sampling = self.preprocessing.sampling
        # A masks block drives annotation-based tile selection. For 'segmentation' it is the
        # slide-manifest ingestion mode (slides + masks → soma-sampled ROIs); for 'slide' it
        # restricts the merged MIL bag to the selected compartment(s) (#110); for 'patient' it
        # restricts every slide's merged bag the same way (tiling/selection identical to
        # 'slide'), and patient-level aggregation then consumes those restricted slide bags
        # (#111). All forward the block into slide2vec's annotation sampling. Tile/detection
        # have no annotation-sampling step, so a masks block there is a config error.
        _masks_dataset_types = {"slide", "patient", "segmentation"}
        if masks is not None and self.dataset_type not in _masks_dataset_types:
            raise ValueError(
                "masks: (annotation-based tile selection) is only valid for "
                f"dataset_type in {sorted(_masks_dataset_types)}, got "
                f"dataset_type={self.dataset_type!r}."
            )
        if sampling is not None and masks is None:
            raise ValueError(
                "sampling: requires a masks: block — it configures how ROIs are sampled from "
                "the annotation masks. Add masks: or drop sampling:."
            )
        if (
            masks is not None
            and sampling is not None
            and sampling.output_mode == "per_annotation"
        ):
            raise ValueError(
                "sampling.output_mode='per_annotation' is not yet supported for feature "
                "extraction (deferred — see soma issue #86). Use 'merged' (one tile encoded "
                "once with its full multi-class mask attached downstream)."
            )
        # feature_mode / augmentation (the live re-encode segmentation path, §13.B).
        # `cached` reads pre-extracted dense grids; `live` re-encodes augmented tiles
        # through the frozen encoder every step. Fail loud rather than silently no-op.
        if self.feature_mode not in {"cached", "live"}:
            raise ValueError(
                f"Invalid feature_mode {self.feature_mode!r}; must be 'cached' or 'live'."
            )
        if self.feature_mode == "live" and self.dataset_type != "segmentation":
            raise ValueError(
                "feature_mode='live' is only supported for dataset_type='segmentation' "
                f"(re-encoding augmented tiles), got dataset_type={self.dataset_type!r}."
            )
        if self.augmentation.is_enabled():
            if self.feature_mode != "live":
                raise ValueError(
                    "augmentation requires feature_mode='live' — cached dense grids cannot "
                    "be augmented (the features are frozen). Set feature_mode='live' or clear "
                    "the augmentation section."
                )
            # dataset_type is already guaranteed 'segmentation' by the live check above.
        if self.task.name == "survival":
            if self.dataset_type == "tile":
                raise ValueError(
                    "dataset_type='tile' is not supported for survival tasks — "
                    "a survival label belongs to a slide or patient, not a single tile. "
                    "Use dataset_type='slide' or 'patient'."
                )
            _survival_incompatible_aggregators = {"clam_sb", "clam_mb", "dtfdmil"}
            if (
                self.aggregator is not None
                and self.aggregator.name in _survival_incompatible_aggregators
            ):
                raise ValueError(
                    f"Aggregator '{self.aggregator.name}' is not supported for survival "
                    "tasks — its label-aware auxiliary loss assumes classification. "
                    "Use a survival-compatible aggregator (e.g. abmil, transmil, mean_pool)."
                )
            survival_loss = self.task.params.get("loss", "nll")
            if survival_loss not in {"nll", "cox"}:
                raise ValueError(
                    f"Unknown survival loss {survival_loss!r}; use 'nll' (discrete-time) "
                    "or 'cox' (continuous-time CoxPH)."
                )
            if survival_loss == "cox":
                # ``cox_window`` is the mode switch: unset/1 = padded mode (the risk
                # set is the batch; single-embedding slide/patient, or padded MIL via
                # masking); >= 2 = prediction-accumulation mode for large variable-size
                # MIL bags (N un-padded forwards per Cox loss, batch_size pinned to 1).
                cox_window = self.task.params.get("cox_window", 1)
                if not isinstance(cox_window, int) or isinstance(cox_window, bool) or cox_window < 1:
                    raise ValueError(
                        f"Cox 'cox_window' must be an integer >= 1, got {cox_window!r}."
                    )
                # Accumulating gradients across windows gives M independent risk sets of
                # size N, never one of size M*N — it does not enlarge the risk set. Reject
                # it in both modes so it is not mistaken for a way to grow the risk set;
                # raise cox_window / batch_size instead.
                if self.training.gradient_accumulation > 1:
                    raise ValueError(
                        "Cox survival loss requires training.gradient_accumulation = 1. "
                        "Gradient accumulation yields several independent risk sets rather "
                        "than one larger one; raise cox_window (accumulation mode) or "
                        "training.batch_size (padded mode) to enlarge the risk set instead."
                    )
                if cox_window >= 2:
                    # Accumulation mode: one bag at a time, no padding, so the loader
                    # batch_size must be 1; the risk set is the cox_window, not the batch.
                    if self.training.batch_size != 1:
                        raise ValueError(
                            "Cox accumulation mode (cox_window >= 2) requires "
                            "training.batch_size = 1 — bags are forwarded un-padded one at a "
                            "time, with cox_window as the risk-set size."
                        )
                    if self.aggregator is None:
                        raise ValueError(
                            "Cox accumulation mode (cox_window >= 2) is for variable-size MIL "
                            "bags and requires an aggregator (e.g. abmil, transmil, mean_pool). "
                            "For single-embedding slide/patient features, omit cox_window and "
                            "use padded mode with batch_size >= 2."
                        )
                else:
                    # Padded / single-embedding mode: the risk set is the batch.
                    if self.training.batch_size < 2:
                        raise ValueError(
                            "Cox survival loss requires training.batch_size >= 2 — the partial "
                            "likelihood's risk set is the batch. (For large MIL bags, use "
                            "accumulation mode by setting task.params.cox_window >= 2.)"
                        )
        # Validate that requested metrics are valid for the task family.
        resolve_metrics(self.task.name, self.evaluation.metrics)
        # Fail fast on unknown encoder / aggregator names — catching these at
        # config construction avoids burning hours of preprocessing before the
        # pipeline would otherwise crash at component-build time.
        if self.encoder is not None and self.composite is not None:
            raise ValueError(
                "Set 'encoder' (single) XOR 'composite' (multi-encoder composite), not both."
            )
        if self.composite is not None:
            if self.dataset_type not in {"segmentation", "detection"}:
                raise ValueError(
                    "Multi-encoder 'composite:' (composite concat) is only supported for "
                    f"dataset_type in {{'segmentation', 'detection'}}, got {self.dataset_type!r}."
                )
            if self.feature_mode != "cached":
                raise ValueError(
                    "'composite:' is cached-only — live multi-encoder (N resident backbones, "
                    "for live augmentation) is a deferred increment; set feature_mode='cached'."
                )
            # Cross-default each member's feature_kind + member_norm from the consumer, and
            # the concat_resolution, mirroring the single-encoder preprocessing.feature_kind
            # cross-default above. composite only reaches here for segmentation/detection,
            # where the consumer signal is the presence of a pixel_classifier.
            consumer_kind = "cls_attention" if self.pixel_classifier is not None else "patch_features"
            resolved_members = []
            for member in self.composite.encoders:
                fk = member.feature_kind or consumer_kind
                norm = member.member_norm or ("l2" if fk == "patch_features" else "none")
                resolved_members.append(replace(member, feature_kind=fk, member_norm=norm))
            concat_resolution = self.composite.concat_resolution or (
                "target" if self.pixel_classifier is not None else "grid"
            )
            object.__setattr__(
                self,
                "composite",
                replace(
                    self.composite,
                    encoders=resolved_members,
                    concat_resolution=concat_resolution,
                ),
            )
        if self.encoder is not None:
            from slide2vec.encoders.registry import encoder_registry

            try:
                encoder_registry.info(self.encoder.name)
            except KeyError as exc:
                available = ", ".join(sorted(encoder_registry.names())) or "(none)"
                raise ValueError(
                    f"Unknown encoder name '{self.encoder.name}'. "
                    f"Available encoders: {available}"
                ) from exc
        if self.composite is not None:
            from slide2vec.encoders.registry import encoder_registry

            for member in self.composite.encoders:
                try:
                    encoder_registry.info(member.name)
                except KeyError as exc:
                    available = ", ".join(sorted(encoder_registry.names())) or "(none)"
                    raise ValueError(
                        f"Unknown encoder name '{member.name}' in 'composite.encoders'. "
                        f"Available encoders: {available}"
                    ) from exc
        if self.aggregator is not None:
            from soma.aggregators.registry import aggregator_registry

            if self.aggregator.name not in aggregator_registry:
                available = ", ".join(sorted(aggregator_registry.list())) or "(none)"
                raise ValueError(
                    f"Unknown aggregator name '{self.aggregator.name}'. "
                    f"Available aggregators: {available}"
                )
        if self.decoder is not None:
            from soma.decoders.registry import decoder_registry

            if self.decoder.name not in decoder_registry:
                available = ", ".join(sorted(decoder_registry.list())) or "(none)"
                raise ValueError(
                    f"Unknown decoder name '{self.decoder.name}'. "
                    f"Available decoders: {available}"
                )
        if self.pixel_classifier is not None:
            from soma.pixel_classifiers import pixel_classifier_registry

            if self.pixel_classifier.name not in pixel_classifier_registry:
                available = ", ".join(sorted(pixel_classifier_registry.list())) or "(none)"
                raise ValueError(
                    f"Unknown pixel_classifier name '{self.pixel_classifier.name}'. "
                    f"Available pixel classifiers: {available}"
                )


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
            "feature_mode": config.feature_mode,
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
        "augmentation": _normalize_yaml_value(asdict(config.augmentation)),
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
    data["decoder"] = (
        _normalize_yaml_value(asdict(config.decoder))
        if config.decoder is not None
        else None
    )
    data["pixel_classifier"] = (
        _normalize_yaml_value(asdict(config.pixel_classifier))
        if config.pixel_classifier is not None
        else None
    )
    data["composite"] = (
        _normalize_yaml_value(asdict(config.composite))
        if config.composite is not None
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


def load_config(
    path: Path | str, overrides: dict[str, Any] | None = None
) -> PipelineConfig:
    """Load a PipelineConfig from YAML, merging bundled defaults first.

    ``overrides`` is an optional nested dict applied to the user-facing layout (the YAML
    structure: ``data``/``run``/``preprocessing``/``encoder``/``decoder``/``task``/
    ``training``/``evaluation``) *before* defaults are merged, so callers can repoint a
    committed config without editing it on disk (e.g. the ``--set`` CLI flag, or a
    reproduction runner substituting data/output paths).
    """
    with open(path) as f:
        raw_data = yaml.safe_load(f) or {}
    if not isinstance(raw_data, dict):
        raise TypeError("Config file must contain a top-level mapping")
    if overrides:
        raw_data = _deep_merge_dicts(raw_data, overrides)
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
        save_probabilities=bool(evaluation_data.get("save_probabilities", False)),
        holdout_test=bool(evaluation_data.get("holdout_test", False)),
    )
