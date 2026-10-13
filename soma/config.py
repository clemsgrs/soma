"""Frozen configuration dataclasses for soma experiments.

These dataclasses form the public configuration surface used by the pipeline,
the CLI, and the docs. Keep the field names stable and document any new field
here so the Sphinx reference stays accurate.

The extraction surface is held, not mirrored (ADR 0019): :class:`FeaturesConfig` holds
slide2vec's ``PreprocessingConfig`` (which composes hs2p's tiling, segmentation, filter,
preview and mask-sampling configs), ``ExecutionOptions`` and ``DenseOptions`` as given.
soma owns only what is soma's: the feature kind, the encoder leaf and the cache policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from importlib import resources
from pathlib import Path
import copy
import math
from numbers import Integral, Real
from typing import Any, Literal

import slide2vec
import yaml

from soma.atomic_io import atomic_write_text

from soma.tasks.registry import task_family_of, task_head_class


RoiBatchSamplingStrategy = Literal["uniform", "class_conditioned"]
ROI_BATCH_SAMPLING_STRATEGIES = frozenset({"uniform", "class_conditioned"})


@lru_cache(maxsize=1)
def _load_default_config_data() -> dict[str, Any]:
    path = resources.files("soma.configs").joinpath("default.yaml")
    with path.open("r", encoding="utf-8") as handle:
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
        "task",
        "representation",
        "evaluation",
        "training",
        "execution",
        "cache",
        "augmentation",
        "normalization",
        "projection",
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
        "task",
        "representation",
        "evaluation",
        "training",
        "execution",
        "cache",
        "augmentation",
        "normalization",
        "projection",
    ):
        if section in data:
            value = data[section]
            if section in ("aggregation", "decoder", "task", "representation", "augmentation"):
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


def _layout_to_pipeline_config(data: dict[str, Any]) -> PipelineConfig:
    from soma._legacy_layout import features_from_layout

    run_data = data.get("run", {})
    data_data = data.get("data", {})
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
        features=features_from_layout(data),
        aggregator=AggregatorConfig(**data["aggregation"]) if data.get("aggregation") else None,
        decoder=DecoderConfig(**data["decoder"]) if data.get("decoder") else None,
        task=_load_task_config(data),
        representation=(
            RepresentationConfig(**data["representation"])
            if data.get("representation")
            else None
        ),
        evaluation=_load_evaluation_config(data),
        training=TrainingConfig(**training_data),
        normalization=NormalizationConfig(**data.get("normalization", {})),
        projection=ProjectionConfig(**data.get("projection", {})),
        heatmaps=HeatmapConfig(**heatmap_data) if heatmap_data is not None else HeatmapConfig(),
        augmentation=_augmentation_leaf(data.get("augmentation")),
        tags=list(run_data.get("tags", [])),
        mirror_root=(
            Path(run_data["mirror_root"])
            if run_data.get("mirror_root") is not None
            else None
        ),
        resume=bool(run_data.get("resume", False)),
        run_id=run_data.get("run_id"),
        folds=(
            tuple(int(fold) for fold in run_data["folds"])
            if run_data.get("folds") is not None
            else None
        ),
    )


_ON_IDENTITY_MISMATCH = ("error", "reextract")


@dataclass(frozen=True)
class CacheConfig:
    """Shared cache policy for tiling and extracted features.

    Reusing the cache keeps repeated experiments from recomputing expensive
    tiling or embedding steps when the upstream configuration has not changed.
    """

    enabled: bool = True
    root_dir: str | Path | None = None
    reuse_policy: str = "strict"
    validate_payloads: bool = False
    # On-disk storage dtype for ALL feature caches (pooled tile/bag/slide/patient/
    # hierarchical AND dense grids): None ⇒ follow the compute precision (fp16 run →
    # fp16 features, else fp32), 'fp16'/'fp32' force it. fp16 halves cache size + the
    # per-epoch read I/O. Folded into every cache key (guarded so legacy fp32 keys stay
    # byte-stable) so an fp16 cache and an fp32 cache never collide.
    dtype: str | None = None
    # What to do with a pooled feature cache whose recorded feature identity (the
    # encoder's image transform, mainly) differs from what the installed slide2vec
    # extracts now: 'error' stops the run and names the differing fields, 'reextract'
    # deletes the cache and extracts it again. Not part of any cache key.
    on_identity_mismatch: str = "error"

    def __post_init__(self) -> None:
        if self.on_identity_mismatch not in _ON_IDENTITY_MISMATCH:
            raise ValueError(
                f"cache.on_identity_mismatch must be one of {list(_ON_IDENTITY_MISMATCH)}, "
                f"got {self.on_identity_mismatch!r}"
            )


@dataclass(frozen=True)
class EncoderLeaf:
    """The ``{name, params}`` leaf naming the frozen encoder (ADR 0019).

    ``name`` is a slide2vec encoder, or ``composite`` for several of them concatenated.
    ``params`` are slide2vec's ``Model.from_preset`` keywords (``output_variant``,
    ``allow_non_recommended_settings``), or the composite's members
    (:class:`soma.encoders.CompositeParams`); the ``encoders`` registry validates them.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def output_variant(self) -> str | None:
        return self.params.get("output_variant")

    @property
    def allow_non_recommended_settings(self) -> bool:
        return bool(self.params.get("allow_non_recommended_settings", False))


_FEATURE_KINDS = ("cached", "live")
_SPACING_POLICIES = ("strict", "native_if_coarser")


def is_annotation_sampling(masks: Any) -> bool:
    """Whether an hs2p ``MaskSamplingConfig`` samples annotations rather than tissue.

    hs2p's own test: the shipped ``{background, tissue}`` vocabulary sampling only
    ``tissue`` is binary tissue tiling; any other vocabulary or sampled set samples
    annotations.
    """
    from hs2p.configs.resolvers import is_binary_tissue_tiling

    return not is_binary_tissue_tiling(masks)


@dataclass(frozen=True)
class FeaturesConfig:
    """Where a run's features come from (ADR 0019).

    soma owns ``kind``, ``encoder``, ``cache`` and ``spacing_policy``. ``extraction``,
    ``execution`` and ``dense`` are slide2vec's ``PreprocessingConfig`` (composing hs2p's
    tiling, segmentation, filter, preview and mask-sampling configs), ``ExecutionOptions``
    and ``DenseOptions``, held as given.

    * ``kind``: ``cached`` extracts once into the feature cache; ``live`` re-encodes
      the (augmented) pixels every step, for the grid shape.
    * ``encoder``: a slide2vec encoder or ``composite`` (a name is accepted for a leaf
      without params).
    * ``extraction``: what slide2vec tiles and how. ``tiling=None`` takes the spacing
      and tile size from ``dense`` in a grid run that sets it, else from the encoder
      preset, when the features are extracted.
    * ``execution``: batch size, precision, GPUs and workers. ``precision=None`` takes
      the encoder's recommendation.
    * ``dense``: the grid shape's read and encode recipe; ``None`` reads at the resolved
      tiling's spacing and tile size with slide2vec's dense defaults. A tiling set beside
      it must name the same spacing, tile size, tolerance and backend.
    * ``spacing_policy``: how a grid run applies the requested spacing to each sample:
      ``strict`` reads every sample at it, ``native_if_coarser`` reads a sample whose
      level 0 is coarser than requested at its native spacing.
    """

    kind: str = "cached"
    encoder: EncoderLeaf | None = None
    extraction: slide2vec.PreprocessingConfig = field(default_factory=slide2vec.PreprocessingConfig)
    execution: slide2vec.ExecutionOptions = field(default_factory=slide2vec.ExecutionOptions)
    dense: slide2vec.DenseOptions | None = None
    cache: CacheConfig = field(default_factory=CacheConfig)
    spacing_policy: str = "strict"

    def __post_init__(self) -> None:
        if isinstance(self.encoder, str):
            object.__setattr__(self, "encoder", EncoderLeaf(name=self.encoder))
        if self.kind not in _FEATURE_KINDS:
            raise ValueError(f"features.kind must be one of {list(_FEATURE_KINDS)}, got {self.kind!r}")
        if self.spacing_policy not in _SPACING_POLICIES:
            raise ValueError(
                f"spacing_policy must be one of {list(_SPACING_POLICIES)}, got {self.spacing_policy!r}"
            )
        for name, expected in (
            ("extraction", slide2vec.PreprocessingConfig),
            ("execution", slide2vec.ExecutionOptions),
            ("cache", CacheConfig),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(
                    f"features.{name} must be a {expected.__module__}.{expected.__qualname__}, "
                    f"got {type(getattr(self, name)).__name__}"
                )
        if self.dense is not None and not isinstance(self.dense, slide2vec.DenseOptions):
            raise TypeError(
                f"features.dense must be a slide2vec.DenseOptions or None, got {type(self.dense).__name__}"
            )
        if self.encoder is not None:
            if not isinstance(self.encoder, EncoderLeaf):
                raise TypeError(f"features.encoder must be an EncoderLeaf, got {type(self.encoder).__name__}")
            from soma.encoders import encoder_registry

            if self.encoder.name not in encoder_registry:
                available = ", ".join(sorted(encoder_registry.list())) or "(none)"
                raise ValueError(
                    f"Unknown encoder name '{self.encoder.name}'. Available encoders: {available}"
                )
            encoder_registry.validate(self.encoder.name, self.encoder.params)

    @property
    def is_composite(self) -> bool:
        """Whether the encoder is soma's ``composite`` of several encoders."""
        return self.encoder is not None and self.encoder.name == "composite"


_NORMALIZATION_METHODS = ("none", "zscore", "l2", "layernorm")
_DEFAULT_NORMALIZATION_EPS = 1e-6


def _is_exact_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_finite_positive_real(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


@dataclass(frozen=True)
class NormalizationConfig:
    """Per-feature normalization applied to frozen encoder features before the model.

    Frozen encoders span 768→4608 dimensions with very different activation scales, so a
    shared aggregator/head and its single externally-calibrated learning rate do not see
    comparable inputs across encoders. This section names the fix, as the first stage of
    the *feature adaptor* — a buffer-carrying front module inserted ahead of the
    aggregator/head (issue #283).

    ``method``:

    * ``none`` (default) — no adaptor stage at all; the model is structurally identical
      to a run that predates this section.
    * ``zscore`` — **fitted**: per-feature center/scale estimated from the Support (train)
      split only, so the transform is leak-free. ``eps`` floors the scale so a constant
      or near-constant channel cannot blow up.
    * ``l2`` — stateless per-feature-vector L2 normalization.
    * ``layernorm`` — stateless per-feature-vector standardization (mean/std over the
      feature axis), with ``eps`` in the denominator.

    This is orthogonal to the composite per-member ``member_norm``, which normalizes each
    member's block *before* concatenation and is untouched by this section.
    """

    method: str = "none"
    eps: float = _DEFAULT_NORMALIZATION_EPS

    def __post_init__(self) -> None:
        if self.method not in _NORMALIZATION_METHODS:
            raise ValueError(
                "NormalizationConfig.method must be one of "
                f"{list(_NORMALIZATION_METHODS)}, got {self.method!r}."
            )
        if not _is_finite_positive_real(self.eps):
            raise ValueError(
                "NormalizationConfig.eps must be a finite positive real number "
                f"(it floors the scale), got {self.eps!r}."
            )
        if self.method == "none" and self.eps != _DEFAULT_NORMALIZATION_EPS:
            raise ValueError(
                "NormalizationConfig.method='none' requires the default eps "
                f"({_DEFAULT_NORMALIZATION_EPS}); got {self.eps!r}."
            )

    @property
    def is_default(self) -> bool:
        """True when this section asks for nothing — the guard for identity folding."""
        return self == NormalizationConfig()


_PROJECTION_METHODS = ("none", "pca", "random")


@dataclass(frozen=True)
class ProjectionConfig:
    """Label-free projection of frozen encoder features to a common width.

    A wider embedding means a larger aggregator, so "smaller encoders are more
    label-efficient" can be manufactured from dimensionality alone. This section is the
    dim-matched ablation that tests whether a ranking survives when that confound is
    removed: every encoder is projected to ``target_dim`` by a **label-free** map, so the
    downstream trainable capacity is equal across the roster (issue #284).

    It is the *second* stage of the feature adaptor, applied **after** ``normalization``
    (order: normalize → project). The two sections are independently configurable.

    ``method``:

    * ``none`` (default) — no projection stage; the aggregator is built against the
      encoder's native dim, exactly as before this section existed.
    * ``pca`` — **fitted**: principal components estimated per fold from the Support
      (train) split only, so the map is leak-free. Centers intrinsically (it stores its
      own mean) and pins a sign convention so repeated fits are byte-identical. No
      whitening. Requires ``n_fit_samples >= target_dim`` and ``target_dim <= D``.
    * ``random`` — a fixed Gaussian matrix, scaled by ``1/sqrt(target_dim)`` to
      approximately preserve inner products. Seeded from ``seed`` combined with the
      encoder identity and the in/out dims, so it is reproducible and constant across
      training trajectories. Unconstrained in ``target_dim``.

    The projection is **frozen** — a buffer, not a learned layer — so it cannot
    reintroduce or relocate the capacity confound it exists to remove.
    """

    method: str = "none"
    target_dim: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.method not in _PROJECTION_METHODS:
            raise ValueError(
                "ProjectionConfig.method must be one of "
                f"{list(_PROJECTION_METHODS)}, got {self.method!r}."
            )
        if not _is_exact_integer(self.seed):
            raise ValueError(
                f"ProjectionConfig.seed must be an integer, got {self.seed!r}."
            )
        if self.method == "none" and (self.target_dim is not None or self.seed != 0):
            raise ValueError(
                "ProjectionConfig.method='none' requires target_dim=None and seed=0; "
                f"got target_dim={self.target_dim!r}, seed={self.seed!r}."
            )
        if self.method == "pca" and self.seed != 0:
            raise ValueError(
                "ProjectionConfig.method='pca' is deterministic and requires seed=0; "
                f"got seed={self.seed!r}."
            )
        if self.method != "none":
            if self.target_dim is None:
                raise ValueError(
                    f"ProjectionConfig.method={self.method!r} requires target_dim — the "
                    "common width every encoder is projected to."
                )
            if not _is_exact_integer(self.target_dim) or self.target_dim < 1:
                raise ValueError(
                    "ProjectionConfig.target_dim must be a positive integer, "
                    f"got {self.target_dim!r}."
                )

    @property
    def is_default(self) -> bool:
        """True when this section asks for nothing — the guard for identity folding."""
        return self == ProjectionConfig()


def _requested_feature_adaptor_stages(
    normalization: NormalizationConfig | None,
    projection: ProjectionConfig | None,
) -> list[str]:
    """Return the active top-level adaptor stages for validation and diagnostics."""
    requested = []
    if normalization is not None and normalization.method != "none":
        requested.append(f"normalization.method={normalization.method!r}")
    if projection is not None and projection.method != "none":
        requested.append(f"projection.method={projection.method!r}")
    return requested


def validate_feature_adaptor_compatibility(
    normalization: NormalizationConfig | None,
    projection: ProjectionConfig | None,
    *,
    feature_mode: str = "cached",
    dataset_type: str | None = None,
    has_composite: bool = False,
    is_hierarchical: bool = False,
) -> None:
    """Enforce which data planes own the top-level feature adaptor.

    Both configuration validation and standalone training APIs call this rule so a path
    cannot become permissive merely by bypassing the orchestrator.
    """
    requested = _requested_feature_adaptor_stages(normalization, projection)
    if not requested:
        return
    requested_text = " and ".join(requested)
    if feature_mode == "live":
        raise ValueError(
            f"{requested_text} requires feature_mode='cached' — the feature adaptor is "
            "fit on the cached Support grids, and the live path re-encodes augmented "
            "tiles every step, so the fitted transform would not match what it "
            "transforms. Set feature_mode='cached', or use {method: none}."
        )

    unsupported = (
        "a composite (multi-encoder) dense stream"
        if has_composite
        else f"dataset_type={dataset_type!r}"
        if dataset_type in {"patient", "tile"}
        else "a hierarchical feature stream"
        if is_hierarchical
        else None
    )
    if unsupported is not None:
        raise ValueError(
            f"{requested_text} is not yet supported for {unsupported} — the feature "
            "adaptor fits on the tile-encoder MIL path, the slide-encoder embedding "
            "path, and the single-encoder dense path. Use {method: none} here "
            "(composites normalize per member via composite.encoders[].member_norm)."
        )


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
class TaskConfig:
    """Task-head selection and constructor parameters.

    ``name`` selects the registered task head. ``params`` are merged with any
    dataset-derived auto parameters before instantiation.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    # Target key -> dataset CSV column, for columns not named after the head's keys
    # (``{"value": "label"}`` trains regression on a ``label`` column).
    targets: dict[str, str] = field(default_factory=dict)


def task_config_dict(task: TaskConfig) -> dict[str, Any]:
    """``asdict(task)`` without an empty ``targets`` mapping.

    ``targets`` was added after runs had been recorded; leaving it out when empty keeps
    every existing experiment identity and saved config byte-identical.
    """
    data = asdict(task)
    if not data.get("targets"):
        data.pop("targets", None)
    return data


@dataclass(frozen=True)
class RepresentationConfig:
    """Fixed task-free representation-evaluation protocol.

    V1 intentionally exposes only CRoMa's published all-neighbours protocol. Keeping
    these fixed values explicit makes the saved config and experiment identity honest.
    """

    kind: str
    confounder_column: str
    split: str
    evaluation_design: str = "all"
    m: int = 5
    alpha: float = 0.10

    def __post_init__(self) -> None:
        if self.kind != "croma":
            raise ValueError(f"representation.kind must be exactly 'croma', got {self.kind!r}.")
        if not isinstance(self.confounder_column, str) or not self.confounder_column.strip():
            raise ValueError("representation.confounder_column must be a non-empty string.")
        if not isinstance(self.split, str) or not self.split.strip():
            raise ValueError("representation.split must be a non-empty string.")
        if self.evaluation_design != "all":
            raise ValueError(
                "representation.evaluation_design must be exactly 'all'; "
                f"got {self.evaluation_design!r}. paired_2x2 is not supported in v1."
            )
        if self.m != 5 or isinstance(self.m, bool):
            raise ValueError(f"representation.m must be exactly 5, got {self.m!r}.")
        if self.alpha != 0.10 or isinstance(self.alpha, bool):
            raise ValueError(
                "representation.alpha must be exactly 0.10 so croma_ltm10 is canonical; "
                f"got {self.alpha!r}."
            )


@dataclass(frozen=True)
class SubgroupConfig:
    """Columns used for subgroup metric breakdowns."""

    columns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation metrics and subgroup analysis configuration.

    Metrics are validated against the selected task family, and subgroup
    columns are used to break down the reported metrics in the run outputs.
    The dense-artifact toggles form a symmetric 2×2 over the segmentation and
    detection paths — cheap visual overlays default on (but suppressible, for a
    metrics-only run), heavy raw outputs default off:
    ``save_segmentation_overlays`` writes the pred/GT color overlays;
    ``save_segmentation_probabilities`` additionally writes a per-tile float16
    ``(C, H, W)`` softmax sidecar under ``probabilities/<split>/`` — opt-in because it is
    ~C×/precision× larger than the always-written argmax raster, and unlocks
    post-hoc soft-Dice/calibration/entropy/ensembling without re-running inference;
    ``save_detection_overlays`` writes the detection point overlays;
    ``save_detection_heatmaps`` writes the raw float16 heatmap sidecar.
    Predicted label maps (``predictions_<split>/``) are always written, so per-sample
    confusion is recomputable from them.
    ``holdout_test`` skips *all* test-split work (no test inference, no
    test predictions file, no ``test`` entries in ``metrics.json``/``summary.json``)
    and reports tune only — the model-selection protocol for benchmark sweeps: rank
    every candidate by tune score, then re-run only the winner with the test held out
    in. The test split may still be declared in ``splits.csv``; it is simply not
    touched. Tune evaluation, threshold sweeps, checkpoint selection, and training
    are unaffected.

    ``overwrite_test`` governs the test-results clobber guard (issue #247): because
    experiment identity is test-invariant, one checkpoint may be scored against several
    test sets over its life. Each test set's result is namespaced by its test identity;
    re-scoring an already-scored test set is refused by default (a loud skip that leaves
    the prior result intact) unless ``overwrite_test`` is set. It is an operational flag,
    not part of the experiment identity.
    """

    metrics: list[str] = field(default_factory=list)
    subgroups: SubgroupConfig = field(default_factory=SubgroupConfig)
    save_segmentation_overlays: bool = True
    save_segmentation_probabilities: bool = False
    save_detection_overlays: bool = True
    save_detection_heatmaps: bool = False
    holdout_test: bool = False
    overwrite_test: bool = False


def _validate_component_leaf(leaf: object, where: str, *, allow_str: bool) -> None:
    """A component leaf is a name (when allowed) or a mapping with a non-empty ``name``."""
    if allow_str and isinstance(leaf, str) and leaf:
        return
    if isinstance(leaf, dict) and isinstance(leaf.get("name"), str) and leaf["name"]:
        params = leaf.get("params")
        if params is not None and not isinstance(params, dict):
            raise ValueError(f"{where}: 'params' must be a mapping, got {params!r}")
        return
    shape = "a name or a {name, params} mapping" if allow_str else "a {name, params} mapping"
    raise ValueError(f"{where} must be {shape}, got {leaf!r}")


@dataclass(frozen=True)
class TrainingConfig:
    """Training-loop hyperparameters and optimizer settings.

    ``batch_size`` and ``gradient_accumulation`` control the effective batch
    size. Exactly one training budget is configured: ``epochs`` counts full
    passes over each fold's training loader, while ``max_steps`` counts
    optimizer updates (one complete gradient-accumulation window). Step-budget
    runs set ``epochs`` to ``None`` and may finish partway through an epoch.
    ``learning_rate``, ``optimizer``, ``scheduler``, and ``patience`` define the
    optimization schedule. ``optimizer`` and ``scheduler`` are component leaves: a name
    (``adam``, ``adamw``, ``sgd``; ``cosine``, ``constant``, ``none``) or a ``{name,
    params}`` mapping whose name is a registry key or a dotted import path; ``scheduler``
    may also be a list of leaves, each stepped at its own ``interval`` (``epoch`` by
    default). ``learning_rate`` and ``weight_decay`` seed the optimizer leaf's params.
    Cosine annealing derives ``T_max`` from the budget: once per epoch with
    ``T_max=epochs`` in epoch mode and once per optimizer update with ``T_max=max_steps``
    in step mode. A warm-up is an ordinary ``constant`` leaf (``factor``, ``total_iters``)
    at interval ``epoch``. ``learner`` optionally names the :class:`soma.learn.Learner`
    subclass (registry key or dotted path) that owns the step logic; the default is the
    shape's learner (``mil`` for sets, ``dense`` for grids). ``monitor`` and
    ``monitor_mode`` choose the ``validation_summary`` key (``loss`` or a metric) used for
    selected-checkpoint selection and early stopping. ``tune_is_test`` ties the tune and test
    splits to the same samples for protocols with a single held-out set: a fold
    may provide either a tune split or a test split (not both), and that split
    is used for both checkpoint selection and test reporting. ``allow_missing_tune``
    enables a deliberate train-as-tune fallback when a fold has no tune split.

    ``tune_every_n_epochs`` evaluates the tune split after every N-th epoch and after the
    final epoch, for budgets where tune evaluation dominates the epoch cost; ``patience``
    then counts tune evaluations rather than epochs.

    ``checkpoint_selection`` governs *which* epoch's weights are evaluated. ``best``
    (default) is the historical behavior: select by the monitored tune metric, with
    early stopping. ``last`` evaluates the **final-epoch** weights — model selection
    comes off the tune metric entirely and early stopping is disabled (so ``patience``
    must be ``None``), while per-epoch tune metrics are still computed and logged as
    diagnostics. It is the protocol for benchmarks that predeclare a fixed epoch budget
    and forbid per-encoder tuning. It is orthogonal to ``allow_missing_tune``: one
    says which checkpoint is evaluated, the other where the diagnostic tune split
    comes from, and neither implies the other.

    ``roi_batch_sampling`` is an opt-in cached-segmentation training-batch
    contract, separate from ``preprocessing.sampling`` (which creates ROI
    coordinates before extraction). ``uniform`` traverses a shuffled ROI
    collection; ``class_conditioned`` follows ``class_request_ratios`` over the
    task's class indices and weights eligible ROI selection by requested-class
    annotated-pixel count. Null ratios request every modeled class equally.
    ``roi_draws_per_epoch`` fixes the common draw budget for controlled arm
    comparisons. When it is null, cached segmentation rounds the training ROI
    population down to whole effective batches (``batch_size *
    gradient_accumulation``), while the loader still emits physical
    ``batch_size`` batches. Selection never alters a chosen mask: the ordinary
    segmentation loss still consumes every annotated pixel in the ROI.
    """

    seed: int = 0
    epochs: int | None = 50
    max_steps: int | None = None
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    # Component leaves (see the class docstring): a name or ``{name, params}``.
    optimizer: str | dict[str, Any] = "adam"
    scheduler: str | dict[str, Any] | list[str | dict[str, Any]] | None = "cosine"
    # Optional ``{name, params}`` leaf for the Learner subclass; ``None`` = the shape's.
    learner: dict[str, Any] | None = None
    # Early-stopping patience in tune evaluations (epochs, when ``tune_every_n_epochs`` is 1),
    # or ``None`` to disable early stopping and always train the full ``epochs`` budget.
    # ``checkpoint_selection='last'`` *requires* ``None``.
    patience: int | None = 10
    # Hold the early stop while the monitor still equals its first tune evaluation, so a
    # thresholded metric sitting at its floor during warm-up cannot end a run before it
    # produces any signal (patience still counts those evaluations; a monitor that never
    # moves trains to the budget). ``False`` gives Lightning's plain
    # ``EarlyStopping(min_delta=0)``: stop after ``patience`` non-improving evaluations
    # whatever the monitor did, keeping the first epoch on an initial plateau.
    hold_patience_until_monitor_moves: bool = True
    # Evaluate the tune split after every N-th epoch (epochs N, 2N, ...) and after the
    # final epoch of the budget. Checkpoint selection and early stopping see only
    # evaluated epochs, so ``patience`` spans ``patience * N`` epochs. Requires
    # ``checkpoint_selection='best'``.
    tune_every_n_epochs: int = 1
    monitor: str = "loss"
    monitor_mode: str = "min"
    batch_size: int = 1
    # Explicit cached-segmentation training-batch contract.  This is intentionally
    # separate from preprocessing.sampling, which creates ROI coordinates before
    # feature extraction.
    roi_batch_sampling: RoiBatchSamplingStrategy | None = None
    class_request_ratios: list[float] | None = None
    roi_draws_per_epoch: int | None = None
    gradient_accumulation: int = 1
    tune_is_test: bool = False
    allow_missing_tune: bool = False
    checkpoint_selection: str = "best"
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True

    def __post_init__(self) -> None:
        if self.epochs is not None and self.epochs < 1:
            raise ValueError("TrainingConfig.epochs must be >= 1")
        _validate_component_leaf(self.optimizer, "TrainingConfig.optimizer", allow_str=True)
        schedulers = self.scheduler if isinstance(self.scheduler, list) else [self.scheduler]
        for leaf in schedulers:
            if leaf is None or leaf == "none":
                continue
            _validate_component_leaf(leaf, "TrainingConfig.scheduler", allow_str=True)
        if self.learner is not None:
            _validate_component_leaf(self.learner, "TrainingConfig.learner", allow_str=False)
        if self.batch_size < 1:
            raise ValueError("TrainingConfig.batch_size must be >= 1")
        if (
            self.roi_batch_sampling is not None
            and self.roi_batch_sampling not in ROI_BATCH_SAMPLING_STRATEGIES
        ):
            raise ValueError(
                "TrainingConfig.roi_batch_sampling must be null, 'uniform', or "
                f"'class_conditioned', got {self.roi_batch_sampling!r}."
            )
        if self.class_request_ratios is not None:
            if self.roi_batch_sampling != "class_conditioned":
                raise ValueError(
                    "TrainingConfig.class_request_ratios requires "
                    "roi_batch_sampling='class_conditioned'."
                )
            if not self.class_request_ratios:
                raise ValueError(
                    "TrainingConfig.class_request_ratios must not be empty."
                )
            invalid_ratios = [
                value
                for value in self.class_request_ratios
                if isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) < 0
            ]
            if invalid_ratios:
                raise ValueError(
                    "TrainingConfig.class_request_ratios must contain finite, "
                    f"non-negative numbers, got {invalid_ratios[0]!r}."
                )
            if sum(float(value) for value in self.class_request_ratios) <= 0:
                raise ValueError(
                    "TrainingConfig.class_request_ratios must contain at least "
                    "one positive value."
                )
        if self.roi_draws_per_epoch is not None and self.roi_draws_per_epoch < 1:
            raise ValueError("TrainingConfig.roi_draws_per_epoch must be >= 1 or null.")
        if self.roi_draws_per_epoch is not None and self.roi_batch_sampling is None:
            raise ValueError(
                "TrainingConfig.roi_draws_per_epoch requires an explicit "
                "roi_batch_sampling strategy."
            )
        if (
            self.roi_draws_per_epoch is not None
            and self.roi_draws_per_epoch % self.batch_size != 0
        ):
            raise ValueError(
                "TrainingConfig.roi_draws_per_epoch must contain whole batches "
                f"of batch_size={self.batch_size}, got {self.roi_draws_per_epoch}."
            )
        if self.gradient_accumulation < 1:
            raise ValueError("TrainingConfig.gradient_accumulation must be >= 1")
        if not isinstance(self.hold_patience_until_monitor_moves, bool):
            raise ValueError("TrainingConfig.hold_patience_until_monitor_moves must be a boolean")
        if self.patience is not None and self.patience < 1:
            raise ValueError("TrainingConfig.patience must be >= 1 (or null to disable)")
        if not self.monitor:
            raise ValueError("TrainingConfig.monitor must be non-empty")
        if self.monitor_mode not in {"min", "max"}:
            raise ValueError("TrainingConfig.monitor_mode must be 'min' or 'max'")
        if self.checkpoint_selection not in {"best", "last"}:
            raise ValueError(
                "TrainingConfig.checkpoint_selection must be 'best' (select the checkpoint "
                "by the monitored tune metric, with early stopping) or 'last' (evaluate the "
                f"final-epoch weights), got {self.checkpoint_selection!r}."
            )
        if self.num_workers < 0:
            raise ValueError("TrainingConfig.num_workers must be >= 0")
        if self.tune_every_n_epochs < 1:
            raise ValueError("TrainingConfig.tune_every_n_epochs must be >= 1")
        if self.checkpoint_selection == "last" and self.tune_every_n_epochs != 1:
            raise ValueError(
                "TrainingConfig.checkpoint_selection='last' keeps the final-epoch weights "
                "and records tune metrics every epoch as diagnostics; "
                "tune_every_n_epochs must be 1."
            )
        if self.checkpoint_selection == "last" and self.patience is not None:
            raise ValueError(
                "TrainingConfig.checkpoint_selection='last' evaluates the final-epoch "
                "weights and never early-stops, so a finite TrainingConfig.patience "
                f"({self.patience}) would be silently ignored. Set patience: null."
            )


def _augmentation_leaf(value: Any) -> dict[str, Any] | None:
    """Normalise the ``augmentation`` section: ``null`` or a ``{name, params}`` leaf.

    ``name`` is an augmentation registry key (``segmentation`` is
    :class:`~soma.learn.SegmentationAugment`) or a dotted import path to a callable
    ``(pixels, target) -> (pixels, target)`` class; ``params`` are its constructor
    arguments (the run's ``ignore_index`` and ``target_size`` are injected where accepted).
    """
    if value is None or value == {}:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("name"), str) or not value["name"]:
        raise ValueError(
            "augmentation must be null or a {name, params} leaf, e.g. "
            "{name: segmentation, params: {horizontal_flip: 0.5}}; the flat "
            "horizontal_flip/... section was replaced by the leaf"
        )
    unknown = sorted(set(value) - {"name", "params"})
    if unknown:
        raise ValueError(f"augmentation leaf has unknown key(s) {unknown}; use {{name, params}}")
    params = value.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("augmentation.params must be a mapping")
    return {"name": value["name"], "params": dict(params)}


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
        features: Where the features come from (:class:`FeaturesConfig`): the
            encoder, slide2vec's extraction and execution settings held as given, the
            dense recipe of a grid run and the cache policy. Its ``encoder`` may be
            ``None`` when the run is handed its features.
        aggregator: MIL aggregator configuration for slide-level bag
            learning, or ``None`` for tile/patient pipelines.
        task: Task-head configuration. Exactly one of ``task`` and
            ``representation`` is required.
        representation: Task-free frozen-feature evaluation configuration.
        evaluation: Metric and subgroup evaluation configuration.
        training: Training hyperparameters.
        normalization: Feature-adaptor normalization applied to frozen encoder
            features ahead of the aggregator/head. Defaults to ``none`` (no
            adaptor at all).
        projection: Feature-adaptor label-free projection to a common width,
            applied after ``normalization``. Defaults to ``none``.
        heatmaps: Attention heatmap rendering settings.
        tags: Free-form labels attached to the experiment metadata.
        mirror_root: Optional shared-storage root for recoverable artifact bundles.
            This operational destination does not change experiment identity.
        resume: When True, reuse the latest existing run dir for this experiment
            instead of minting a fresh one, and skip folds that already wrote
            ``metrics.json`` (issue #244). Ignored when ``run_id`` is set.
        run_id: Pin the run to this exact run id (resume into it if it exists,
            else create it under that name). Takes precedence over ``resume``.
        folds: Train only these fold indices in this launch and leave the other pending
            folds untouched, so several launches pinned to one ``run_id`` can each take a
            fold on its own GPU. ``summary.json`` is written by the first launch that finds
            every fold complete. ``None`` trains every pending fold.
            All three are invocation-time directives: they are not part of the
            experiment identity and are not written back into the saved
            ``config.yaml``.
    """

    dataset_csv: str | Path
    splits_csv: str | Path
    output_root: str | Path
    dataset_type: str
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    aggregator: AggregatorConfig | None = None
    decoder: DecoderConfig | None = None
    task: TaskConfig | None = None
    representation: RepresentationConfig | None = None
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    heatmaps: HeatmapConfig = field(default_factory=HeatmapConfig)
    # ``{name, params}`` augmentation of a live grid run's train split, or ``None``.
    augmentation: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)
    mirror_root: Path | None = None
    # Run-lifecycle directives (issue #244) — not part of experiment identity.
    resume: bool = False
    run_id: str | None = None
    folds: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        self._name_registered_components_by_key()
        if self.folds is not None:
            if not self.folds or len(set(self.folds)) != len(self.folds) or min(self.folds) < 0:
                raise ValueError(
                    f"run.folds must list distinct non-negative fold indices, got {list(self.folds)}."
                )
            if not (self.run_id or self.resume):
                raise ValueError(
                    "run.folds needs run.run_id (or run.resume) so launches that split the "
                    "folds between them share one run dir."
                )
        if (self.training.epochs is None) == (self.training.max_steps is None):
            raise ValueError(
                "training budget requires exactly one of training.epochs and "
                "training.max_steps to be set."
            )
        if self.training.max_steps is not None and (
            not isinstance(self.training.max_steps, int)
            or isinstance(self.training.max_steps, bool)
            or self.training.max_steps < 1
        ):
            raise ValueError("training.max_steps must be an integer >= 1.")
        if (self.task is None) == (self.representation is None):
            raise TypeError(
                "PipelineConfig requires exactly one of 'task' and 'representation'."
            )
        _valid_dataset_types = {
            "slide",
            "tile",
            "patient",
            "segmentation",
            "detection",
        }
        if self.dataset_type not in _valid_dataset_types:
            raise ValueError(
                f"Invalid dataset_type {self.dataset_type!r}. "
                f"Must be one of: {sorted(_valid_dataset_types)}"
            )
        if self.training.roi_batch_sampling is not None and not (
            self.dataset_type == "segmentation"
            and self.features.kind == "cached"
            and self.decoder is not None
        ):
            raise ValueError(
                "training.roi_batch_sampling is only supported for cached neural "
                "segmentation with a decoder; other task and data paths would silently "
                "ignore the requested training-batch contract."
            )
        if self.representation is not None:
            if self.dataset_type != "tile":
                raise ValueError("representation v1 requires data.dataset_type='tile'.")
            if self.training.seed != 0:
                raise ValueError(
                    "representation v1 is deterministic and requires run.seed=0; "
                    f"got {self.training.seed!r}. Repeated launches are attempts, not seeds."
                )
            if self.features.kind != "cached":
                raise ValueError("representation v1 requires data.feature_mode='cached'.")
            if self.aggregator is not None or self.decoder is not None or self.features.is_composite:
                raise ValueError(
                    "representation v1 does not use aggregation, decoder or composite task "
                    "components."
                )
            if (
                self.normalization.method != "none"
                or self.projection.method != "none"
                or self.augmentation is not None
            ):
                raise ValueError(
                    "representation v1 requires normalization.method='none', "
                    "projection.method='none', and augmentation disabled."
                )
            return
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
            if self.decoder is None:
                raise ValueError(
                    "dataset_type='segmentation' requires a decoder (e.g. decoder: "
                    "{name: lightweight_conv})."
                )
            if self.aggregator is not None:
                raise ValueError(
                    "aggregator must be None for dataset_type='segmentation' — "
                    "segmentation uses a decoder (dense per-pixel), not MIL aggregation."
                )
            if task_family_of(self.task.name) != "segmentation":
                raise ValueError(
                    "dataset_type='segmentation' requires a task head of the "
                    f"'segmentation' family, got {self.task.name!r}."
                )
            # The training class scheme (task.params.classes / ignore) is validated here so
            # a bad scheme fails at config load, not after extraction. A bare num_classes
            # (masks already hold class indices) stays the pipeline's to require.
            task_params = self.task.params
            masks = self.features.extraction.masks
            annotation_rasters = is_annotation_sampling(masks)
            if annotation_rasters or "classes" in task_params or "ignore" in task_params:
                from soma.dense.reader import check_classes_declared, resolve_class_scheme

                resolve_class_scheme(task_params, annotation_rasters=annotation_rasters)
                if annotation_rasters:
                    check_classes_declared(task_params, masks.pixel_mapping)
        elif self.dataset_type == "detection":
            # Detection-v1 reuses the segmentation dense front half: a decoder regresses a
            # per-class peak heatmap that the DetectionHead turns into points (design §6).
            if self.decoder is None:
                raise ValueError(
                    "dataset_type='detection' requires a decoder (the heatmap regressor; "
                    "e.g. decoder: {name: lightweight_conv})."
                )
            if self.aggregator is not None:
                raise ValueError(
                    "aggregator must be None for dataset_type='detection' — detection is a "
                    "dense per-pixel task, not MIL aggregation."
                )
            if task_family_of(self.task.name) != "detection":
                raise ValueError(
                    "dataset_type='detection' requires a task head of the 'detection' "
                    f"family, got {self.task.name!r}."
                )
            # A bad class scheme fails at load, not at the first fold. A bare ``num_classes``
            # (point files already hold class indices) stays the pipeline's to require.
            if "classes" in self.task.params or "drop" in self.task.params:
                from soma.class_scheme import resolve_classes

                resolve_classes(self.task.params, excluded_key="drop", subject="detection")
            if "score_threshold" in self.task.params:
                raise ValueError(
                    "task.params.score_threshold was removed: the head emits every peak "
                    "above task.params.min_score, training-time validation thresholds at "
                    "task.params.validation_threshold, and the test operating point is swept "
                    "on the tune split."
                )
            for knob in ("min_score", "validation_threshold"):
                value = self.task.params.get(knob)
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0.0 <= float(value) <= 1.0
                ):
                    raise ValueError(f"task.params.{knob} must be a score in [0, 1], got {value!r}.")
        else:
            if self.decoder is not None:
                raise ValueError(
                    f"decoder must be None for dataset_type={self.dataset_type!r} — "
                    "decoders are only used for dataset_type='segmentation' or 'detection'."
                )
        # An annotation vocabulary in hs2p's masks drives annotation-based tile selection.
        # For 'segmentation' over a slide manifest it samples the ROIs; for 'slide' it
        # restricts the merged MIL bag to the selected compartment(s) (#110); for 'patient'
        # it restricts every slide's merged bag the same way (#111). Tile and detection runs
        # have no annotation-sampling step, so an annotation vocabulary there is an error.
        masks = self.features.extraction.masks
        _masks_dataset_types = {"slide", "patient", "segmentation"}
        if is_annotation_sampling(masks):
            if self.dataset_type not in _masks_dataset_types:
                raise ValueError(
                    "extraction.masks samples annotations (annotation-based tile selection), "
                    f"which is only valid for dataset_type in {sorted(_masks_dataset_types)}, "
                    f"got dataset_type={self.dataset_type!r}."
                )
            if masks.output_mode == "per_annotation":
                raise ValueError(
                    "extraction.masks.output_mode='per_annotation' is not yet supported for "
                    "feature extraction (deferred, see soma issue #86). Use 'merged' (one tile "
                    "encoded once with its full multi-class mask attached downstream)."
                )
        # The live path re-encodes augmented tiles through the frozen encoder every step;
        # it serves the grid shape only. Fail loud rather than silently no-op.
        if self.features.kind == "live" and self.dataset_type not in {"segmentation", "detection"}:
            raise ValueError(
                "feature_mode='live' re-encodes tiles for the grid shape (dataset_type "
                f"'segmentation' or 'detection'), got dataset_type={self.dataset_type!r}."
            )
        validate_feature_adaptor_compatibility(
            self.normalization,
            self.projection,
            feature_mode=self.features.kind,
            dataset_type=self.dataset_type,
            has_composite=self.features.is_composite,
        )
        if self.augmentation is not None:
            if self.features.kind != "live":
                raise ValueError(
                    "augmentation requires feature_mode='live' — cached dense grids cannot "
                    "be augmented (the features are frozen). Set feature_mode='live' or clear "
                    "the augmentation section."
                )
            if self.dataset_type == "detection":
                raise ValueError(
                    "point targets cannot be augmented yet: joint point augmentation waits "
                    "for torchvision 0.23 tv_tensors.KeyPoints (soma#586). Clear the "
                    "augmentation section for live detection."
                )
            from soma.learn.augment import augment_registry

            try:
                augment_registry.resolve(self.augmentation["name"])
            except KeyError as error:
                raise ValueError(f"Unknown augmentation {self.augmentation['name']!r}: {error}") from error
        if "dropout" in self.task.params:
            # Fail at config load, not at head construction after extraction: the head
            # re-validates, but a full pipeline run should not spend the extraction
            # stage before discovering a bad probability.
            head_dropout = self.task.params["dropout"]
            if (
                isinstance(head_dropout, bool)
                or not isinstance(head_dropout, (int, float))
                or not 0.0 <= head_dropout < 1.0
            ):
                raise ValueError(
                    f"task.params.dropout must be a probability in [0.0, 1.0), got {head_dropout!r}."
                )
        if task_family_of(self.task.name) == "survival":
            from soma.tasks.survival import resolve_survival_task

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
            _, survival_loss = resolve_survival_task(self.task.name, self.task.params)
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
        # Validate that requested metrics and the training monitor are valid for the task
        # head (its family's metrics, or the ones its params select).  ``last`` ignores the
        # monitor for checkpoint selection, but the monitor is still part of the declared
        # protocol and must name a diagnostic the run computes.
        # A configured ``training.learner`` owns its ``validation_summary`` keys, so its
        # monitor is checked by the Checkpoint callback against the real summary instead.
        resolved_metrics = task_head_class(self.task.name).resolve_metrics(
            self.task.params, self.evaluation.metrics
        )
        if (
            self.training.learner is None
            and self.training.monitor != "loss"
            and self.training.monitor not in resolved_metrics
        ):
            available = ", ".join(resolved_metrics)
            raise ValueError(
                f"Training monitor {self.training.monitor!r} is not an effective metric for "
                f"task {self.task.name!r}. Available metrics: {available}; use 'loss' "
                "to monitor the tune loss."
            )
        # Fail fast on unknown aggregator / decoder names (the encoder leaf is checked by
        # FeaturesConfig): catching these at config construction avoids burning hours of
        # preprocessing before the pipeline would otherwise crash at component-build time.
        if self.features.is_composite:
            if self.dataset_type not in {"segmentation", "detection"}:
                raise ValueError(
                    "The composite encoder (several encoders concatenated) is only supported "
                    f"for dataset_type in {{'segmentation', 'detection'}}, got {self.dataset_type!r}."
                )
            if self.features.kind != "cached":
                raise ValueError(
                    "The composite encoder is cached-only: live multi-encoder (N resident "
                    "backbones, for live augmentation) is a deferred increment; set "
                    "feature_mode='cached'."
                )
        if self.aggregator is not None:
            from soma.aggregators.registry import aggregator_registry

            try:
                aggregator_registry.resolve(self.aggregator.name)
            except KeyError as exc:
                available = ", ".join(sorted(aggregator_registry.list())) or "(none)"
                raise ValueError(
                    f"Unknown aggregator name '{self.aggregator.name}'. "
                    f"Available aggregators: {available}"
                ) from exc
        if self.decoder is not None:
            from soma.decoders.registry import decoder_registry

            try:
                decoder_registry.resolve(self.decoder.name)
            except KeyError as exc:
                available = ", ".join(sorted(decoder_registry.list())) or "(none)"
                raise ValueError(
                    f"Unknown decoder name '{self.decoder.name}'. "
                    f"Available decoders: {available}"
                ) from exc

    def _name_registered_components_by_key(self) -> None:
        """Store a dotted path naming a registered class as that entry's key.

        Such a path is that entry, so the name-keyed checks and routing (the survival
        ``loss`` selector, ``clam_mb``'s branch-aware head, ``hipt``'s hierarchy) must see
        the key. An unregistered dotted path stays as written.
        """
        dotted = [
            attribute
            for attribute in ("task", "aggregator", "decoder")
            if getattr(self, attribute) is not None and "." in getattr(self, attribute).name
        ]
        if not dotted:
            return
        import soma.tasks  # noqa: F401 - registers the built-in heads
        from soma.aggregators.registry import aggregator_registry
        from soma.decoders.registry import decoder_registry
        from soma.tasks.registry import task_registry

        registries = {"task": task_registry, "aggregator": aggregator_registry, "decoder": decoder_registry}
        for attribute in dotted:
            component = getattr(self, attribute)
            key = registries[attribute].key_of(component.name)
            if key != component.name:
                object.__setattr__(self, attribute, replace(component, name=key))


# --- YAML serialization ---


def _normalize_yaml_value(obj: Any) -> Any:
    """Recursively normalize dataclass output into YAML-safe primitives."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (tuple, list)):
        return [_normalize_yaml_value(value) for value in obj]
    if isinstance(obj, dict):
        return {key: _normalize_yaml_value(value) for key, value in obj.items()}
    return obj


def _config_to_layout_dict(config: PipelineConfig) -> dict[str, Any]:
    """Convert a PipelineConfig into the canonical nested YAML layout."""
    from soma._legacy_layout import features_to_layout

    data = {
        "run": {
            "output_root": _normalize_yaml_value(config.output_root),
            "seed": config.training.seed,
            "tags": _normalize_yaml_value(config.tags),
            "mirror_root": _normalize_yaml_value(config.mirror_root),
        },
        "data": {
            "dataset_csv": _normalize_yaml_value(config.dataset_csv),
            "splits_csv": _normalize_yaml_value(config.splits_csv),
            "dataset_type": config.dataset_type,
        },
        "task": (
            _normalize_yaml_value(task_config_dict(config.task))
            if config.task is not None
            else None
        ),
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
        "augmentation": _normalize_yaml_value(config.augmentation),
        "normalization": _normalize_yaml_value(asdict(config.normalization)),
        "projection": _normalize_yaml_value(asdict(config.projection)),
        "reports": {
            "heatmaps": _normalize_yaml_value(asdict(config.heatmaps)),
        },
    }
    features = features_to_layout(config.features, dataset_type=config.dataset_type)
    data["data"]["feature_mode"] = features.pop("feature_mode")
    data.update(_normalize_yaml_value(features))
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
    if config.representation is not None:
        data["representation"] = _normalize_yaml_value(asdict(config.representation))
    return data


def config_yaml_dict(config: PipelineConfig) -> dict[str, Any]:
    """Return the fully-resolved nested YAML dict that :func:`save_config` writes.

    Exposed so callers can compare configs by their persisted form without
    round-tripping through a file — e.g. the resume drift guard (issue #244).
    The run-lifecycle directives ``resume`` / ``run_id`` / ``folds`` are deliberately not part
    of this dict, so a run's saved ``config.yaml`` is identical whether it was the
    original launch or a resume of it.
    """
    return _deep_merge_dicts(_load_default_config_data(), _config_to_layout_dict(config))


#: ``ExecutionOptions`` fields slide2vec resolves for the machine when it is built: the
#: GPU count and the CPU-capped preprocessing workers. A saved ``config.yaml`` records the
#: resolved values; they are operational, not part of a run's recipe.
_MACHINE_RESOLVED_EXECUTION_FIELDS = ("num_gpus", "num_preprocessing_workers")


def without_machine_resolved_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """A ``config.yaml`` payload without the execution fields resolved for the machine,
    for comparing two runs' recipes (the resume drift guards)."""
    stripped = copy.deepcopy(payload)
    execution = stripped.get("execution")
    if isinstance(execution, dict):
        for name in _MACHINE_RESOLVED_EXECUTION_FIELDS:
            execution.pop(name, None)
    return stripped


def save_config(config: PipelineConfig, path: Path | str) -> None:
    """Serialize a PipelineConfig to a fully resolved nested YAML file."""
    data = config_yaml_dict(config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, yaml.safe_dump(data, default_flow_style=False, sort_keys=False))


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


def _load_task_config(data: dict[str, Any]) -> TaskConfig | None:
    task_data = data.get("task")
    if task_data is None:
        return None
    if not task_data or "name" not in task_data:
        raise ValueError(
            "Config is missing required 'task.name' (e.g. task: {name: binary_classification})"
        )
    return TaskConfig(
        name=task_data["name"],
        params=task_data.get("params", {}),
        targets=dict(task_data.get("targets") or {}),
    )


def _load_evaluation_config(data: dict[str, Any]) -> EvalConfig:
    evaluation_data = data.get("evaluation", {})
    if "save_segmentation_confusion_evidence" in (evaluation_data or {}):
        raise ValueError(
            "evaluation.save_segmentation_confusion_evidence was removed: every run writes "
            "its predicted label maps (predictions_<split>/), from which per-sample "
            "confusion is recomputable."
        )
    subgroup_data = evaluation_data.get("subgroups", {})
    columns = subgroup_data.get("columns", []) if subgroup_data else []
    return EvalConfig(
        metrics=evaluation_data.get("metrics", []),
        subgroups=SubgroupConfig(columns=columns),
        save_segmentation_overlays=bool(evaluation_data.get("save_segmentation_overlays", True)),
        save_segmentation_probabilities=bool(
            evaluation_data.get("save_segmentation_probabilities", False)
        ),
        save_detection_overlays=bool(evaluation_data.get("save_detection_overlays", True)),
        save_detection_heatmaps=bool(evaluation_data.get("save_detection_heatmaps", False)),
        holdout_test=bool(evaluation_data.get("holdout_test", False)),
        overwrite_test=bool(evaluation_data.get("overwrite_test", False)),
    )
