"""The experiment configuration: seven sections, each holding one public callable's
keywords, so a knob that cannot be passed in Python cannot appear in YAML (ADR 0020).

Every swappable component is a :class:`Component` (``{name, params}``). YAML hydrates the
dataclasses generically from their field types; cross-section checks, defaults and what
the cohort decides belong to :mod:`soma.plan`.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args, get_origin

import numpy as np
import slide2vec
import yaml

from soma.atomic_io import atomic_write_text

__all__ = [
    "CRoMaOptions", "CacheOptions", "CohortConfig", "Component", "EncoderLeaf", "EvaluationConfig",
    "ExperimentConfig", "FeaturesConfig", "LearnerConfig", "LoadersConfig", "RunConfig",
    "TrainerConfig", "hydrate", "is_annotation_sampling", "plain",
]


# --- {name, params} leaves -------------------------------------------------------------


@dataclass(frozen=True)
class Component:
    """A ``{name, params}`` leaf: a registry key or a dotted import path, and its params."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"a component leaf needs a non-empty name, got {self.name!r}")
        params = {} if self.params is None else self.params
        if not isinstance(params, Mapping):
            raise TypeError(f"the params of '{self.name}' must be a mapping, got {type(params).__name__}")
        object.__setattr__(self, "params", dict(params))

    @classmethod
    def of(cls, value: Any) -> Any:
        """``value`` as a leaf of this class: a leaf, a name or a ``{name, params}`` mapping."""
        if value is None or type(value) is cls:
            return value
        if isinstance(value, Component):
            return cls(value.name, value.params)
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, Mapping):
            unknown = sorted(set(value) - {"name", "params"})
            if unknown or "name" not in value:
                raise ValueError(f"a component leaf is a name or a {{name, params}} mapping, got {dict(value)!r}")
            return cls(value["name"], value.get("params") or {})
        raise TypeError(f"a component leaf is a name or a {{name, params}} mapping, got {value!r}")


@dataclass(frozen=True)
class EncoderLeaf(Component):
    """The frozen encoder: a slide2vec encoder, or ``composite`` for several concatenated.

    ``params`` are slide2vec's ``Model.from_preset`` keywords (``output_variant``,
    ``allow_non_recommended_settings``), or the composite's members
    (:class:`soma.encoders.CompositeParams`); the ``encoders`` registry validates them.
    """

    @property
    def output_variant(self) -> str | None:
        return self.params.get("output_variant")

    @property
    def allow_non_recommended_settings(self) -> bool:
        return bool(self.params.get("allow_non_recommended_settings", False))


# --- features ------------------------------------------------------------------------

@dataclass(frozen=True)
class CacheOptions:
    """``features.cache``: the feature and tiling cache policy. ``dtype`` (``None``
    follows the compute precision) enters every cache key; ``on_identity_mismatch``
    (``error`` or ``reextract``) handles a pooled cache from another slide2vec identity."""

    enabled: bool = True
    root_dir: str | Path | None = None
    reuse_policy: str = "strict"
    validate_payloads: bool = False
    dtype: str | None = None
    on_identity_mismatch: str = "error"

    def __post_init__(self) -> None:
        _check_choice("cache.on_identity_mismatch", self.on_identity_mismatch, ("error", "reextract"))


def is_annotation_sampling(masks: Any) -> bool:
    """Whether an hs2p ``MaskSamplingConfig`` samples annotations rather than binary
    tissue (hs2p's own test)."""
    from hs2p.configs.resolvers import is_binary_tissue_tiling

    return not is_binary_tissue_tiling(masks)


@dataclass(frozen=True)
class FeaturesConfig:
    """``features``: where a run's features come from (ADR 0019).

    ``kind``: ``cached`` extracts once; ``live`` re-encodes the (augmented) pixels every
    step (grid shape). ``encoder``: a slide2vec encoder or ``composite`` (``None`` when the
    run is handed its features). ``extraction``, ``execution`` and ``dense`` are
    slide2vec's ``PreprocessingConfig``, ``ExecutionOptions`` and ``DenseOptions``, held as
    given; an unset tiling comes from ``dense`` (grid) or the encoder preset.
    ``spacing_policy: native_if_coarser`` reads a coarser sample at its native spacing.
    """

    kind: str = "cached"
    encoder: EncoderLeaf | None = None
    extraction: slide2vec.PreprocessingConfig = field(default_factory=slide2vec.PreprocessingConfig)
    execution: slide2vec.ExecutionOptions = field(default_factory=slide2vec.ExecutionOptions)
    dense: slide2vec.DenseOptions | None = None
    cache: CacheOptions = field(default_factory=CacheOptions)
    spacing_policy: str = "strict"

    def __post_init__(self) -> None:
        object.__setattr__(self, "encoder", EncoderLeaf.of(self.encoder))
        _check_choice("features.kind", self.kind, ("cached", "live"))
        _check_choice("features.spacing_policy", self.spacing_policy, ("strict", "native_if_coarser"))
        _check_types(self, "features.", ("extraction", "execution", "dense", "cache"))
        if self.encoder is not None:
            from soma.encoders import encoder_registry

            if self.encoder.name not in encoder_registry:
                available = ", ".join(sorted(encoder_registry.list())) or "(none)"
                raise ValueError(f"Unknown encoder name '{self.encoder.name}'. Available encoders: {available}")
            encoder_registry.validate(self.encoder.name, self.encoder.params)

    @property
    def is_composite(self) -> bool:
        """Whether the encoder is soma's ``composite`` of several encoders."""
        return self.encoder is not None and self.encoder.name == "composite"


# --- the other sections ----------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """``run``: :class:`~soma.pipeline.Pipeline`'s keywords. ``resume`` reuses the latest
    run dir, ``run_id`` pins one, ``folds`` trains only those folds in this launch (launches
    pinned to one ``run_id`` split the folds); ``mirror_root`` mirrors fold artifacts."""

    output_dir: str | Path = "runs"
    seed: int = 0
    folds: tuple[int, ...] | None = None
    run_id: str | None = None
    resume: bool = False
    mirror_root: str | Path | None = None

    def __post_init__(self) -> None:
        if self.folds is not None:
            folds = tuple(int(fold) for fold in self.folds)
            if not folds or len(set(folds)) != len(folds) or min(folds) < 0:
                raise ValueError(f"run.folds must list distinct non-negative fold indices, got {list(folds)}.")
            if not (self.run_id or self.resume):
                raise ValueError(
                    "run.folds needs run.run_id (or run.resume) so launches that split the folds "
                    "between them share one run dir."
                )
            object.__setattr__(self, "folds", folds)


@dataclass(frozen=True)
class CohortConfig:
    """``cohort``: :meth:`~soma.data.Cohort.from_csv`'s files. ``collapse: patient_id``
    makes one sample per patient; ``tune_is_test`` uses one held-out split for selection
    and test; ``allow_missing_tune`` lets a fold without tune select on train."""

    records: str | Path
    splits: str | Path
    collapse: str | None = None
    tune_is_test: bool = False
    allow_missing_tune: bool = False

    def __post_init__(self) -> None:
        if self.collapse is not None:
            _check_choice("cohort.collapse", self.collapse, ("patient_id",))


@dataclass(frozen=True)
class LoadersConfig:
    """``loaders``: :func:`soma.learn.loaders`' keywords. ``sampler`` is a training-batch
    sampler leaf (cached segmentation ROI strategies); ``augment`` augments a live grid
    run's train split."""

    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    sampler: Component | None = None
    augment: Component | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sampler", Component.of(self.sampler))
        object.__setattr__(self, "augment", Component.of(self.augment))
        _check_count("loaders.batch_size", self.batch_size, minimum=1)
        _check_count("loaders.num_workers", self.num_workers, minimum=0)


#: The ``learner.params`` keys holding a single component leaf.
LEAF_PARAMS = ("aggregator", "decoder", "head", "optimizer")


@dataclass(frozen=True)
class LearnerConfig(Component):
    """``learner``: the :class:`~soma.learn.Learner` and the model it trains.

    ``params`` holds the model's leaves (``aggregator`` for the set shape, ``decoder`` for
    the grid shape, ``head`` with its ``targets``: target key -> cohort column), the
    feature adaptor (``normalization``, ``projection``), the ``optimizer`` leaf, the
    ``scheduler`` leaf or list of leaves, and any other keyword of the learner class.
    """

    name: str = "mil"

    def __post_init__(self) -> None:
        super().__post_init__()
        params = dict(self.params)
        for key in LEAF_PARAMS:
            if params.get(key) is not None:
                params[key] = Component.of(params[key])
        if "scheduler" in params:
            schedulers = params["scheduler"]
            schedulers = schedulers if isinstance(schedulers, (list, tuple)) else [schedulers]
            params["scheduler"] = [_scheduler_leaf(leaf) for leaf in schedulers if leaf not in (None, "none")]
        object.__setattr__(self, "params", params)


def _scheduler_leaf(leaf: Any) -> Any:
    """A scheduler leaf; a mapping with keys beyond ``{name, params}`` (``interval``) stays one."""
    if isinstance(leaf, Mapping) and set(leaf) - {"name", "params"}:
        if not isinstance(leaf.get("name"), str):
            raise ValueError(f"a scheduler leaf needs a name, got {dict(leaf)!r}")
        return dict(leaf)
    return Component.of(leaf)


def _default_callbacks() -> list[Component]:
    return [Component("checkpoint"), Component("early_stopping")]


@dataclass(frozen=True)
class TrainerConfig:
    """``trainer``: :class:`~soma.learn.Trainer`'s keywords. A step budget (``max_steps``)
    replaces the epoch budget; ``callbacks`` are the selection policy (``checkpoint``,
    ``early_stopping``; leave the latter out to train the whole budget)."""

    max_epochs: int | None = 50
    max_steps: int | None = None
    accumulate_grad_batches: int = 1
    val_every_n_epochs: int = 1
    device: str = "auto"
    precision: str = "fp32"
    callbacks: list[Component] = field(default_factory=_default_callbacks)

    def __post_init__(self) -> None:
        object.__setattr__(self, "callbacks", [Component.of(leaf) for leaf in self.callbacks or ()])
        if self.max_steps is not None:
            _check_count("trainer.max_steps", self.max_steps, minimum=1)
            object.__setattr__(self, "max_epochs", None)
        elif self.max_epochs is None:
            raise ValueError("trainer needs a budget: set max_epochs or max_steps.")
        else:
            _check_count("trainer.max_epochs", self.max_epochs, minimum=1)
        _check_count("trainer.accumulate_grad_batches", self.accumulate_grad_batches, minimum=1)
        _check_count("trainer.val_every_n_epochs", self.val_every_n_epochs, minimum=1)
        _check_choice("trainer.precision", self.precision, ("fp32", "bf16"))


@dataclass(frozen=True)
class CRoMaOptions:
    """``evaluation.representation``: CRoMa's published all-neighbours protocol, fixed."""

    confounder_column: str
    split: str
    evaluation_design: str = "all"
    m: int = 5
    alpha: float = 0.10

    def __post_init__(self) -> None:
        for name in ("confounder_column", "split"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"representation.{name} must be a non-empty string.")
        for name, fixed in (("evaluation_design", "all"), ("m", 5), ("alpha", 0.10)):
            value = getattr(self, name)
            if value != fixed or isinstance(value, bool):
                raise ValueError(f"representation.{name} must be exactly {fixed!r} (v1 protocol), got {value!r}.")


@dataclass(frozen=True)
class EvaluationConfig:
    """``evaluation``: :func:`soma.evaluate`'s ``metrics`` and ``subgroups`` (columns), and
    what a run scores and saves. ``holdout_test`` skips test work; ``overwrite_test``
    re-scores a test set. The grid shape saves overlays, the segmentation softmax and the
    detection heatmaps; ``attention_heatmaps`` (``cmap``, ``alpha``, ``blur_sigma``)
    renders tile attention; ``representation`` makes the run a task-free CRoMa evaluation.
    """

    metrics: list[Any] = field(default_factory=list)
    subgroups: list[str] = field(default_factory=list)
    holdout_test: bool = False
    overwrite_test: bool = False
    save_overlays: bool = True
    save_probabilities: bool = False
    save_heatmaps: bool = False
    attention_heatmaps: dict[str, Any] | None = None
    representation: CRoMaOptions | None = None

    def __post_init__(self) -> None:
        if self.attention_heatmaps is not None:
            unknown = sorted(set(self.attention_heatmaps) - {"cmap", "alpha", "blur_sigma"})
            if unknown:
                raise ValueError(f"evaluation.attention_heatmaps has no key(s) {unknown}; use cmap, alpha, blur_sigma")


# --- the experiment --------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    """A whole experiment: the seven sections. Only ``cohort`` is required; a run that
    trains needs a ``learner`` with a ``head``."""

    run: RunConfig = field(default_factory=RunConfig)
    cohort: CohortConfig
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    loaders: LoadersConfig = field(default_factory=LoadersConfig)
    learner: LearnerConfig | None = None
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "learner", LearnerConfig.of(self.learner))
        _check_types(self, "", [f.name for f in dataclasses.fields(self)])

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentConfig:
        """Hydrate the sections from a nested mapping (a parsed YAML document)."""
        return hydrate(cls, data)

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: Mapping[str, Any] | None = None) -> ExperimentConfig:
        """Read a YAML config; ``overrides`` (nested like the file) are merged in first, so
        a committed config can be repointed (``--set``, a benchmark's paths) unedited."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, Mapping):
            raise TypeError(f"{path}: an experiment config is a mapping of sections")
        return cls.from_dict(_merged(data, overrides or {}))

    def to_dict(self) -> dict[str, Any]:
        """The config as YAML-safe primitives, every field written."""
        return plain(self)

    def to_yaml(self, path: str | Path | None = None) -> str:
        """The config as YAML; written to ``path`` (atomically) when given."""
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(Path(path), text)
        return text


# --- generic hydration -----------------------------------------------------------------


def hydrate(cls: type, data: Any, where: str = "") -> Any:
    """Build dataclass ``cls`` from ``data``, hydrating nested fields from their types.

    A field typed as a dataclass (soma's, slide2vec's or hs2p's) is built from a mapping,
    a :class:`Component` from a name or a ``{name, params}`` mapping, a tuple from a list;
    anything else is passed as given and the dataclass validates it. An unknown key raises.
    """
    if isinstance(data, cls):
        return data
    where = where or cls.__name__
    if not isinstance(data, Mapping):
        raise TypeError(f"{where} must be a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls) if f.init]
    unknown = sorted(set(data) - set(names))
    if unknown:
        raise ValueError(f"{where} has no key(s) {unknown}; its keys are {names}")
    return cls(**{key: _hydrate_value(hints[key], value, f"{where}.{key}") for key, value in data.items()})


def _hydrate_value(hint: Any, value: Any, where: str) -> Any:
    if value is None:
        return None
    origin = get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        options = [option for option in get_args(hint) if option is not type(None)]
        for option in options:
            if isinstance(value, Mapping) and _is_structured(option):
                return _hydrate_value(option, value, where)
            if isinstance(value, list) and get_origin(option) in (tuple, list):
                return _hydrate_value(option, value, where)
        if isinstance(value, str) and Path in options and str not in options:
            return Path(value)
        return value
    if isinstance(hint, type) and issubclass(hint, Component):
        return hint.of(value)
    if dataclasses.is_dataclass(hint) and isinstance(value, Mapping):
        return hydrate(hint, value, where)
    if origin is tuple and isinstance(value, list):
        return tuple(value)
    if origin is list and isinstance(value, (list, tuple)):
        (item,) = get_args(hint) or (Any,)
        return [_hydrate_value(item, element, where) for element in value]
    if hint is Path and isinstance(value, str):
        return Path(value)
    return value


def _merged(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        nested = isinstance(merged.get(key), Mapping) and isinstance(value, Mapping)
        merged[key] = _merged(merged[key], value) if nested else value
    return merged


def _is_structured(hint: Any) -> bool:
    return dataclasses.is_dataclass(hint) or (isinstance(hint, type) and issubclass(hint, Component))


def plain(value: Any) -> Any:
    """``value`` as YAML-safe primitives: dataclasses to mappings of their fields, tuples
    to lists, paths to strings, numpy scalars and arrays to Python values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value) if f.init}
    if isinstance(value, Mapping):
        return {plain(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _check_types(obj: Any, where: str, names) -> None:
    """Each of ``names`` holds a value of its annotated type (a section, slide2vec's options)."""
    hints = typing.get_type_hints(type(obj))
    for name in names:
        hint, value = hints[name], getattr(obj, name)
        if not isinstance(value, hint):
            label = str(hint)
            if isinstance(hint, type):
                label = hint.__qualname__ if hint.__module__ == __name__ else f"{hint.__module__}.{hint.__qualname__}"
            raise TypeError(f"{where}{name} must be a {label}, got {type(value).__name__}")


def _check_choice(name: str, value: Any, choices: tuple[Any, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}")


def _check_count(name: str, value: Any, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
