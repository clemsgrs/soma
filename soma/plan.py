"""``resolve(config, cohort) -> Plan``: what a run will actually do, before it does it.

A :class:`Plan` is an :class:`~soma.config.ExperimentConfig` with every default filled in
from the params dataclasses the components declared at registration (ADR 0018) and every
value the cohort decides (class counts, label maps, survival bins) merged **under** the
explicit ones: the config wins, and a contradiction between the two raises. A dotted-path
class that is not registered keeps its params verbatim. What only the features decide
(the feature dimension) is not in the plan: it is a constructor argument the model bundle
records, so the plan is complete before anything is extracted.

A run writes its config as given (``config.yaml``) and its plan (``plan.yaml``); a resume
compares plans (:meth:`Plan.comparable`). :func:`check` holds the cross-section rules a
config must satisfy; ``resolve`` runs it first.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from soma.config import (
    LEAF_PARAMS,
    Component,
    ExperimentConfig,
    LearnerConfig,
    is_annotation_sampling,
    plain,
    seal,
)

__all__ = [
    "Plan",
    "check",
    "component_leaves",
    "encoder_level",
    "expand",
    "head_family",
    "learner_shape",
    "load_cohort",
    "resolve",
    "revalidated",
    "sample_unit",
]

#: The sections of a config and of its plan, in order.
SECTIONS = ("run", "cohort", "features", "loaders", "learner", "trainer", "evaluation")
#: ``run`` fields a resume may change: where a launch writes and which of the run's folds
#: it takes on (launches pinned to one ``run_id`` split the folds), not what it trains.
RESUME_EXCLUDED_RUN_FIELDS = ("output_dir", "run_id", "resume", "mirror_root", "folds")
#: The ``learner.params`` keys that build the model rather than the learner.
MODEL_PARAMS = (*LEAF_PARAMS[:3], "normalization", "projection")
#: The ``learner.params`` keys that are not the learner class's own keywords: the model's
#: and the optimization leaves (passed as ``optimizer=`` / ``schedulers=``).
LEARNER_OWN_EXCLUDED = (*MODEL_PARAMS, "optimizer", "scheduler")
#: The optimizer a learner leaf without one trains with (``Learner.DEFAULT_OPTIMIZER``).
DEFAULT_OPTIMIZER = Component("adam", {"lr": 1e-4, "weight_decay": 1e-5})
#: The set-shape aggregators whose label-aware auxiliary loss assumes classification.
_SURVIVAL_INCOMPATIBLE_AGGREGATORS = frozenset({"clam_sb", "clam_mb", "dtfdmil"})


class Plan(Mapping):
    """A resolved config, frozen: section name -> section.

    ``plan.learner`` (or ``plan["learner"]``) is the resolved section; its sections refuse
    field writes. :meth:`to_yaml` / :meth:`from_yaml` round-trip it, and two plans are
    equal when their YAML forms are.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = seal(copy.deepcopy(config))

    @property
    def config(self) -> ExperimentConfig:
        """The plan as an :class:`~soma.config.ExperimentConfig` (every default filled in)."""
        return self._config

    def __getitem__(self, section: str) -> Any:
        if section not in SECTIONS:
            raise KeyError(section)
        return getattr(self._config, section)

    def __getattr__(self, name: str) -> Any:
        if name in SECTIONS:
            return getattr(self._config, name)
        raise AttributeError(name)

    def __iter__(self) -> Iterator[str]:
        return iter(SECTIONS)

    def __len__(self) -> int:
        return len(SECTIONS)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Plan) and self.to_dict() == other.to_dict()

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"Plan({self._config!r})"

    def to_dict(self) -> dict[str, Any]:
        return self._config.to_dict()

    def to_yaml(self, path: str | Path | None = None) -> str:
        return self._config.to_yaml(path)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Plan:
        """A plan from its dict form (a parsed ``plan.yaml``), taken as resolved."""
        return cls(ExperimentConfig.from_dict(data))

    @classmethod
    def from_yaml(cls, path: str | Path) -> Plan:
        return cls(ExperimentConfig.from_yaml(path))

    def comparable(self) -> dict[str, Any]:
        """The plan a resume must match: without the launch-only ``run`` fields
        (``output_dir``, ``run_id``, ``resume``, ``mirror_root``, ``folds``). ``run.seed``
        stays."""
        return comparable(self.to_dict())

    @property
    def shape(self) -> str:
        """``set`` or ``grid``: the shape the learner trains."""
        return learner_shape(self._config)

    @property
    def task_family(self) -> str | None:
        """The head's task family (``None`` for a task-free run)."""
        return head_family(self._config)

    @property
    def unit(self) -> str:
        """What one sample of a set run is: ``slide``, ``tile`` or ``patient``."""
        return sample_unit(self._config)


def comparable(payload: Mapping[str, Any]) -> dict[str, Any]:
    """:meth:`Plan.comparable` over a plan's dict form (a parsed ``plan.yaml``)."""
    payload = copy.deepcopy(dict(payload))
    run = payload.get("run")
    if isinstance(run, dict):
        for name in RESUME_EXCLUDED_RUN_FIELDS:
            run.pop(name, None)
    return payload


# --- what a config decides -------------------------------------------------------------


def _registries():
    import soma.models.heads  # noqa: F401 - registers the built-in heads
    from soma.models.aggregators.registry import aggregator_registry
    from soma.models.decoders.registry import decoder_registry
    from soma.learn import (
        augment_registry,
        callback_registry,
        learner_registry,
        optimizer_registry,
        sampler_registry,
        scheduler_registry,
    )
    from soma.models.heads.registry import task_registry

    return {
        "aggregator": aggregator_registry,
        "decoder": decoder_registry,
        "head": task_registry,
        "learner": learner_registry,
        "optimizer": optimizer_registry,
        "scheduler": scheduler_registry,
        "callback": callback_registry,
        "sampler": sampler_registry,
        "augment": augment_registry,
    }


def _resolve_name(registry: Any, name: str, what: str) -> Any:
    try:
        return registry.resolve(name)
    except KeyError as error:
        raise ValueError(f"Unknown {what} {name!r}: {error.args[0]}") from None


def learner_class(config: ExperimentConfig) -> Any:
    if config.learner is None:
        return None
    return _resolve_name(_registries()["learner"], config.learner.name, "learner")


def _is_learner(cls: Any) -> bool:
    from soma.learn import Learner

    return isinstance(cls, type) and issubclass(cls, Learner)


def learner_shape(config: ExperimentConfig) -> str:
    """``grid`` for a :class:`~soma.learn.DenseLearner` (or a subclass), else ``set``."""
    from soma.learn import DenseLearner

    cls = learner_class(config)
    return "grid" if isinstance(cls, type) and issubclass(cls, DenseLearner) else "set"


def _head(config: ExperimentConfig) -> Component | None:
    return None if config.learner is None else config.learner.params.get("head")


def head_family(config: ExperimentConfig) -> str | None:
    head = _head(config)
    if head is None:
        return None
    from soma.models.heads.registry import task_head_class

    return str(task_head_class(head.name).task_family)


def encoder_level(name: str) -> str:
    """What the encoder ``name`` embeds: ``tile``, ``slide`` or ``patient``, as slide2vec
    resolves it. The ``composite`` encoder concatenates tile encoders, so it is ``tile``."""
    if name == "composite":
        return "tile"
    from slide2vec.encoders.registry import encoder_registry, resolve_encoder_level

    return str(resolve_encoder_level(name, encoder_registry.info(name)))


def sample_unit(config: ExperimentConfig) -> str:
    """What one sample is: ``patient`` (``cohort.collapse``), ``tile`` (a tile encoder
    without an aggregator embeds each image whole) or ``slide``. A grid sample is a tile."""
    if config.cohort.collapse == "patient_id":
        return "patient"
    if learner_shape(config) == "grid":
        return "tile"
    encoder = config.features.encoder
    aggregator = None if config.learner is None else config.learner.params.get("aggregator")
    if aggregator is None and encoder is not None and encoder_level(encoder.name) == "tile":
        return "tile"
    return "slide"


def _head_params(head: Component) -> dict[str, Any]:
    """The head's constructor params: its leaf params without the cohort's ``targets``."""
    return {key: value for key, value in head.params.items() if key != "targets"}


def target_columns(config: ExperimentConfig) -> tuple[dict[str, str], dict[str, type]]:
    """``(target key -> cohort column, target key -> dtype)`` the run's cohort must carry."""
    head = _head(config)
    if head is None:
        return {"label": "label"}, {"label": object}
    from soma.models.heads.registry import task_head_class

    schema = task_head_class(head.name).resolve_target_schema(_head_params(head))
    renames = dict(head.params.get("targets") or {})
    unknown = sorted(set(renames) - set(schema))
    if unknown:
        raise ValueError(
            f"head targets name key(s) {unknown} the head does not declare; its target keys are {sorted(schema)}."
        )
    return {key: str(renames.get(key, key)) for key in schema}, schema


def load_cohort(config: ExperimentConfig):
    """The :class:`~soma.data.Cohort` ``config`` trains on: its records and splits, with
    the target columns its head declares (``tune_is_test`` applied)."""
    from soma.data import Cohort

    targets, dtypes = target_columns(config)
    cohort = Cohort.from_csv(
        config.cohort.records,
        config.cohort.splits,
        targets=targets,
        dtypes=dtypes,
        unit=config.cohort.collapse,
        allow_missing_test=config.cohort.tune_is_test,
    )
    head = _head(config)
    derives_bins = head is not None and head.params.get("binning") == "train_quantiles"
    if derives_bins and cohort.records and "bin" in cohort.records[0].metadata:
        raise ValueError(
            "cohort.records has a 'bin' column, but the head derives bins from the training split "
            "(binning: train_quantiles); drop the column or set binning: column."
        )
    return cohort.with_test_from_tune() if config.cohort.tune_is_test else cohort


# --- check: the cross-section rules ----------------------------------------------------


def revalidated(config: ExperimentConfig) -> ExperimentConfig:
    """``config`` rebuilt from its dict form, so every section validates its fields again:
    a field write runs no check of its own, and the rebuilt copy holds what a run reads."""
    return ExperimentConfig.from_dict(config.to_dict())


def check(config: ExperimentConfig) -> None:
    """Refuse a config whose sections contradict each other, before anything runs."""
    _check(revalidated(config))


def _check(config: ExperimentConfig) -> None:
    registries = _registries()
    features, evaluation = config.features, config.evaluation
    if evaluation.representation is not None:
        _check_representation(config)
        return
    if config.learner is None:
        raise ValueError("A run needs a learner with a head, or evaluation.representation for a task-free run.")
    learner = config.learner
    learner_cls = _resolve_name(registries["learner"], learner.name, "learner")
    params = dict(learner.params)
    if not _is_learner(learner_cls):
        registries["learner"].validate(learner.name, params)
        return
    head = params.get("head")
    if head is None:
        raise ValueError(f"learner '{learner.name}' needs a head: learner.params.head = {{name, params}}.")
    if "schedulers" in params:
        raise ValueError("learner.params spells the scheduler leaf (or list of leaves) as 'scheduler'.")
    own = {key: value for key, value in params.items() if key not in LEARNER_OWN_EXCLUDED}
    registries["learner"].validate(learner.name, own)
    _validate_leaf(registries["optimizer"], params.get("optimizer") or DEFAULT_OPTIMIZER, "optimizer")
    for leaf in params.get("scheduler") or ():
        _validate_leaf(registries["scheduler"], leaf, "scheduler")
    from soma.models.heads.registry import task_head_class

    head_cls = task_head_class(head.name)
    family = str(head_cls.task_family)
    head_params = _head_params(head)
    if family == "survival" and "bin_edges" in head_params:
        from soma.models.heads.survival import DiscreteSurvivalHead

        if issubclass(head_cls, DiscreteSurvivalHead):
            raise ValueError(
                f"head '{head.name}': bin_edges cannot be set from a config: they are fitted per fold "
                "on the training split (binning: train_quantiles)."
            )
    registries["head"].validate(head.name, head_params)
    target_columns(config)
    aggregator, decoder = params.get("aggregator"), params.get("decoder")
    for leaf, kind in ((aggregator, "aggregator"), (decoder, "decoder")):
        if leaf is not None:
            _validate_leaf(registries[kind], leaf, kind)
    shape = learner_shape(config)
    grid_families = {"segmentation", "detection"}
    if shape == "grid":
        if family not in grid_families:
            raise ValueError(f"the grid learner '{learner.name}' trains a segmentation or detection head, got '{head.name}' ({family}).")
        if decoder is None:
            raise ValueError(f"a {family} run needs a decoder (e.g. learner.params.decoder: {{name: lightweight_conv}}).")
        if aggregator is not None:
            raise ValueError(f"a {family} run is dense per-pixel: it takes a decoder, not an aggregator.")
        level = None if features.encoder is None else encoder_level(features.encoder.name)
        if level not in (None, "tile"):
            raise ValueError(
                f"encoder '{features.encoder.name}' gives one vector per {level}, but a {family} run "
                "encodes each tile into a token grid: a dense run needs a tile-level encoder."
            )
    else:
        if family in grid_families:
            raise ValueError(f"a {family} head trains the grid shape: use learner {{name: dense}}.")
        if decoder is not None:
            raise ValueError("a decoder is for the grid shape (segmentation, detection); the set shape takes an aggregator.")
    unit = sample_unit(config)
    if unit == "patient" and aggregator is not None:
        raise ValueError("cohort.collapse: patient_id trains on patient embeddings, which take no aggregator.")
    if features.encoder is not None and not features.is_composite and encoder_level(features.encoder.name) == "patient" and unit != "patient":
        raise ValueError(f"encoder '{features.encoder.name}' embeds patients: set cohort.collapse: patient_id.")
    if aggregator is not None and features.encoder is not None and encoder_level(features.encoder.name) == "slide":
        raise ValueError(
            f"encoder '{features.encoder.name}' gives one vector per slide, so there is no bag for an "
            "aggregator to pool: drop learner.params.aggregator or use a tile-level encoder."
        )
    masks = features.extraction.masks
    if family == "segmentation":
        annotation = is_annotation_sampling(masks)
        # A null field is unset, like an omitted one (``soma init`` spells them as null).
        if annotation or head_params.get("classes") is not None or head_params.get("ignore") is not None:
            from soma.extract.dense.reader import check_classes_declared, resolve_class_scheme

            resolve_class_scheme(head_params, annotation_rasters=annotation)
            if annotation:
                check_classes_declared(head_params, masks.pixel_mapping)
    elif family == "detection" and (head_params.get("classes") or head_params.get("drop")):
        from soma._internal.class_scheme import resolve_classes

        resolve_classes(head_params, excluded_key="drop", subject="detection")
    if is_annotation_sampling(masks):
        if not (unit in {"slide", "patient"} if shape == "set" else family == "segmentation"):
            raise ValueError(
                "extraction.masks samples annotations, which only slide or patient bags and "
                "slide-manifest segmentation use."
            )
        if masks.output_mode == "per_annotation":
            raise ValueError(
                "extraction.masks.output_mode='per_annotation' is not yet supported for feature "
                "extraction (soma issue #86); use 'merged'."
            )
    if features.kind == "live" and shape != "grid":
        raise ValueError("features.kind='live' re-encodes tiles for the grid shape (learner 'dense').")
    if features.is_composite and (shape != "grid" or features.kind != "cached"):
        raise ValueError("the composite encoder serves cached grid runs only (learner 'dense', features.kind 'cached').")
    if shape == "set" and unit != "tile" and features.encoder is not None:
        _check_pooled_tiling(config)
    normalization, projection = _adaptor_params(params)
    from soma.learn.feature_adaptor import validate_feature_adaptor_compatibility

    validate_feature_adaptor_compatibility(
        normalization,
        projection,
        feature_mode=features.kind,
        unit=unit if shape == "set" else None,
        has_composite=features.is_composite,
    )
    _check_loaders(config, registries, family=family, shape=shape, decoder=decoder)
    if family == "survival":
        _check_survival(config, head, unit=unit, aggregator=aggregator)
    metrics = head_cls.resolve_metrics(head_params, evaluation.metrics)
    _check_callbacks(config, registries, metrics=metrics, custom_learner=learner.name not in ("mil", "dense"))


def _validate_leaf(registry: Any, leaf: Any, what: str) -> None:
    name = leaf["name"] if isinstance(leaf, Mapping) else leaf.name
    _resolve_name(registry, name, what)
    registry.validate(name, (leaf.get("params") if isinstance(leaf, Mapping) else leaf.params) or {})


def _adaptor_params(params: Mapping[str, Any]):
    from soma.learn.feature_adaptor import NormalizationParams, ProjectionParams

    def build(cls, value, key):
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError(f"learner.params.{key} must be a mapping, got {value!r}")
        return cls(**value)

    return (
        build(NormalizationParams, params.get("normalization"), "normalization"),
        build(ProjectionParams, params.get("projection"), "projection"),
    )


def _check_pooled_tiling(config: ExperimentConfig) -> None:
    """Refuse the tiling a slide or patient run extracts at when its encoder does not
    support it, as preprocessing would: a tile size the encoder cannot take, and a spacing,
    tile size or precision off its recommendation unless ``allow_non_recommended_settings``
    is set. An unset tiling takes the encoder's preset, which the encoder supports."""
    from slide2vec.api import EncoderInputContract
    from slide2vec.encoders import encoder_registry, resolve_preprocessing_requirements
    from slide2vec.encoders.validation import validate_encoder_config

    from soma.models.encoders import resolve_encoder_precision

    features = config.features
    encoder, tiling = features.encoder, features.extraction.tiling
    requested = {
        "spacing": None if tiling is None else float(tiling.requested_spacing_um),
        "tile size": None if tiling is None else int(tiling.requested_tile_size_px),
        "precision": resolve_encoder_precision(features.execution.precision, encoder_name=encoder.name),
    }
    if tiling is not None:
        try:
            # The capability alone, which the flag does not lift: the flag's permission for
            # an off-preset size is the recommendation check below.
            EncoderInputContract.declared_pooled(
                encoder.name, requested_tile_size_px=requested["tile size"], allow_non_recommended_settings=True
            )
        except ValueError as error:
            raise ValueError(
                f"encoder '{encoder.name}' cannot take {requested['tile size']} px tiles "
                f"(features.extraction.tiling): {error} "
                "features.encoder.params.allow_non_recommended_settings does not lift this."
            ) from None
    if encoder.allow_non_recommended_settings:
        return
    try:
        validate_encoder_config(
            encoder.name,
            requested_spacing_um=requested["spacing"],
            requested_tile_size_px=requested["tile size"],
            precision=requested["precision"],
        )
    except ValueError:
        pass
    else:
        return
    # slide2vec's message names its own settings; name soma's.
    info = encoder_registry.info(encoder.name)
    supported = {
        "spacing": info.get("supported_spacing_um"),
        "tile size": resolve_preprocessing_requirements(encoder.name, info)["tile_size_px"],
        "precision": info.get("precision"),
    }
    known = [name for name, value in supported.items() if value is not None and requested[name] is not None]
    raise ValueError(
        f"encoder '{encoder.name}' supports {_settings(supported, known)}; this run extracts at "
        f"{_settings(requested, known)} (features.extraction.tiling, features.execution.precision). Set "
        "features.encoder.params.allow_non_recommended_settings: true to extract at other settings."
    )


def _settings(values: Mapping[str, Any], names: list[str]) -> str:
    """``spacing 0.25 or 0.5 µm, tile size 224 px and precision fp16``."""

    def alternatives(value: Any) -> str:
        options = [f"{float(option):g}" for option in (value if isinstance(value, (list, tuple)) else [value])]
        return options[0] if len(options) == 1 else f"{', '.join(options[:-1])} or {options[-1]}"

    units = {"spacing": " µm", "tile size": " px", "precision": ""}
    parts = [
        f"{name} {alternatives(values[name]) if name == 'spacing' else values[name]}{units[name]}" for name in names
    ]
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"


def _check_representation(config: ExperimentConfig) -> None:
    features = config.features
    if config.learner is not None:
        raise ValueError("evaluation.representation is a task-free evaluation: drop the learner section.")
    if config.run.seed != 0:
        raise ValueError(
            f"representation v1 is deterministic and requires run.seed=0; got {config.run.seed!r}. "
            "Repeated launches are attempts, not seeds."
        )
    if features.kind != "cached" or features.is_composite:
        raise ValueError("representation v1 reads cached single-encoder features (features.kind 'cached').")
    if features.encoder is not None and encoder_level(features.encoder.name) != "tile":
        raise ValueError("representation v1 scores tile embeddings: use a tile-level encoder.")
    if config.loaders.augment is not None:
        raise ValueError("representation v1 takes no augmentation.")


def _check_loaders(config: ExperimentConfig, registries, *, family: str, shape: str, decoder) -> None:
    loaders, features = config.loaders, config.features
    if loaders.augment is not None:
        if features.kind != "live":
            raise ValueError(
                "loaders.augment requires features.kind='live': cached grids are frozen features "
                "and cannot be augmented."
            )
        _validate_leaf(registries["augment"], loaders.augment, "augmentation")
    sampler = loaders.sampler
    if sampler is None:
        return
    if not (shape == "grid" and family == "segmentation" and features.kind == "cached" and decoder is not None):
        raise ValueError(
            "loaders.sampler is the cached neural segmentation training-batch contract; other "
            "task and data paths would silently ignore it."
        )
    _validate_leaf(registries["sampler"], sampler, "sampler")
    params = sampler.params
    draws = params.get("draws_per_epoch")
    if draws is not None and draws % loaders.batch_size != 0:
        raise ValueError(
            f"loaders.sampler draws_per_epoch must contain whole batches of batch_size={loaders.batch_size}, got {draws}."
        )
    ratios = params.get("class_request_ratios")
    if ratios is not None:
        import math
        from numbers import Real

        bad = [r for r in ratios if isinstance(r, bool) or not isinstance(r, Real) or not math.isfinite(float(r)) or r < 0]
        if not ratios or bad or sum(float(r) for r in ratios) <= 0:
            raise ValueError(
                "loaders.sampler class_request_ratios must be finite, non-negative and contain a "
                f"positive value, got {ratios!r}."
            )


def _check_survival(config: ExperimentConfig, head: Component, *, unit: str, aggregator) -> None:
    from soma.models.heads.survival import CoxSurvivalHead
    from soma.models.heads.registry import task_head_class

    if unit == "tile":
        raise ValueError(
            "survival labels belong to a slide or patient, not a single tile: use an aggregator "
            "or a slide- or patient-level encoder."
        )
    registries = _registries()
    if aggregator is not None and registries["aggregator"].key_of(aggregator.name) in _SURVIVAL_INCOMPATIBLE_AGGREGATORS:
        raise ValueError(
            f"Aggregator '{aggregator.name}' is not supported for survival tasks: its label-aware "
            "auxiliary loss assumes classification. Use e.g. abmil, transmil or mean_pool."
        )
    if not issubclass(task_head_class(head.name), CoxSurvivalHead):
        return
    cox_window = head.params.get("cox_window", 1)
    batch_size = config.loaders.batch_size
    if config.trainer.accumulate_grad_batches > 1:
        raise ValueError(
            "Cox survival requires trainer.accumulate_grad_batches = 1: accumulation gives several "
            "independent risk sets, not one larger one; raise cox_window or loaders.batch_size."
        )
    if cox_window >= 2:
        if batch_size != 1:
            raise ValueError(
                "Cox accumulation mode (cox_window >= 2) requires loaders.batch_size = 1: bags are "
                "forwarded un-padded one at a time, with cox_window as the risk-set size."
            )
        if aggregator is None:
            raise ValueError(
                "Cox accumulation mode (cox_window >= 2) is for variable-size MIL bags and needs an "
                "aggregator; for single embeddings use padded mode with batch_size >= 2."
            )
    elif batch_size < 2:
        raise ValueError(
            "Cox survival requires loaders.batch_size >= 2: the partial likelihood's risk set is the "
            "batch (or set head cox_window >= 2 for large MIL bags)."
        )


def _check_callbacks(config: ExperimentConfig, registries, *, metrics: list[str], custom_learner: bool) -> None:
    callbacks = config.trainer.callbacks
    names = []
    for leaf in callbacks:
        _validate_leaf(registries["callback"], leaf, "callback")
        names.append(registries["callback"].key_of(leaf.name))
    selections = [leaf.params.get("selection", "best") for leaf, key in zip(callbacks, names) if key == "checkpoint"]
    if "last" in selections:
        if "early_stopping" in names:
            raise ValueError(
                "a checkpoint with selection 'last' keeps the final epoch and never early-stops; "
                "remove the early_stopping callback."
            )
        if config.trainer.val_every_n_epochs != 1:
            raise ValueError(
                "a checkpoint with selection 'last' records tune metrics every epoch as "
                "diagnostics; trainer.val_every_n_epochs must be 1."
            )
    if custom_learner:  # its own validation_summary keys; the Checkpoint checks them at fit time
        return
    for leaf, key in zip(callbacks, names):
        monitor = leaf.params.get("monitor", "loss")
        if key in ("checkpoint", "early_stopping") and monitor != "loss" and monitor not in metrics:
            raise ValueError(
                f"{key} monitor {monitor!r} is not an effective metric of the head. Available "
                f"metrics: {', '.join(metrics)}; use 'loss' to monitor the tune loss."
            )


# --- resolve ---------------------------------------------------------------------------


def resolve(config: ExperimentConfig, cohort: Any) -> Plan:
    """Expand ``config`` into its :class:`Plan` against ``cohort``.

    Every ``{name, params}`` leaf takes its registered params dataclass's defaults; the
    head's ``infer`` hook adds what the cohort decides, under the explicit params (a
    differing explicit value raises); the head's ``targets`` are spelled out. The plan is
    normalized through its YAML form, so a Python config and the YAML it saves resolve to
    one plan.
    """
    return Plan(_spelled_out(config, cohort, infer=True))


def expand(config: ExperimentConfig) -> ExperimentConfig:
    """``config`` with every ``{name, params}`` leaf's registered defaults spelled out and
    nothing the cohort decides: what ``soma init`` prints. Its plan is ``config``'s."""
    return _spelled_out(config, None, infer=False)


def _spelled_out(config: ExperimentConfig, cohort: Any, *, infer: bool) -> ExperimentConfig:
    config = revalidated(config)
    _check(config)
    registries = _registries()
    learner = config.learner
    if learner is not None:
        learner = _resolve_learner(config, learner, cohort, registries, infer=infer)
    encoder = config.features.encoder
    if encoder is not None:
        from soma.models.encoders import encoder_registry

        encoder = replace(encoder, params=encoder_registry.canonical_params(encoder.name, encoder.params))
    loaders = config.loaders
    resolved = replace(
        config,
        features=replace(config.features, encoder=encoder),
        loaders=replace(
            loaders,
            sampler=_expand(registries["sampler"], loaders.sampler),
            augment=_expand(registries["augment"], loaders.augment),
        ),
        learner=learner,
        trainer=replace(
            config.trainer, callbacks=[_expand(registries["callback"], leaf) for leaf in config.trainer.callbacks]
        ),
    )
    return ExperimentConfig.from_dict(plain(resolved))


def _expand(registry: Any, leaf: Any) -> Any:
    """``leaf`` with its params as its registered params dataclass holds them."""
    if leaf is None:
        return None
    if isinstance(leaf, Mapping):  # a scheduler leaf carrying ``interval``
        return {**leaf, "params": registry.canonical_params(leaf["name"], leaf.get("params"))}
    return replace(leaf, params=registry.canonical_params(leaf.name, leaf.params))


def _resolve_learner(
    config: ExperimentConfig, learner: LearnerConfig, cohort: Any, registries, *, infer: bool
) -> LearnerConfig:
    cls = registries["learner"].resolve(learner.name)
    params = dict(learner.params)
    if not _is_learner(cls):
        return replace(learner, params=registries["learner"].canonical_params(learner.name, params))
    resolved: dict[str, Any] = {}
    for key in ("aggregator", "decoder"):
        if params.get(key) is not None:
            resolved[key] = _expand(registries[key], params[key])
    head = params["head"]
    if infer:
        resolved["head"] = _resolve_head(config, head, cohort, registries["head"])
    else:
        defaults = registries["head"].canonical_params(head.name, _head_params(head))
        resolved["head"] = replace(head, params={**defaults, "targets": target_columns(config)[0]})
    normalization, projection = _adaptor_params(params)
    resolved["normalization"] = dataclasses.asdict(normalization)
    resolved["projection"] = dataclasses.asdict(projection)
    resolved["optimizer"] = _expand(registries["optimizer"], params.get("optimizer") or DEFAULT_OPTIMIZER)
    resolved["scheduler"] = [_expand(registries["scheduler"], leaf) for leaf in params.get("scheduler") or ()]
    own = {key: value for key, value in params.items() if key not in LEARNER_OWN_EXCLUDED}
    expanded = registries["learner"].canonical_params(learner.name, own)
    for key in ("optimizer", "schedulers"):  # the leaves above, spelled once
        if key not in own:
            expanded.pop(key, None)
    return replace(learner, params={**resolved, **expanded})


def _resolve_head(config: ExperimentConfig, head: Component, cohort: Any, registry: Any) -> Component:
    from soma.models.heads.registry import task_head_class

    head_cls = task_head_class(head.name)
    explicit = _head_params(head)
    # A null param is unset, as the params defaults and the infer hooks read it: the
    # cohort decides it, so it neither contradicts nor shadows the inferred value.
    given = {key: value for key, value in explicit.items() if value is not None}
    if head_cls.task_family == "survival":
        from soma.models.heads.survival import resolve_survival_task, validate_survival_dataset

        _, loss = resolve_survival_task(head.name)
        validate_survival_dataset(
            cohort.records,
            sample_unit(config),
            loss=loss,
            num_bins=given.get("num_bins"),
            binning=given.get("binning", "column"),
        )
    inferred = plain(registry.infer(head.name, cohort, given))
    contradictions = sorted(key for key in set(inferred) & set(given) if plain(given[key]) != inferred[key])
    if contradictions:
        raise ValueError(
            f"head '{head.name}': {', '.join(f'{key}={given[key]!r}' for key in contradictions)} "
            f"contradicts what the cohort shows ({', '.join(f'{key}={inferred[key]!r}' for key in contradictions)})."
        )
    params = registry.canonical_params(head.name, {**explicit, **inferred, **given})
    # The head's fit hook runs per fold at train time; a fold it cannot fit (too few
    # uncensored times for the survival bins) fails here, before anything is extracted.
    # What it fits stays out of the plan, which is the same for every fold.
    for fold, records in enumerate(_fold_training_records(config, cohort)):
        try:
            registry.fit(head.name, records, params)
        except ValueError as error:
            raise ValueError(f"Fold {fold}: {error}") from None
    targets, _ = target_columns(config)
    return replace(head, params={**params, "targets": targets})


def _fold_training_records(config: ExperimentConfig, cohort: Any) -> Iterator[list[Any]]:
    """Each fold's training records, one per sample the fold trains on: a patient run's
    patients by their first record in cohort order, so a patient with several slides
    counts once."""
    if sample_unit(config) != "patient":
        for fold in cohort.folds:
            yield [cohort.record(sample_id) for sample_id in fold.train]
        return
    first: dict[str, Any] = {}
    for record in cohort.records:
        first.setdefault(record.patient_id, record)
    for fold in cohort.folds:
        patients = dict.fromkeys(cohort.record(sample_id).patient_id for sample_id in fold.train)
        yield [first[patient_id] for patient_id in patients]


# --- provenance ------------------------------------------------------------------------


#: The learner params that hold a component leaf, in the order a record lists them.
_LEARNER_LEAVES = ("aggregator", "decoder", "head", "optimizer", "scheduler")


def component_leaves(plan: Plan | ExperimentConfig) -> list[tuple[str, str, Any]]:
    """``(kind, name, object)`` for every component leaf of ``plan``.

    The leaves are the encoder, the sampler, the augmentation, each trainer callback, the
    learner and its leaves (aggregator or decoder, head, optimizer, schedulers), and each
    ``evaluation.metrics`` entry, in that order. ``name`` is spelled as the plan spells it
    (a registry key or a dotted path) and ``object`` is the class or function it resolves
    to; it is ``None`` for a metric that only declares a direction.
    """
    from soma.models.encoders import encoder_registry

    config = plan.config if isinstance(plan, Plan) else plan
    registries = {**_registries(), "encoder": encoder_registry}
    named: list[tuple[str, str]] = []
    if config.features.encoder is not None:
        named.append(("encoder", config.features.encoder.name))
    for kind, leaf in (("sampler", config.loaders.sampler), ("augment", config.loaders.augment)):
        if leaf is not None:
            named.append((kind, leaf.name))
    named += [("callback", leaf.name) for leaf in config.trainer.callbacks]
    if config.learner is not None:
        named.append(("learner", config.learner.name))
        params = config.learner.params
        for kind in _LEARNER_LEAVES:
            value = params.get(kind)
            for leaf in value if kind == "scheduler" and isinstance(value, (list, tuple)) else [value]:
                if leaf is not None:
                    named.append((kind, _leaf_name(leaf)))
    leaves = [(kind, name, registries[kind].resolve(name)) for kind, name in named]
    for entry in config.evaluation.metrics:
        name = _leaf_name(entry)
        leaves.append(("metric", name, _metric_object(name)))
    return leaves


def _leaf_name(leaf: Any) -> str:
    """The name of a leaf spelled as a name, a ``Component`` or a ``{name, params}`` mapping."""
    if isinstance(leaf, Component):
        return leaf.name
    if isinstance(leaf, Mapping):
        return str(leaf["name"])
    return str(leaf)


def _metric_object(name: str) -> Any:
    """The function a metric name scores with: a registered metric's, or a dotted path's."""
    from soma.evaluation.metrics import metric_registry

    if name in metric_registry:
        return metric_registry.get(name).fn
    if "." in name:
        module_name, _, attribute = name.rpartition(".")
        return getattr(importlib.import_module(module_name), attribute)
    return None
