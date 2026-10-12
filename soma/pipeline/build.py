"""Config → objects: the run's feature source and, per fold, the :class:`FoldSetup`.

Everything a ``PipelineConfig`` decides lives here, so :func:`~soma.pipeline.folds.run_fold`
holds only the canonical order. The ``dataset_type`` branches stay here until slice 6
replaces them with ``{name, params}`` leaves:

* :func:`attach_source` — the supplied source, or one extracted (or built live) from the
  config;
* :func:`build_set_model` / :func:`build_grid_model` — the two model builders. They are
  also what :func:`soma.load` calls to rebuild a bundle, so a bundle is rebuilt by the
  same code that built the model;
* :func:`build_set_fold` / :func:`build_grid_fold` — one fold's shape-specific steps,
  as deferred callables ``run_fold`` applies in the canonical order.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import json
import logging
import os
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from soma.aggregators.registry import aggregator_registry
from soma.cache import resolve_cache_root
from soma.config import (
    AggregatorConfig,
    EncoderConfig,
    EncoderMemberConfig,
    EvalConfig,
    NormalizationConfig,
    PipelineConfig,
    PreprocessingConfig,
    ProjectionConfig,
    TrainingConfig,
    validate_feature_adaptor_compatibility,
)
from soma.data import (
    AnnotationManifest,
    Cohort,
    FoldSplit,
    GridGeometry,
    GridSource,
    ImageManifest,
    TileImageSource,
    require_coverage,
)
from soma.data.cohort import targets_frame
from soma.decoders.registry import build_decoder_for_grid
from soma.dense.composite import CompositeDenseFeatureStore
from soma.dense.reader import CLASS_SCHEME_KEYS, resolve_class_scheme
from soma.evaluation import Baseline
from soma.evaluation.metrics import resolve_metrics
from soma.extraction import FeatureExtractor, _release_parent_cuda_state
from soma.learn import (
    Checkpoint,
    EarlyStopping,
    Learner,
    Progress,
    learner_registry,
    record_builder,
    resolve_leaf,
)
from soma.learn.grid import grid_loaders
from soma.pipeline import _legacy
from soma.pipeline._legacy import LegacyFolds, LegacyRecord, LegacySamples
from soma.preprocessing.resolution import resolve_pipeline_preprocessing
from soma.tasks.classification import BranchAwareClassificationHead, MulticlassClassificationHead
from soma.tasks.registry import task_registry
from soma.training.bag_dataset import BagDataset, HierarchicalBagDataset
from soma.training.collate import bag_collate_fn, cox_window_collate, hierarchical_bag_collate_fn
from soma.training.feature_adaptor import (
    FeatureAdaptor,
    build_feature_adaptor,
    feature_adaptor_output_dim,
    write_feature_adapter_sidecar,
)
from soma.training.fold_planning import plan_dense_fold
from soma.training.model import DenseModel, EmbeddingModel, MILModel
from soma.training.patient_dataset import PatientDataset, patient_collate_fn
from soma.training.sample_dataset import SampleDataset, sample_collate_fn

__all__ = [
    "FoldLoaders",
    "FoldSetup",
    "RunInputs",
    "SourceContext",
    "SplitPlan",
    "attach_source",
    "build_composite_dense_store",
    "build_fold",
    "build_grid_fold",
    "build_grid_model",
    "build_set_fold",
    "build_set_model",
    "composite_member_extraction_spec",
    "extraction_shape",
    "extraction_unit",
    "live_encoder_for",
    "live_encoder_spec",
    "load_pipeline_data",
    "resolve_task_class",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# What run_fold applies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitPlan:
    """How one split is scored: its loader, its targets, and the ids the baseline scores.

    ``targets`` is what :func:`soma.evaluate` scores against (a targets frame for the set
    shape, a target source for the grid). ``placeholder_ids`` are the split's samples
    without features (set shape), scored by the training :class:`~soma.evaluation.Baseline`.
    ``write`` says whether ``predictions_<name>`` lands in the fold dir.
    """

    name: str
    loader: DataLoader
    targets: Any
    placeholder_ids: tuple[str, ...] = ()
    subgroups: tuple[str, ...] = ()
    write: bool = True
    callbacks: tuple[Any, ...] = ()


@dataclass(frozen=True)
class FoldLoaders:
    """Step 4's output: the training loaders and the scored splits (tune first)."""

    train: DataLoader
    tune: DataLoader | None
    splits: tuple[SplitPlan, ...]
    #: Side files written once the fit returns (the ROI batch sampler's audit).
    after_fit: Callable[[Path], None] | None = None


@dataclass(frozen=True)
class FoldSetup:
    """One fold's shape-specific steps, deferred so ``run_fold`` applies them in order.

    ``fit_adaptor(fold_dir)`` fits the feature adaptor on the train split (step 2);
    ``build_learner(adaptor)`` builds the model and its learner (step 3);
    ``build_loaders(learner, fold_dir)`` the loaders and scored splits (step 4);
    ``fit_baseline(head)``
    the placeholder predictor (step 5, set shape); ``callbacks()`` and ``trainer`` the
    fit (step 6); ``tune_step(predictions, split, head)`` the tune-time threshold
    (step 7, detection); ``metrics(head)`` what every split is scored with (step 8);
    ``preprocessing(head, threshold)`` and ``provenance`` the bundle's metadata (step 9).
    """

    fold: int
    num_folds: int
    seed: int
    fit_adaptor: Callable[[Path], FeatureAdaptor | None]
    build_learner: Callable[[FeatureAdaptor | None], Learner]
    build_loaders: Callable[[Learner, Path], FoldLoaders]
    callbacks: Callable[[], list[Any]]
    trainer: dict[str, Any]
    metrics: Callable[[Any], list[Any]]
    preprocessing: Callable[[Any, Any], dict[str, Any]]
    provenance: dict[str, Any]
    fit_baseline: Callable[[Any], Baseline] | None = None
    tune_step: Callable[[Any, SplitPlan, Any], Any] | None = None


@dataclass
class RunInputs:
    """What every fold of a run shares: the config, the attached source, the records."""

    config: PipelineConfig
    preprocessing: PreprocessingConfig
    evaluation: EvalConfig
    source: Any
    cohort: Cohort
    dataset: LegacySamples
    splits: LegacyFolds
    run_id: str
    run_dir: Path
    source_identity: dict[str, Any]
    encoder_stage: Any = None
    roi_population_cache_root: Path | None = None
    _roi_population: Any = field(default=None, repr=False)
    _roi_caches: tuple[dict, dict] = field(default_factory=lambda: ({}, {}), repr=False)

    @property
    def num_folds(self) -> int:
        return self.splits.num_folds

    def roi_population(self):
        """The run's segmentation ROI population (explicit ROI batch sampling only),
        resolved once and shared by every fold."""
        if self._roi_population is None:
            records = _legacy.segmentation_run_records(self.dataset, self.splits)
            head = _build_segmentation_head(self, self.source.geometry(records[0].sample_id))
            self._roi_population = _resolve_segmentation_roi_population(
                cache_root=self.roi_population_cache_root or self.run_dir / "segmentation_roi_population",
                records=records,
                head=head,
                label_maps=_legacy.segmentation_label_maps(head, records),
                source_fingerprint_cache=self._roi_caches[0],
                resolved_backend_cache=self._roi_caches[1],
            )
        return self._roi_population


def build_fold(inputs: RunInputs, fold: int, fold_split: FoldSplit) -> FoldSetup:
    """The fold's :class:`FoldSetup`, from the builder of the config's shape."""
    if extraction_shape(inputs.config.dataset_type) == "grid":
        return build_grid_fold(inputs, fold, fold_split)
    return build_set_fold(inputs, fold, fold_split)


# ---------------------------------------------------------------------------
# Models: built by the pipeline, rebuilt by soma.load
# ---------------------------------------------------------------------------

#: ``adaptor=`` default of the builders: build the adaptor unfitted from its sections
#: (its fitted buffers load from the bundle's weights).
UNFITTED: Any = object()


def _adaptor_sections(normalization, projection) -> tuple[NormalizationConfig | None, ProjectionConfig | None]:
    return (
        None if normalization is None else NormalizationConfig(**normalization),
        None if projection is None else ProjectionConfig(**projection),
    )


def _section(config: Any) -> dict[str, Any] | None:
    return None if config is None else dataclasses.asdict(config)


def _component_name(registry, configured: str, cls: type) -> str:
    """The configured registry key when it names ``cls``, else ``cls``'s dotted path."""
    try:
        if registry.resolve(configured) is cls:
            return configured
    except KeyError:
        pass
    return f"{cls.__module__}.{cls.__qualname__}"


def build_set_model(
    *,
    kind: str,
    feature_dim: int,
    head: dict[str, Any],
    aggregator: dict[str, Any] | None = None,
    normalization: dict[str, Any] | None = None,
    projection: dict[str, Any] | None = None,
    encoder_identity: str = "",
    adaptor: Any = UNFITTED,
) -> nn.Module:
    """A set-shape model: ``MILModel`` (``kind="mil"``) or ``EmbeddingModel``.

    ``head`` and ``aggregator`` are ``{name, params}`` sections with fully resolved
    params (class count, label map, HIPT geometry). The aggregator is built before the
    head, both against the adaptor's output width; ``adaptor`` is the fitted
    :class:`FeatureAdaptor` (``None`` for none) or, by default, rebuilt unfitted from
    ``normalization`` / ``projection``.
    """
    if adaptor is UNFITTED:
        adaptor = build_feature_adaptor(
            *_adaptor_sections(normalization, projection),
            num_features=feature_dim,
            encoder_identity=encoder_identity,
        )
    width = feature_adaptor_output_dim(adaptor, num_features=feature_dim)
    head_cls = task_registry.resolve(head["name"])
    if kind == "embedding":
        model: nn.Module = EmbeddingModel(
            task_head=head_cls(input_dim=width, **head["params"]), feature_adaptor=adaptor
        )
    elif kind == "mil":
        if aggregator is None:
            raise ValueError("a MIL model needs an aggregator section")
        name = aggregator["name"]
        params = aggregator_registry.validate(name, aggregator["params"])
        agg = aggregator_registry.resolve(name)(input_dim=width, **params)
        model = MILModel(
            aggregator=agg,
            task_head=head_cls(input_dim=agg.output_dim, **head["params"]),
            feature_adaptor=adaptor,
        )
    else:
        raise ValueError(f"kind must be 'mil' or 'embedding', got {kind!r}")
    return record_builder(
        model,
        build_set_model,
        kind=kind,
        feature_dim=feature_dim,
        head=head,
        aggregator=aggregator,
        normalization=normalization,
        projection=projection,
        encoder_identity=encoder_identity,
    )


#: Live encoder stages, keyed by their spec, alive while any model holds one: a fold
#: ensemble of bundles loads one stage, not one per fold.
_LIVE_STAGES: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def _live_stage(spec: dict[str, Any]):
    from soma import extract

    key = json.dumps(spec, sort_keys=True, default=str)
    stage = _LIVE_STAGES.get(key)
    if stage is None:
        stage = extract.live_encoder(**spec)
        _LIVE_STAGES[key] = stage
    return stage


def build_grid_model(
    *,
    feature_dim: int,
    decoder: dict[str, Any],
    head: dict[str, Any],
    normalization: dict[str, Any] | None = None,
    projection: dict[str, Any] | None = None,
    encoder_identity: str = "",
    encoder: dict[str, Any] | None = None,
    adaptor: Any = UNFITTED,
    stage: Any = None,
) -> DenseModel:
    """A grid-shape :class:`DenseModel`: the head, then the decoder against the adaptor's
    output width, behind an optional frozen encoder stage.

    ``encoder`` is the :func:`soma.extract.live_encoder` spec of a live run; without a
    ``stage`` it is built here, eagerly, so a bundle whose gated weights are missing
    fails at load time. The pipeline passes the run's stage, built once for every fold.
    """
    if stage is None and encoder is not None:
        stage = _live_stage(encoder)
    if adaptor is UNFITTED:
        adaptor = build_feature_adaptor(
            *_adaptor_sections(normalization, projection),
            num_features=feature_dim,
            encoder_identity=encoder_identity,
        )
    head_module = task_registry.resolve(head["name"])(**head["params"])
    decoder_module = build_decoder_for_grid(
        decoder["name"],
        decoder["params"],
        geometry=head["params"]["geometry"],
        input_dim=feature_adaptor_output_dim(adaptor, num_features=feature_dim),
        num_classes=head_module.num_classes,
    )
    model = DenseModel(decoder_module, head_module, encoder=stage, adaptor=adaptor)
    return record_builder(
        model,
        build_grid_model,
        feature_dim=feature_dim,
        decoder=decoder,
        head=head,
        normalization=normalization,
        projection=projection,
        encoder_identity=encoder_identity,
        encoder=encoder,
    )


# ---------------------------------------------------------------------------
# Learner and fit (step 6)
# ---------------------------------------------------------------------------


def _optimizer_leaf(training: TrainingConfig) -> dict[str, object]:
    """The config's optimizer leaf with ``learning_rate`` / ``weight_decay`` as its params."""
    leaf = training.optimizer if isinstance(training.optimizer, dict) else {"name": training.optimizer}
    params = {"lr": training.learning_rate, "weight_decay": training.weight_decay}
    params.update(leaf.get("params") or {})
    return {**leaf, "params": params}


def _scheduler_leaves(training: TrainingConfig) -> tuple[object, ...]:
    schedulers = training.scheduler if isinstance(training.scheduler, list) else [training.scheduler]
    return tuple(leaf for leaf in schedulers if leaf is not None and leaf != "none")


def build_learner(model: nn.Module, training: TrainingConfig, *, shape: str) -> Learner:
    """Wrap ``model`` in the configured learner (``training.learner``) or the shape's."""
    cls, params, _ = resolve_leaf(training.learner or {"name": shape}, learner_registry)
    return cls(model, optimizer=_optimizer_leaf(training), schedulers=_scheduler_leaves(training), **params)


def _fit_callbacks(training: TrainingConfig, *, fold: int, num_folds: int) -> Callable[[], list[Any]]:
    """``Checkpoint`` (selection in memory, no file) and, unless ``patience`` is ``None``
    (required by ``checkpoint_selection='last'``), ``EarlyStopping``; plus ``Progress``."""

    def callbacks() -> list[Any]:
        installed: list[Any] = [
            Checkpoint(
                monitor=training.monitor,
                mode=training.monitor_mode,
                selection=training.checkpoint_selection,
            )
        ]
        if training.patience is not None:
            installed.append(
                EarlyStopping(
                    monitor=training.monitor,
                    mode=training.monitor_mode,
                    patience=training.patience,
                    hold_until_moves=training.hold_patience_until_monitor_moves,
                )
            )
        installed.append(Progress(label=f"fold {fold + 1}/{num_folds}" if num_folds > 1 else None))
        return installed

    return callbacks


def _trainer_kwargs(training: TrainingConfig) -> dict[str, Any]:
    return {
        "max_epochs": training.epochs,
        "max_steps": training.max_steps,
        "accumulate_grad_batches": training.gradient_accumulation,
        "val_every_n_epochs": training.tune_every_n_epochs,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


# ---------------------------------------------------------------------------
# Shared fold helpers
# ---------------------------------------------------------------------------


def _loader_kwargs(training: TrainingConfig) -> dict[str, object]:
    """DataLoader kwargs from ``TrainingConfig``; ``persistent_workers`` needs workers."""
    return {
        "batch_size": training.batch_size,
        "num_workers": training.num_workers,
        "pin_memory": training.pin_memory,
        "persistent_workers": training.persistent_workers and training.num_workers > 0,
    }


def _format_fold_summary(
    fold: int,
    train_count: int,
    tune_count: int,
    tests_counts: dict[str, int],
    empty_sample_ids_by_split: dict[str, list[str]] | None = None,
) -> str:
    tests_str = " ".join(f"{name}={count}" for name, count in sorted(tests_counts.items()))
    base = f"Fold {fold}: train={train_count} tune={tune_count} {tests_str}"
    if not empty_sample_ids_by_split:
        return base
    empty_counts = {split: len(ids) for split, ids in empty_sample_ids_by_split.items()}
    if not any(empty_counts.values()):
        return base
    extras: list[str] = []
    if empty_counts.get("train", 0):
        extras.append(f"train empty dropped={empty_counts['train']}")
    fallback = {split: count for split, count in empty_counts.items() if split != "train" and count}
    if fallback:
        extras.append(
            "eval empty fallback=" + ", ".join(f"{split}={count}" for split, count in sorted(fallback.items()))
        )
    return f"{base} | {' | '.join(extras)}" if extras else base


class _SupportFeatures:
    """A re-iterable stream of the train split's feature tensors, one sample at a time.

    The feature adaptor fits its stages in order, so a fitted projection needs a second
    pass over rows the normalize stage has already transformed; this restarts the stream
    on every ``__iter__`` while never holding more than one sample at once.
    """

    def __init__(self, feature_store, records) -> None:
        self._feature_store = feature_store
        self._sample_ids = [record.sample_id for record in records]

    def __iter__(self):
        return (self._feature_store.load(sample_id) for sample_id in self._sample_ids)


class _SupportGrids(_SupportFeatures):
    """The train ROIs' dense grids, **channel-axis last**: every grid position is one
    feature vector, which is the row layout :meth:`FeatureAdaptor.fit` expects."""

    def __iter__(self):
        return (self._feature_store.load(sample_id).movedim(0, -1) for sample_id in self._sample_ids)


def _adaptor_step(
    config: PipelineConfig,
    *,
    support: _SupportFeatures,
    num_support: int,
    feature_dim: int,
    encoder_identity: str,
) -> Callable[[Path], FeatureAdaptor | None]:
    """Step 2: fit the fold's feature adaptor on the train split alone ("K means K").

    It happens before anything trainable is constructed, so the aggregator/decoder/head
    are built against the adaptor's (possibly rewired) output width. Held-out samples
    only pass through ``forward()``, so the transform is leak-free by construction.
    """

    def fit_adaptor(fold_dir: Path) -> FeatureAdaptor | None:
        adaptor = build_feature_adaptor(
            config.normalization,
            config.projection,
            num_features=feature_dim,
            encoder_identity=encoder_identity,
        )
        if adaptor is None:
            return None
        adaptor.fit(support)
        logger.info(
            "Fitted feature adaptor on %d Support sample(s): normalization=%s "
            "(%d channel(s) at the eps floor), projection=%s (%d → %d dims)",
            num_support,
            adaptor.method,
            adaptor.num_eps_floored,
            adaptor.projection_method,
            feature_dim,
            adaptor.output_dim,
        )
        write_feature_adapter_sidecar(adaptor, fold_dir, n_support_samples=num_support)
        return adaptor

    return fit_adaptor


def _no_adaptor(fold_dir: Path) -> None:
    return None


def _encoder_identity(config: PipelineConfig) -> str:
    return config.encoder.name if config.encoder is not None else ""


def _jsonable(value: Any) -> Any:
    """``value`` with numpy scalars, arrays and tuples turned into JSON types."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


# ---------------------------------------------------------------------------
# Set shape
# ---------------------------------------------------------------------------


def _resolve_tune_is_test_split(fold_split: FoldSplit, fold_label: str) -> str:
    test_split_names = fold_split.test_split_names
    if len(test_split_names) != 1:
        raise ValueError(
            f"{fold_label} has {len(test_split_names)} test splits; "
            "training.tune_is_test=True requires exactly one test split"
        )
    return test_split_names[0]


def _bag_head_class(task_cls: type, *, task_name: str, aggregator_name: str) -> type:
    """Pick the head class for a bag-level run.

    ``clam_mb`` emits one branch per class and needs a branch-aware head; every other
    task family has no per-class branch to route through. A registered head that already
    handles branch representations is kept as is; the ordinary multiclass head is
    swapped for the built-in branch-aware one.
    """
    if aggregator_name != "clam_mb":
        return task_cls
    task_family = str(task_cls.task_family)
    if task_family != "multiclass_classification":
        raise ValueError(
            f"clam_mb does not support task '{task_name}' (family "
            f"'{task_family}'); it only supports the multiclass_classification "
            "family. Use a different aggregator."
        )
    if getattr(task_cls, "supports_branch_representation", False):
        return task_cls
    if task_cls is MulticlassClassificationHead:
        return BranchAwareClassificationHead
    raise ValueError(
        f"clam_mb requires custom task head '{task_name}' to support branch "
        "representations. Subclass BranchAwareClassificationHead to preserve "
        "your custom loss and metrics."
    )


def _resolve_hipt_params(preprocessing: PreprocessingConfig, aggregator: AggregatorConfig) -> dict[str, object]:
    patch_size = preprocessing.requested_tile_size_px or preprocessing.read_tile_size_px
    region_size = preprocessing.requested_region_size_px or preprocessing.read_region_size_px
    tile_multiple = preprocessing.region_tile_multiple
    if patch_size is None or region_size is None:
        raise ValueError("hierarchical preprocessing must resolve patch and region sizes")
    patch_size = int(patch_size)
    region_size = int(region_size)
    if tile_multiple is None:
        if region_size % patch_size != 0:
            raise ValueError(
                "hierarchical preprocessing requires requested_region_size_px to be divisible by requested_tile_size_px"
            )
        tile_multiple = region_size // patch_size
    tile_multiple = int(tile_multiple)
    if region_size != patch_size * tile_multiple:
        raise ValueError(
            "hierarchical preprocessing requires requested_region_size_px to equal "
            "requested_tile_size_px × region_tile_multiple"
        )
    params = {
        key: value
        for key, value in aggregator.params.items()
        if key not in {"region_size", "patch_size", "tile_multiple"}
    }
    params.update({"region_size": region_size, "patch_size": patch_size})
    return params


def _event_balanced_train_loader(
    dataset,
    collate_fn,
    *,
    events: list[int],
    training: TrainingConfig,
    min_events_per_window: int,
    window_size: int | None = None,
) -> DataLoader:
    """Training loader whose batches each hold >= one event (for batched Cox).

    The Cox partial likelihood is undefined on an event-free batch, so the risk set must
    contain events. ``window_size`` is ``training.batch_size`` in padded mode and
    ``cox_window`` in accumulation mode; the matching ``collate_fn`` decides which.
    """
    from soma.training.survival_sampler import EventBalancedBatchSampler

    sampler = EventBalancedBatchSampler(
        events,
        batch_size=window_size if window_size is not None else training.batch_size,
        min_events_per_window=min_events_per_window,
        seed=training.seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=training.num_workers,
        pin_memory=training.pin_memory,
        persistent_workers=training.persistent_workers and training.num_workers > 0,
    )


@dataclass(frozen=True)
class _SetFoldRecords:
    """Which records each split of a set fold trains or scores on."""

    train: list[Any]  # records (or patient ids on the patient path)
    tune: list[Any]
    tests: dict[str, list[Any]]
    tune_output_ids: tuple[str, ...]
    tune_placeholders: list[str]
    test_output_ids: dict[str, tuple[str, ...]]
    test_placeholders: dict[str, list[str]]
    records_by_id: dict[str, LegacyRecord]
    train_records: list[LegacyRecord]


def _feature_statuses(source) -> dict[str, str] | None:
    """The source's per-sample feature-manifest statuses, or ``None`` when it has no
    manifest (an in-memory, directory or grouped source supplied by the caller)."""
    if not getattr(source, "has_feature_manifest", False):
        return None
    return source.feature_statuses


def _set_fold_records(inputs: RunInputs, fold: int, fold_split: FoldSplit) -> _SetFoldRecords:
    """Resolve the fold's train / tune / test membership against the features on disk.

    Samples the feature manifest records as ``empty`` (no tissue, no bag) leave training
    and are scored with the training baseline; ``tune_is_test``, ``allow_missing_tune``
    and ``evaluation.holdout_test`` reshape the splits as their docs say.
    """
    config, dataset, feature_store = inputs.config, inputs.dataset, inputs.source
    training, evaluation = config.training, inputs.evaluation
    dataset_type = config.dataset_type
    all_test_ids = {sid for ids in fold_split.tests.values() for sid in ids}
    _fp = f"Fold {fold}" if inputs.num_folds > 1 else "Run"
    statuses = _feature_statuses(feature_store)
    if dataset_type == "patient":
        # Patient-level: features are indexed by patient_id; splits are slide-based.
        train_records = [dataset.samples[sid] for sid in fold_split.train]
        tune_records = [dataset.samples[sid] for sid in fold_split.tune]
        test_records_by_split = {
            name: [dataset.samples[sid] for sid in ids] for name, ids in fold_split.tests.items()
        }
        train_ids = _legacy.patient_ids_for_records(train_records)
        tune_ids = _legacy.patient_ids_for_records(tune_records)
        test_ids = {
            name: _legacy.patient_ids_for_records(records) for name, records in test_records_by_split.items()
        }
        raw_train_ids, raw_tune_ids = list(train_ids), list(tune_ids)
        raw_test_ids = {name: list(ids) for name, ids in test_ids.items()}
        split_ids = set(train_ids) | set(tune_ids) | {pid for ids in test_ids.values() for pid in ids}
        if statuses is not None:
            missing = sorted(split_ids - set(statuses))
            if missing:
                raise ValueError(
                    f"Patient feature manifest is missing patient(s) required by fold {fold}: {missing}"
                )
            expected = {pid for pid, status in statuses.items() if status == "success"}
            empty = {pid for pid, status in statuses.items() if status == "empty"}
            unexpected_missing = sorted(expected - set(feature_store.sample_ids))
            if unexpected_missing:
                raise ValueError(
                    "Feature store is missing expected patient embedding(s) for fold "
                    f"{fold}: {unexpected_missing}"
                )
            train_ids = [pid for pid in train_ids if pid in expected]
            tune_ids = [pid for pid in tune_ids if pid in expected]
            test_ids = {name: [pid for pid in ids if pid in expected] for name, ids in test_ids.items()}
            empty_by_split: dict[str, list[str]] | None = {
                "train": [pid for pid in raw_train_ids if pid in empty],
                "tune": [pid for pid in raw_tune_ids if pid in empty],
                **{name: [pid for pid in ids if pid in empty] for name, ids in raw_test_ids.items()},
            }
        else:
            missing = sorted(split_ids - set(feature_store.sample_ids))
            if missing:
                raise ValueError(
                    "Patient feature store is missing patient embedding(s) required by fold "
                    f"{fold}: {missing}"
                )
            empty_by_split = None
    elif statuses is not None:
        split_ids = set(fold_split.train) | set(fold_split.tune) | all_test_ids
        missing = sorted(split_ids - set(statuses))
        if missing:
            raise ValueError(f"Feature manifest is missing sample(s) required by fold {fold}: {missing}")
        expected = {sid for sid, status in statuses.items() if status == "success"}
        empty = {sid for sid, status in statuses.items() if status == "empty"}
        unexpected_missing = sorted(expected - set(feature_store.sample_ids))
        if unexpected_missing:
            raise ValueError(f"Feature store is missing expected sample(s) for fold {fold}: {unexpected_missing}")
        empty_by_split = {
            "train": [sid for sid in fold_split.train if sid in empty],
            "tune": [sid for sid in fold_split.tune if sid in empty],
            **{name: [sid for sid in ids if sid in empty] for name, ids in fold_split.tests.items()},
        }
        train_records = _legacy.records_for_sample_ids(dataset, fold_split.train, expected)
        tune_records = _legacy.records_for_sample_ids(dataset, fold_split.tune, expected)
        test_records_by_split = {
            name: _legacy.records_for_sample_ids(dataset, ids, expected) for name, ids in fold_split.tests.items()
        }
    else:
        # No manifest: every listed sample must have a feature file.
        train_records = [dataset.samples[sid] for sid in fold_split.train]
        tune_records = [dataset.samples[sid] for sid in fold_split.tune]
        test_records_by_split = {
            name: [dataset.samples[sid] for sid in ids] for name, ids in fold_split.tests.items()
        }
        empty_by_split = None

    tune_from_test: str | None = None
    if training.tune_is_test:
        tune_from_test = _resolve_tune_is_test_split(fold_split, _fp)
        logger.warning(
            "%s uses test split '%s' as tune because tune_is_test=True; "
            "checkpoint selection and test reporting use the same samples",
            _fp,
            tune_from_test,
        )
        tune_records = list(test_records_by_split[tune_from_test])
        if dataset_type == "patient":
            tune_ids = list(test_ids[tune_from_test])
            raw_tune_ids = list(raw_test_ids[tune_from_test])
        if empty_by_split is not None:
            empty_by_split["tune"] = list(empty_by_split.get(tune_from_test, []))

    if evaluation.holdout_test:
        # Tune-only model-selection run: no test loader, no test inference; metrics and
        # summary carry tune only. Done after tune_is_test resolved tune from test.
        test_records_by_split = {}
        if dataset_type == "patient":
            test_ids, raw_test_ids = {}, {}
        if empty_by_split is not None:
            empty_by_split = {k: v for k, v in empty_by_split.items() if k in ("train", "tune")}

    patient = dataset_type == "patient"
    train_items = train_ids if patient else train_records
    tune_items = tune_ids if patient else tune_records
    tests_items = test_ids if patient else test_records_by_split
    unit = "patients" if patient else "samples"
    if not train_items:
        raise ValueError(f"{_fp} has no training {unit} with available features")
    fallback_to_train = not tune_items and not (empty_by_split and empty_by_split.get("tune"))
    if fallback_to_train and not training.allow_missing_tune:
        raise ValueError(f"{_fp} has no tuning {unit} with available features")
    if fallback_to_train:
        logger.warning(
            "%s has no tuning %s with available features; "
            "using train split as tune because allow_missing_tune=True",
            _fp,
            unit,
        )
        tune_items = list(train_items)
        tune_records = list(train_records)
        if patient:
            raw_tune_ids = list(raw_train_ids)
    for name, items in tests_items.items():
        split_empty = empty_by_split.get(name, []) if empty_by_split else []
        if not items and not split_empty:
            raise ValueError(f"{_fp} has no {unit} with available features in split '{name}'")
    logger.info(
        _format_fold_summary(
            fold=fold,
            train_count=len(train_items),
            tune_count=len(tune_items),
            tests_counts={name: len(items) for name, items in tests_items.items()},
            empty_sample_ids_by_split=empty_by_split,
        )
    )

    empty_eval = empty_by_split or {}
    if patient:
        # Patient ids score against the patient's first slide in cohort order, as
        # Cohort.collapse does.
        tune_output = tuple(raw_train_ids if fallback_to_train else raw_tune_ids)
        test_output = {name: tuple(raw_test_ids[name]) for name in tests_items}
        records_by_id = dataset.patient_record_map
    else:
        if fallback_to_train:
            tune_output = tuple(fold_split.train)
        elif tune_from_test is not None:
            tune_output = tuple(fold_split.tests[tune_from_test])
        else:
            tune_output = tuple(fold_split.tune)
        test_output = {name: tuple(fold_split.tests[name]) for name in tests_items}
        records_by_id = dataset.samples
    return _SetFoldRecords(
        train=list(train_items),
        tune=list(tune_items),
        tests={name: list(items) for name, items in tests_items.items()},
        tune_output_ids=tune_output,
        tune_placeholders=list(empty_eval.get("train" if fallback_to_train else "tune", [])),
        test_output_ids=test_output,
        test_placeholders={name: list(empty_eval.get(name, [])) for name in tests_items},
        records_by_id=records_by_id,
        train_records=list(train_records),
    )


def build_set_fold(inputs: RunInputs, fold: int, fold_split: FoldSplit) -> FoldSetup:
    """One set-shape fold (slide, patient, tile, hierarchical): bags or embeddings."""
    config, dataset, feature_store = inputs.config, inputs.dataset, inputs.source
    training, evaluation, task = config.training, inputs.evaluation, config.task
    dataset_type, aggregator = config.dataset_type, config.aggregator
    records = _set_fold_records(inputs, fold, fold_split)

    if evaluation.subgroups.columns:
        sample = next(iter(dataset.samples.values()))
        missing = [c for c in evaluation.subgroups.columns if c not in sample.metadata]
        if missing:
            raise ValueError(
                f"Subgroup column(s) not found in dataset metadata: {missing}. "
                f"Available: {sorted(sample.metadata)}"
            )

    task_cls = task_registry.resolve(task.name)
    # Validate survival columns before the head's infer hook reads them, so a
    # missing/malformed column raises a clear message instead of a cryptic KeyError.
    if task_cls.task_family == "survival":
        from soma.tasks.survival import resolve_survival_task, validate_survival_dataset

        task_cls, survival_loss = resolve_survival_task(task.name, task.params)
        validate_survival_dataset(
            dataset.records, dataset_type, loss=survival_loss, num_bins=task.params.get("num_bins")
        )
    # The cohort decides what the config leaves unset (class count, label map, survival
    # bins); explicit task.params win.
    head_name = _component_name(task_registry, task.name, task_cls)
    inferred = task_registry.infer(head_name, inputs.cohort, task.params)
    task_params = {**inferred, **task.params, "metrics": evaluation.metrics}
    task_params.pop("loss", None)
    task_family = task_cls.task_family

    # A transform a path cannot honor is refused rather than silently dropped.
    validate_feature_adaptor_compatibility(
        config.normalization,
        config.projection,
        dataset_type=dataset_type,
        is_hierarchical=feature_store.rank == 3,
    )
    feature_dim = feature_store.feature_dim
    encoder_identity = _encoder_identity(config)
    fits_adaptor = False
    aggregator_spec = None
    head_cls = task_cls
    if dataset_type == "patient":
        if aggregator is not None:
            raise ValueError("aggregator must be None for dataset_type='patient'")
        kind = "embedding"
    elif dataset_type == "tile" or feature_store.rank == 1:
        if feature_store.rank == 1 and aggregator is not None:
            raise ValueError("aggregator must be None for slide-level features")
        kind, fits_adaptor = "embedding", True
    elif feature_store.rank == 3:
        if aggregator is None:
            raise ValueError("aggregator must be provided for hierarchical features")
        if aggregator.name != "hipt":
            raise ValueError("hierarchical features require the hipt aggregator")
        kind = "mil"
        aggregator_spec = {"name": aggregator.name, "params": _resolve_hipt_params(inputs.preprocessing, aggregator)}
    else:
        if aggregator is None:
            raise ValueError("aggregator must be provided for tile-level features")
        kind, fits_adaptor = "mil", True
        aggregator_spec = {"name": aggregator.name, "params": dict(aggregator.params)}
        head_cls = _bag_head_class(task_cls, task_name=task.name, aggregator_name=aggregator.name)
    model_spec = {
        "kind": kind,
        "feature_dim": feature_dim,
        "head": {"name": _component_name(task_registry, task.name, head_cls), "params": task_params},
        "aggregator": aggregator_spec,
        "normalization": _section(config.normalization),
        "projection": _section(config.projection),
        "encoder_identity": encoder_identity,
    }

    def build_fold_learner(adaptor: FeatureAdaptor | None) -> Learner:
        return build_learner(build_set_model(**model_spec, adaptor=adaptor), training, shape="mil")

    def build_loaders(learner: Learner, fold_dir: Path) -> FoldLoaders:
        return _set_loaders(inputs, records, learner.head, kind=kind)

    def fit_baseline(head) -> Baseline:
        # Samples without features are scored with the training split's baseline (class
        # prior / mean / constant risk), flagged as placeholders.
        return Baseline.fit(
            targets_frame(records.train_records),
            task_family=task_family,
            classes=head.classes,
            target_keys=head.target_keys,
        )

    def metrics(head) -> list[Any]:
        # The head resolved its metrics for its own outputs (a multi-output regression
        # reports mean_pearson, not the family's single-value defaults).
        return list(getattr(head, "metrics", None) or resolve_metrics(task_family, evaluation.metrics))

    def preprocessing(head, threshold) -> dict[str, Any]:
        return {
            "shape": "set",
            "dataset_type": dataset_type,
            "task_family": task_family,
            "target_schema": {key: getattr(t, "__name__", str(t)) for key, t in head.target_schema.items()},
            "classes": _jsonable(head.classes),
            "metrics": _jsonable(metrics(head)),
            "feature_dim": int(feature_dim),
        }

    return FoldSetup(
        fold=fold,
        num_folds=inputs.num_folds,
        seed=training.seed,
        fit_adaptor=(
            _adaptor_step(
                config,
                support=_SupportFeatures(feature_store, records.train_records),
                num_support=len(records.train_records),
                feature_dim=feature_dim,
                encoder_identity=encoder_identity,
            )
            if fits_adaptor
            else _no_adaptor
        ),
        build_learner=build_fold_learner,
        build_loaders=build_loaders,
        fit_baseline=fit_baseline,
        callbacks=_fit_callbacks(training, fold=fold, num_folds=inputs.num_folds),
        trainer=_trainer_kwargs(training),
        metrics=metrics,
        preprocessing=preprocessing,
        provenance={"run_id": inputs.run_id, "feature_source": inputs.source_identity},
    )


def _set_loaders(inputs: RunInputs, records: _SetFoldRecords, head, *, kind: str) -> FoldLoaders:
    """Step 4 for the set shape: the train / tune / test loaders and the scored splits."""
    config, feature_store, training = inputs.config, inputs.source, inputs.config.training
    loader_kwargs = _loader_kwargs(training)
    target_fn = head.extract_targets
    patient = config.dataset_type == "patient"
    if patient:
        record_map = inputs.dataset.patient_record_map
        collate = functools.partial(patient_collate_fn, target_dtypes=head.target_dtypes)

        def dataset(items):
            return PatientDataset(items, record_map, feature_store, target_fn)

        def events(items) -> list[int]:
            return [int(record_map[pid].targets["event"]) for pid in items]

    else:
        if kind == "embedding":
            dataset_cls, collate_fn = SampleDataset, sample_collate_fn
        elif feature_store.rank == 3:
            dataset_cls, collate_fn = HierarchicalBagDataset, hierarchical_bag_collate_fn
        else:
            dataset_cls, collate_fn = BagDataset, bag_collate_fn
        collate = functools.partial(collate_fn, target_dtypes=head.target_dtypes)

        def dataset(items):
            return dataset_cls(items, feature_store, target_fn)

        def events(items) -> list[int]:
            return [int(record.targets["event"]) for record in items]

    def loader(items, *, shuffle: bool) -> DataLoader:
        return DataLoader(dataset(items), shuffle=shuffle, collate_fn=collate, **loader_kwargs)

    train_loader = loader(records.train, shuffle=True)
    tune_loader = loader(records.tune, shuffle=False)
    test_loaders = {name: loader(items, shuffle=False) for name, items in records.tests.items()}
    if getattr(head, "needs_event_balanced_batches", False):
        # Cox: batches each hold an event. Tune/test keep the ordinary loaders.
        window_size = None
        train_collate = collate
        if not patient and getattr(head, "accumulates_predictions", False):
            # Accumulation mode: windows of cox_window un-padded bags.
            train_collate = functools.partial(cox_window_collate, target_dtypes=head.target_dtypes)
            window_size = head.accumulation_window
        train_loader = _event_balanced_train_loader(
            dataset(records.train),
            train_collate,
            events=events(records.train),
            training=training,
            min_events_per_window=head.min_events_per_window,
            window_size=window_size,
        )

    def targets(output_ids: tuple[str, ...]) -> pd.DataFrame:
        frame = targets_frame([records.records_by_id[sid] for sid in output_ids])
        frame.index = pd.Index([str(sid) for sid in output_ids], name="sample_id", dtype=object)
        return frame

    subgroups = tuple(inputs.evaluation.subgroups.columns)
    splits = [
        SplitPlan(
            name="tune",
            loader=tune_loader,
            targets=targets(records.tune_output_ids),
            placeholder_ids=tuple(records.tune_placeholders),
            write=False,
        )
    ]
    for name, test_loader in test_loaders.items():
        splits.append(
            SplitPlan(
                name=name,
                loader=test_loader,
                targets=targets(records.test_output_ids[name]),
                placeholder_ids=tuple(records.test_placeholders[name]),
                subgroups=subgroups,
            )
        )
    return FoldLoaders(train=train_loader, tune=tune_loader, splits=tuple(splits))


# ---------------------------------------------------------------------------
# Grid shape
# ---------------------------------------------------------------------------


def _segmentation_head_spec(inputs: RunInputs, geometry) -> dict[str, Any]:
    """The registered segmentation head bound to ``geometry``, as a ``{name, params}``
    section (a user subclass with its own loss keeps its class)."""
    config, preprocessing = inputs.config, inputs.preprocessing
    task, masks = config.task, config.preprocessing.masks
    head_cls = task_registry.resolve(task.name)
    num_classes, class_names, label_remap = resolve_class_scheme(task.params, annotation_rasters=masks is not None)
    seg_params = {key: value for key, value in task.params.items() if key not in CLASS_SCHEME_KEYS}
    mask_spacing_um = preprocessing.requested_spacing_um if preprocessing is not None else None
    params = {
        "num_classes": num_classes,
        "geometry": geometry,
        "metrics": inputs.evaluation.metrics,
        "spacing_um": float(mask_spacing_um) if mask_spacing_um is not None else None,
        "spacing_policy": preprocessing.spacing_policy if preprocessing is not None else "strict",
        "backend": preprocessing.mask_backend if preprocessing is not None else "auto",
        "image_backend": preprocessing.backend if preprocessing is not None else "auto",
        "tolerance": float(preprocessing.tolerance) if preprocessing is not None else 0.05,
        "label_remap": label_remap,
        "pixel_mapping": masks.pixel_mapping if masks is not None else None,
        "class_names": class_names if len(class_names) == num_classes else None,
        **seg_params,
    }
    return {"name": _component_name(task_registry, task.name, head_cls), "params": params}


def _build_segmentation_head(inputs: RunInputs, geometry):
    spec = _segmentation_head_spec(inputs, geometry)
    return task_registry.resolve(spec["name"])(**spec["params"])


def _resolve_segmentation_roi_population(
    *,
    cache_root: str | Path,
    records: list[LegacyRecord],
    head,
    label_maps,
    source_fingerprint_cache: dict[str, dict[str, object]] | None,
    resolved_backend_cache: dict[tuple[str, str, bool], str] | None,
):
    """Resolve one ROI population with the run's shared cache policy."""
    from soma.learn.roi_population import resolve_segmentation_roi_population

    return resolve_segmentation_roi_population(
        cache_root,
        records,
        lambda record: {"mask": label_maps.load(record.sample_id)},
        num_classes=head.num_classes,
        target_identity=head.target_identity,
        workers=max(1, min(8, os.cpu_count() or 1)),
        source_fingerprint_cache=source_fingerprint_cache,
        resolved_backend_cache=resolved_backend_cache,
    )


def _segmentation_roi_batch_sampler(records, population, *, training: TrainingConfig, fold: int):
    """The explicit cached-grid training-batch contract, when requested."""
    from soma.learn.samplers import sampler_registry

    sample_ids = [record.sample_id for record in records]
    params: dict[str, Any] = {"draws_per_epoch": training.roi_draws_per_epoch}
    if training.class_request_ratios is not None:
        params["class_request_ratios"] = training.class_request_ratios
    strategy = training.roi_batch_sampling
    return sampler_registry.get(strategy)(
        sample_ids=sample_ids,
        class_pixel_counts=population.subset(sample_ids).class_pixel_counts,
        batch_size=training.batch_size,
        gradient_accumulation=training.gradient_accumulation,
        seed=training.seed + fold,
        **sampler_registry.validate(strategy, params),
    )


def _grid_augment(augmentation: dict | None, head, geometry):
    """Build the configured ``{name, params}`` augmentation, injecting the run's
    ``ignore_index`` and ``target_size`` where the augmentation accepts them."""
    if augmentation is None:
        return None
    from soma.learn import augment_registry

    cls, params, _ = resolve_leaf(augmentation, augment_registry)
    try:
        accepted = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        accepted = {}
    injected = {
        "ignore_index": getattr(head, "ignore_index", None),
        "target_size": tuple(int(v) for v in geometry.target_size),
    }
    for key, value in injected.items():
        if key in accepted and key not in params and value is not None:
            params[key] = value
    return cls(**params)


def _uniform_spacing(source, sample_ids: Sequence[str]) -> float:
    """The one µm/px every sample of a detection run is read at."""
    spacings = {round(float(source.spacing(sid)), 9) for sid in sample_ids}
    if len(spacings) != 1:
        raise ValueError(
            "Detection requires one effective spacing across the run (its distances are in "
            f"µm); found {sorted(spacings)}."
        )
    return float(next(iter(spacings)))


def build_grid_fold(inputs: RunInputs, fold: int, fold_split: FoldSplit) -> FoldSetup:
    """One grid-shape fold (segmentation or detection) over cached grids or live pixels.

    The source is a :class:`~soma.data.GridSource` (cached grids) or, with the run's
    frozen encoder stage, an :class:`~soma.data.ImageSource` re-encoded every step, where
    ``augmentation`` augments the train split. Detection sweeps its per-class threshold
    on tune (without a tune split it scores at the head's ``validation_threshold``).
    """
    config, feature_store, stage = inputs.config, inputs.source, inputs.encoder_stage
    training, evaluation, decoder = config.training, inputs.evaluation, config.decoder
    dataset_type = config.dataset_type
    if decoder is None:
        raise ValueError(f"dataset_type={dataset_type!r} requires a decoder configuration")
    _fp = f"Fold {fold}" if inputs.num_folds > 1 else "Run"
    live = stage is not None
    validate_feature_adaptor_compatibility(
        config.normalization,
        config.projection,
        feature_mode="live" if live else "cached",
        has_composite=isinstance(feature_store, CompositeDenseFeatureStore),
    )
    fold_plan = plan_dense_fold(
        dataset=inputs.dataset,
        fold_split=fold_split,
        training=training,
        fold_label=_fp,
        logger=logger,
        holdout_test=evaluation.holdout_test,
    )
    train_records, tune_records = fold_plan.train_records, fold_plan.tune_records
    test_records_by_split, all_records = fold_plan.test_records_by_split, fold_plan.all_records
    all_ids = [record.sample_id for record in all_records]
    require_coverage(feature_store, all_ids, what="pixels" if live else "features")

    # The head's geometry is bound here: the encoder stage's frame (live) or the cached
    # grids' (the loaders check every sample shares it), anchored at level 0 by the source.
    reference = feature_store.geometry(train_records[0].sample_id)
    if live:
        geometry = GridGeometry.from_dense(
            stage.geometry,
            origin_level0=reference.origin_level0,
            level0_px_per_token_px=reference.level0_px_per_token_px,
        )
        feature_dim = int(stage.feature_dim)
    else:
        geometry = reference
        feature_dim = int(feature_store.feature_dim)
        if dataset_type == "segmentation":
            _require_configured_spacing(inputs, all_records)

    if dataset_type == "segmentation":
        head_spec = _segmentation_head_spec(inputs, geometry)
        point_source = None
    else:
        head_spec, point_source = _detection_head_spec(inputs, geometry, all_records, all_ids)
    encoder_identity = _encoder_identity(config)
    model_spec = {
        "feature_dim": feature_dim,
        "decoder": {"name": decoder.name, "params": dict(decoder.params)},
        "head": head_spec,
        "normalization": _section(config.normalization),
        "projection": _section(config.projection),
        "encoder_identity": encoder_identity,
        "encoder": live_encoder_spec(config) if live else None,
    }

    def fit_adaptor(fold_dir: Path) -> FeatureAdaptor | None:
        if live:  # refused above for live: the stage's grids are never seen whole
            return None
        return _adaptor_step(
            config,
            support=_SupportGrids(feature_store, train_records),
            num_support=len(train_records),
            feature_dim=feature_dim,
            encoder_identity=encoder_identity,
        )(fold_dir)

    def build_fold_learner(adaptor: FeatureAdaptor | None) -> Learner:
        model = build_grid_model(**model_spec, adaptor=adaptor, stage=stage)
        if model.consumes_image and tuple(geometry.encoded_size) != tuple(geometry.target_size):
            # An image-prior decoder emits logits at the mask resolution; the head's
            # resize-to-encoded + crop must then be the identity.
            raise ValueError(
                f"decoder '{decoder.name}' consumes the image and emits logits at the mask "
                f"size {tuple(geometry.target_size)}, but the encoder input is padded to "
                f"{tuple(geometry.encoded_size)}; image-prior decoders need an unpadded tile."
            )
        return build_learner(model, training, shape="dense")

    def build_loaders(learner: Learner, fold_dir: Path) -> FoldLoaders:
        head, consumes_image = learner.head, learner.model.consumes_image
        if dataset_type == "segmentation":
            targets = _legacy.segmentation_label_maps(head, all_records)
        else:
            targets = point_source
        images = None
        if consumes_image and not live:
            images = _legacy.flat_tile_images(all_records, size=geometry.target_size)
            if images is None:
                raise ValueError(
                    f"decoder '{decoder.name}' reads each tile's pixels; slide-manifest ROIs "
                    "carry no flat tile to read"
                )
        sampler = None
        if training.roi_batch_sampling is not None:
            if live or dataset_type != "segmentation":
                raise ValueError("Explicit ROI batch sampling is a cached segmentation training contract.")
            population = inputs.roi_population().subset(all_ids)
            (fold_dir / "segmentation_roi_population.json").write_text(
                json.dumps(population.provenance(), indent=2, sort_keys=True), encoding="utf-8"
            )
            sampler = _segmentation_roi_batch_sampler(train_records, population, training=training, fold=fold)

        def write_sampler_audit(directory: Path) -> None:
            (directory / "roi_batch_sampling.json").write_text(
                json.dumps(sampler.audit(), indent=2, sort_keys=True), encoding="utf-8"
            )

        loader_kwargs = _loader_kwargs(training)
        fold_loaders = grid_loaders(
            feature_store,
            head=head,
            train=[record.sample_id for record in train_records],
            tune=[record.sample_id for record in tune_records],
            tests={name: [r.sample_id for r in recs] for name, recs in test_records_by_split.items()},
            batch_size=loader_kwargs.pop("batch_size"),
            num_workers=loader_kwargs.pop("num_workers"),
            targets=targets,
            augment=_grid_augment(config.augmentation, head, geometry) if live else None,
            # An image-prior decoder reads the raw pixels, so the stage preprocesses them.
            preprocess=stage.preprocessor() if live and not consumes_image else None,
            sampler=sampler,
            images=images,
            loader_kwargs=loader_kwargs,
        )
        logger.info(
            _format_fold_summary(
                fold=fold,
                train_count=len(train_records),
                tune_count=len(tune_records),
                tests_counts={name: len(recs) for name, recs in test_records_by_split.items()},
            )
        )
        overlays = images or _legacy.overlay_images(
            all_records, size=geometry.target_size, preprocessing=inputs.preprocessing
        )

        def plan(name: str, loader, records) -> SplitPlan:
            return SplitPlan(
                name=name,
                loader=loader,
                targets=targets.subset([record.sample_id for record in records]),
                callbacks=tuple(_grid_callbacks(inputs, fold_dir, name, overlays)),
            )

        splits = [plan("tune", fold_loaders.tune, tune_records)]
        splits += [plan(name, fold_loaders.tests[name], recs) for name, recs in test_records_by_split.items()]
        return FoldLoaders(
            train=fold_loaders.train,
            tune=fold_loaders.tune,
            splits=tuple(splits),
            after_fit=write_sampler_audit if sampler is not None else None,
        )

    def metrics(head) -> list[Any]:
        names: list[Any] = list(resolve_metrics(head.task_family, evaluation.metrics))
        if dataset_type == "detection":
            return [{"name": name, "params": {"matching": head.matching}} for name in names]
        return names

    def tune_step(predictions, split: SplitPlan, head):
        # Without a tune split, train stands in and a sweep there would pick an in-sample
        # cut: score at the head's fixed validation threshold instead.
        if fold_plan.tune_from_train:
            return head.validation_threshold
        from soma.evaluation import sweep_threshold

        threshold, _ = sweep_threshold(predictions, split.targets, per_class=True, matching=head.matching)
        return threshold

    def preprocessing(head, threshold) -> dict[str, Any]:
        metadata = {
            "shape": "grid",
            "dataset_type": dataset_type,
            "task_family": head.task_family,
            "num_classes": int(head.num_classes),
            "class_names": _jsonable(head.class_names),
            "geometry": _jsonable(dataclasses.asdict(geometry)),
            "tile_size": _jsonable(geometry.target_size),
            "spacing_um": head_spec["params"].get("spacing_um"),
            "feature_mode": "live" if live else "cached",
            "metrics": _jsonable(metrics(head)),
        }
        if dataset_type == "segmentation":
            metadata["ignore_index"] = int(head.ignore_index)
            metadata["label_remap"] = _jsonable(head.label_remap)
        else:
            metadata["match_distance"] = float(head.match_distance)
            metadata["matching"] = head.matching
            metadata["min_score"] = float(head.min_score)
            # Keyed by class name so ``PointPredictions.with_threshold`` applies it as recorded.
            metadata["score_threshold"] = (
                {head.class_names[int(k)]: float(v) for k, v in threshold.items()}
                if isinstance(threshold, dict)
                else float(threshold)
            )
        return metadata

    return FoldSetup(
        fold=fold,
        num_folds=inputs.num_folds,
        seed=training.seed,
        fit_adaptor=fit_adaptor,
        build_learner=build_fold_learner,
        build_loaders=build_loaders,
        callbacks=_fit_callbacks(training, fold=fold, num_folds=inputs.num_folds),
        trainer=_trainer_kwargs(training),
        metrics=metrics,
        preprocessing=preprocessing,
        provenance={"run_id": inputs.run_id, "feature_source": inputs.source_identity},
        tune_step=tune_step if dataset_type == "detection" else None,
    )


def _require_configured_spacing(inputs: RunInputs, records) -> None:
    """Cached segmentation grids must be extracted at the spacing the config reads masks
    at (per record under ``native_if_coarser``); see :func:`soma.dense.require_declared_spacing`."""
    from soma.dense import require_declared_spacing

    preprocessing = inputs.preprocessing
    requested = preprocessing.requested_spacing_um if preprocessing is not None else None
    require_declared_spacing(
        inputs.source,
        {
            record.sample_id: (
                None if requested is None else preprocessing.effective_spacing_um(record.spacing_at_level_0)
            )
            for record in records
        },
    )


def _detection_head_spec(inputs: RunInputs, geometry, records, sample_ids):
    """``({name, params}, PointSource)`` of a detection run: classes from the task's class
    scheme, distances converted at the run's one spacing."""
    task = inputs.config.task
    num_classes, class_names, point_source = _legacy.detection_points(task, records)
    params = {k: v for k, v in task.params.items() if k not in ("num_classes", "classes", "drop")}
    if "match_distance" not in params:
        raise ValueError(
            "dataset_type='detection' requires task.params.match_distance (the F1 "
            "matching distance δ, in µm)."
        )
    head_cls = task_registry.resolve(task.name)
    spec = {
        "name": _component_name(task_registry, task.name, head_cls),
        "params": {
            "num_classes": num_classes,
            "geometry": geometry,
            "spacing_um": _uniform_spacing(inputs.source, sample_ids),
            "metrics": inputs.evaluation.metrics,
            "class_names": class_names,
            **params,
        },
    }
    return spec, point_source


def _grid_callbacks(inputs: RunInputs, fold_dir: Path, split_name: str, overlay_images) -> list[Any]:
    """The predict callbacks writing a grid split's side artefacts under ``fold_dir``."""
    from soma import evaluation as verbs

    evaluation, dataset_type = inputs.evaluation, inputs.config.dataset_type
    out: list[Any] = []
    overlays = (
        evaluation.save_segmentation_overlays if dataset_type == "segmentation" else evaluation.save_detection_overlays
    )
    if overlays:
        out.append(verbs.SaveOverlays(fold_dir / "overlays" / split_name, images=overlay_images))
    if dataset_type == "segmentation" and evaluation.save_segmentation_probabilities:
        out.append(verbs.SaveProbabilities(fold_dir / "probabilities" / split_name))
    if dataset_type == "detection" and evaluation.save_detection_heatmaps:
        out.append(verbs.SaveHeatmaps(fold_dir / "heatmaps" / split_name))
    return out


# ---------------------------------------------------------------------------
# The run's feature source: supplied, extracted, live or composite
# ---------------------------------------------------------------------------


def extraction_shape(dataset_type: str) -> str:
    """The feature shape a ``dataset_type`` trains on: ``grid`` for dense tasks, else ``set``."""
    return "grid" if dataset_type in {"segmentation", "detection"} else "set"


def extraction_unit(config: PipelineConfig) -> str:
    """What one manifest row is to extraction: a whole slide or a pre-cropped image."""
    if config.dataset_type in {"slide", "patient"}:
        return "slide"
    if config.dataset_type in {"segmentation", "detection"} and config.preprocessing.masks is not None:
        return "slide"
    return "tile"


def resolve_task_class(config: PipelineConfig):
    """The registered head class a config trains (survival resolves its loss variant)."""
    if config.task is None:
        return None
    task_cls = task_registry.resolve(config.task.name)
    if task_cls.task_family == "survival":
        from soma.tasks.survival import resolve_survival_task

        task_cls, _ = resolve_survival_task(config.task.name, config.task.params)
    return task_cls


def load_pipeline_data(config: PipelineConfig) -> tuple[Cohort, ImageManifest, AnnotationManifest | None]:
    """Read the cohort and manifests a config names, validated against its task head.

    The head's ``resolve_target_schema(task.params)`` says which target keys the cohort
    must carry (``label``, ``time``/``event``, ``value`` or ``value_<k>`` ...);
    ``task.targets`` maps a key to a differently named CSV column.
    ``training.tune_is_test`` reuses each fold's tune split for test reporting via
    :meth:`Cohort.with_test_from_tune`.
    """
    task_cls = resolve_task_class(config)
    schema = task_cls.resolve_target_schema(config.task.params) if task_cls is not None else {}
    if config.representation is not None and config.task is None:
        schema = {"label": object}
    renames = dict(config.task.targets) if config.task is not None and config.task.targets else {}
    unknown = sorted(set(renames) - set(schema))
    if unknown:
        raise ValueError(
            f"task.targets names key(s) {unknown} the task head does not declare; "
            f"its target keys are {sorted(schema)}."
        )
    targets = {key: renames.get(key, key) for key in schema}
    cohort = Cohort.from_csv(
        config.dataset_csv,
        config.splits_csv,
        targets=targets,
        dtypes=schema,
        unit="patient_id" if config.dataset_type == "patient" else None,
        allow_missing_test=config.training.tune_is_test,
    )
    if config.training.tune_is_test:
        cohort = cohort.with_test_from_tune()
    image_manifest = ImageManifest.from_csv(config.dataset_csv)
    annotation_manifest = None
    if config.dataset_type in {"segmentation", "detection"}:
        masks = config.preprocessing.masks
        annotation_manifest = AnnotationManifest.from_csv(
            config.dataset_csv,
            pixel_mapping=masks.pixel_mapping if masks is not None else None,
        )
        if config.dataset_type == "segmentation" and not annotation_manifest.has_label_masks:
            raise ValueError("dataset_type='segmentation' needs a label_mask_path column.")
        if config.dataset_type == "detection" and not annotation_manifest.has_points:
            raise ValueError("dataset_type='detection' needs a points_path column.")
    return cohort, image_manifest, annotation_manifest


@dataclass(frozen=True)
class SourceContext:
    """The run's feature source and the records it trains against.

    Extraction may change the records (annotation sampling turns slides into ROIs), so
    ``dataset`` / ``splits`` are the ones the source serves. On the live grid path
    ``source`` is the pixels' image source and ``encoder_stage`` the frozen encoder.
    """

    source: Any
    dataset: LegacySamples
    splits: LegacyFolds
    encoder_stage: Any = None


def attach_source(
    config: PipelineConfig,
    *,
    source: Any,
    dataset: LegacySamples,
    splits: LegacyFolds,
    image_manifest: ImageManifest,
    run_dir: Path,
) -> SourceContext:
    """Attach the supplied ``source``, or extract one from the config.

    A supplied image source on the live grid path gets the run's frozen encoder stage; a
    supplied :class:`~soma.data.GridSource` (cached grids) takes the cached path, so it
    cannot be augmented. Without a source: the live grid path builds its frozen encoder
    stage and an image source; a composite extracts every member; otherwise the
    configured encoder extracts (or reuses from the feature cache) set or grid features.
    """
    is_dense = extraction_shape(config.dataset_type) == "grid"
    annotation_sampled = config.dataset_type == "segmentation" and config.preprocessing.masks is not None
    if annotation_sampled and source is not None:
        raise ValueError(
            "a supplied source is not supported for annotation-sampled segmentation; "
            "FeatureExtractor owns the effective ROI dataset and cache identity."
        )
    if annotation_sampled and config.feature_mode == "live":
        raise NotImplementedError(
            "Live extraction over annotation-sampled segmentation ROIs is outside "
            "the persistent FeatureExtractor contract."
        )
    if annotation_sampled and config.composite is not None:
        raise NotImplementedError(
            "Multi-encoder extraction over annotation-sampled segmentation ROIs is not implemented."
        )
    live = is_dense and config.feature_mode == "live"
    if source is not None:
        # A supplied image source under ``feature_mode: live`` is re-encoded by the
        # run's frozen encoder stage, like the one built below; supplied grids are cached.
        cached_grids = isinstance(source, GridSource)
        if live and cached_grids and config.augmentation is not None:
            raise ValueError(
                "augmentation needs pixels re-encoded by an encoder stage, but the supplied "
                "source is cached grids (frozen features); supply an ImageSource or clear "
                "the augmentation section."
            )
        stage = live_encoder_for(config) if live and not cached_grids else None
        return SourceContext(source=source, dataset=dataset, splits=splits, encoder_stage=stage)
    if live:
        preprocessing = resolve_pipeline_preprocessing(config)
        images = TileImageSource.from_manifest(
            image_manifest,
            size=int(preprocessing.requested_tile_size_px),
            spacing_um=float(preprocessing.requested_spacing_um),
            spacing_policy=preprocessing.spacing_policy,
            backend=preprocessing.backend,
            tolerance=float(preprocessing.tolerance),
        )
        return SourceContext(source=images, dataset=dataset, splits=splits, encoder_stage=live_encoder_for(config))
    if is_dense and config.composite is not None:
        from soma.dense import DenseSourceProvenance

        store = build_composite_dense_store(config, dataset, output_root=run_dir / "features")
        store.provenance = DenseSourceProvenance(kind="composite_dense_cache", dataset_csv=config.dataset_csv)
        return SourceContext(source=store, dataset=dataset, splits=splits)
    if config.encoder is None:
        raise ValueError("PipelineConfig.encoder is required when no source is supplied.")
    cache_config = replace(
        config.cache,
        root_dir=resolve_cache_root(config.cache, feature_dir=run_dir, output_root=config.output_root),
    )
    extraction = FeatureExtractor(
        dataset,
        config.encoder,
        resolve_pipeline_preprocessing(config),
        shape=extraction_shape(config.dataset_type),
        unit=extraction_unit(config),
        output_root=run_dir,
        execution=config.execution,
        cache=cache_config,
    ).extract()
    return SourceContext(
        source=extraction.source,
        dataset=extraction.dataset,
        splits=splits.project(extraction.dataset),
    )


def live_encoder_spec(config: PipelineConfig) -> dict[str, Any]:
    """The :func:`soma.extract.live_encoder` arguments a ``feature_mode='live'`` config
    describes: what a live bundle records to rebuild its frozen encoder stage."""
    from soma.encoders.validation import resolve_encoder_precision

    if config.encoder is None:
        raise ValueError("PipelineConfig.encoder is required for feature_mode='live'.")
    preprocessing = resolve_pipeline_preprocessing(config)
    if preprocessing.requested_tile_size_px is None:
        raise ValueError(
            "feature_mode='live' requires preprocessing.requested_tile_size_px (the "
            "supervision tile size)."
        )
    if preprocessing.requested_spacing_um is None:
        raise ValueError(
            "feature_mode='live' requires a spacing: set preprocessing.requested_spacing_um "
            "or use an encoder that advertises a single supported_spacing_um."
        )
    return {
        "encoder": config.encoder.name,
        "target_size": int(preprocessing.requested_tile_size_px),
        "spacing_um": float(preprocessing.requested_spacing_um),
        "window": preprocessing.dense_window_size,
        "overlap": float(preprocessing.dense_window_overlap),
        "feature_kind": preprocessing.feature_kind or "patch_features",
        "precision": resolve_encoder_precision(config.encoder, encoder_name=config.encoder.name),
        "output_variant": config.encoder.output_variant,
        "allow_non_recommended_settings": config.encoder.allow_non_recommended_settings,
        "tolerance": float(preprocessing.tolerance),
        "backend": preprocessing.backend,
        "attention_blocks": tuple(preprocessing.attention.blocks),
        "attention_include_registers": bool(preprocessing.attention.include_registers),
    }


def live_encoder_for(config: PipelineConfig):
    """The frozen encoder stage a ``feature_mode='live'`` config re-encodes with.

    Built through :func:`soma.extract.live_encoder` from :func:`live_encoder_spec`, once
    per run and shared by every fold. Whole-image inference over a finished run loads
    its bundles instead (:meth:`soma.dense.predict.SlidingWindowSegmentationPredictor.from_bundles`).
    """
    from soma import extract

    return extract.live_encoder(**live_encoder_spec(config), num_gpus=config.execution.num_gpus)


def log_cuda_memory(tag: str) -> None:
    """Log the torch CUDA allocator state (nvidia-smi "used" minus this is framework overhead)."""
    if not torch.cuda.is_available():
        return
    gib = 1024**3
    logger.info(
        "CUDA memory [%s]: allocated=%.2f GiB  reserved=%.2f GiB  max_reserved=%.2f GiB",
        tag,
        torch.cuda.memory_allocated() / gib,
        torch.cuda.memory_reserved() / gib,
        torch.cuda.max_memory_reserved() / gib,
    )
    # Reset the peak so a later read reflects the decoder-training phase only.
    torch.cuda.reset_peak_memory_stats()


def composite_member_extraction_spec(
    member: EncoderMemberConfig, preprocessing: PreprocessingConfig
) -> tuple[EncoderConfig, PreprocessingConfig]:
    """The ``(EncoderConfig, PreprocessingConfig)`` one composite member extracts with.

    Mirrors the single-encoder path exactly, so a member's dense cache key is identical to
    the key the same encoder produces in a single-encoder run (composite reuse of
    already-extracted per-encoder caches depends on this). The per-member sliding window
    falls back to the run's shared window when the member leaves it unset.
    """
    from soma.encoders.validation import resolve_preprocessing_config

    member_encoder = EncoderConfig(
        name=member.name,
        precision=member.precision,
        batch_size=member.batch_size,
        adaptive_batching=member.adaptive_batching,
        output_variant=member.output_variant,
        allow_non_recommended_settings=member.allow_non_recommended_settings,
    )
    member_prep = replace(
        resolve_preprocessing_config(member_encoder, preprocessing),
        feature_kind=member.feature_kind,
        attention=member.attention,
        dense_window_size=(
            member.dense_window_size if member.dense_window_size is not None else preprocessing.dense_window_size
        ),
        dense_window_overlap=(
            member.dense_window_overlap
            if member.dense_window_overlap is not None
            else preprocessing.dense_window_overlap
        ),
    )
    return member_encoder, member_prep


def build_composite_dense_store(config: PipelineConfig, dataset: LegacySamples, *, output_root: Path):
    """Extract every composite member into its own cache; return the concat view.

    Public so that out-of-core consumers (a project protocol's rescore step) rebuild the
    exact store a composite run trained on. Members share the run's spacing and
    supervision ``target_size``; their token grids are combined at load time by
    :class:`CompositeDenseFeatureStore` per ``concat_resolution`` / ``concat_grid_size``.
    """
    from soma.dense_extraction import _DenseImageExtractor

    composite = config.composite
    if composite is None:
        raise ValueError("build_composite_dense_store requires config.composite.")
    preprocessing = resolve_pipeline_preprocessing(config)
    target_size = preprocessing.requested_tile_size_px
    if target_size is None:
        raise ValueError(
            "multi-encoder extraction requires "
            "preprocessing.requested_tile_size_px (the mask/tile supervision size)."
        )
    if preprocessing.requested_spacing_um is None:
        raise ValueError(
            "multi-encoder extraction requires an explicit or auto-resolved "
            "preprocessing.requested_spacing_um."
        )
    cache_config = replace(
        config.cache,
        root_dir=resolve_cache_root(config.cache, feature_dir=output_root, output_root=config.output_root),
    )
    member_stores = []
    try:
        for index, member in enumerate(composite.encoders):
            member_encoder, member_prep = composite_member_extraction_spec(member, preprocessing)
            extractor = _DenseImageExtractor(
                dataset,
                member_encoder,
                target_size=int(target_size),
                spacing_um=float(preprocessing.requested_spacing_um),
                backend=preprocessing.backend,
                tolerance=float(preprocessing.tolerance),
                window_size=member_prep.dense_window_size,
                overlap=float(member_prep.dense_window_overlap),
                execution=config.execution,
                cache=cache_config,
                preprocessing=member_prep,
            )
            member_dir = Path(output_root) / f"member_{index}_{member.name}"
            member_stores.append(extractor.run(feature_dir=member_dir))
    finally:
        _release_parent_cuda_state()
        log_cuda_memory("after composite dense extraction release")
    return CompositeDenseFeatureStore(
        member_stores,
        concat_resolution=composite.concat_resolution or "target",
        concat_grid_size=composite.concat_grid_size,
        member_norms=[m.member_norm or "none" for m in composite.encoders],
    )
