"""Plan → objects: the run's feature source and, per fold, the :class:`FoldSetup`.

Everything a :class:`~soma.plan.Plan` decides lives here, so
:func:`~soma.pipeline.folds.run_fold` holds only the canonical order. The builders read
plan sections: the learner's shape picks the builder, the head's task family the grid
task, and the cohort's ``collapse`` the patient path:

* :func:`extraction_options` — :func:`soma.extract`'s keywords for the plan's features
  (the :class:`~soma.pipeline.Pipeline` extracts through the verb);
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

from soma.models.aggregators.registry import aggregator_registry
from soma.extract.cache import resolve_cache_root
from slide2vec import DenseOptions, PreprocessingConfig

from soma._internal.class_scheme import resolve_classes
from soma.config import (
    Component,
    EvaluationConfig,
    ExperimentConfig,
    LoadersConfig,
    TrainerConfig,
    is_annotation_sampling,
)
from soma.data import (
    AnnotationManifest,
    Cohort,
    FoldSplit,
    GridGeometry,
    ImageManifest,
    LabelMapSource,
    PointSource,
    SampleRecord,
    TileImageSource,
    require_coverage,
)
from soma.data.cohort import targets_frame
from soma.data.records import targets_equal
from soma.models.decoders.registry import build_decoder_for_grid
from soma.extract.dense.composite import CompositeDenseFeatureStore
from soma.extract.dense.reader import CLASS_SCHEME_KEYS, resolve_class_scheme
from soma.evaluation import Baseline
from soma.evaluation.metrics import resolve_metrics
from soma.extract.extractor import _release_parent_cuda_state
from soma.extract.rois import RoiSet, label_map_identity
from soma.learn import (
    Learner,
    Progress,
    callback_registry,
    learner_registry,
    record_builder,
    resolve_leaf,
)
from soma.learn.grid import grid_loaders
from soma.plan import LEARNER_OWN_EXCLUDED, Plan, load_cohort
from soma.extract.preprocessing.resolution import resolve_pipeline_dense, resolve_pipeline_preprocessing
from soma.models.heads.classification import BranchAwareClassificationHead, MulticlassClassificationHead
from soma.models.heads.registry import task_registry
from soma.learn.bag_dataset import BagDataset, HierarchicalBagDataset
from soma.learn.collate import bag_collate_fn, cox_window_collate, hierarchical_bag_collate_fn
from soma.learn.feature_adaptor import (
    FeatureAdaptor,
    NormalizationParams,
    ProjectionParams,
    build_feature_adaptor,
    feature_adaptor_output_dim,
    validate_feature_adaptor_compatibility,
    write_feature_adapter_sidecar,
)
from soma.pipeline.fold_planning import plan_dense_fold
from soma.models.model import DenseModel, EmbeddingModel, MILModel
from soma.learn.patient_dataset import PatientDataset, patient_collate_fn
from soma.learn.sample_dataset import SampleDataset, sample_collate_fn

__all__ = [
    "FoldLoaders",
    "FoldSetup",
    "RunInputs",
    "SplitPlan",
    "build_composite_dense_store",
    "build_fold",
    "build_grid_fold",
    "build_grid_model",
    "build_set_fold",
    "build_set_model",
    "extraction_options",
    "extraction_unit",
    "live_encoder_for",
    "live_encoder_spec",
    "load_pipeline_data",
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
    """What every fold of a run shares: the plan, the source, the cohort it serves.

    ``cohort`` is the one the source is keyed by: an annotation-sampled run's ROI cohort
    (:meth:`~soma.data.Cohort.derived`), whose ``rois`` the source carries. ``images`` and
    ``annotations`` are the cohort CSV's manifests, read by the target sources.
    """

    plan: Plan
    preprocessing: PreprocessingConfig
    evaluation: EvaluationConfig
    source: Any
    cohort: Cohort
    images: ImageManifest
    annotations: AnnotationManifest | None
    run_id: str
    run_dir: Path
    source_identity: dict[str, Any]
    encoder_stage: Any = None
    #: A grid run's slide2vec ``DenseOptions`` (its read spacing, tolerance and backend).
    dense: DenseOptions | None = None
    roi_population_cache_root: Path | None = None
    #: ``{top-level package: version}`` of the plan's dotted-path components.
    packages: dict[str, str | None] = field(default_factory=dict)
    #: The ROIs of an annotation-sampled run, keyed like the source (``source.rois``).
    rois: RoiSet | None = None
    _roi_population: Any = field(default=None, repr=False)
    _roi_caches: tuple[dict, dict] = field(default_factory=lambda: ({}, {}), repr=False)

    @property
    def num_folds(self) -> int:
        return self.cohort.num_folds

    @property
    def provenance(self) -> dict[str, Any]:
        """What every fold bundle's ``provenance.json`` records beyond soma's own: the run,
        the feature source's identity and one ``<package>_version`` entry per top-level
        package a dotted-path component imports from."""
        return {
            "run_id": self.run_id,
            "feature_source": self.source_identity,
            **{f"{package}_version": version for package, version in self.packages.items()},
        }

    def roi_population(self):
        """The run's segmentation ROI population (explicit ROI batch sampling only),
        resolved once and shared by every fold."""
        if self._roi_population is None:
            named = {sid for fold in self.cohort.folds for sid in fold.all_sample_ids}
            # Every ROI a fold names, once, in cohort order.
            sample_ids = [sid for sid in self.cohort.sample_ids if sid in named]
            head = _build_segmentation_head(self, self.source.geometry(sample_ids[0]))
            self._roi_population = _resolve_segmentation_roi_population(
                cache_root=self.roi_population_cache_root or self.run_dir / "segmentation_roi_population",
                identity=label_map_identity(
                    sample_ids, images=self.images, annotations=self.annotations, rois=self.rois
                ),
                head=head,
                label_maps=_label_maps(self, head, sample_ids),
                source_fingerprint_cache=self._roi_caches[0],
                resolved_backend_cache=self._roi_caches[1],
            )
        return self._roi_population


def build_fold(inputs: RunInputs, fold: int, fold_split: FoldSplit) -> FoldSetup:
    """The fold's :class:`FoldSetup`, from the builder of the learner's shape."""
    if inputs.plan.shape == "grid":
        return build_grid_fold(inputs, fold, fold_split)
    return build_set_fold(inputs, fold, fold_split)


# ---------------------------------------------------------------------------
# Models: built by the pipeline, rebuilt by soma.load
# ---------------------------------------------------------------------------

#: ``adaptor=`` default of the builders: build the adaptor unfitted from its sections
#: (its fitted buffers load from the bundle's weights).
UNFITTED: Any = object()


def _adaptor_sections(normalization, projection) -> tuple[NormalizationParams | None, ProjectionParams | None]:
    return (
        None if normalization is None else NormalizationParams(**normalization),
        None if projection is None else ProjectionParams(**projection),
    )


def _adaptor_params(plan: Plan) -> tuple[dict[str, Any], dict[str, Any]]:
    """The plan's ``normalization`` and ``projection`` sections (both always present)."""
    params = plan.learner.params
    return dict(params["normalization"]), dict(params["projection"])


def head_params(head: Component) -> dict[str, Any]:
    """A head leaf's constructor params: the plan's params without the cohort's ``targets``."""
    return {key: value for key, value in head.params.items() if key != "targets"}


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


def build_learner(model: nn.Module, plan: Plan) -> Learner:
    """Wrap ``model`` in the plan's learner, with its optimizer and scheduler leaves."""
    learner = plan.learner
    own = {key: value for key, value in learner.params.items() if key not in LEARNER_OWN_EXCLUDED}
    cls, params, _ = resolve_leaf({"name": learner.name, "params": own}, learner_registry)
    return cls(
        model,
        optimizer=learner.params["optimizer"],
        schedulers=tuple(learner.params.get("scheduler") or ()),
        **params,
    )


def _fit_callbacks(trainer: TrainerConfig, *, fold: int, num_folds: int) -> Callable[[], list[Any]]:
    """The plan's callbacks (``checkpoint`` selects in memory, no file), plus ``Progress``."""

    def callbacks() -> list[Any]:
        installed: list[Any] = []
        for leaf in trainer.callbacks:
            cls, params, _ = resolve_leaf(leaf, callback_registry)
            installed.append(cls(**params))
        installed.append(Progress(label=f"fold {fold + 1}/{num_folds}" if num_folds > 1 else None))
        return installed

    return callbacks


def _trainer_kwargs(trainer: TrainerConfig) -> dict[str, Any]:
    return {
        "max_epochs": trainer.max_epochs,
        "max_steps": trainer.max_steps,
        "accumulate_grad_batches": trainer.accumulate_grad_batches,
        "val_every_n_epochs": trainer.val_every_n_epochs,
        "device": trainer.device,
        "precision": trainer.precision,
    }


# ---------------------------------------------------------------------------
# Shared fold helpers
# ---------------------------------------------------------------------------


def _loader_kwargs(loaders: LoadersConfig) -> dict[str, object]:
    """DataLoader kwargs from the ``loaders`` section; ``persistent_workers`` needs workers."""
    return {
        "batch_size": loaders.batch_size,
        "num_workers": loaders.num_workers,
        "pin_memory": loaders.pin_memory,
        "persistent_workers": loaders.persistent_workers and loaders.num_workers > 0,
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
    plan: Plan,
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
            *_adaptor_sections(*_adaptor_params(plan)),
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


def _encoder_identity(plan: Plan) -> str:
    encoder = plan.features.encoder
    return encoder.name if encoder is not None else ""


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
            "cohort.tune_is_test requires exactly one test split"
        )
    return test_split_names[0]


def _bag_head_class(task_cls: type, *, task_name: str, aggregator_key: str) -> type:
    """Pick the head class for a bag-level run.

    ``clam_mb`` emits one branch per class and needs a branch-aware head; every other
    task family has no per-class branch to route through. A registered head that already
    handles branch representations is kept as is; the ordinary multiclass head is
    swapped for the built-in branch-aware one.
    """
    if aggregator_key != "clam_mb":
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


def _resolve_hipt_params(preprocessing: PreprocessingConfig, aggregator: Component) -> dict[str, object]:
    patch_size = preprocessing.tiling.requested_tile_size_px if preprocessing.tiling is not None else None
    region_size = preprocessing.requested_region_size_px
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
    loaders: LoadersConfig,
    seed: int,
    min_events_per_window: int,
    window_size: int | None = None,
) -> DataLoader:
    """Training loader whose batches each hold >= one event (for batched Cox).

    The Cox partial likelihood is undefined on an event-free batch, so the risk set must
    contain events. ``window_size`` is ``loaders.batch_size`` in padded mode and
    ``cox_window`` in accumulation mode; the matching ``collate_fn`` decides which.
    """
    from soma.learn.survival_sampler import EventBalancedBatchSampler

    sampler = EventBalancedBatchSampler(
        events,
        batch_size=window_size if window_size is not None else loaders.batch_size,
        min_events_per_window=min_events_per_window,
        seed=seed,
    )
    kwargs = _loader_kwargs(loaders)
    kwargs.pop("batch_size")
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, **kwargs)


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
    records_by_id: dict[str, SampleRecord]
    train_records: list[SampleRecord]


def _patient_ids(records: Sequence[SampleRecord]) -> list[str]:
    """The patients of ``records``, in order of first appearance."""
    seen: set[str] = set()
    patient_ids: list[str] = []
    for record in records:
        if record.patient_id is None:
            raise ValueError(
                f"Sample '{record.sample_id}' has no patient_id. "
                "All samples must have a patient_id for a patient run (cohort.collapse: patient_id)."
            )
        if record.patient_id not in seen:
            seen.add(record.patient_id)
            patient_ids.append(record.patient_id)
    return patient_ids


def _patient_records(cohort: Cohort) -> dict[str, SampleRecord]:
    """One representative record per patient, its first sample in cohort order, as
    :meth:`~soma.data.Cohort.collapse` keeps it; a patient's samples must share targets."""
    representatives: dict[str, SampleRecord] = {}
    for record in cohort.records:
        if record.patient_id is None:
            raise ValueError(
                f"Sample '{record.sample_id}' is missing a patient_id. "
                "All rows must have a patient_id for patient-level pipelines."
            )
        head = representatives.setdefault(record.patient_id, record)
        if not targets_equal(head.targets, record.targets):
            raise ValueError(
                f"Patient '{record.patient_id}' has inconsistent targets across slides: "
                f"{head.targets} vs {record.targets}. "
                "All slides for a patient must share the same targets."
            )
    return representatives


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
    plan, samples, feature_store = inputs.plan, inputs.cohort, inputs.source
    cohort, evaluation = plan.cohort, inputs.evaluation
    patient = plan.unit == "patient"
    all_test_ids = {sid for ids in fold_split.tests.values() for sid in ids}
    _fp = f"Fold {fold}" if inputs.num_folds > 1 else "Run"
    statuses = _feature_statuses(feature_store)
    if patient:
        # Patient-level: features are indexed by patient_id; splits are slide-based.
        train_records = [samples.record(sid) for sid in fold_split.train]
        tune_records = [samples.record(sid) for sid in fold_split.tune]
        test_records_by_split = {
            name: [samples.record(sid) for sid in ids] for name, ids in fold_split.tests.items()
        }
        train_ids = _patient_ids(train_records)
        tune_ids = _patient_ids(tune_records)
        test_ids = {name: _patient_ids(records) for name, records in test_records_by_split.items()}
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
        train_records = [samples.record(sid) for sid in fold_split.train if sid in expected]
        tune_records = [samples.record(sid) for sid in fold_split.tune if sid in expected]
        test_records_by_split = {
            name: [samples.record(sid) for sid in ids if sid in expected] for name, ids in fold_split.tests.items()
        }
    else:
        # No manifest: every listed sample must have a feature file.
        train_records = [samples.record(sid) for sid in fold_split.train]
        tune_records = [samples.record(sid) for sid in fold_split.tune]
        test_records_by_split = {
            name: [samples.record(sid) for sid in ids] for name, ids in fold_split.tests.items()
        }
        empty_by_split = None

    tune_from_test: str | None = None
    if cohort.tune_is_test:
        tune_from_test = _resolve_tune_is_test_split(fold_split, _fp)
        logger.warning(
            "%s uses test split '%s' as tune because cohort.tune_is_test is set; "
            "checkpoint selection and test reporting use the same samples",
            _fp,
            tune_from_test,
        )
        tune_records = list(test_records_by_split[tune_from_test])
        if patient:
            tune_ids = list(test_ids[tune_from_test])
            raw_tune_ids = list(raw_test_ids[tune_from_test])
        if empty_by_split is not None:
            empty_by_split["tune"] = list(empty_by_split.get(tune_from_test, []))

    if evaluation.holdout_test:
        # Tune-only model-selection run: no test loader, no test inference; metrics and
        # summary carry tune only. Done after tune_is_test resolved tune from test.
        test_records_by_split = {}
        if patient:
            test_ids, raw_test_ids = {}, {}
        if empty_by_split is not None:
            empty_by_split = {k: v for k, v in empty_by_split.items() if k in ("train", "tune")}

    train_items = train_ids if patient else train_records
    tune_items = tune_ids if patient else tune_records
    tests_items = test_ids if patient else test_records_by_split
    unit = "patients" if patient else "samples"
    if not train_items:
        raise ValueError(f"{_fp} has no training {unit} with available features")
    fallback_to_train = not tune_items and not (empty_by_split and empty_by_split.get("tune"))
    if fallback_to_train and not cohort.allow_missing_tune:
        raise ValueError(f"{_fp} has no tuning {unit} with available features")
    if fallback_to_train:
        logger.warning(
            "%s has no tuning %s with available features; "
            "using train split as tune because cohort.allow_missing_tune is set",
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
        records_by_id = _patient_records(samples)
    else:
        if fallback_to_train:
            tune_output = tuple(fold_split.train)
        elif tune_from_test is not None:
            tune_output = tuple(fold_split.tests[tune_from_test])
        else:
            tune_output = tuple(fold_split.tune)
        test_output = {name: tuple(fold_split.tests[name]) for name in tests_items}
        records_by_id = {record.sample_id: record for record in samples.records}
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
    plan, feature_store = inputs.plan, inputs.source
    evaluation, head = inputs.evaluation, plan.learner.params["head"]
    aggregator = plan.learner.params.get("aggregator")
    records = _set_fold_records(inputs, fold, fold_split)

    if evaluation.subgroups:
        sample = inputs.cohort.records[0]
        missing = [c for c in evaluation.subgroups if c not in sample.metadata]
        if missing:
            raise ValueError(
                f"Subgroup column(s) not found in dataset metadata: {missing}. "
                f"Available: {sorted(sample.metadata)}"
            )

    # The plan already holds what the cohort decided (class count, label map, survival
    # bins) under the explicit head params; the survival columns were validated there.
    task_cls = task_registry.resolve(head.name)
    task_params = {**head_params(head), "metrics": evaluation.metrics}
    task_family = task_cls.task_family

    # A transform a path cannot honor is refused rather than silently dropped.
    normalization, projection = _adaptor_params(plan)
    validate_feature_adaptor_compatibility(
        *_adaptor_sections(normalization, projection),
        unit=plan.unit,
        is_hierarchical=feature_store.rank == 3,
    )
    feature_dim = feature_store.feature_dim
    encoder_identity = _encoder_identity(plan)
    fits_adaptor = False
    aggregator_spec = None
    head_cls = task_cls
    aggregator_key = None if aggregator is None else aggregator_registry.key_of(aggregator.name)
    if plan.unit == "patient":
        kind = "embedding"
    elif feature_store.rank == 1:
        if aggregator is not None:
            raise ValueError("an aggregator pools a bag; these features are one vector per sample")
        kind, fits_adaptor = "embedding", True
    elif feature_store.rank == 3:
        if aggregator_key != "hipt":
            raise ValueError("hierarchical features require the hipt aggregator")
        kind = "mil"
        aggregator_spec = {"name": aggregator.name, "params": _resolve_hipt_params(inputs.preprocessing, aggregator)}
    else:
        if aggregator is None:
            raise ValueError("bags of tile features need an aggregator (learner.params.aggregator)")
        kind, fits_adaptor = "mil", True
        aggregator_spec = {"name": aggregator.name, "params": dict(aggregator.params)}
        head_cls = _bag_head_class(task_cls, task_name=head.name, aggregator_key=aggregator_key)
    model_spec = {
        "kind": kind,
        "feature_dim": feature_dim,
        "head": {"name": _component_name(task_registry, head.name, head_cls), "params": task_params},
        "aggregator": aggregator_spec,
        "normalization": normalization,
        "projection": projection,
        "encoder_identity": encoder_identity,
    }

    def build_fold_learner(adaptor: FeatureAdaptor | None) -> Learner:
        return build_learner(build_set_model(**model_spec, adaptor=adaptor), plan)

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
            "unit": plan.unit,
            "task_family": task_family,
            "target_schema": {key: getattr(t, "__name__", str(t)) for key, t in head.target_schema.items()},
            "classes": _jsonable(head.classes),
            "metrics": _jsonable(metrics(head)),
            "feature_dim": int(feature_dim),
        }

    return FoldSetup(
        fold=fold,
        num_folds=inputs.num_folds,
        seed=plan.run.seed,
        fit_adaptor=(
            _adaptor_step(
                plan,
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
        callbacks=_fit_callbacks(plan.trainer, fold=fold, num_folds=inputs.num_folds),
        trainer=_trainer_kwargs(plan.trainer),
        metrics=metrics,
        preprocessing=preprocessing,
        provenance=inputs.provenance,
    )


def _set_loaders(inputs: RunInputs, records: _SetFoldRecords, head, *, kind: str) -> FoldLoaders:
    """Step 4 for the set shape: the train / tune / test loaders and the scored splits."""
    plan, feature_store = inputs.plan, inputs.source
    loader_kwargs = _loader_kwargs(plan.loaders)
    target_fn = head.extract_targets
    patient = plan.unit == "patient"
    if patient:
        record_map = _patient_records(inputs.cohort)
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
            loaders=plan.loaders,
            seed=plan.run.seed,
            min_events_per_window=head.min_events_per_window,
            window_size=window_size,
        )

    def targets(output_ids: tuple[str, ...]) -> pd.DataFrame:
        frame = targets_frame([records.records_by_id[sid] for sid in output_ids])
        frame.index = pd.Index([str(sid) for sid in output_ids], name="sample_id", dtype=object)
        return frame

    subgroups = tuple(inputs.evaluation.subgroups)
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
    plan, dense = inputs.plan, inputs.dense
    head, masks = plan.learner.params["head"], plan.features.extraction.masks
    annotation_rasters = is_annotation_sampling(masks)
    tiling = inputs.preprocessing.tiling if inputs.preprocessing is not None else None
    head_cls = task_registry.resolve(head.name)
    scheme = head_params(head)
    num_classes, class_names, label_remap = resolve_class_scheme(scheme, annotation_rasters=annotation_rasters)
    seg_params = {key: value for key, value in scheme.items() if key not in CLASS_SCHEME_KEYS}
    params = {
        "num_classes": num_classes,
        "geometry": geometry,
        "metrics": inputs.evaluation.metrics,
        "spacing_um": float(dense.spacing_um) if dense is not None else None,
        "spacing_policy": plan.features.spacing_policy,
        "backend": tiling.mask_backend if tiling is not None else "auto",
        "image_backend": dense.backend if dense is not None else "auto",
        "tolerance": float(dense.tolerance) if dense is not None else 0.05,
        "label_remap": label_remap,
        "pixel_mapping": masks.pixel_mapping if annotation_rasters else None,
        "class_names": class_names if len(class_names) == num_classes else None,
        **seg_params,
    }
    return {"name": _component_name(task_registry, head.name, head_cls), "params": params}


def _build_segmentation_head(inputs: RunInputs, geometry):
    spec = _segmentation_head_spec(inputs, geometry)
    return task_registry.resolve(spec["name"])(**spec["params"])


def _label_maps(inputs: RunInputs, head, sample_ids: Sequence[str]):
    """The label-map target source of a segmentation run's samples, the class scheme
    folded in: an annotation-sampled run's ROIs read the mask crops ROI sampling stored
    (``rois.label_maps``); pre-cropped tiles read their rasters through
    :meth:`~soma.data.LabelMapSource.from_manifests`."""
    params = head.label_map_read_params
    if inputs.rois is not None:
        return inputs.rois.subset(sample_ids).label_maps(
            size=params["size"], label_remap=params["label_remap"], ignore_index=head.ignore_index
        )
    if inputs.annotations is None:
        raise ValueError("a segmentation run needs the cohort's annotation manifest (label_mask_path).")
    return LabelMapSource.from_manifests(inputs.annotations, inputs.images, **params).subset(sample_ids)


def _resolve_segmentation_roi_population(
    *,
    cache_root: str | Path,
    identity: list[dict[str, Any]],
    head,
    label_maps,
    source_fingerprint_cache: dict[str, dict[str, object]] | None,
    resolved_backend_cache: dict[tuple[str, str, bool], str] | None,
):
    """Resolve one ROI population with the run's shared cache policy."""
    from soma.learn.roi_population import resolve_segmentation_roi_population

    return resolve_segmentation_roi_population(
        cache_root,
        identity,
        lambda sample_id: {"mask": label_maps.load(sample_id)},
        num_classes=head.num_classes,
        target_identity=head.target_identity,
        workers=max(1, min(8, os.cpu_count() or 1)),
        source_fingerprint_cache=source_fingerprint_cache,
        resolved_backend_cache=resolved_backend_cache,
    )


def _segmentation_roi_batch_sampler(records, population, *, plan: Plan, fold: int):
    """The explicit cached-grid training-batch contract (``loaders.sampler``)."""
    from soma.learn.samplers import sampler_registry

    sample_ids = [record.sample_id for record in records]
    cls, params, _ = resolve_leaf(plan.loaders.sampler, sampler_registry)
    return cls(
        sample_ids=sample_ids,
        class_pixel_counts=population.subset(sample_ids).class_pixel_counts,
        batch_size=plan.loaders.batch_size,
        gradient_accumulation=plan.trainer.accumulate_grad_batches,
        seed=plan.run.seed + fold,
        **params,
    )


def _grid_augment(augmentation: Component | None, head, geometry):
    """Build the configured ``{name, params}`` augmentation, injecting the run's
    ``ignore_index`` and ``target_size`` where the augmentation accepts them and its
    params leave them unset."""
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
        if key in accepted and params.get(key) is None and value is not None:
            params[key] = value
    params = {key: value for key, value in params.items() if value is not None or key not in injected}
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
    ``loaders.augment`` augments the train split. Detection sweeps its per-class threshold
    on tune (without a tune split it scores at the head's ``validation_threshold``).
    """
    plan, feature_store, stage = inputs.plan, inputs.source, inputs.encoder_stage
    evaluation, decoder = inputs.evaluation, plan.learner.params["decoder"]
    family = plan.task_family
    _fp = f"Fold {fold}" if inputs.num_folds > 1 else "Run"
    live = stage is not None
    normalization, projection = _adaptor_params(plan)
    validate_feature_adaptor_compatibility(
        *_adaptor_sections(normalization, projection),
        feature_mode="live" if live else "cached",
        has_composite=isinstance(feature_store, CompositeDenseFeatureStore),
    )
    fold_plan = plan_dense_fold(
        samples=inputs.cohort,
        fold_split=fold_split,
        cohort=plan.cohort,
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
        if family == "segmentation":
            _require_configured_spacing(inputs, all_ids)

    if family == "segmentation":
        head_spec = _segmentation_head_spec(inputs, geometry)
        point_source = None
    else:
        head_spec, point_source = _detection_head_spec(inputs, geometry, all_ids)
    encoder_identity = _encoder_identity(plan)
    model_spec = {
        "feature_dim": feature_dim,
        "decoder": {"name": decoder.name, "params": dict(decoder.params)},
        "head": head_spec,
        "normalization": normalization,
        "projection": projection,
        "encoder_identity": encoder_identity,
        "encoder": live_encoder_spec(plan) if live else None,
    }

    def fit_adaptor(fold_dir: Path) -> FeatureAdaptor | None:
        if live:  # refused above for live: the stage's grids are never seen whole
            return None
        return _adaptor_step(
            plan,
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
        return build_learner(model, plan)

    def build_loaders(learner: Learner, fold_dir: Path) -> FoldLoaders:
        head, consumes_image = learner.head, learner.model.consumes_image
        if family == "segmentation":
            targets = _label_maps(inputs, head, all_ids)
        else:
            targets = point_source
        images = None
        if consumes_image and not live:
            if inputs.rois is not None:
                raise ValueError(
                    f"decoder '{decoder.name}' reads each tile's pixels; slide-manifest ROIs "
                    "carry no flat tile to read"
                )
            images = TileImageSource.from_manifest(inputs.images, size=geometry.target_size).subset(all_ids)
        sampler = None
        if plan.loaders.sampler is not None:
            if live or family != "segmentation":
                raise ValueError("Explicit ROI batch sampling is a cached segmentation training contract.")
            population = inputs.roi_population().subset(all_ids)
            (fold_dir / "segmentation_roi_population.json").write_text(
                json.dumps(population.provenance(), indent=2, sort_keys=True), encoding="utf-8"
            )
            sampler = _segmentation_roi_batch_sampler(train_records, population, plan=plan, fold=fold)

        def write_sampler_audit(directory: Path) -> None:
            (directory / "roi_batch_sampling.json").write_text(
                json.dumps(sampler.audit(), indent=2, sort_keys=True), encoding="utf-8"
            )

        loader_kwargs = _loader_kwargs(plan.loaders)
        fold_loaders = grid_loaders(
            feature_store,
            head=head,
            train=[record.sample_id for record in train_records],
            tune=[record.sample_id for record in tune_records],
            tests={name: [r.sample_id for r in recs] for name, recs in test_records_by_split.items()},
            batch_size=loader_kwargs.pop("batch_size"),
            num_workers=loader_kwargs.pop("num_workers"),
            targets=targets,
            augment=_grid_augment(plan.loaders.augment, head, geometry) if live else None,
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
        overlays = images or _overlay_images(inputs, all_ids, size=geometry.target_size)

        def split_plan(name: str, loader, records) -> SplitPlan:
            return SplitPlan(
                name=name,
                loader=loader,
                targets=targets.subset([record.sample_id for record in records]),
                callbacks=tuple(_grid_callbacks(inputs, fold_dir, name, overlays)),
            )

        splits = [split_plan("tune", fold_loaders.tune, tune_records)]
        splits += [split_plan(name, fold_loaders.tests[name], recs) for name, recs in test_records_by_split.items()]
        return FoldLoaders(
            train=fold_loaders.train,
            tune=fold_loaders.tune,
            splits=tuple(splits),
            after_fit=write_sampler_audit if sampler is not None else None,
        )

    def metrics(head) -> list[Any]:
        names: list[Any] = list(resolve_metrics(head.task_family, evaluation.metrics))
        if family == "detection":
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
            "task_family": head.task_family,
            "num_classes": int(head.num_classes),
            "class_names": _jsonable(head.class_names),
            "geometry": _jsonable(dataclasses.asdict(geometry)),
            "tile_size": _jsonable(geometry.target_size),
            "spacing_um": head_spec["params"].get("spacing_um"),
            "feature_mode": "live" if live else "cached",
            "metrics": _jsonable(metrics(head)),
        }
        if family == "segmentation":
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
        seed=plan.run.seed,
        fit_adaptor=fit_adaptor,
        build_learner=build_fold_learner,
        build_loaders=build_loaders,
        callbacks=_fit_callbacks(plan.trainer, fold=fold, num_folds=inputs.num_folds),
        trainer=_trainer_kwargs(plan.trainer),
        metrics=metrics,
        preprocessing=preprocessing,
        provenance=inputs.provenance,
        tune_step=tune_step if family == "detection" else None,
    )


def _overlay_images(inputs: RunInputs, sample_ids: Sequence[str], *, size):
    """The pixels :class:`~soma.evaluation.SaveOverlays` blends over: the flat tiles, or
    each slide-manifest ROI's window of its whole slide at the run spacing (read as the
    grid run's slide2vec ``DenseOptions`` read it)."""
    dense, spacing_policy = inputs.dense, inputs.plan.features.spacing_policy
    spacing_um = dense.spacing_um if dense is not None else None
    backend = dense.backend if dense is not None else "auto"
    tolerance = float(dense.tolerance) if dense is not None else 0.05
    if inputs.rois is not None and spacing_um is not None:
        return inputs.rois.subset(sample_ids).pixels(
            size=size, spacing_um=spacing_um, spacing_policy=spacing_policy, backend=backend, tolerance=tolerance
        )
    if inputs.rois is not None:
        return None
    return TileImageSource.from_manifest(
        inputs.images,
        size=size,
        spacing_um=spacing_um,
        spacing_policy=spacing_policy,
        backend=backend,
        tolerance=tolerance,
    ).subset(sample_ids)


def _spacing_at_level_0(inputs: RunInputs, sample_id: str) -> float | None:
    """A sample's declared level-0 spacing: an ROI's is its slide's."""
    if inputs.rois is not None and sample_id in inputs.rois:
        sample_id = inputs.rois[sample_id].slide_id
    return inputs.images[sample_id].spacing_at_level_0


def _require_configured_spacing(inputs: RunInputs, sample_ids: Sequence[str]) -> None:
    """Cached segmentation grids must be extracted at the spacing the config reads masks
    at (per sample under ``native_if_coarser``); see :func:`soma.extract.dense.require_declared_spacing`."""
    from soma.extract.dense import require_declared_spacing

    from soma.extract.spacing import resolve_effective_spacing_um

    dense = inputs.dense
    require_declared_spacing(
        inputs.source,
        {
            sample_id: (
                None
                if dense is None
                else resolve_effective_spacing_um(
                    requested_spacing_um=dense.spacing_um,
                    spacing_at_level_0=_spacing_at_level_0(inputs, sample_id),
                    tolerance=dense.tolerance,
                    policy=inputs.plan.features.spacing_policy,
                )
            )
            for sample_id in sample_ids
        },
    )


def _detection_head_spec(inputs: RunInputs, geometry, sample_ids):
    """``({name, params}, PointSource)`` of a detection run: classes from the head's class
    scheme (``classes`` / ``drop`` / ``num_classes``), distances converted at the run's
    one spacing."""
    head = inputs.plan.learner.params["head"]
    scheme = head_params(head)
    num_classes, class_names, class_remap, dropped = resolve_classes(
        scheme, excluded_key="drop", subject="detection"
    )
    if inputs.annotations is None:
        raise ValueError("a detection run needs the cohort's annotation manifest (points_path).")
    point_source = PointSource.from_manifest(
        inputs.annotations, class_remap=class_remap, drop=dropped
    ).subset(sample_ids)
    params = {k: v for k, v in scheme.items() if k not in ("num_classes", "classes", "drop")}
    if params.get("match_distance") is None:
        raise ValueError(
            "a detection head requires match_distance (the F1 matching distance δ, in µm)."
        )
    head_cls = task_registry.resolve(head.name)
    spec = {
        "name": _component_name(task_registry, head.name, head_cls),
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

    evaluation, family = inputs.evaluation, inputs.plan.task_family
    out: list[Any] = []
    if evaluation.save_overlays:
        out.append(verbs.SaveOverlays(fold_dir / "overlays" / split_name, images=overlay_images))
    if family == "segmentation" and evaluation.save_probabilities:
        out.append(verbs.SaveProbabilities(fold_dir / "probabilities" / split_name))
    if family == "detection" and evaluation.save_heatmaps:
        out.append(verbs.SaveHeatmaps(fold_dir / "heatmaps" / split_name))
    return out


# ---------------------------------------------------------------------------
# The run's feature source: supplied, extracted, live or composite
# ---------------------------------------------------------------------------


def annotation_sampled(plan: Plan | ExperimentConfig) -> bool:
    """Whether a segmentation run samples its ROIs from whole slides (the slide-manifest
    input mode): the cohort has a ``label_mask_path`` column of annotation rasters and
    hs2p's masks declare their annotation vocabulary. A pre-cropped segmentation cohort
    has a ``label_mask_path`` column too, but its masks keep hs2p's binary tissue
    vocabulary (ADR 0019)."""
    from soma.plan import head_family

    if head_family(plan) != "segmentation" or not is_annotation_sampling(plan.features.extraction.masks):
        return False
    return "label_mask_path" in pd.read_csv(plan.cohort.records, nrows=0).columns


def extraction_unit(plan: Plan | ExperimentConfig) -> str:
    """What one manifest row is to extraction: a whole slide or a pre-cropped image."""
    from soma.plan import learner_shape, sample_unit

    if learner_shape(plan) == "set":
        return "tile" if sample_unit(plan) == "tile" else "slide"
    return "slide" if annotation_sampled(plan) else "tile"


def load_pipeline_data(config: ExperimentConfig) -> tuple[Cohort, ImageManifest, AnnotationManifest | None]:
    """Read the cohort and manifests a config names, validated against its head.

    The head's ``resolve_target_schema`` says which target keys the cohort must carry
    (``label``, ``time``/``event``, ``value`` or ``value_<k>`` ...); the head's
    ``targets`` map a key to a differently named cohort column
    (:func:`soma.plan.load_cohort`). A grid run also reads the annotations.
    """
    from soma.plan import head_family

    records = config.cohort.records
    roi_columns = sorted({"region_x", "region_y"} & set(pd.read_csv(records, nrows=0).columns))
    if roi_columns:
        # ROIs are extraction-internal (ADR 0021): their table is provenance, and its rows
        # read as whole slides would be sampled again.
        raise ValueError(
            f"cohort.records {str(records)!r} has the ROI columns {roi_columns}: it is the ROI "
            "table an annotation-sampled extraction wrote. Point cohort.records at the slide "
            "manifest the ROIs were sampled from; each ROI inherits its slide's splits."
        )
    cohort = load_cohort(config)
    image_manifest = ImageManifest.from_csv(records)
    annotation_manifest = None
    family = head_family(config)
    if family in {"segmentation", "detection"}:
        masks = config.features.extraction.masks
        annotation_manifest = AnnotationManifest.from_csv(
            records,
            pixel_mapping=masks.pixel_mapping if is_annotation_sampling(masks) else None,
        )
        if family == "segmentation" and not annotation_manifest.has_label_masks:
            raise ValueError("a segmentation run needs a label_mask_path column in the cohort records.")
        if family == "detection" and not annotation_manifest.has_points:
            raise ValueError("a detection run needs a points_path column in the cohort records.")
    return cohort, image_manifest, annotation_manifest


def extraction_options(plan: Plan, *, run_dir: Path) -> dict[str, Any]:
    """:func:`soma.extract`'s keywords for the plan's ``features``: its encoder, the
    extraction (tiling, segmentation, masks) and execution, a grid run's dense options,
    what one manifest row is, and the cache rooted for this run."""
    features = plan.features
    return {
        "encoder": features.encoder,
        "cache": replace(
            features.cache,
            root_dir=resolve_cache_root(features.cache, feature_dir=run_dir, output_root=plan.run.output_dir),
        ),
        "unit": extraction_unit(plan),
        "execution": features.execution,
        "extraction": resolve_pipeline_preprocessing(plan),
        "dense": resolve_pipeline_dense(plan) if plan.shape == "grid" else None,
        "spacing_policy": features.spacing_policy,
        "output_dir": run_dir,
    }


def live_encoder_spec(plan: Plan) -> dict[str, Any]:
    """The :func:`soma.extract.live_encoder` arguments a ``features.kind='live'`` plan
    describes: what a live bundle records to rebuild its frozen encoder stage."""
    from soma.models.encoders.validation import resolve_encoder_precision

    encoder = plan.features.encoder
    if encoder is None:
        raise ValueError("features.encoder is required for features.kind='live'.")
    dense = resolve_pipeline_dense(plan)
    return {
        "encoder": encoder.name,
        "target_size": int(dense.target_size),
        "spacing_um": float(dense.spacing_um),
        "window": dense.window_size,
        "overlap": float(dense.overlap),
        "feature_kind": dense.feature_kind,
        "precision": resolve_encoder_precision(plan.features.execution.precision, encoder_name=encoder.name),
        "output_variant": encoder.output_variant,
        "allow_non_recommended_settings": encoder.allow_non_recommended_settings,
        "tolerance": float(dense.tolerance),
        "backend": dense.backend,
        "pad_mode": dense.pad_mode,
        "image_pad_value": dense.image_pad_value,
        "attention_blocks": tuple(dense.attention_blocks),
        "attention_include_registers": bool(dense.attention_include_registers),
    }


def live_encoder_for(plan: Plan):
    """The frozen encoder stage a ``features.kind='live'`` plan re-encodes with.

    Built through :func:`soma.extract.live_encoder` from :func:`live_encoder_spec`, once
    per run and shared by every fold. Whole-image inference over a finished run loads
    its bundles instead (:meth:`soma.evaluation.sliding_window.SlidingWindowSegmentationPredictor.from_bundles`).
    """
    from soma import extract

    return extract.live_encoder(**live_encoder_spec(plan), num_gpus=plan.features.execution.num_gpus)


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


def build_composite_dense_store(plan: Plan | ExperimentConfig, images: ImageManifest, *, output_root: Path):
    """Extract every composite member of ``images`` into its own cache; return the concat
    view.

    Public so that out-of-core consumers (a project protocol's rescore step) rebuild the
    exact store a composite run trained on. Each member extracts through
    :func:`soma.extract` with the run's slide2vec ``DenseOptions`` and
    ``ExecutionOptions`` plus its own overrides, exactly as that encoder does in a
    single-encoder run, so its dense cache is the one a single-encoder run of it uses. The
    token grids are combined at load time by :class:`CompositeDenseFeatureStore` per
    ``concat_resolution`` / ``concat_grid_size``.
    """
    from soma.models.encoders import CompositeParams
    from soma.models.encoders.composite import member_norm_for
    from soma.config import EncoderLeaf
    from soma.extract import extract

    features = plan.features
    if not features.is_composite:
        raise ValueError("build_composite_dense_store requires the composite encoder.")
    composite = CompositeParams(**features.encoder.params)
    dense = resolve_pipeline_dense(plan)
    cache_config = replace(
        features.cache,
        root_dir=resolve_cache_root(features.cache, feature_dir=output_root, output_root=plan.run.output_dir),
    )
    member_stores, member_norms = [], []
    try:
        for index, member in enumerate(composite.members):
            member_dense = member.dense_options(dense)
            member_norms.append(member_norm_for(member, member_dense))
            member_stores.append(
                extract(
                    images,
                    encoder=EncoderLeaf(name=member.name, params=dict(member.params)),
                    cache=cache_config,
                    unit="tile",
                    dense=member_dense,
                    execution=member.execution_options(features.execution),
                    output_dir=Path(output_root) / f"member_{index}_{member.name}",
                )
            )
    finally:
        _release_parent_cuda_state()
        log_cuda_memory("after composite dense extraction release")
    return CompositeDenseFeatureStore(
        member_stores,
        concat_resolution=composite.concat_resolution,
        concat_grid_size=composite.concat_grid_size,
        member_norms=member_norms,
    )
