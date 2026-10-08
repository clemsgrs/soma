"""kaiko-ai/eva slide-level benchmarks — ``eva/camelyon16_small`` and ``eva/panda_small``.

The two slide-level classification tasks of the `kaiko-ai/eva
<https://github.com/kaiko-ai/eva>`_ pathology leaderboard join the ``eva`` family next to
the tile-classification (:mod:`soma.benchmarks.eva`) and segmentation
(:mod:`soma.benchmarks.eva_segmentation`) members, sharing their encoder table, reference
ledger and ``soma reproduce eva`` fan-out. They reproduce EVA's offline configs
``configs/vision/pathology/offline/classification/{camelyon16_small,panda_small}.yaml``
on soma's slide path (``dataset_type="slide"``) with components that exist only for this
protocol:

* ``eva_abmil`` — EVA's ``ABMIL`` bag encoder: ``Linear(D -> 128)`` projection, then
  gated attention pooling (``tanh`` x ``sigmoid``, hidden width 128) over the projected
  tiles. No dropout. Its 128-d bag vector feeds the head.
* ``eva_mil_binary`` / ``eva_mil_multiclass`` — EVA's ``ABMIL`` classifier MLP
  (``128 -> 128 -> 64 -> out``, ReLU, no dropout). The binary head emits one logit
  trained with ``BCEWithLogitsLoss`` and predicts positive when the logit is above 0
  (``sigmoid > 0.5``, torchmetrics' threshold); the multiclass head emits one logit per
  class trained with cross entropy.

They are registered under ``eva_`` names when this module is imported but are
benchmark-private: they are not general soma components and are not documented as such.

Protocol points that matter for matching the leaderboard:

* The curators (:mod:`soma.curation.eva_slide`) choose EVA's tiles: a shuffled level-0
  grid filtered by EVA's saturation mask, at most 1000 tiles at 0.25 µm/px
  (Camelyon16Small) or 200 at 0.5 µm/px (PANDASmall). They write them as hs2p tiling
  artifacts named by the manifest's ``coordinates_path``, so soma skips its own tiling.
* AdamW ``lr=1e-3`` (torch's default ``weight_decay=0.01``; EVA sets only ``lr``) behind
  EVA's default ``ConstantLR`` warm-up (lr/3 for the first five epochs), batch size 32
  (bags padded and masked, shuffled, last batch kept), 100 epochs (EVA's leaderboard
  quotes a 12,500-step cap that these configs never reach), early-stopping patience 20
  validation rounds, best checkpoint on the validation balanced accuracy, 20 seeds.
* Both datasets have a real validation split (soma ``tune``) and test split, so the run
  reports on soma's ``test`` split (``tune_is_test=False``): the leaderboard numbers are
  test balanced accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from soma.aggregators.base import Aggregator, AggregatorOutput
from soma.aggregators.mil.attention_pool import AttentionPool
from soma.aggregators.registry import aggregator_registry
from soma.benchmarks.eva import (
    DEFAULT_ENCODER,
    ENCODERS,
    REFERENCE_ENVIRONMENT,
    REFERENCE_NAME,
    _cache_from_overrides,
)
from soma.benchmarks.registry import (
    Facet,
    ReferenceRow,
    expected_rows,
    register_benchmark,
    score_from_summary,
)
from soma.config import (
    AggregatorConfig,
    CacheConfig,
    EncoderConfig,
    EvalConfig,
    ExecutionConfig,
    PipelineConfig,
    PreprocessingConfig,
    TaskConfig,
    TrainingConfig,
)
from soma.curation.eva_slide import (
    PANDA_NUM_CLASSES,
    SLIDE_DATASETS,
    TILE_PX,
    curate_eva_slide_dataset,
)
from soma.curation.manifest import CuratedManifest
from soma.evaluation.metrics import compute_metrics, resolve_metrics
from soma.tasks.base import TaskHead
from soma.tasks.registry import task_registry

if TYPE_CHECKING:
    from soma.dataset import Dataset, SampleRecord

# --- Protocol constants (eva offline slide-level classification configs) --------------
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 0.01  # eva sets only lr; torch.optim.AdamW default weight_decay
#: eva's ``HeadModule`` defaults ``lr_scheduler`` to torch ``ConstantLR`` (factor 1/3 for
#: five epochs, stepped per epoch); neither config overrides it.
WARMUP_EPOCHS = 5
WARMUP_FACTOR = 1 / 3
PATIENCE = 20
#: eva runs each slide-level head over twenty seeds (``N_RUNS``) and averages.
CANONICAL_SEEDS: tuple[int, ...] = tuple(range(20))
PRIMARY_METRIC = "test/balanced_accuracy"
METRICS = ["balanced_accuracy", "accuracy"]

AGGREGATOR_NAME = "eva_abmil"
BINARY_HEAD_NAME = "eva_mil_binary"
MULTICLASS_HEAD_NAME = "eva_mil_multiclass"
#: ``ABMIL(projected_input_size=128, hidden_size_attention=128, hidden_sizes_mlp=(128, 64))``.
PROJECTED_DIM = 128
ATTENTION_HIDDEN_DIM = 128
MLP_HIDDEN_DIMS: tuple[int, ...] = (128, 64)


# --- Benchmark-private components -----------------------------------------------------


class EvaABMIL(Aggregator):
    """EVA's ``ABMIL`` bag encoder: ``Linear(D -> 128)``, then gated attention pooling.

    ``(B, N, D) -> (B, 128)``. Padded tiles are zeroed before the projection (as EVA does)
    and get zero attention weight. The attention logits are returned as
    ``tile_attention``. EVA's final attention layer also has a bias, which shifts every
    logit equally and so leaves the pooling unchanged; soma's :class:`AttentionPool`
    omits it.
    """

    def __init__(
        self,
        input_dim: int,
        projected_dim: int = PROJECTED_DIM,
        hidden_dim: int = ATTENTION_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self._output_dim = int(projected_dim)
        self.projector = nn.Linear(input_dim, projected_dim, bias=True)
        self.pool = AttentionPool(
            input_dim=projected_dim, hidden_dim=hidden_dim, activation="tanh", gated=True
        )

    def forward(self, X: Tensor, mask: Tensor | None = None) -> AggregatorOutput:
        if mask is not None:
            X = X.masked_fill(~mask.unsqueeze(-1), 0)
        projected = self.projector(X)
        z, attention_logits = self.pool(projected, mask=mask)
        return AggregatorOutput(bag_representation=z, tile_attention=attention_logits)

    @property
    def output_dim(self) -> int:
        return self._output_dim


def _eva_mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    """EVA's ``MLP(hidden_layer_sizes=(128, 64), hidden_activation_fn=ReLU)``, no dropout."""
    layers: list[nn.Module] = []
    previous = input_dim
    for size in MLP_HIDDEN_DIMS:
        layers += [nn.Linear(previous, size), nn.ReLU()]
        previous = size
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


def _class_index(record: "SampleRecord", num_classes: int) -> dict[str, int]:
    """The curated ``label`` is EVA's class index; refuse anything outside the head."""
    label = int(record.label)
    if not 0 <= label < num_classes:
        raise ValueError(
            f"Sample {record.sample_id!r} has label {record.label!r}; the EVA head expects "
            f"a class index in [0, {num_classes})."
        )
    return {"label": label}


class EvaMILBinaryHead(TaskHead):
    """EVA's ``ABMIL`` classifier for a binary task: one logit, ``BCEWithLogitsLoss``.

    The label is the class index (0 or 1). Predictions are positive when the logit is
    above 0 (torchmetrics' ``sigmoid > 0.5``), and probabilities are reported as
    ``[1 - p, p]`` so soma's binary metrics and prediction files read them as usual.
    """

    target_dtypes = {"label": torch.long}
    task_family = "binary_classification"

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        metrics: list[str] | None = None,
    ) -> None:
        super().__init__()
        if num_classes != 2:
            raise ValueError(f"{type(self).__name__} requires num_classes=2, got {num_classes}.")
        self.num_classes = 2
        self.mlp = _eva_mlp(input_dim, 1)
        self.metrics = resolve_metrics("binary_classification", metrics or [])

    def extract_targets(self, record: "SampleRecord") -> dict[str, int]:
        return _class_index(record, self.num_classes)

    def forward(self, X: Tensor) -> Tensor:
        return self.mlp(X)

    def compute_loss(self, predictions: Tensor, targets: dict[str, Tensor]) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            predictions.squeeze(-1), targets["label"].to(predictions.dtype)
        )

    def _probabilities(self, raw_output: Tensor) -> tuple[Any, Any]:
        logit = raw_output.detach().squeeze(-1).float().cpu()
        positive = torch.sigmoid(logit)
        probabilities = torch.stack([1.0 - positive, positive], dim=1).numpy()
        return probabilities, (logit > 0).long().numpy()

    def postprocess(self, raw_output: Tensor) -> dict[str, Any]:
        probabilities, predicted = self._probabilities(raw_output)
        return {"probabilities": probabilities, "predicted_labels": predicted}

    def compute_metrics(self, raw_output: Tensor, targets: dict[str, Tensor]) -> dict[str, float]:
        probabilities, predicted = self._probabilities(raw_output)
        y_true = targets["label"].detach().cpu().numpy()
        return compute_metrics(
            "binary_classification", self.metrics, y_true, predicted, y_prob=probabilities
        )


class EvaMILMulticlassHead(TaskHead):
    """EVA's ``ABMIL`` classifier for a multiclass task: one logit per class, cross entropy."""

    target_dtypes = {"label": torch.long}
    task_family = "multiclass_classification"

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        metrics: list[str] | None = None,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"{type(self).__name__} requires num_classes >= 2, got {num_classes}.")
        self.num_classes = int(num_classes)
        self.mlp = _eva_mlp(input_dim, self.num_classes)
        self.metrics = resolve_metrics("multiclass_classification", metrics or [])

    @classmethod
    def auto_params(cls, dataset: "Dataset") -> dict[str, Any]:
        # The labels are class indices; the benchmark config pins num_classes anyway.
        return {"num_classes": max(int(r.label) for r in dataset.samples.values()) + 1}

    def extract_targets(self, record: "SampleRecord") -> dict[str, int]:
        return _class_index(record, self.num_classes)

    def forward(self, X: Tensor) -> Tensor:
        return self.mlp(X)

    def compute_loss(self, predictions: Tensor, targets: dict[str, Tensor]) -> Tensor:
        return F.cross_entropy(predictions, targets["label"])

    def postprocess(self, raw_output: Tensor) -> dict[str, Any]:
        probabilities = torch.softmax(raw_output.detach().float(), dim=1).cpu().numpy()
        return {"probabilities": probabilities, "predicted_labels": probabilities.argmax(axis=1)}

    def compute_metrics(self, raw_output: Tensor, targets: dict[str, Tensor]) -> dict[str, float]:
        processed = self.postprocess(raw_output)
        y_true = targets["label"].detach().cpu().numpy()
        return compute_metrics(
            "multiclass_classification",
            self.metrics,
            y_true,
            processed["predicted_labels"],
            y_prob=processed["probabilities"],
        )


if AGGREGATOR_NAME not in aggregator_registry:
    aggregator_registry.register(
        AGGREGATOR_NAME,
        EvaABMIL,
        metadata={
            "description": "EVA ABMIL bag encoder: Linear(D->128) + gated attention "
            "(benchmark-private, eva/camelyon16_small + eva/panda_small)"
        },
    )
if BINARY_HEAD_NAME not in task_registry:
    task_registry.register(BINARY_HEAD_NAME, EvaMILBinaryHead)
if MULTICLASS_HEAD_NAME not in task_registry:
    task_registry.register(MULTICLASS_HEAD_NAME, EvaMILMulticlassHead)


# --- Datasets -------------------------------------------------------------------------


@dataclass(frozen=True)
class SlideBenchmarkSpec:
    """Per-dataset EVA slide-level protocol parameters."""

    task: str  # benchmark-private head
    num_classes: int
    target_mpp: float
    max_tiles: int  # EVA's N_PATCHES: the sampler cap (bags are never larger)


DATASETS: dict[str, SlideBenchmarkSpec] = {
    "camelyon16_small": SlideBenchmarkSpec(
        BINARY_HEAD_NAME,
        2,
        SLIDE_DATASETS["camelyon16_small"].target_mpp,
        SLIDE_DATASETS["camelyon16_small"].max_samples,
    ),
    "panda_small": SlideBenchmarkSpec(
        MULTICLASS_HEAD_NAME,
        PANDA_NUM_CLASSES,
        SLIDE_DATASETS["panda_small"].target_mpp,
        SLIDE_DATASETS["panda_small"].max_samples,
    ),
}


def _require_dataset(dataset: str) -> SlideBenchmarkSpec:
    try:
        return DATASETS[dataset]
    except KeyError as exc:
        raise ValueError(
            f"Unknown EVA slide dataset {dataset!r}. Supported: {', '.join(DATASETS)}"
        ) from exc


def _build_eva_slide_config(
    *,
    dataset: str,
    encoder: str,
    dataset_csv: str | Path,
    splits_csv: str | Path,
    output_root: str | Path,
    seed: int = 0,
    epochs: int | None = None,
    patience: int | None = None,
    encoder_batch_size: int = 32,
    head_num_workers: int = 0,
    execution: ExecutionConfig | None = None,
    cache: CacheConfig | None = None,
) -> PipelineConfig:
    """Assemble an EVA-faithful slide-level :class:`~soma.config.PipelineConfig`.

    ``epochs`` defaults to EVA's :data:`EPOCHS` and ``patience`` to :data:`PATIENCE`.
    Pass overrides for smoke runs.
    """
    spec = _require_dataset(dataset)
    enc = ENCODERS.get(encoder)
    task_params: dict[str, Any] = {}
    if spec.task == MULTICLASS_HEAD_NAME:
        task_params["num_classes"] = spec.num_classes
    return PipelineConfig(
        dataset_csv=str(dataset_csv),
        splits_csv=str(splits_csv),
        output_root=Path(output_root),
        dataset_type="slide",
        execution=execution or ExecutionConfig(),
        cache=cache or CacheConfig(enabled=True),
        # The curated manifest supplies every slide's tiles (coordinates_path); these are
        # the geometry the artifacts were sampled at, which staging checks.
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=TILE_PX,
            requested_spacing_um=spec.target_mpp,
            # Required by the configuration but unused: supplied coordinates skip tissue
            # segmentation (the curator applied EVA's saturation mask).
            tissue_method="otsu",
        ),
        encoder=EncoderConfig(
            name=encoder,
            output_variant=enc.output_variant if enc is not None else None,
            batch_size=encoder_batch_size,
            # The spacing is EVA's, not the encoder's recommended regime.
            allow_non_recommended_settings=True,
        ),
        aggregator=AggregatorConfig(name=AGGREGATOR_NAME),
        task=TaskConfig(name=spec.task, params=task_params),
        evaluation=EvalConfig(metrics=list(METRICS)),
        training=TrainingConfig(
            seed=seed,
            epochs=EPOCHS if epochs is None else epochs,
            max_steps=None,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            optimizer="adamw",
            scheduler="none",
            lr_warmup_epochs=WARMUP_EPOCHS,
            lr_warmup_factor=WARMUP_FACTOR,
            patience=PATIENCE if patience is None else patience,
            monitor="balanced_accuracy",
            monitor_mode="max",
            batch_size=BATCH_SIZE,
            tune_is_test=False,
            num_workers=head_num_workers,
            pin_memory=False,
            persistent_workers=False,
        ),
        tags=["eva", dataset, encoder, "slide"],
    )


class EvaSlideClassificationBenchmark:
    """One EVA slide-level dataset registered as ``eva/<dataset>`` (protocol-as-code).

    Fixes the dataset and EVA's ABMIL recipe, varies the ``encoder`` axis, and reads
    ``test/balanced_accuracy`` from the run's ``summary.json``.
    """

    canonical_seeds = CANONICAL_SEEDS
    primary_metric = PRIMARY_METRIC
    reference_environment = REFERENCE_ENVIRONMENT

    def __init__(self, dataset: str) -> None:
        spec = _require_dataset(dataset)
        self.dataset = dataset
        self.name = f"eva/{dataset}"
        self.facet = Facet(
            fixed={"dataset": dataset, "task": spec.task, "protocol": "eva-slide-abmil"},
            varied=("encoder",),
        )

    def curate(self, raw_root: str | Path, out_dir: str | Path) -> CuratedManifest:
        """Sample EVA's tiles per slide and write the slide manifest (delegates to the curator)."""
        return curate_eva_slide_dataset(self.dataset, raw_root, out_dir)

    def build_config(
        self,
        *,
        encoder: str = DEFAULT_ENCODER,
        dataset_csv: str | Path | None = None,
        splits_csv: str | Path | None = None,
        output_root: str | Path | None = None,
        seed: int | None = None,
        overrides: dict[str, Any] | None = None,
        epochs: int | None = None,
        patience: int | None = None,
        encoder_batch_size: int = 32,
        execution: ExecutionConfig | None = None,
    ) -> PipelineConfig:
        """Build the EVA-faithful config for this dataset and the ``encoder`` axis."""
        return _build_eva_slide_config(
            dataset=self.dataset,
            encoder=encoder,
            dataset_csv=dataset_csv if dataset_csv is not None else "dataset.csv",
            splits_csv=splits_csv if splits_csv is not None else "splits.csv",
            output_root=output_root if output_root is not None else "output/eva",
            seed=0 if seed is None else int(seed),
            epochs=epochs,
            patience=patience,
            encoder_batch_size=encoder_batch_size,
            execution=execution,
            cache=_cache_from_overrides(overrides),
        )

    def expected(self, **axes: Any) -> list[ReferenceRow]:
        """Keyed reference row(s) for this dataset × the resolved encoder axis."""
        merged: dict[str, Any] = {"dataset": self.dataset, "encoder": DEFAULT_ENCODER}
        merged.update({k: v for k, v in axes.items() if v is not None})
        return expected_rows(REFERENCE_NAME, **merged)

    def score(self, run_dir: str | Path) -> dict[str, float]:
        """DEFAULT scorer: read the run's ``summary.json`` (balanced accuracy per split)."""
        return score_from_summary(run_dir)


EVA_SLIDE_BENCHMARKS: dict[str, EvaSlideClassificationBenchmark] = {}
for _dataset in DATASETS:
    _bench = EvaSlideClassificationBenchmark(_dataset)
    EVA_SLIDE_BENCHMARKS[_bench.name] = _bench
    register_benchmark(_bench)

__all__ = [
    "EVA_SLIDE_BENCHMARKS",
    "EvaABMIL",
    "EvaMILBinaryHead",
    "EvaMILMulticlassHead",
    "EvaSlideClassificationBenchmark",
]
