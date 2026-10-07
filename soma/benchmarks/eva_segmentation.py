"""kaiko-ai/eva patch-segmentation benchmarks — ``eva/consep`` and ``eva/monusac`` (issue #522).

The two segmentation tasks of the `kaiko-ai/eva <https://github.com/kaiko-ai/eva>`_
pathology leaderboard join the ``eva`` family next to the tile-classification members of
:mod:`soma.benchmarks.eva`. They share that module's encoder table, reference ledger and
``soma reproduce eva`` fan-out, and run on soma's dense (``dataset_type="segmentation"``)
path with components that exist only to reproduce EVA's decoder protocol:

* ``eva_conv_ms`` — EVA's ``ConvDecoderMS``: ``Upsample(x2) -> Conv3x3(in, 64) ->
  Upsample(x2) -> Conv3x3(64, classes)``, no normalisation, no activation.
* ``eva_segmentation`` — a :class:`~soma.tasks.segmentation.SegmentationHead` whose loss is
  EVA's pure soft Dice (MONAI ``DiceLoss(softmax=True, batch=True)``, background included,
  no cross-entropy term) and whose confusion counts follow EVA's metric wrapper.

Both are registered under ``eva_`` names when this module is imported (``soma list tasks``
/ ``soma list decoders`` show them) but are benchmark-private: they are not general soma
components and are not documented as such.

Protocol points that matter for matching the leaderboard (offline segmentation configs
``configs/vision/pathology/offline/segmentation/{consep,monusac}.yaml``):

* The curators (:mod:`soma.curation.eva_segmentation`) materialise EVA's geometry: CoNSeP
  250 px grid tiles at native 0.25 µm/px resized to 224, MoNuSAC whole images resized on
  the short side to 224 and centre-cropped. soma trains on the flat 224 px PNGs.
* Dense features are the last block's patch-token grid taken **before** the backbone's
  final LayerNorm (timm ``features_only`` with ``norm=False``): slide2vec's
  ``patch_features_prenorm`` feature kind.
* AdamW ``lr=2e-3`` (torch's default ``weight_decay=0.01``) behind eva's default
  ``ConstantLR`` warm-up (lr/3 for the first five epochs), batch size 64, a fixed
  budget of ``max_steps=2000`` optimizer updates, no scheduler, best checkpoint on the
  validation foreground Dice, early-stopping patience 200 (CoNSeP) / 100 (MoNuSAC)
  validation rounds, five seeds.
* The reported metric is MONAI's ``DiceMetric(include_background=False)`` with empty
  targets skipped: soma's ``foreground_mean_dice``. Both datasets report on EVA's
  validation split, so the run uses ``tune_is_test=True`` and soma's ``test`` split *is*
  that split.
* MoNuSAC's test-only ``Ambiguous`` class (index 5) is the ``ignore_index``: dropped
  from the loss, and treated as EVA's metric wrapper treats it (see
  :func:`eva_confusion_counts`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from soma.benchmarks.eva import (
    CANONICAL_SEEDS,
    ENCODERS,
    DEFAULT_ENCODER,
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
    CacheConfig,
    DecoderConfig,
    EncoderConfig,
    EvalConfig,
    ExecutionConfig,
    PipelineConfig,
    PreprocessingConfig,
    TaskConfig,
    TrainingConfig,
)
from soma.curation.eva_segmentation import (
    CONSEP_CLASSES,
    MONUSAC_CLASSES,
    MONUSAC_IGNORE_INDEX,
    NOMINAL_SPACING_UM,
    OUTPUT_PX,
    curate_eva_segmentation_dataset,
)
from soma.curation.manifest import CuratedManifest
from soma.decoders.base import Decoder
from soma.decoders.registry import decoder_registry
from soma.tasks.dense_metrics import _validate_dense_shapes
from soma.tasks.registry import task_registry
from soma.tasks.segmentation import SegmentationHead

# --- Protocol constants (eva offline segmentation configs) ----------------------------
MAX_STEPS = 2000
BATCH_SIZE = 64
LEARNING_RATE = 2.0e-3
WEIGHT_DECAY = 0.01  # eva sets only lr; torch.optim.AdamW default weight_decay
# eva's ``SemanticSegmentationModule`` defaults ``lr_scheduler`` to torch ``ConstantLR``
# (factor 1/3 for 5 epochs, stepped per epoch by Lightning) and no segmentation config
# overrides it. Without this warm-up the first AdamW steps at 2e-3 on the unnormalised
# pre-norm grid (|x| up to ~700 for uni2) saturate the softmax and some seeds never recover.
WARMUP_EPOCHS = 5
WARMUP_FACTOR = 1 / 3
DECODER_HIDDEN_DIM = 64  # ConvDecoderMS's fixed intermediate width
#: MONAI DiceLoss smoothing constants (numerator / denominator).
DICE_SMOOTH = 1.0e-5

PRIMARY_METRIC = "test/foreground_mean_dice"
METRICS = ["foreground_mean_dice", "mean_dice", "mean_iou"]

DECODER_NAME = "eva_conv_ms"
HEAD_NAME = "eva_segmentation"
#: Dense feature tap: the pre-norm last-block patch grid (timm ``features_only``), as EVA.
FEATURE_KIND = "patch_features_prenorm"


# --- Benchmark-private components -----------------------------------------------------


class EvaConvMSDecoder(Decoder):
    """EVA's ``ConvDecoderMS`` (the DINOv2 ``+ms`` decoder), byte-faithful.

    ``(B, d, h, w) -> (B, C, 4h, 4w)``: nearest ×2 upsample, 3x3 conv to 64 channels,
    nearest ×2 upsample, 3x3 conv to the classes. No normalisation, no activation; the
    head resizes the logits to the mask size.
    """

    def __init__(self, *, input_dim: int, num_classes: int) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        self._num_classes = int(num_classes)
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(input_dim, DECODER_HIDDEN_DIM, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(DECODER_HIDDEN_DIM, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.layers(X)

    @property
    def num_classes(self) -> int:
        return self._num_classes


def eva_dice_loss(
    logits: Tensor, mask: Tensor, *, num_classes: int, ignore_index: int
) -> Tensor:
    """EVA's training objective: MONAI ``DiceLoss(softmax=True, batch=True)``.

    Softmax probabilities, one Dice per class over the whole batch (``batch=True``
    reduces the batch and spatial axes together), background included, smoothing
    ``1e-5`` on both the numerator and the denominator, ``1 - mean_c Dice_c``.

    ``ignore_index`` pixels are dropped from both the probabilities and the targets. EVA's
    wrapper instead zeroes the logits there (a uniform softmax) and maps the target to
    background; the two agree on every EVA training split, where the ignored class never
    occurs (MoNuSAC's ``Ambiguous`` is test-only).
    """
    _validate_dense_shapes(logits, mask, num_classes)
    probs = logits.softmax(dim=1)
    valid = (mask != ignore_index).unsqueeze(1).to(probs.dtype)
    safe_mask = mask.clone()
    safe_mask[mask == ignore_index] = 0
    onehot = nn.functional.one_hot(safe_mask, num_classes).permute(0, 3, 1, 2).to(probs.dtype)
    probs = probs * valid
    onehot = onehot * valid
    dims = (0, 2, 3)
    intersection = (probs * onehot).sum(dims)
    denominator = probs.sum(dims) + onehot.sum(dims)
    dice = (2.0 * intersection + DICE_SMOOTH) / (denominator + DICE_SMOOTH)
    return 1.0 - dice.mean()


def eva_confusion_counts(
    logits: Tensor, mask: Tensor, *, num_classes: int, ignore_index: int
) -> Tensor:
    """Per-image, per-class ``(intersection, pred_area, target_area)`` the EVA way.

    EVA's ``MonaiDiceScore`` wrapper removes the ignored class from the one-hot *targets*
    but masks the predictions by the ignored channel of the **predictions**, which a
    decoder with ``num_classes`` outputs never emits. Pixels labelled ``ignore_index``
    therefore contribute no target and no intersection, yet still count towards the
    predicted area of whatever class was predicted there. Matching that keeps the
    MoNuSAC Dice comparable with the leaderboard; without an ignored class the counts
    equal :func:`soma.tasks.dense_metrics.dense_confusion_counts`.
    """
    _validate_dense_shapes(logits, mask, num_classes)
    pred = logits.argmax(dim=1)
    counts = torch.zeros(logits.shape[0], num_classes, 3, dtype=torch.long, device=logits.device)
    for c in range(num_classes):
        pred_c = pred == c
        target_c = mask == c
        counts[:, c, 0] = (pred_c & target_c).sum(dim=(1, 2))
        counts[:, c, 1] = pred_c.sum(dim=(1, 2))
        counts[:, c, 2] = target_c.sum(dim=(1, 2))
    return counts


class EvaSegmentationHead(SegmentationHead):
    """``SegmentationHead`` with EVA's pure-Dice loss and EVA's confusion counting.

    Everything else — the geometry, the mask reading, the metric reduction (including
    ``foreground_mean_dice``) — is the built-in head's. The loss knobs the built-in head
    takes (``dice_weight``, ``class_weights``, Tversky and focal shapes) are accepted for
    construction parity but have no effect: the objective is fixed by the protocol.
    """

    def compute_loss(self, predictions: Tensor, targets: dict[str, Tensor]) -> Tensor:
        return eva_dice_loss(
            predictions,
            targets["mask"],
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

    def dense_stats(self, raw_output: Tensor, targets: dict[str, Tensor]) -> Tensor:
        return eva_confusion_counts(
            raw_output,
            targets["mask"],
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )


if DECODER_NAME not in decoder_registry:
    decoder_registry.register(
        DECODER_NAME,
        EvaConvMSDecoder,
        metadata={"description": "EVA ConvDecoderMS (benchmark-private, eva/consep + eva/monusac)"},
    )
if HEAD_NAME not in task_registry:
    task_registry.register(HEAD_NAME, EvaSegmentationHead)


# --- Datasets -------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentationDatasetSpec:
    """Per-dataset EVA segmentation protocol parameters."""

    class_names: tuple[str, ...]  # decoder classes, background first
    patience: int  # eva's per-dataset EarlyStopping patience (validation rounds)
    ignore_index: int | None  # mask value excluded from the loss (MoNuSAC's Ambiguous)


DATASETS: dict[str, SegmentationDatasetSpec] = {
    "consep": SegmentationDatasetSpec(CONSEP_CLASSES, 200, None),
    # The decoder predicts five classes; Ambiguous (5) exists only in the test masks.
    "monusac": SegmentationDatasetSpec(MONUSAC_CLASSES[:MONUSAC_IGNORE_INDEX], 100, MONUSAC_IGNORE_INDEX),
}


def _require_dataset(dataset: str) -> SegmentationDatasetSpec:
    try:
        return DATASETS[dataset]
    except KeyError as exc:
        raise ValueError(
            f"Unknown EVA segmentation dataset {dataset!r}. Supported: {', '.join(DATASETS)}"
        ) from exc


def _build_eva_segmentation_config(
    *,
    dataset: str,
    encoder: str,
    dataset_csv: str | Path,
    splits_csv: str | Path,
    output_root: str | Path,
    seed: int = 0,
    max_steps: int | None = None,
    patience: int | None = None,
    encoder_batch_size: int = 32,
    head_num_workers: int = 0,
    execution: ExecutionConfig | None = None,
    cache: CacheConfig | None = None,
) -> PipelineConfig:
    """Assemble an EVA-faithful segmentation :class:`~soma.config.PipelineConfig`.

    ``max_steps`` defaults to eva's fixed :data:`MAX_STEPS` budget; ``patience`` defaults
    to the dataset's eva value. Pass overrides for smoke runs.
    """
    spec = _require_dataset(dataset)
    if max_steps is None:
        max_steps = MAX_STEPS
    if patience is None:
        patience = spec.patience
    task_params: dict[str, Any] = {"num_classes": len(spec.class_names)}
    if spec.ignore_index is not None:
        task_params["ignore_index"] = spec.ignore_index

    return PipelineConfig(
        dataset_csv=str(dataset_csv),
        splits_csv=str(splits_csv),
        output_root=Path(output_root),
        dataset_type="segmentation",
        execution=execution or ExecutionConfig(),
        cache=cache or CacheConfig(enabled=True),
        # The curated PNGs declare NOMINAL_SPACING_UM; requesting the same value reads
        # EVA's 224 px samples pixel for pixel (no resampling).
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=OUTPUT_PX,
            requested_spacing_um=NOMINAL_SPACING_UM,
            # EVA builds its backbones with timm ``features_only=True, out_indices=1``:
            # the last block's patch grid *before* the final norm. Match that tap.
            feature_kind=FEATURE_KIND,
        ),
        encoder=EncoderConfig(
            name=encoder,
            # Dense extraction returns the patch-token grid; the pooled output variant
            # (virchow2's CLS-only pin for the tile benchmarks) does not apply.
            output_variant=None,
            batch_size=encoder_batch_size,
            # The tiles carry only a nominal spacing; the encoder's recommended regime is moot.
            allow_non_recommended_settings=True,
        ),
        decoder=DecoderConfig(name=DECODER_NAME),
        task=TaskConfig(name=HEAD_NAME, params=task_params),
        evaluation=EvalConfig(metrics=list(METRICS)),
        training=TrainingConfig(
            seed=seed,
            epochs=None,
            max_steps=max_steps,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            optimizer="adamw",
            scheduler="none",
            lr_warmup_epochs=WARMUP_EPOCHS,
            lr_warmup_factor=WARMUP_FACTOR,
            patience=patience,
            monitor="foreground_mean_dice",
            monitor_mode="max",
            batch_size=BATCH_SIZE,
            tune_is_test=True,
            num_workers=head_num_workers,
            pin_memory=False,
            persistent_workers=False,
        ),
        tags=["eva", dataset, encoder, "segmentation"],
    )


class EvaSegmentationBenchmark:
    """One EVA patch-segmentation dataset registered as ``eva/<dataset>`` (protocol-as-code).

    Fixes the dataset and EVA's decoder recipe, varies the ``encoder`` axis, and reads
    ``test/foreground_mean_dice`` from the run's ``summary.json``.
    """

    canonical_seeds = CANONICAL_SEEDS
    primary_metric = PRIMARY_METRIC
    reference_environment = REFERENCE_ENVIRONMENT

    def __init__(self, dataset: str) -> None:
        _require_dataset(dataset)
        self.dataset = dataset
        self.name = f"eva/{dataset}"
        self.facet = Facet(
            fixed={
                "dataset": dataset,
                "task": HEAD_NAME,
                "protocol": "eva-segmentation-decoder",
            },
            varied=("encoder",),
        )

    def curate(self, raw_root: str | Path, out_dir: str | Path) -> CuratedManifest:
        """Curate this dataset into 224 px tiles + class-index masks (delegates to the curator)."""
        return curate_eva_segmentation_dataset(self.dataset, raw_root, out_dir)

    def build_config(
        self,
        *,
        encoder: str = DEFAULT_ENCODER,
        dataset_csv: str | Path | None = None,
        splits_csv: str | Path | None = None,
        output_root: str | Path | None = None,
        seed: int | None = None,
        overrides: dict[str, Any] | None = None,
        max_steps: int | None = None,
        patience: int | None = None,
        encoder_batch_size: int = 32,
        execution: ExecutionConfig | None = None,
    ) -> PipelineConfig:
        """Build the EVA-faithful config for this dataset and the ``encoder`` axis."""
        return _build_eva_segmentation_config(
            dataset=self.dataset,
            encoder=encoder,
            dataset_csv=dataset_csv if dataset_csv is not None else "dataset.csv",
            splits_csv=splits_csv if splits_csv is not None else "splits.csv",
            output_root=output_root if output_root is not None else "output/eva",
            seed=0 if seed is None else int(seed),
            max_steps=max_steps,
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
        """DEFAULT scorer: read the run's ``summary.json`` (Dice per split)."""
        return score_from_summary(run_dir)


EVA_SEGMENTATION_BENCHMARKS: dict[str, EvaSegmentationBenchmark] = {}
for _dataset in DATASETS:
    _bench = EvaSegmentationBenchmark(_dataset)
    EVA_SEGMENTATION_BENCHMARKS[_bench.name] = _bench
    register_benchmark(_bench)

__all__ = [
    "ENCODERS",
    "EvaConvMSDecoder",
    "EvaSegmentationBenchmark",
    "EvaSegmentationHead",
    "EVA_SEGMENTATION_BENCHMARKS",
    "eva_confusion_counts",
    "eva_dice_loss",
]
