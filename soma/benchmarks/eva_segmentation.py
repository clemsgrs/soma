"""kaiko-ai/eva patch-segmentation benchmarks — ``eva/consep`` and ``eva/monusac`` (issue #522).

The two segmentation tasks of the `kaiko-ai/eva <https://github.com/kaiko-ai/eva>`_
pathology leaderboard join the ``eva`` family next to the tile-classification members of
:mod:`soma.benchmarks.eva`. They share that module's encoder table, reference ledger and
``soma reproduce eva`` fan-out, and run on soma's dense (grid) segmentation
path, starting from the :func:`~soma.presets.segmentation` preset, with components that
exist only to reproduce EVA's decoder protocol:

* ``eva_conv_with_image`` — EVA's online ``ConvDecoderWithImage``, the leaderboard
  decoder: it concatenates the ImageNet-normalised input tile to the upsampled grid, so
  it rides soma's image-prior decoder seam (``Decoder.consumes_image``).
* ``eva_conv_ms`` — EVA's offline ``ConvDecoderMS``: ``Upsample(x2) -> Conv3x3(in, 64)
  -> Upsample(x2) -> Conv3x3(64, classes)``, no normalisation, no activation. Kept for
  reference; EVA publishes no numbers for it.
* ``eva_segmentation`` — a :class:`~soma.tasks.segmentation.SegmentationHead` whose loss is
  EVA's pure soft Dice (MONAI ``DiceLoss(softmax=True, batch=True)``, background included,
  no cross-entropy term) and whose confusion counts follow EVA's metric wrapper.

Both are registered under ``eva_`` names when this module is imported (``soma list tasks``
/ ``soma list decoders`` show them) but are benchmark-private: they are not general soma
components and are not documented as such.

Protocol points that matter for matching the leaderboard (online segmentation configs
``configs/vision/pathology/online/segmentation/{consep,monusac}.yaml``, which EVA's
``replicate_evaluations`` guide prescribes for segmentation):

* The curators (:mod:`soma.curation.eva_segmentation`) materialise EVA's geometry: CoNSeP
  250 px grid tiles at native 0.25 µm/px resized to 224 (deterministic, so the run reads
  cached grids), MoNuSAC test images resized on the short side to 224 and centre-cropped.
  MoNuSAC train images stay whole: EVA draws a ``RandomResizedCrop(224)`` from them every
  step, so that run re-encodes live (``feature_mode='live'``) with
  ``augmentation: {name: segmentation, params: {crop_scale: [0.08, 1.0]}}``.
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
  :func:`eva_confusion_counts`) both in checkpoint selection and in the reported test
  Dice, which :meth:`EvaSegmentationBenchmark.score` re-scores from the run's saved
  label maps (the run's ``summary.json`` holds soma's standard counts).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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
from dataclasses import replace as _replace

from hs2p import TilingConfig
from slide2vec import DenseOptions, ExecutionOptions, PreprocessingConfig

from soma import presets
from soma.config import (
    CacheOptions,
    CohortConfig,
    Component,
    EncoderLeaf,
    EvaluationConfig,
    ExperimentConfig,
    LoadersConfig,
    RunConfig,
    TrainerConfig,
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
from soma.params import NoParams
from soma.tasks.dense_metrics import _validate_dense_shapes
from soma.tasks.registry import task_registry
from soma.tasks.segmentation import SegmentationHead, SegmentationParams

# --- Protocol constants (eva online segmentation configs) -----------------------------
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

DECODER_NAME = "eva_conv_with_image"
OFFLINE_DECODER_NAME = "eva_conv_ms"
# eva's online configs normalise every backbone's input with these ImageNet statistics
# (``NORMALIZE_MEAN`` / ``NORMALIZE_STD`` defaults); the decoder sees the same tensor.
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
IMAGE_HIDDEN_DIMS = (64, 32, 32)  # ConvDecoderWithImage's fixed widths
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


class Conv2dBnReLU(nn.Sequential):
    """``Conv2d(3x3, pad 1) -> BatchNorm2d -> ReLU``, eva's decoder block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class EvaConvWithImageDecoder(Decoder):
    """EVA's ``ConvDecoderWithImage`` (the leaderboard's online decoder), byte-faithful.

    ``(B, d, h, w) + (B, 3, H, W) -> (B, C, H, W)``: nearest x2 upsample, 3x3 conv-BN-ReLU
    to 64 channels, bilinear resize to the image size, concatenation with the normalised
    RGB image, two 3x3 conv-BN-ReLU blocks (32, 32), and a 1x1 classifier. The pixels
    arrive in ``[0, 1]`` and are normalised here with the ImageNet statistics eva feeds
    the backbone, so the decoder sees exactly eva's ``DecoderInputs.images``.
    """

    consumes_image = True

    def __init__(self, *, input_dim: int, num_classes: int) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        self._num_classes = int(num_classes)
        first, second, third = IMAGE_HIDDEN_DIMS
        self.layers = nn.Sequential(nn.Upsample(scale_factor=2), Conv2dBnReLU(input_dim, first))
        self.image_block = nn.Sequential(Conv2dBnReLU(first + 3, second), Conv2dBnReLU(second, third))
        self.classifier = nn.Conv2d(third, num_classes, kernel_size=1)
        self.register_buffer("_mean", torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(IMAGE_STD).view(1, 3, 1, 1))

    def forward(self, X: Tensor, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3 or image.shape[0] != X.shape[0]:
            raise ValueError(
                f"{type(self).__name__} expects an RGB image batch (B, 3, H, W) matching the "
                f"grid batch; got {tuple(image.shape)} for grid {tuple(X.shape)}"
            )
        features = nn.functional.interpolate(
            self.layers(X), size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        normalised = (image.to(features.dtype) - self._mean) / self._std
        return self.classifier(self.image_block(torch.cat([features, normalised], dim=1)))

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


def eva_label_map_counts(
    prediction: np.ndarray, target: Any, *, num_classes: int, ignore_index: int
) -> np.ndarray:
    """:func:`eva_confusion_counts` over one saved ``(H, W)`` label map: ``(C, 3)`` counts.

    soma's standard :func:`~soma.evaluation.grid.label_map_counts`, plus the predicted
    area on ``ignore_index`` target pixels, which EVA's metric wrapper still counts.
    """
    from soma.evaluation.grid import label_map_counts

    target = target.detach().cpu().numpy() if torch.is_tensor(target) else np.asarray(target)
    prediction = np.asarray(prediction).astype(np.int64)
    counts = label_map_counts(prediction, target, num_classes=num_classes, ignore_index=ignore_index)
    on_ignored = prediction[target == int(ignore_index)]
    outside = (on_ignored < 0) | (on_ignored >= num_classes)
    if outside.any():
        raise ValueError(
            f"predicted label value(s) {sorted(np.unique(on_ignored[outside]).tolist())} on "
            f"ignored pixels lie outside [0, num_classes={num_classes})"
        )
    counts[:, 1] += np.bincount(on_ignored, minlength=num_classes)
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


if OFFLINE_DECODER_NAME not in decoder_registry:
    decoder_registry.register(
        OFFLINE_DECODER_NAME,
        EvaConvMSDecoder,
        params=NoParams,
        metadata={"description": "EVA ConvDecoderMS, the offline-config decoder (benchmark-private)"},
    )
if DECODER_NAME not in decoder_registry:
    decoder_registry.register(
        DECODER_NAME,
        EvaConvWithImageDecoder,
        params=NoParams,
        metadata={
            "description": "EVA ConvDecoderWithImage, the leaderboard's online decoder "
            "(benchmark-private, eva/consep + eva/monusac)"
        },
    )
if HEAD_NAME not in task_registry:
    task_registry.register(HEAD_NAME, EvaSegmentationHead, params=SegmentationParams)


# --- Datasets -------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentationDatasetSpec:
    """Per-dataset EVA segmentation protocol parameters."""

    class_names: tuple[str, ...]  # decoder classes, background first
    patience: int  # eva's per-dataset EarlyStopping patience (validation rounds)
    ignore_index: int | None  # mask value excluded from the loss (MoNuSAC's Ambiguous)
    #: eva's online train ``RandomResizedCrop`` area scale; ``None`` = deterministic
    #: transforms, so the cached grids are eva's samples.
    train_crop_scale: tuple[float, float] | None = None


DATASETS: dict[str, SegmentationDatasetSpec] = {
    "consep": SegmentationDatasetSpec(CONSEP_CLASSES, 200, None),
    # The decoder predicts five classes; Ambiguous (5) exists only in the test masks.
    # Online MoNuSAC trains on ``RandomResizedCrop(224)`` (torchvision's default scale) of
    # the whole image every step, so the run re-encodes live.
    "monusac": SegmentationDatasetSpec(
        MONUSAC_CLASSES[:MONUSAC_IGNORE_INDEX], 100, MONUSAC_IGNORE_INDEX, (0.08, 1.0)
    ),
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
    execution: ExecutionOptions | None = None,
    cache: CacheOptions | None = None,
) -> ExperimentConfig:
    """Assemble an EVA-faithful segmentation :class:`~soma.config.ExperimentConfig`.

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

    live = spec.train_crop_scale is not None
    monitor = {"monitor": "foreground_mean_dice", "mode": "max"}
    cfg = presets.segmentation(encoder, DECODER_NAME)
    cfg.run = RunConfig(output_dir=Path(output_root), seed=seed)
    cfg.cohort = CohortConfig(records=str(dataset_csv), splits=str(splits_csv), tune_is_test=True)
    cfg.features.kind = "live" if live else "cached"
    # Dense extraction returns the patch-token grid, so no pooled output variant applies;
    # the tiles carry only a nominal spacing, so the encoder's recommended regime is moot.
    cfg.features.encoder = EncoderLeaf(name=encoder, params={"allow_non_recommended_settings": True})
    cfg.features.extraction = PreprocessingConfig(
        tiling=TilingConfig(
            requested_spacing_um=NOMINAL_SPACING_UM,
            requested_tile_size_px=OUTPUT_PX,
            tolerance=0.05,
            overlap=0.0,
        ),
    )
    # The curated PNGs declare NOMINAL_SPACING_UM; requesting the same value reads EVA's
    # 224 px samples pixel for pixel (no resampling). EVA builds its backbones with timm
    # ``features_only=True, out_indices=1``: the last block's patch grid *before* the final
    # norm. Match that tap.
    cfg.features.dense = DenseOptions(
        spacing_um=NOMINAL_SPACING_UM,
        target_size=OUTPUT_PX,
        feature_kind=FEATURE_KIND,
    )
    cfg.features.execution = _replace(execution or ExecutionOptions(), batch_size=encoder_batch_size)
    cfg.features.cache = cache or CacheOptions(enabled=True)
    cfg.loaders = LoadersConfig(
        batch_size=BATCH_SIZE,
        num_workers=head_num_workers,
        augment=Component("segmentation", {"crop_scale": list(spec.train_crop_scale)}) if live else None,
    )
    cfg.learner.params["head"] = Component(HEAD_NAME, task_params)
    cfg.learner.params["optimizer"] = Component("adamw", {"lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY})
    # EVA's warm-up: torch ConstantLR (factor 1/3 for 5 epochs) and no main schedule.
    cfg.learner.params["scheduler"] = [Component("constant", {"factor": WARMUP_FACTOR, "total_iters": WARMUP_EPOCHS})]
    cfg.trainer = TrainerConfig(
        max_steps=max_steps,
        callbacks=[
            Component("checkpoint", monitor),
            Component("early_stopping", {**monitor, "patience": patience}),
        ],
    )
    cfg.evaluation = EvaluationConfig(metrics=list(METRICS))
    return cfg


class EvaSegmentationBenchmark:
    """One EVA patch-segmentation dataset registered as ``eva/<dataset>`` (protocol-as-code).

    Fixes the dataset and EVA's decoder recipe, varies the ``encoder`` axis, and scores
    ``test/foreground_mean_dice`` with EVA's counting (see :meth:`score`).
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
        execution: ExecutionOptions | None = None,
    ) -> ExperimentConfig:
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
        """The run's test-split metrics, counted the EVA way.

        With an ignored class (MoNuSAC's ``Ambiguous``) soma's ``summary.json`` holds the
        standard counts, which drop ignored pixels from the predicted area; EVA keeps
        them there. This re-scores the run's saved ``predictions_<split>/`` label maps
        with :func:`eva_label_map_counts` and the registry reducers. Without one (CoNSeP)
        the two countings agree, so the summary is read as is.
        """
        if DATASETS[self.dataset].ignore_index is None:
            return score_from_summary(run_dir)
        return _eva_rescore(run_dir)


def _eva_rescore(run_dir: str | Path) -> dict[str, float]:
    """``{<split>/<metric>: value}`` of a finished single-fold run's saved test label maps."""
    from soma.benchmarks.ocelot import _locate_run_config
    from soma.config import ExperimentConfig
    from soma.data import AnnotationManifest, ImageManifest, LabelMapSource
    from soma.data.targets import resolve_class_scheme
    from soma.evaluation import LabelMapPredictions
    from soma.evaluation.metrics import metric_registry, resolve_metrics

    config_path = _locate_run_config(Path(run_dir))
    folder = config_path.parent
    cfg = ExperimentConfig.from_yaml(config_path)
    split_dirs = sorted(
        path
        for path in folder.glob("predictions_*")
        if path.is_dir() and path.name != "predictions_tune"
    )
    if not split_dirs:
        raise FileNotFoundError(
            f"no predictions_<split>/ label maps in {folder}; EVA scoring re-counts the run's "
            "saved test predictions"
        )
    _, _, label_remap = resolve_class_scheme(cfg.learner.params["head"].params, annotation_rasters=False)
    from soma.preprocessing.resolution import resolve_pipeline_dense

    dense = resolve_pipeline_dense(cfg)
    tiling = cfg.features.extraction.tiling
    annotations = AnnotationManifest.from_csv(cfg.cohort.records)
    images = ImageManifest.from_csv(cfg.cohort.records)
    metrics = resolve_metrics("segmentation", cfg.evaluation.metrics)
    scores: dict[str, float] = {}
    for split_dir in split_dirs:
        predictions = LabelMapPredictions.load(split_dir)
        if not predictions.sample_ids:
            continue
        num_classes = predictions.num_classes
        targets = LabelMapSource.from_manifests(
            annotations,
            images,
            size=predictions[predictions.sample_ids[0]].shape,
            label_remap=label_remap,
            spacing_um=dense.spacing_um,
            spacing_policy=cfg.features.spacing_policy,
            tolerance=float(dense.tolerance),
            backend=tiling.mask_backend if tiling is not None else "auto",
            image_backend=dense.backend,
        )
        rows = []
        for sample_id in predictions.sample_ids:
            prediction, target = predictions[sample_id], targets.load(sample_id)
            if tuple(prediction.shape) != tuple(target.shape):
                raise ValueError(
                    f"label map for '{sample_id}' is {tuple(prediction.shape)} but its target "
                    f"is {tuple(target.shape)}"
                )
            rows.append(
                eva_label_map_counts(
                    prediction, target, num_classes=num_classes, ignore_index=predictions.ignore_index
                )
            )
        counts = torch.from_numpy(np.stack(rows)).to(torch.long)
        split = split_dir.name.removeprefix("predictions_")
        for name in metrics:
            value = metric_registry.get(name).fn(counts, num_classes=num_classes)
            if isinstance(value, Mapping):
                scores.update({f"{split}/{key}": float(v) for key, v in value.items()})
            else:
                scores[f"{split}/{name}"] = float(value)
    return scores


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
    "eva_label_map_counts",
]
