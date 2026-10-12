"""Extension seam: a user-registered task head, decoder and learner train through the real CLI.

The customisation API is subclass-and-override: subclass the family's built-in head (or
``Decoder``, or the shape's ``Learner``), override the methods that should differ, and
reference the subclass from the config, either under a name registered with its params
dataclass (and, for a head whose class count the cohort decides, the family's ``infer``
hook) or by its dotted import path, which resolves with a warning and passes its params
through unvalidated (ADR 0018). Validation keys on the registered class's
``task_family``, so the custom head passes everywhere the built-in one does. A custom set
head passes the ``soma.testing.check_task_head`` conformance suite, and a learner's
``predict_step`` override adds a column to the run's predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import soma
from soma.decoders.base import Decoder
from soma.decoders.registry import decoder_registry
from soma.evaluation import Predictions
from soma.learn import DenseLearner, MILLearner, PredictOutput, learner_registry
from soma.tasks import ClassificationParams, SegmentationParams, infer_label_map
from soma.tasks.classification import BinaryClassificationHead
from soma.tasks.dense_metrics import soft_dice_loss
from soma.tasks.registry import task_registry
from soma.tasks.segmentation import SegmentationHead
from soma.testing import check_task_head
from soma.training.sample_dataset import SampleBatch
from tests.e2e.harness import CPU_EXECUTION, CPU_LOADER, run_soma
from tests.e2e.synthetic import SPACING_UM, make_dense_dataset, make_slide_cohort

CUSTOM_HEAD = "e2e_dice_only_segmentation"
CUSTOM_DECODER = "e2e_two_stage_conv"
CUSTOM_SET_HEAD = "e2e_label_smoothing_classification"

# Call counters: the scenario proves the overrides actually ran, not just registered.
CALLS = {"loss": 0, "decoder": 0, "entropy": 0, "smoothing": 0, "predict": 0, "smooth_grid": 0}


class AttentionEntropyLearner(MILLearner):
    """``MILLearner`` plus an attention-entropy regulariser: one ``training_step`` override.

    Referenced from the config by dotted path (``training.learner``), so nothing is
    registered; ``weight`` arrives through the leaf's ``params``.
    """

    def __init__(self, model, *, weight: float, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.weight = float(weight)

    def training_step(self, batch) -> Tensor:
        output = self.forward_batch(batch)
        attention = output.tile_attention.masked_fill(~batch.mask, float("-inf")).softmax(-1)
        entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1).mean()
        CALLS["entropy"] += 1
        return self.compute_loss(output, batch) + self.weight * entropy

    def predict_step(self, batch) -> PredictOutput:
        """The head's columns plus each bag's attention entropy (one extra column)."""
        step = super().predict_step(batch)
        attention = step.raw.tile_attention.masked_fill(~batch.mask, float("-inf")).softmax(-1)
        entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1)
        CALLS["predict"] += 1
        return PredictOutput(columns={**step.columns, "attention_entropy": entropy}, raw=step.raw)


class SmoothGridLearner(DenseLearner):
    """``DenseLearner`` plus a total-variation penalty on the predicted class probabilities:
    one ``training_step`` override on the grid shape, named by dotted path in the config."""

    def __init__(self, model, *, weight: float, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.weight = float(weight)

    def training_step(self, batch) -> Tensor:
        output = self.forward_batch(batch)
        probabilities = output.logits.softmax(dim=1)
        variation = (probabilities[..., 1:, :] - probabilities[..., :-1, :]).abs().mean() + (
            probabilities[..., 1:] - probabilities[..., :-1]
        ).abs().mean()
        CALLS["smooth_grid"] += 1
        return self.compute_loss(output, batch) + self.weight * variation


class LabelSmoothingHead(BinaryClassificationHead):
    """``BinaryClassificationHead`` trained with label-smoothed cross-entropy."""

    def compute_loss(self, predictions: Tensor, targets: dict[str, Tensor]) -> Tensor:
        CALLS["smoothing"] += 1
        return F.cross_entropy(predictions, targets["label"], label_smoothing=0.1)


class DiceOnlySegmentationHead(SegmentationHead):
    """``SegmentationHead`` with the cross-entropy term dropped: pure soft Dice."""

    def compute_loss(self, predictions: Tensor, targets: dict[str, Tensor]) -> Tensor:
        CALLS["loss"] += 1
        return soft_dice_loss(
            predictions,
            targets["mask"],
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )


@dataclass
class TwoStageConvParams:
    """``TwoStageConvDecoder``'s params: registering them validates a config's values."""

    hidden_dim: int = 16

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {self.hidden_dim}")


class TwoStageConvDecoder(Decoder):
    """Upsample, 3x3 conv, upsample, 3x3 conv to class logits — no norm, no activation."""

    def __init__(self, *, input_dim: int, num_classes: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self._num_classes = int(num_classes)
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, X: Tensor) -> Tensor:
        CALLS["decoder"] += 1
        return self.layers(X)

    @property
    def num_classes(self) -> int:
        return self._num_classes


@pytest.fixture
def registered_extensions(monkeypatch):
    """Register the custom heads and decoder with their params for one test, then restore
    the registries.

    The set head inherits nothing from its base's registration, so it names the
    categorical heads' params and their ``infer`` hook (the cohort decides its class
    count and label map). Without the restore, the names would leak into every later test
    that lists the registries (CI runs the unit and E2E suites in a single pytest session).
    """
    monkeypatch.setattr(task_registry, "_entries", dict(task_registry._entries))
    monkeypatch.setattr(decoder_registry, "_entries", dict(decoder_registry._entries))
    monkeypatch.setattr(learner_registry, "_warned", set())
    soma.register("heads", CUSTOM_HEAD, DiceOnlySegmentationHead, params=SegmentationParams)
    soma.register(
        "heads", CUSTOM_SET_HEAD, LabelSmoothingHead, params=ClassificationParams, infer=infer_label_map
    )
    soma.register("decoders", CUSTOM_DECODER, TwoStageConvDecoder, params=TwoStageConvParams)


def test_custom_head_and_decoder_train_through_cli(
    tmp_path, encoder_name, encoded_images, new_artifact, registered_extensions
):
    CALLS["loss"] = CALLS["decoder"] = 0
    artifact = new_artifact("extension_custom_head_decoder")

    roi_px = 96
    dataset_csv, splits_csv = make_dense_dataset(tmp_path / "data", "segmentation", size=roi_px)
    config = {
        "run": {"output_root": str(tmp_path / "runs"), "seed": 0},
        "data": {
            "dataset_csv": str(dataset_csv),
            "splits_csv": str(splits_csv),
            "dataset_type": "segmentation",
        },
        "preprocessing": {
            "backend": "openslide",
            "requested_tile_size_px": roi_px,
            "requested_spacing_um": SPACING_UM,
        },
        "execution": CPU_EXECUTION,
        "cache": {"enabled": True, "root_dir": str(tmp_path / "cache")},
        "encoder": {"name": encoder_name, "precision": "fp32", "batch_size": 8},
        "decoder": {"name": CUSTOM_DECODER, "params": {"hidden_dim": 16}},
        "task": {"name": CUSTOM_HEAD, "params": {"num_classes": 3}},
        "evaluation": {"metrics": ["mean_dice", "mean_iou"]},
        "normalization": {"method": "zscore"},
        "training": {
            "epochs": 20,
            "learning_rate": 3e-3,
            "batch_size": 4,
            "patience": 20,
            **CPU_LOADER,
        },
    }
    run = run_soma(config, tmp_path / "config.yaml")

    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_equal("rois_encoded", sum(encoded_images), 12)
    artifact.check_at_least("custom_loss_calls", CALLS["loss"], 1)
    artifact.check_at_least("custom_decoder_calls", CALLS["decoder"], 1)
    artifact.check_at_least("test/mean_dice", run.summary["test/mean_dice"], 0.8)
    artifact.assert_passed()


def test_custom_learner_by_dotted_path_trains_through_cli(
    tmp_path, encoder_name, new_artifact, registered_extensions
):
    """A ``MILLearner`` subclass adding an intermediate loss trains through the CLI via
    the ``training.learner`` leaf, named by dotted import path and given its params; its
    ``predict_step`` adds a column to the predictions. The registered custom head it
    trains passes ``check_task_head``."""
    CALLS["entropy"] = CALLS["smoothing"] = CALLS["predict"] = 0
    artifact = new_artifact("extension_custom_learner")
    cohort = make_slide_cohort(tmp_path / "cohort")
    epochs = 25
    config = {
        "run": {"output_root": str(tmp_path / "runs"), "seed": 0},
        "data": {
            "dataset_csv": str(cohort.manifest("binary_classification")),
            "splits_csv": str(cohort.splits_csv),
            "dataset_type": "slide",
        },
        "preprocessing": {
            "backend": "openslide",
            "requested_tile_size_px": 32,
            "requested_spacing_um": SPACING_UM,
            "tissue_method": "otsu",
            "seg_downsample": 16,
            "a_t": 1,
            "min_coverage": {"tissue": 0.5},
        },
        "execution": CPU_EXECUTION,
        "cache": {"enabled": True, "root_dir": str(tmp_path / "cache")},
        "encoder": {"name": encoder_name, "precision": "fp32", "batch_size": 256},
        "aggregation": {"name": "abmil"},
        "task": {"name": CUSTOM_SET_HEAD},
        "evaluation": {"metrics": ["auroc", "balanced_accuracy"]},
        "normalization": {"method": "zscore"},
        "training": {
            "epochs": epochs,
            "learning_rate": 5e-3,
            "batch_size": 1,
            "patience": epochs,
            "learner": {
                "name": "tests.e2e.test_extension.AttentionEntropyLearner",
                "params": {"weight": 0.01},
            },
            **CPU_LOADER,
        },
    }
    with pytest.warns(UserWarning, match=r"learners 'tests\.e2e\.test_extension\.AttentionEntropyLearner' is not registered"):
        run = run_soma(config, tmp_path / "config.yaml")

    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_at_least("custom_training_steps", CALLS["entropy"], epochs)
    artifact.check_at_least("test/auroc", run.summary["test/auroc"], 0.95)
    history = run.histories()[""]
    artifact.check_equal("history/columns_without_tune_prefix", "tune_loss" in history[0], False)
    artifact.check_equal("history/loss_present", "loss" in history[0], True)

    artifact.check_at_least("custom_head_loss_calls", CALLS["smoothing"], epochs)
    artifact.check_at_least("custom_predict_steps", CALLS["predict"], 1)
    predictions = Predictions.from_parquet(run.run_dir / "predictions_test.parquet")
    entropy = predictions.frame["attention_entropy"].to_numpy()
    artifact.check_equal("predictions/extra_column_role", predictions.column_roles["attention_entropy"], "extra")
    artifact.check_equal("predictions/extra_column_finite", bool(np.isfinite(entropy).all()), True)
    artifact.check_equal("predictions/declared_columns", predictions.prediction_columns, ("prob_0", "prob_1", "predicted_label"))

    torch.manual_seed(0)
    check_task_head(
        LabelSmoothingHead(input_dim=8, num_classes=2),
        SampleBatch(
            features=torch.randn(8, 8),
            targets={"label": torch.tensor([0, 1] * 4)},
            sample_ids=tuple(f"s{i}" for i in range(8)),
        ),
    )
    artifact.check_equal("custom_head/check_task_head", "passed", "passed")
    artifact.assert_passed()


def test_custom_grid_learner_by_dotted_path_trains_through_cli(
    tmp_path, encoder_name, new_artifact, registered_extensions
):
    """A ``DenseLearner`` subclass adding a regulariser trains a cached segmentation run
    through the CLI via the ``training.learner`` leaf."""
    CALLS["smooth_grid"] = 0
    artifact = new_artifact("extension_custom_grid_learner")
    roi_px = 96
    dataset_csv, splits_csv = make_dense_dataset(tmp_path / "data", "segmentation", size=roi_px)
    epochs = 20
    config = {
        "run": {"output_root": str(tmp_path / "runs"), "seed": 0},
        "data": {
            "dataset_csv": str(dataset_csv),
            "splits_csv": str(splits_csv),
            "dataset_type": "segmentation",
        },
        "preprocessing": {
            "backend": "openslide",
            "requested_tile_size_px": roi_px,
            "requested_spacing_um": SPACING_UM,
        },
        "execution": CPU_EXECUTION,
        "cache": {"enabled": True, "root_dir": str(tmp_path / "cache")},
        "encoder": {"name": encoder_name, "precision": "fp32", "batch_size": 8},
        "decoder": {"name": "lightweight_conv"},
        "task": {"name": "segmentation", "params": {"num_classes": 3}},
        "evaluation": {"metrics": ["mean_dice", "mean_iou"]},
        "normalization": {"method": "zscore"},
        "training": {
            "epochs": epochs,
            "learning_rate": 3e-3,
            "batch_size": 4,
            "patience": epochs,
            "learner": {
                "name": "tests.e2e.test_extension.SmoothGridLearner",
                "params": {"weight": 0.1},
            },
            **CPU_LOADER,
        },
    }
    run = run_soma(config, tmp_path / "config.yaml")

    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_at_least("custom_training_steps", CALLS["smooth_grid"], epochs)
    artifact.check_at_least("test/mean_dice", run.summary["test/mean_dice"], 0.8)
    artifact.assert_passed()
