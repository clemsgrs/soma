"""Extension seam: a user-registered task head and decoder train through the real CLI.

The customisation API is subclass-and-override: subclass the family's built-in head (or
``Decoder``), override the methods that should differ, register the subclass under a new
name, and reference that name from the config. Validation keys on the registered class's
``task_family``, so the custom head passes everywhere the built-in one does.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from soma.decoders.base import Decoder
from soma.decoders.registry import decoder_registry
from soma.tasks.dense_metrics import soft_dice_loss
from soma.tasks.registry import task_registry
from soma.tasks.segmentation import SegmentationHead
from tests.e2e.harness import CPU_EXECUTION, CPU_LOADER, run_soma
from tests.e2e.synthetic import SPACING_UM, make_dense_dataset

CUSTOM_HEAD = "e2e_dice_only_segmentation"
CUSTOM_DECODER = "e2e_two_stage_conv"

# Call counters: the scenario proves the overrides actually ran, not just registered.
CALLS = {"loss": 0, "decoder": 0}


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


def _register() -> None:
    if CUSTOM_HEAD not in task_registry:
        task_registry.register(CUSTOM_HEAD, DiceOnlySegmentationHead)
    if CUSTOM_DECODER not in decoder_registry:
        decoder_registry.register(CUSTOM_DECODER, TwoStageConvDecoder)


def test_custom_head_and_decoder_train_through_cli(
    tmp_path, encoder_name, encoded_images, new_artifact
):
    _register()
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
