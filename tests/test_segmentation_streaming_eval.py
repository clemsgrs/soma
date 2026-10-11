"""Streaming dense evaluation: ``DenseLearner`` validation accumulates compact
per-image confusion counts for segmentation instead of concatenating full logits.

The load-bearing assertion is that the streamed metric is the **per-image-macro**
value (the §9 monitor default), which differs from dataset-global on an uneven
fixture — a regression that summed counts to ``(C, 3)`` would silently produce the
dataset-global number and still "return a number".
"""

from __future__ import annotations

from functools import partial

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from soma.dense import compute_dense_geometry
from soma.learn import Checkpoint, DenseLearner, EarlyStopping, Trainer
from soma.tasks.segmentation import SegmentationHead
from soma.training.model import SegmentationModelOutput
from soma.training.segmentation_dataset import segmentation_collate_fn


def _logits_from_pred(pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(pred, num_classes).permute(0, 3, 1, 2).float() * 10.0


class _FixedLogitsModel(nn.Module):
    """Stub seg model: returns preset target-res logits per forward (loader order).

    Lets the test pin predictions exactly without depending on decoder init. Carries
    a dummy parameter so an optimizer has something to bind to.
    """

    def __init__(self, task_head: SegmentationHead, logits_in_order: list[torch.Tensor]) -> None:
        super().__init__()
        self.task_head = task_head
        self._dummy = nn.Parameter(torch.zeros(1))
        self._logits = list(logits_in_order)
        self._i = 0

    def forward(self, X: torch.Tensor) -> SegmentationModelOutput:
        logits = self._logits[self._i]
        self._i += 1
        return SegmentationModelOutput(logits=logits)


class _SteeredSegmentationHead(SegmentationHead):
    """Drive one deterministic parameter step while retaining real dense metrics."""

    def compute_loss(self, raw_output, targets):
        return -raw_output[:, 1].mean()


class _FlippingSegmentationModel(nn.Module):
    """Predict class 0 until one SGD step moves ``signal`` above zero."""

    def __init__(self, task_head: SegmentationHead) -> None:
        super().__init__()
        self.task_head = task_head
        self.signal = nn.Parameter(torch.tensor(-1.5))

    def forward(self, X: torch.Tensor) -> SegmentationModelOutput:
        class_0 = torch.zeros(
            (X.shape[0], X.shape[-2], X.shape[-1]),
            dtype=X.dtype,
            device=X.device,
        )
        class_1 = self.signal.expand_as(class_0)
        return SegmentationModelOutput(logits=torch.stack([class_0, class_1], dim=1))


def _validate(head, items, logits_in_order) -> dict[str, float]:
    loader = DataLoader(
        items,
        batch_size=1,  # two single-image batches -> exercises cross-batch cat
        shuffle=False,
        collate_fn=partial(segmentation_collate_fn, target_dtypes=head.target_dtypes),
    )
    learner = DenseLearner(_FixedLogitsModel(head, logits_in_order))
    learner.eval()
    return learner.validation_summary([learner.validation_step(batch) for batch in loader])


def test_tune_streams_per_image_macro_not_dataset_global(tmp_path):
    # Two 2x2 images, both all class 0:
    #   A: predicted all 0 -> per-image dice = 1.0
    #   B: predicted all 1 -> per-image dice = 0.0
    # per_image_macro = mean(1.0, 0.0) = 0.5 ; dataset_global ≈ 0.333.
    geom = compute_dense_geometry(target_size=2, patch_size=1)
    head = SegmentationHead(num_classes=2, geometry=geom)

    grid = torch.zeros(1, 2, 2)  # (d, h, w); ignored by the stub but collated
    mask = torch.zeros(2, 2, dtype=torch.long)
    items = [(grid, {"mask": mask}, "a"), (grid, {"mask": mask}, "b")]
    logits_a = _logits_from_pred(torch.zeros(1, 2, 2, dtype=torch.long), 2)
    logits_b = _logits_from_pred(torch.ones(1, 2, 2, dtype=torch.long), 2)

    metrics = _validate(head, items, [logits_a, logits_b])

    assert metrics["mean_dice"] == pytest.approx(0.5)  # per-image macro
    assert metrics["mean_dice"] != pytest.approx(1 / 3)  # NOT dataset-global
    assert "mean_iou" in metrics
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_dataset_global_mean_dice_sums_counts_before_class_macro():
    # Three-class fixture with class 2 absent from the entire dataset:
    #   A: one perfect class-0 pixel -> per-image macro Dice 1.0
    #   B: 100 class-0 pixels all predicted class 1 -> per-image macro Dice 0.0
    # Legacy mean_dice = (1 + 0) / 2 = 0.5.
    # Dataset-global class Dice = [2 / 102, 0, undefined], so the existing
    # skip-absent-class convention gives dataset_global_mean_dice = 1 / 102.
    counts = torch.tensor(
        [
            [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 100], [0, 100, 0], [0, 0, 0]],
        ]
    )
    geom = compute_dense_geometry(target_size=2, patch_size=1)
    head = SegmentationHead(
        num_classes=3,
        geometry=geom,
        metrics=["mean_dice", "dataset_global_mean_dice"],
    )

    metrics = head.finalize_eval_metrics(counts)

    assert metrics == {
        "mean_dice": pytest.approx(0.5),
        "dataset_global_mean_dice": pytest.approx(1 / 102),
    }


def _fit_with_dataset_global_monitor(tmp_path):
    geom = compute_dense_geometry(target_size=2, patch_size=1)
    head = _SteeredSegmentationHead(
        num_classes=2,
        geometry=geom,
        metrics=["dataset_global_mean_dice"],
    )
    grid = torch.zeros(1, 2, 2)
    mask = torch.zeros(2, 2, dtype=torch.long)
    loader = DataLoader(
        [(grid, {"mask": mask}, "sample")],
        batch_size=1,
        collate_fn=partial(segmentation_collate_fn, target_dtypes=head.target_dtypes),
    )
    learner = DenseLearner(
        _FlippingSegmentationModel(head), optimizer={"name": "sgd", "params": {"lr": 1.0}}
    )
    checkpoint = Checkpoint(monitor="dataset_global_mean_dice", mode="max", dirpath=tmp_path)
    stopping = EarlyStopping(monitor="dataset_global_mean_dice", mode="max", patience=1)
    trainer = Trainer(max_epochs=5, callbacks=[checkpoint, stopping], device="cpu")
    return trainer.fit(learner, loader, loader)


def test_dataset_global_mean_dice_is_recorded_in_training_history(tmp_path):
    result = _fit_with_dataset_global_monitor(tmp_path)

    assert result.history["dataset_global_mean_dice"].tolist() == [
        pytest.approx(1.0),
        pytest.approx(0.0),
    ]


def test_dataset_global_mean_dice_selects_checkpoint(tmp_path):
    result = _fit_with_dataset_global_monitor(tmp_path)

    assert (result.selected_epoch, result.selection_monitor) == (
        0,
        ("dataset_global_mean_dice", pytest.approx(1.0)),
    )


def test_dataset_global_mean_dice_drives_early_stopping(tmp_path):
    result = _fit_with_dataset_global_monitor(tmp_path)

    assert len(result.history) == 2


def test_dataset_global_monitor_is_named_in_checkpoint_metadata(tmp_path):
    result = _fit_with_dataset_global_monitor(tmp_path)

    checkpoint = torch.load(result.checkpoint_path, weights_only=True)
    assert checkpoint["selection"] == {
        "strategy": "best",
        "monitor": "dataset_global_mean_dice",
        "mode": "max",
        "value": pytest.approx(1.0),
    }
    assert checkpoint["metrics"]["dataset_global_mean_dice"] == pytest.approx(1.0)


def test_segmentation_head_declares_streaming_eval():
    geom = compute_dense_geometry(target_size=2, patch_size=1)
    head = SegmentationHead(num_classes=2, geometry=geom)
    assert head.accumulates_eval_metrics is True


def test_finalize_matches_compute_metrics_no_drift():
    # The batched (compute_metrics) and streamed (finalize) paths must agree.
    geom = compute_dense_geometry(target_size=2, patch_size=1)
    head = SegmentationHead(num_classes=2, geometry=geom, metrics=["mean_dice", "mean_iou"])
    logits = _logits_from_pred(torch.tensor([[[0, 1], [1, 0]]]), 2)
    targets = {"mask": torch.tensor([[[0, 1], [0, 0]]])}
    batched = head.compute_metrics(logits, targets)
    streamed = head.finalize_eval_metrics(head.dense_stats(logits, targets))
    assert batched == streamed
