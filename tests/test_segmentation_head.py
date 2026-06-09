"""Tests for the segmentation head/model and the pure dense loss/metric functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from soma.decoders import LinearDecoder
from soma.tasks import task_registry
from soma.tasks.dense_metrics import (
    cross_entropy_dice_loss,
    dense_confusion_counts,
    reduce_dice_iou,
    soft_dice_loss,
)
from soma.tasks.segmentation import SegmentationHead, load_mask
from soma.dense import compute_dense_geometry
from soma.training.model import SegmentationModel, SegmentationModelOutput


def _logits_from_pred(pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Sharp logits whose argmax equals ``pred`` (B, H, W) -> (B, C, H, W)."""
    return F.one_hot(pred, num_classes).permute(0, 3, 1, 2).float() * 10.0


# --------------------------------------------------------------------------- #
# Pure metrics — exact values on a tiny mask with an ignore pixel + absent class.
# --------------------------------------------------------------------------- #


def test_dice_iou_exact_values_with_ignore_and_absent_class():
    # mask: [[0, 1], [1, 255]] -> (1,1) is ignore_index. Classes present: {0, 1}.
    mask = torch.tensor([[[0, 1], [1, 255]]])  # (1, 2, 2)
    pred = torch.tensor([[[0, 1], [0, 0]]])  # (1,0) wrong (pred 0, target 1); (1,1) ignored
    logits = _logits_from_pred(pred, num_classes=3)  # class 2 absent in pred & target

    counts = dense_confusion_counts(logits, mask, num_classes=3, ignore_index=255)
    # class 0: inter=1, pred_area=2, target_area=1 ; class 1: inter=1, pred=1, target=2 ; class 2: 0,0,0
    assert counts.tolist() == [[[1, 2, 1], [1, 1, 2], [0, 0, 0]]]

    out = reduce_dice_iou(counts, num_classes=3)
    # dice_0 = 2*1/(2+1)=2/3 ; dice_1 = 2*1/(1+2)=2/3 ; class 2 undefined -> excluded
    assert out["mean_dice"] == pytest.approx(2 / 3)
    # iou_0 = 1/(2+1-1)=1/2 ; iou_1 = 1/(1+2-1)=1/2
    assert out["mean_iou"] == pytest.approx(0.5)
    assert out["dice_class_0"] == pytest.approx(2 / 3)
    assert out["dice_class_1"] == pytest.approx(2 / 3)
    assert out["dice_class_2"] == 0.0  # fully undefined -> reported 0.0


def test_per_class_dice_is_dataset_global_not_per_image_mean():
    # Image A (1 pixel): perfect class-0 -> per-image dice_0 = 1.
    # Image B (large): all class-0 target but predicted class-1 -> inter=0, large denom.
    # counts: [inter, pred_area, target_area] per class.
    counts = torch.tensor(
        [
            [[1, 1, 1], [0, 0, 0]],          # image A: class0 perfect, class1 absent
            [[0, 0, 100], [0, 100, 0]],      # image B: class0 all missed (target 100, pred 0)
        ]
    )
    out = reduce_dice_iou(counts, num_classes=2)
    # dataset-global class0 = 2*(1+0)/((1+1)+(100+0)) = 2/102 ≈ 0.0196, NOT the
    # per-image mean (1 + 0)/2 = 0.5.
    assert out["dice_class_0"] == pytest.approx(2 / 102, abs=1e-4)
    assert out["dice_class_0"] < 0.05


def test_perfect_prediction_scores_one_and_dataset_global_matches():
    mask = torch.tensor([[[0, 1], [1, 0]]])
    logits = _logits_from_pred(mask.clone(), num_classes=2)
    counts = dense_confusion_counts(logits, mask, num_classes=2, ignore_index=255)
    macro = reduce_dice_iou(counts, num_classes=2)
    glob = reduce_dice_iou(counts, num_classes=2, aggregation="dataset_global")
    assert macro["mean_dice"] == pytest.approx(1.0)
    assert macro["mean_iou"] == pytest.approx(1.0)
    assert glob["mean_dice"] == pytest.approx(1.0)


def test_soft_dice_and_ce_loss_ignore_and_finite():
    mask = torch.tensor([[[0, 1], [1, 255]]])
    perfect = _logits_from_pred(torch.tensor([[[0, 1], [1, 0]]]), num_classes=2)
    loss = cross_entropy_dice_loss(perfect, mask, num_classes=2, ignore_index=255)
    assert torch.isfinite(loss) and float(loss) < 0.1  # near-perfect -> small

    wrong = _logits_from_pred(torch.tensor([[[1, 0], [0, 0]]]), num_classes=2)
    worse = cross_entropy_dice_loss(wrong, mask, num_classes=2, ignore_index=255)
    assert float(worse) > float(loss)


def test_soft_dice_loss_gradients_flow():
    mask = torch.tensor([[[0, 1], [1, 0]]])
    logits = torch.randn(1, 2, 2, 2, requires_grad=True)
    soft_dice_loss(logits, mask, num_classes=2, ignore_index=255).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


# --------------------------------------------------------------------------- #
# SegmentationHead forward (resize + crop) and the model composition.
# --------------------------------------------------------------------------- #


def test_head_forward_no_pad_interpolates_to_target():
    geom = compute_dense_geometry(target_size=8, patch_size=4)  # encoded 8, grid 2, no pad
    head = SegmentationHead(num_classes=3, geometry=geom)
    out = head(torch.randn(2, 3, 2, 2))  # decoder grid logits
    assert tuple(out.shape) == (2, 3, 8, 8)


def test_head_forward_padded_crops_back_to_target():
    geom = compute_dense_geometry(target_size=6, patch_size=4)  # encoded 8, grid 2, crop->6
    assert geom.encoded_size == (8, 8) and geom.crop_box == (0, 0, 6, 6)
    head = SegmentationHead(num_classes=3, geometry=geom)
    out = head(torch.randn(1, 3, 2, 2))
    assert tuple(out.shape) == (1, 3, 6, 6)  # cropped to target, not 8


def test_segmentation_model_produces_target_res_logits():
    geom = compute_dense_geometry(target_size=8, patch_size=4)
    model = SegmentationModel(
        decoder=LinearDecoder(input_dim=16, num_classes=3),
        task_head=SegmentationHead(num_classes=3, geometry=geom),
    )
    out = model(torch.randn(2, 16, 2, 2))  # (B, d, h, w)
    assert isinstance(out, SegmentationModelOutput)
    assert tuple(out.logits.shape) == (2, 3, 8, 8)
    # the trainer's contract: compute_loss/compute_metrics consume out.logits
    mask = torch.zeros(2, 8, 8, dtype=torch.long)
    assert torch.isfinite(model.task_head.compute_loss(out.logits, {"mask": mask}))
    metrics = model.task_head.compute_metrics(out.logits, {"mask": mask})
    assert "mean_dice" in metrics and "mean_iou" in metrics


# --------------------------------------------------------------------------- #
# Head config + mask loader.
# --------------------------------------------------------------------------- #


def test_head_registered_and_ignore_index_validated():
    assert task_registry.get("segmentation") is SegmentationHead
    geom = compute_dense_geometry(target_size=8, patch_size=4)
    with pytest.raises(ValueError, match="ignore_index"):
        SegmentationHead(num_classes=3, geometry=geom, ignore_index=1)  # 1 in [0,3)
    SegmentationHead(num_classes=3, geometry=geom, ignore_index=255)  # ok


def test_load_mask_accepts_2d_integer(tmp_path: Path):
    arr = np.array([[0, 1], [2, 255]], dtype=np.uint8)
    path = tmp_path / "m.png"
    Image.fromarray(arr).save(path)
    mask = load_mask(path)
    assert mask.dtype == torch.long and mask.tolist() == [[0, 1], [2, 255]]
    with pytest.raises(ValueError, match="expected"):
        load_mask(path, expected_size=(4, 4))


def test_load_mask_rejects_rgb(tmp_path: Path):
    path = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="2-D single-channel"):
        load_mask(path)


def test_load_mask_rejects_palette(tmp_path: Path):
    path = tmp_path / "palette.png"
    img = Image.fromarray(np.array([[0, 1], [2, 3]], dtype=np.uint8)).convert("P")
    img.save(path)
    assert Image.open(path).mode == "P"
    with pytest.raises(ValueError, match="palette image"):
        load_mask(path)


def test_compute_metrics_honors_configured_selection():
    geom = compute_dense_geometry(target_size=4, patch_size=1)
    logits = _logits_from_pred(torch.zeros(1, 4, 4, dtype=torch.long), num_classes=2)
    targets = {"mask": torch.zeros(1, 4, 4, dtype=torch.long)}

    default = SegmentationHead(num_classes=2, geometry=geom).compute_metrics(logits, targets)
    assert set(default) == {"mean_dice", "mean_iou"}  # DEFAULT excludes per-class

    only_iou = SegmentationHead(num_classes=2, geometry=geom, metrics=["mean_iou"])
    assert set(only_iou.compute_metrics(logits, targets)) == {"mean_iou"}

    with_per_class = SegmentationHead(
        num_classes=2, geometry=geom, metrics=["mean_dice", "mean_iou", "dice_per_class"]
    )
    keys = set(with_per_class.compute_metrics(logits, targets))
    assert keys == {"mean_dice", "mean_iou", "dice_class_0", "dice_class_1"}


def test_extract_targets_rejects_out_of_range_label(tmp_path: Path):
    from soma.dataset import SampleRecord

    path = tmp_path / "m.png"
    Image.fromarray(np.array([[0, 1], [2, 0]], dtype=np.uint8)).save(path)  # label 2 with C=2
    geom = compute_dense_geometry(target_size=2, patch_size=1)
    head = SegmentationHead(num_classes=2, geometry=geom, ignore_index=255)
    record = SampleRecord(sample_id="s0", image_path=path, label=None, mask_path=path)
    with pytest.raises(ValueError, match=r"label value\(s\) \[2\] outside"):
        head.extract_targets(record)


def test_loss_is_zero_not_nan_for_all_ignore_batch():
    mask = torch.full((1, 2, 2), 255, dtype=torch.long)  # every pixel ignored
    logits = torch.randn(1, 2, 2, 2, requires_grad=True)
    loss = cross_entropy_dice_loss(logits, mask, num_classes=2, ignore_index=255)
    assert torch.isfinite(loss) and float(loss) == 0.0
    loss.backward()  # graph-connected; no-op step, no nan
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
