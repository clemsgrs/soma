"""Image-prior decoders: the pixel feed through the segmentation seam.

A decoder may declare ``consumes_image``; the datasets then attach the sample's pixels
at the mask resolution as ``targets["image"]`` (float in [0, 1], ``(3, H, W)``), the
collate stacks them like the mask, and the segmentation models hand them to the
decoder. Decoders that do not ask see no change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import Tensor

from soma.decoders.base import Decoder
from soma.tasks.segmentation import SegmentationHead
from soma.training.model import SegmentationModel
from soma.training.segmentation_dataset import (
    attach_image_targets,
    segmentation_collate_fn,
)
from soma.dataset import SampleRecord
from soma.dense.geometry import compute_dense_geometry
from soma.training.segmentation_dataset import LiveSegmentationDataset


class _GridOnly(Decoder):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 2, 1)

    def forward(self, X: Tensor) -> Tensor:
        return self.conv(X)

    @property
    def num_classes(self) -> int:
        return 2


class _WithImage(Decoder):
    consumes_image = True

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 2, 1)
        self.seen: list[tuple[int, ...]] = []

    def forward(self, X: Tensor, image: Tensor) -> Tensor:
        self.seen.append(tuple(image.shape))
        logits = torch.nn.functional.interpolate(self.conv(X), size=image.shape[-2:])
        return logits + image[:, :2].mean(dim=(2, 3), keepdim=True)

    @property
    def num_classes(self) -> int:
        return 2


def _head() -> SegmentationHead:
    return SegmentationHead(num_classes=2, geometry=compute_dense_geometry(target_size=16, patch_size=8))


def test_decoders_do_not_consume_the_image_by_default():
    assert Decoder.consumes_image is False
    assert _GridOnly().consumes_image is False
    assert _WithImage().consumes_image is True


def test_segmentation_model_passes_the_image_only_to_consuming_decoders():
    grid = torch.zeros(2, 4, 2, 2)
    image = torch.rand(2, 3, 16, 16)

    plain = SegmentationModel(decoder=_GridOnly(), task_head=_head())
    assert tuple(plain(grid, image=image).logits.shape) == (2, 2, 16, 16)

    consuming = _WithImage()
    model = SegmentationModel(decoder=consuming, task_head=_head())
    assert tuple(model(grid, image=image).logits.shape) == (2, 2, 16, 16)
    assert consuming.seen == [(2, 3, 16, 16)]
    with pytest.raises(ValueError, match="consumes_image"):
        model(grid)


def test_collate_stacks_every_target_key_including_the_image():
    items = [
        (torch.zeros(4, 2, 2), {"mask": torch.zeros(16, 16, dtype=torch.long), "image": torch.rand(3, 16, 16)}, "a"),
        (torch.zeros(4, 2, 2), {"mask": torch.ones(16, 16, dtype=torch.long), "image": torch.rand(3, 16, 16)}, "b"),
    ]
    batch = segmentation_collate_fn(items, target_dtypes={"mask": torch.long})
    assert tuple(batch.targets["mask"].shape) == (2, 16, 16)
    assert tuple(batch.targets["image"].shape) == (2, 3, 16, 16)
    assert batch.targets["image"].dtype == torch.float32


def test_attach_image_targets_loads_the_sample_pixels_at_the_mask_size(tmp_path: Path):
    pixels = np.zeros((16, 16, 3), dtype=np.uint8)
    pixels[..., 0] = 255
    image_path = tmp_path / "a.png"
    Image.fromarray(pixels).save(image_path)
    record = SampleRecord(sample_id="a", image_path=image_path, label=None, label_mask_path=image_path)

    target_fn = attach_image_targets(lambda rec: {"mask": torch.zeros(16, 16, dtype=torch.long)})
    targets = target_fn(record)

    assert set(targets) == {"mask", "image"}
    assert tuple(targets["image"].shape) == (3, 16, 16)
    assert targets["image"].dtype == torch.float32
    assert torch.equal(targets["image"][0], torch.ones(16, 16))
    assert torch.equal(targets["image"][1], torch.zeros(16, 16))


def test_attach_image_targets_rejects_a_pixel_size_that_differs_from_the_mask(tmp_path: Path):
    image_path = tmp_path / "a.png"
    Image.fromarray(np.zeros((8, 16, 3), dtype=np.uint8)).save(image_path)
    record = SampleRecord(sample_id="a", image_path=image_path, label=None, label_mask_path=image_path)

    target_fn = attach_image_targets(lambda rec: {"mask": torch.zeros(16, 16, dtype=torch.long)})
    with pytest.raises(ValueError, match="image.*8.*16.*mask.*16.*16"):
        target_fn(record)


def test_live_dataset_emits_the_augmented_pixels_at_the_mask_size_when_asked(tmp_path: Path):
    pixels = np.zeros((16, 16, 3), dtype=np.uint8)
    pixels[..., 2] = 255
    Image.fromarray(pixels).save(tmp_path / "tile.png")
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="L").save(tmp_path / "tile_mask.png")
    record = SampleRecord(
        sample_id="t0", image_path=tmp_path / "tile.png", label=None, label_mask_path=tmp_path / "tile_mask.png"
    )
    geometry = compute_dense_geometry(target_size=16, patch_size=8)
    common = dict(
        geometry=geometry, preprocessor=lambda item: item.float() * 0, spacing_um=None,
        backend="auto", tolerance=0.05, num_classes=2, ignore_index=255,
    )

    silent = LiveSegmentationDataset([record], **common)
    assert set(silent[0][1]) == {"mask"}

    emitting = LiveSegmentationDataset([record], emit_image=True, **common)
    _, targets, _ = emitting[0]
    assert set(targets) == {"mask", "image"}
    assert tuple(targets["image"].shape) == (3, 16, 16) and targets["image"].dtype == torch.float32
    assert torch.equal(targets["image"][2], torch.ones(16, 16))  # raw pixels, not the kit's normalisation
