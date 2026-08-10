"""SegmentationDataset — dense feature grids paired with per-pixel mask targets.

Each item is ``(grid (d, h, w), targets {"mask": (H, W)}, sample_id)``: the grid is
loaded from a :class:`~soma.dense.DenseFeatureSource`, and ``targets`` come from the
injected ``target_fn`` (the ``SegmentationHead.extract_targets``, a later slice,
which loads ``mask_path``) — mirroring how :class:`SampleDataset` defers target
semantics to the head.

The mask stays at the supervision ``target_size`` (e.g. 512); the head crops/resizes
the decoder logits to it (design §5b), so the data plane does not pad masks. Collation
stacks grids ``(B, d, h, w)`` and masks ``(B, H, W)`` as ``long`` (preserving
``ignore_index``), failing loud if shapes are non-uniform within a batch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from soma.dataset import SampleRecord
from soma.dense import DenseFeatureSource
from soma.dense.geometry import DenseGridGeometry


class SegmentationDataset(Dataset):
    """Dataset pairing dense ``(d, h, w)`` grids with dense mask targets.

    Args:
        records: SampleRecords (with ``mask_path``) for this split.
        feature_store: DenseFeatureSource for loading cached dense grids.
        target_fn: Callable mapping a SampleRecord to its targets dict, which must
            contain a ``"mask"`` tensor of shape ``(H, W)`` (the head's
            ``extract_targets``).
    """

    def __init__(
        self,
        records: list[SampleRecord],
        feature_store: DenseFeatureSource,
        target_fn: Callable[[SampleRecord], dict[str, Tensor]],
    ) -> None:
        self._records = records
        self._store = feature_store
        self._target_fn = target_fn

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor], str]:
        record = self._records[idx]
        grid = self._store.load(record.sample_id)  # (d, h, w)
        targets = self._target_fn(record)
        if "mask" not in targets:
            raise ValueError(
                f"segmentation target_fn must return a 'mask' for '{record.sample_id}'; "
                f"got keys {sorted(targets)}"
            )
        mask = targets["mask"]
        # Fail loud if the mask does not match the grid's recorded supervision size:
        # a silent mismatch would misregister the loss against the features.
        target_size = self._store.metadata(record.sample_id).get("target_size")
        if target_size is not None and tuple(int(s) for s in mask.shape[-2:]) != tuple(
            int(s) for s in target_size
        ):
            raise ValueError(
                f"mask for '{record.sample_id}' is {tuple(int(s) for s in mask.shape[-2:])} "
                f"but the dense grid records target_size {tuple(int(s) for s in target_size)}."
            )
        return grid, targets, record.sample_id


class LiveSegmentationDataset(Dataset):
    """Image+mask pairs read and (optionally) augmented for the live re-encode path.

    The live counterpart of :class:`SegmentationDataset`: instead of a cached grid +
    a head-loaded mask, it yields the **augmented image tensor** ``(C, Henc, Wenc)``
    that :class:`~soma.training.model.LiveSegmentationModel` re-encodes on-GPU, plus
    the jointly-transformed mask ``(H, W)`` at the supervision ``target_size``. The
    collated batch reuses :func:`segmentation_collate_fn` (now stacking images, not
    grids).

    Why a dedicated dataset (it bypasses the head's ``extract_targets``): augmentation
    requires the image and mask to be loaded *together* and transformed *jointly*
    (geometric ops must agree pixel-for-pixel), so the head's fixed-mask ``target_fn``
    cannot be used. Both are read spacing-aware (so they register against the dense
    grid by construction), wrapped as ``tv_tensors.Image``/``Mask``, passed through the
    joint v2 transform, then the image is normalized with the encoder's dense transform
    and padded to ``encoded_size`` while the mask stays at ``target_size`` (the head
    crops the decoded logits back to it).

    Args:
        records: SampleRecords (with ``image_path`` and ``mask_path``) for this split.
        geometry: The run's :class:`DenseGridGeometry` (target/encoded size + pad).
        preprocessor: The public kit's serializable CPU item preprocessor. It owns
            normalization, geometry validation, and padding after Soma augmentation.
        spacing_um: µm/px to read both image and mask at (``None`` = flat PIL read).
        backend: hs2p backend for spacing-aware reads.
        tolerance: hs2p spacing tolerance.
        num_classes / ignore_index: validate mask label values (fail loud, with the
            sample id, before a cryptic device-side one_hot/CE assert).
        augment: Joint ``(image, mask)`` v2 transform, or ``None`` for no augmentation.
    """

    def __init__(
        self,
        records: list[SampleRecord],
        *,
        geometry: DenseGridGeometry,
        preprocessor: Callable,
        spacing_um: float | None,
        backend: str,
        tolerance: float,
        num_classes: int,
        ignore_index: int,
        augment: Callable | None = None,
    ) -> None:
        self._records = records
        self._geometry = geometry
        self._preprocessor = preprocessor
        self._spacing_um = float(spacing_um) if spacing_um is not None else None
        self._backend = backend
        self._tolerance = float(tolerance)
        self._num_classes = int(num_classes)
        self._ignore_index = int(ignore_index)
        self._augment = augment

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor], str]:
        from soma.dense.reader import read_image_at_spacing, read_mask_at_spacing

        record = self._records[idx]
        if record.mask_path is None:
            raise ValueError(f"segmentation sample '{record.sample_id}' has no mask_path")
        # Read both at the same spacing so they register against the dense grid.
        image_array = read_image_at_spacing(
            record.image_path,
            spacing_um=self._spacing_um,
            backend=self._backend,
            tolerance=self._tolerance,
        )
        mask_array = read_mask_at_spacing(
            record.mask_path,
            spacing_um=self._spacing_um,
            backend=self._backend,
            tolerance=self._tolerance,
        )

        if self._augment is not None:
            from torchvision import tv_tensors

            image_tv = tv_tensors.Image(
                torch.from_numpy(np.ascontiguousarray(image_array)).permute(2, 0, 1)
            )  # (C, H, W) uint8
            mask_tv = tv_tensors.Mask(
                torch.from_numpy(np.ascontiguousarray(mask_array).astype(np.int64))
            )  # (H, W)
            image_tv, mask_tv = self._augment(image_tv, mask_tv)
            image = image_tv.as_subclass(torch.Tensor).to(torch.uint8)
            mask = mask_tv.as_subclass(torch.Tensor).to(torch.long)
        else:
            image = torch.from_numpy(np.ascontiguousarray(image_array)).permute(2, 0, 1)
            mask = torch.from_numpy(np.ascontiguousarray(mask_array).astype(np.int64))

        # Soma owns joint augmentation; the public kit owns every encoder-specific step
        # after that handoff (normalization, geometry validation, and padding).
        padded = self._preprocessor(image.contiguous())

        if tuple(int(s) for s in mask.shape[-2:]) != self._geometry.target_size:
            raise ValueError(
                f"mask for '{record.sample_id}' is {tuple(int(s) for s in mask.shape[-2:])} "
                f"but the run's target_size is {self._geometry.target_size}."
            )
        allowed = set(range(self._num_classes)) | {self._ignore_index}
        invalid = sorted(v for v in torch.unique(mask).tolist() if v not in allowed)
        if invalid:
            raise ValueError(
                f"mask for '{record.sample_id}' has label value(s) {invalid} outside "
                f"[0, num_classes={self._num_classes}) ∪ {{ignore_index={self._ignore_index}}}."
            )
        return padded, {"mask": mask}, record.sample_id


@dataclass
class SegmentationBatch:
    """A collated batch of dense grids + mask targets.

    Attributes:
        features: Dense feature grids, shape ``(B, d, h, w)``.
        targets: ``{"mask": (B, H, W)}`` with the mask as ``long`` (CE-ready),
            ``ignore_index`` values preserved.
        sample_ids: Tuple of sample IDs.
    """

    features: Tensor
    targets: dict[str, Tensor]
    sample_ids: tuple[str, ...]


def segmentation_collate_fn(
    batch: list[tuple[Tensor, dict[str, Tensor], str]],
    target_dtypes: dict[str, torch.dtype],
) -> SegmentationBatch:
    """Collate dense grids + masks; fail loud on non-uniform shapes within a batch."""
    grids, target_dicts, sample_ids = zip(*batch)
    try:
        features = torch.stack(list(grids))
    except RuntimeError as exc:
        shapes = sorted({tuple(g.shape) for g in grids})
        raise ValueError(
            f"dense grids in a batch must share shape (d, h, w); got {shapes}. "
            "v1 assumes a fixed tile size per run."
        ) from exc
    mask_dtype = target_dtypes.get("mask", torch.long)
    try:
        masks = torch.stack([t["mask"] for t in target_dicts]).to(mask_dtype)
    except RuntimeError as exc:
        shapes = sorted({tuple(t["mask"].shape) for t in target_dicts})
        raise ValueError(
            f"masks in a batch must share shape (H, W); got {shapes}."
        ) from exc
    return SegmentationBatch(
        features=features,
        targets={"mask": masks},
        sample_ids=tuple(sample_ids),
    )
