"""SegmentationDataset — dense feature grids paired with per-pixel mask targets.

Each item is ``(grid (d, h, w), targets {"mask": (H, W)}, sample_id)``: the grid is
loaded from a :class:`~soma.dense.DenseFeatureStore`, and ``targets`` come from the
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

import torch
from torch import Tensor
from torch.utils.data import Dataset

from soma.dataset import SampleRecord
from soma.dense import DenseFeatureStore


class SegmentationDataset(Dataset):
    """Dataset pairing dense ``(d, h, w)`` grids with dense mask targets.

    Args:
        records: SampleRecords (with ``mask_path``) for this split.
        feature_store: DenseFeatureStore for loading cached dense grids.
        target_fn: Callable mapping a SampleRecord to its targets dict, which must
            contain a ``"mask"`` tensor of shape ``(H, W)`` (the head's
            ``extract_targets``).
    """

    def __init__(
        self,
        records: list[SampleRecord],
        feature_store: DenseFeatureStore,
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
