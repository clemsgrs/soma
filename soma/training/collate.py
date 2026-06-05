"""Bag collation — pad variable-length bags and construct masks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def stack_targets(
    target_dicts: tuple[dict[str, int | float], ...],
    target_dtypes: dict[str, torch.dtype],
) -> dict[str, Tensor]:
    """Stack per-sample target dicts into a dict of (B,) tensors.

    Each key in ``target_dtypes`` becomes one tensor with the declared dtype,
    collected across the batch in order. The head owns the keys; collation is
    task-agnostic.
    """
    return {
        key: torch.tensor([t[key] for t in target_dicts], dtype=dtype)
        for key, dtype in target_dtypes.items()
    }


@dataclass
class BagBatch:
    """A collated batch of bags with padding and masks.

    Attributes:
        features: Padded tile features, shape (B, N_max, D).
        mask: Boolean mask, shape (B, N_max). True = valid tile.
        targets: Ground-truth targets keyed by the head's target_dtypes,
            each shape (B,).
        sample_ids: Tuple of sample IDs.
    """

    features: Tensor
    mask: Tensor
    targets: dict[str, Tensor]
    sample_ids: tuple[str, ...]


def bag_collate_fn(
    batch: list[tuple[Tensor, dict[str, int | float], str]],
    target_dtypes: dict[str, torch.dtype],
) -> BagBatch:
    """Collate variable-length bags by padding to max length.

    Args:
        batch: List of (features, targets, sample_id) tuples from BagDataset.
        target_dtypes: Mapping of target key to tensor dtype (from the head).

    Returns:
        BagBatch with padded features and boolean mask.
    """
    features_list, target_dicts, sample_ids = zip(*batch)

    max_len = max(f.shape[0] for f in features_list)
    feat_dim = features_list[0].shape[1]

    padded = torch.zeros(len(batch), max_len, feat_dim)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, f in enumerate(features_list):
        n = f.shape[0]
        padded[i, :n] = f
        mask[i, :n] = True

    return BagBatch(
        features=padded,
        mask=mask,
        targets=stack_targets(target_dicts, target_dtypes),
        sample_ids=tuple(sample_ids),
    )


@dataclass
class HierarchicalBagBatch:
    """A collated batch of hierarchical bags with region padding and masks.

    Attributes:
        features: Padded hierarchical features, shape (B, M_max, P, D).
        mask: Boolean region mask, shape (B, M_max). True = valid region.
        targets: Ground-truth targets keyed by the head's target_dtypes,
            each shape (B,).
        sample_ids: Tuple of sample IDs.
    """

    features: Tensor
    mask: Tensor
    targets: dict[str, Tensor]
    sample_ids: tuple[str, ...]

    @property
    def region_mask(self) -> Tensor:
        return self.mask


def hierarchical_bag_collate_fn(
    batch: list[tuple[Tensor, dict[str, int | float], str]],
    target_dtypes: dict[str, torch.dtype],
) -> HierarchicalBagBatch:
    """Collate variable-length hierarchical bags by padding the region axis."""
    features_list, target_dicts, sample_ids = zip(*batch)

    max_regions = max(f.shape[0] for f in features_list)
    tiles_per_region = features_list[0].shape[1]
    feat_dim = features_list[0].shape[2]

    padded = torch.zeros(
        len(batch),
        max_regions,
        tiles_per_region,
        feat_dim,
        dtype=features_list[0].dtype,
        device=features_list[0].device,
    )
    mask = torch.zeros(len(batch), max_regions, dtype=torch.bool, device=features_list[0].device)

    for i, f in enumerate(features_list):
        if f.shape[1] != tiles_per_region or f.shape[2] != feat_dim:
            raise ValueError("All hierarchical features in a batch must share P and D")
        n = f.shape[0]
        padded[i, :n] = f
        mask[i, :n] = True

    targets = stack_targets(target_dicts, target_dtypes)
    targets = {key: value.to(features_list[0].device) for key, value in targets.items()}
    return HierarchicalBagBatch(
        features=padded,
        mask=mask,
        targets=targets,
        sample_ids=tuple(sample_ids),
    )
