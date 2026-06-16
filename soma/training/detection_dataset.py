"""DetectionDataset — dense feature grids paired with point-heatmap + GT-point targets.

The detection counterpart of :class:`~soma.training.segmentation_dataset.SegmentationDataset`.
Each item is ``(grid (d, h, w), targets, sample_id)`` where ``targets`` is
``{"heatmap": (C, H, W), "gt_points": (K, 3)}`` from the head's ``extract_targets``: the
heatmap is the regression target, the GT points are kept (variable length) for F1@δ
matching at eval. :func:`detection_collate_fn` stacks the grids and heatmaps and **pads**
the GT points to the batch-max count with NaN rows (stripped before matching), so the
batch stays a plain dict-of-tensors the trainer moves to device unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset

from soma.dataset import SampleRecord
from soma.dense import DenseFeatureStore
from soma.training.segmentation_dataset import SegmentationBatch

_PAD = float("nan")


class DetectionDataset(Dataset):
    """Dataset pairing dense ``(d, h, w)`` grids with detection targets.

    Args:
        records: SampleRecords (with ``points_path``) for this split.
        feature_store: DenseFeatureStore for loading cached dense grids.
        target_fn: Callable mapping a SampleRecord to its targets dict, which must
            contain ``"heatmap"`` ``(C, H, W)`` and ``"gt_points"`` ``(K, 3)`` (the
            head's ``extract_targets``).
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
        if "heatmap" not in targets or "gt_points" not in targets:
            raise ValueError(
                f"detection target_fn must return 'heatmap' and 'gt_points' for "
                f"'{record.sample_id}'; got keys {sorted(targets)}"
            )
        heatmap = targets["heatmap"]
        target_size = self._store.metadata(record.sample_id).get("target_size")
        if target_size is not None and tuple(int(s) for s in heatmap.shape[-2:]) != tuple(
            int(s) for s in target_size
        ):
            raise ValueError(
                f"heatmap for '{record.sample_id}' is {tuple(int(s) for s in heatmap.shape[-2:])} "
                f"but the dense grid records target_size {tuple(int(s) for s in target_size)}."
            )
        return grid, targets, record.sample_id


def detection_collate_fn(
    batch: list[tuple[Tensor, dict[str, Tensor], str]],
    target_dtypes: dict[str, torch.dtype],
) -> SegmentationBatch:
    """Collate grids + heatmaps (stacked) + GT points (NaN-padded to batch-max)."""
    grids, target_dicts, sample_ids = zip(*batch)
    try:
        features = torch.stack(list(grids))
    except RuntimeError as exc:
        shapes = sorted({tuple(g.shape) for g in grids})
        raise ValueError(
            f"dense grids in a batch must share shape (d, h, w); got {shapes}. "
            "v1 assumes a fixed tile size per run."
        ) from exc

    hm_dtype = target_dtypes.get("heatmap", torch.float32)
    try:
        heatmaps = torch.stack([t["heatmap"] for t in target_dicts]).to(hm_dtype)
    except RuntimeError as exc:
        shapes = sorted({tuple(t["heatmap"].shape) for t in target_dicts})
        raise ValueError(f"heatmaps in a batch must share shape (C, H, W); got {shapes}.") from exc

    gt_list = [t["gt_points"] for t in target_dicts]
    k_max = max((int(g.shape[0]) for g in gt_list), default=0)
    padded = torch.full((len(gt_list), k_max, 3), _PAD, dtype=torch.float32)
    for i, g in enumerate(gt_list):
        if g.shape[0]:
            padded[i, : g.shape[0]] = g.to(torch.float32)

    return SegmentationBatch(
        features=features,
        targets={"heatmap": heatmaps, "gt_points": padded},
        sample_ids=tuple(sample_ids),
    )
