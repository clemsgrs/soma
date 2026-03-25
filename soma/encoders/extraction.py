"""Feature extraction pipeline: dataset, collator, sampler, and extract_features."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from soma.encoders.base import Encoder
from soma.encoders.tile_reader import SuperTileIndex, build_supertile_index
from soma.preprocessing.tiling import TilingResult
from soma.wsi.reader import SlideReader


# ---------------------------------------------------------------------------
# Dataset + Collator + Sampler
# ---------------------------------------------------------------------------


class TileIndexDataset(Dataset):
    """Yields tile indices. Actual reading happens in the collator."""

    def __init__(self, num_tiles: int):
        self._num_tiles = num_tiles

    def __len__(self) -> int:
        return self._num_tiles

    def __getitem__(self, idx: int) -> int:
        return idx


class TileBatchCollator:
    """Reads tiles from a slide and applies the encoder's transform.

    When super-tiles are available, reads one large region per super-tile
    and crops individual tiles from it. Otherwise falls back to one
    ``read_region`` per tile.
    """

    def __init__(
        self,
        reader: SlideReader,
        tiling_result: TilingResult,
        transform,
        st_index: SuperTileIndex | None = None,
    ):
        self._reader = reader
        self._coords = tiling_result.coordinates
        self._tile_size_lv0 = tiling_result.tile_size_lv0
        self._read_level = tiling_result.read_level
        self._tile_size_px = tiling_result.effective_tile_size_px
        self._downsample = (
            reader.level_downsamples[tiling_result.read_level]
            if hasattr(reader, "level_downsamples")
            else 1.0
        )
        self._transform = transform
        self._st_index = st_index

    def __call__(self, batch_indices: list[int]) -> tuple[Tensor, Tensor]:
        indices_t = torch.tensor(batch_indices, dtype=torch.long)
        images = []

        if self._st_index is not None:
            images = self._read_with_supertiles(batch_indices)
        else:
            images = self._read_individually(batch_indices)

        batch_tensor = torch.stack(images)
        return indices_t, batch_tensor

    def _read_individually(self, batch_indices: list[int]) -> list[Tensor]:
        images = []
        for idx in batch_indices:
            x, y = int(self._coords[idx, 0]), int(self._coords[idx, 1])
            region = self._reader.read_region(
                (x, y), self._read_level, (self._tile_size_px, self._tile_size_px)
            )
            images.append(self._transform(region))
        return images

    def _read_with_supertiles(self, batch_indices: list[int]) -> list[Tensor]:
        """Read tiles using super-tile grouping for fewer read_region calls."""
        st_index = self._st_index
        assert st_index is not None

        # Group batch indices by super-tile
        st_groups: dict[int, list[int]] = {}
        for idx in batch_indices:
            st_id = int(st_index.tile_to_st[idx])
            st_groups.setdefault(st_id, []).append(idx)

        # Cache for read regions
        region_cache: dict[int, np.ndarray] = {}
        images: list[tuple[int, Tensor]] = []  # (position_in_batch, tensor)

        for st_id, tile_indices in st_groups.items():
            if st_id not in region_cache:
                st = st_index.supertiles[st_id]
                read_size = max(1, int(st.read_size_lv0 / self._downsample))
                region_cache[st_id] = self._reader.read_region(
                    (st.x_lv0, st.y_lv0),
                    self._read_level,
                    (read_size, read_size),
                )

            region = region_cache[st_id]
            ts = self._tile_size_px

            for tile_idx in tile_indices:
                cx = int(st_index.tile_crop_x[tile_idx] / self._downsample)
                cy = int(st_index.tile_crop_y[tile_idx] / self._downsample)
                tile_img = region[cy : cy + ts, cx : cx + ts]
                pos = batch_indices.index(tile_idx)
                images.append((pos, self._transform(tile_img)))

        # Sort by batch position to maintain order
        images.sort(key=lambda x: x[0])
        return [img for _, img in images]


class SuperTileBatchSampler:
    """Batch sampler that keeps super-tile groups intact.

    Greedily packs whole super-tile groups into batches of approximately
    ``batch_size`` tiles. A group that exceeds batch_size is emitted as-is.
    """

    def __init__(self, groups: list[np.ndarray], batch_size: int):
        self.batches: list[list[int]] = []
        current: list[int] = []
        for group in groups:
            positions = group.tolist()
            if current and len(current) + len(positions) > batch_size:
                self.batches.append(current)
                current = positions
            else:
                current.extend(positions)
        if current:
            self.batches.append(current)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@torch.inference_mode()
def extract_features(
    encoder: Encoder,
    reader: SlideReader,
    tiling_result: TilingResult,
    *,
    batch_size: int = 32,
    num_workers: int = 4,
    precision: str = "fp16",
    use_supertiles: bool = True,
) -> Tensor:
    """Extract features for all tiles. Returns (N, D) float32 tensor.

    Feature row i corresponds to coordinate row i in tiling_result.
    """
    num_tiles = len(tiling_result.coordinates)
    if num_tiles == 0:
        return torch.empty(0, encoder.encode_dim, dtype=torch.float32)

    # Build super-tile index
    st_index: SuperTileIndex | None = None
    if use_supertiles and num_tiles >= 2:
        st_index = build_supertile_index(tiling_result)

    # Build dataloader components
    transform = encoder.get_transform()
    collator = TileBatchCollator(reader, tiling_result, transform, st_index)

    if st_index is not None:
        # Build groups from ordered_indices
        groups = _build_sampler_groups(st_index)
        sampler = SuperTileBatchSampler(groups, batch_size)
        loader = DataLoader(
            TileIndexDataset(num_tiles),
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=False,
        )
    else:
        loader = DataLoader(
            TileIndexDataset(num_tiles),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=False,
        )

    # Extract features
    features = torch.empty(num_tiles, encoder.encode_dim, dtype=torch.float32)
    device = encoder.device
    use_amp = precision == "fp16" and device.type == "cuda"

    for indices, images in loader:
        images = images.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            feats = encoder.encode(images)
        features[indices] = feats.float().cpu()

    return features


def _build_sampler_groups(st_index: SuperTileIndex) -> list[np.ndarray]:
    """Build per-super-tile groups of ordered dataset positions."""
    groups: list[np.ndarray] = []
    current_st = -1
    start = 0

    for pos, tile_idx in enumerate(st_index.ordered_indices):
        st_id = int(st_index.tile_to_st[tile_idx])
        if st_id != current_st:
            if pos > start:
                groups.append(np.arange(start, pos, dtype=np.int64))
            current_st = st_id
            start = pos

    if start < len(st_index.ordered_indices):
        groups.append(
            np.arange(start, len(st_index.ordered_indices), dtype=np.int64)
        )

    return groups


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_features(features: Tensor, output_dir: Path, slide_id: str) -> Path:
    """Atomically save features tensor. Returns the .pt path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_path = output_dir / f"{slide_id}.pt"
    tmp_path = output_dir / f"{slide_id}.pt.tmp"
    torch.save(features, tmp_path)
    os.replace(tmp_path, pt_path)
    return pt_path
