"""Feature extraction pipeline: dataset, collator, sampler, and extraction helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from soma.encoders.base import SlideEncoder, TileEncoder
from soma.encoders.tile_reader import SuperTileIndex, build_supertile_index
from soma.preprocessing.tiling import TilingResult
from soma.wsi.reader import SlideReader


class TileIndexDataset(Dataset):
    """Yields tile indices. Actual reading happens in the collator."""

    def __init__(self, num_tiles: int):
        self._num_tiles = num_tiles

    def __len__(self) -> int:
        return self._num_tiles

    def __getitem__(self, idx: int) -> int:
        return idx


class TileBatchCollator:
    """Reads tiles from a slide and applies the encoder's transform."""

    def __init__(
        self,
        reader: SlideReader,
        tiling_result: TilingResult,
        transform,
        st_index: SuperTileIndex | None = None,
    ):
        self._reader = reader
        self._coords = tiling_result.coordinates
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
        images = (
            self._read_with_supertiles(batch_indices)
            if self._st_index is not None
            else self._read_individually(batch_indices)
        )
        return indices_t, torch.stack(images)

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
        st_index = self._st_index
        assert st_index is not None

        st_groups: dict[int, list[int]] = {}
        for idx in batch_indices:
            st_id = int(st_index.tile_to_st[idx])
            st_groups.setdefault(st_id, []).append(idx)

        region_cache: dict[int, np.ndarray] = {}
        images: list[tuple[int, Tensor]] = []

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

        images.sort(key=lambda x: x[0])
        return [img for _, img in images]


class SuperTileBatchSampler:
    """Batch sampler that keeps super-tile groups intact."""

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


@dataclass(frozen=True)
class SlideExtractionResult:
    slide_features: Tensor
    tile_features: Tensor | None = None


def _build_loader(
    tile_encoder: TileEncoder,
    reader: SlideReader,
    tiling_result: TilingResult,
    *,
    batch_size: int,
    num_workers: int,
    use_supertiles: bool,
) -> DataLoader:
    num_tiles = len(tiling_result.coordinates)
    st_index: SuperTileIndex | None = None
    if use_supertiles and num_tiles >= 2:
        st_index = build_supertile_index(tiling_result)

    collator = TileBatchCollator(
        reader,
        tiling_result,
        tile_encoder.get_transform(),
        st_index,
    )
    if st_index is not None:
        groups = _build_sampler_groups(st_index)
        sampler = SuperTileBatchSampler(groups, batch_size)
        return DataLoader(
            TileIndexDataset(num_tiles),
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=False,
        )
    return DataLoader(
        TileIndexDataset(num_tiles),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=False,
    )


@torch.inference_mode()
def extract_tile_features(
    tile_encoder: TileEncoder,
    reader: SlideReader,
    tiling_result: TilingResult,
    *,
    batch_size: int = 32,
    num_workers: int = 4,
    precision: str = "fp16",
    use_supertiles: bool = True,
) -> Tensor:
    """Extract tile features for one slide. Returns an ``(N, D)`` float32 tensor."""
    num_tiles = len(tiling_result.coordinates)
    if num_tiles == 0:
        return torch.empty(0, tile_encoder.encode_dim, dtype=torch.float32)

    loader = _build_loader(
        tile_encoder,
        reader,
        tiling_result,
        batch_size=batch_size,
        num_workers=num_workers,
        use_supertiles=use_supertiles,
    )

    features = torch.empty(num_tiles, tile_encoder.encode_dim, dtype=torch.float32)
    device = tile_encoder.device
    use_amp = precision == "fp16" and device.type == "cuda"

    for indices, images in loader:
        images = images.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            feats = tile_encoder.encode_tiles(images)
        features[indices] = feats.float().cpu()

    return features


@torch.inference_mode()
def extract_slide_features(
    slide_encoder: SlideEncoder,
    tile_encoder: TileEncoder,
    reader: SlideReader,
    tiling_result: TilingResult,
    *,
    batch_size: int = 32,
    num_workers: int = 4,
    precision: str = "fp16",
    use_supertiles: bool = True,
    return_tile_features: bool = False,
) -> SlideExtractionResult:
    """Extract one pooled slide embedding, optionally returning tile features too."""
    tile_features = extract_tile_features(
        tile_encoder,
        reader,
        tiling_result,
        batch_size=batch_size,
        num_workers=num_workers,
        precision=precision,
        use_supertiles=use_supertiles,
    )
    coordinates = torch.as_tensor(tiling_result.coordinates, dtype=torch.long)
    prepared = slide_encoder.prepare_coordinates(
        coordinates,
        base_spacing_um=float(getattr(reader, "spacing", tiling_result.effective_spacing_um)),
        target_spacing_um=float(tiling_result.effective_spacing_um),
    )
    slide_features = slide_encoder.encode_slide(
        tile_features.to(slide_encoder.device),
        prepared.to(slide_encoder.device),
        tile_size_lv0=int(tiling_result.tile_size_lv0),
    )
    slide_features = slide_features.detach().float().cpu()
    if slide_features.ndim > 1:
        slide_features = slide_features.squeeze(0)
    return SlideExtractionResult(
        slide_features=slide_features,
        tile_features=tile_features if return_tile_features else None,
    )


def _build_sampler_groups(st_index: SuperTileIndex) -> list[np.ndarray]:
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
        groups.append(np.arange(start, len(st_index.ordered_indices), dtype=np.int64))

    return groups


def save_features(features: Tensor, output_dir: Path, slide_id: str) -> Path:
    """Atomically save features tensor. Returns the ``.pt`` path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_path = output_dir / f"{slide_id}.pt"
    tmp_path = output_dir / f"{slide_id}.pt.tmp"
    torch.save(features, tmp_path)
    os.replace(tmp_path, pt_path)
    return pt_path
