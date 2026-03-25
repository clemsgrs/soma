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
from soma.wsi.reader import BatchRegionReader, SlideReader


class TileIndexDataset(Dataset):
    """Yields tile indices. Actual reading happens in the collator."""

    def __init__(self, tile_indices: np.ndarray):
        self._tile_indices = np.asarray(tile_indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self._tile_indices))

    def __getitem__(self, idx: int) -> int:
        return int(self._tile_indices[idx])


class TileBatchCollator:
    """Reads tiles from a slide and applies the encoder's transform."""

    def __init__(
        self,
        reader: SlideReader,
        tiling_result: TilingResult,
        transform,
        st_index: SuperTileIndex | None = None,
        num_workers: int = 0,
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
        self._num_workers = num_workers

    def __call__(self, batch_indices: list[int]) -> tuple[Tensor, Tensor]:
        indices_t = torch.tensor(batch_indices, dtype=torch.long)
        requests = self._build_read_requests(batch_indices)
        images = self._read_requests(requests, len(batch_indices))
        return indices_t, torch.stack(images)

    def _build_read_requests(self, batch_indices: list[int]) -> list[_ReadRequest]:
        position_by_idx = {tile_idx: pos for pos, tile_idx in enumerate(batch_indices)}
        if self._st_index is None:
            return [
                _ReadRequest(
                    location=(int(self._coords[idx, 0]), int(self._coords[idx, 1])),
                    size=(self._tile_size_px, self._tile_size_px),
                    crops=(_TileCrop(pos=position_by_idx[idx], crop_x=0, crop_y=0),),
                )
                for idx in batch_indices
            ]

        st_groups: dict[int, list[int]] = {}
        for idx in batch_indices:
            st_id = int(self._st_index.tile_to_st[idx])
            st_groups.setdefault(st_id, []).append(idx)

        requests: list[_ReadRequest] = []
        for st_id, tile_indices in st_groups.items():
            st = self._st_index.supertiles[st_id]
            read_size = max(1, int(round(st.read_size_lv0 / self._downsample)))
            crops = tuple(
                _TileCrop(
                    pos=position_by_idx[tile_idx],
                    crop_x=int(round(self._st_index.tile_crop_x[tile_idx] / self._downsample)),
                    crop_y=int(round(self._st_index.tile_crop_y[tile_idx] / self._downsample)),
                )
                for tile_idx in tile_indices
            )
            requests.append(
                _ReadRequest(
                    location=(int(st.x_lv0), int(st.y_lv0)),
                    size=(read_size, read_size),
                    crops=crops,
                )
            )
        return requests

    def _read_requests(
        self, requests: list[_ReadRequest], num_images: int
    ) -> list[Tensor]:
        if isinstance(self._reader, BatchRegionReader):
            return self._read_requests_batched(requests, num_images)
        return self._read_requests_sequential(requests, num_images)

    def _read_requests_sequential(
        self, requests: list[_ReadRequest], num_images: int
    ) -> list[Tensor]:
        images: list[Tensor | None] = [None] * num_images
        for request in requests:
            region = self._reader.read_region(
                request.location,
                self._read_level,
                request.size,
            )
            self._store_request_images(images, request, region)
        return [image for image in images if image is not None]

    def _read_requests_batched(
        self, requests: list[_ReadRequest], num_images: int
    ) -> list[Tensor]:
        images: list[Tensor | None] = [None] * num_images
        grouped: dict[tuple[int, tuple[int, int]], list[_ReadRequest]] = {}
        for request in requests:
            grouped.setdefault((self._read_level, request.size), []).append(request)

        for (level, size), grouped_requests in grouped.items():
            locations = [request.location for request in grouped_requests]
            regions = self._reader.read_regions(
                locations,
                level,
                size,
                num_workers=self._num_workers,
            )
            for request, region in zip(grouped_requests, regions):
                self._store_request_images(images, request, region)

        return [image for image in images if image is not None]

    def _store_request_images(
        self,
        images: list[Tensor | None],
        request: _ReadRequest,
        region: np.ndarray,
    ) -> None:
        ts = self._tile_size_px
        for crop in request.crops:
            tile_img = region[crop.crop_y : crop.crop_y + ts, crop.crop_x : crop.crop_x + ts]
            images[crop.pos] = self._transform(tile_img)


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


@dataclass(frozen=True)
class _TileCrop:
    pos: int
    crop_x: int
    crop_y: int


@dataclass(frozen=True)
class _ReadRequest:
    location: tuple[int, int]
    size: tuple[int, int]
    crops: tuple[_TileCrop, ...]


def _build_loader(
    tile_encoder: TileEncoder,
    reader: SlideReader,
    tiling_result: TilingResult,
    *,
    batch_size: int,
    adaptive_batching: bool,
    num_workers: int,
    use_supertiles: bool,
) -> DataLoader:
    num_tiles = len(tiling_result.coordinates)
    st_index: SuperTileIndex | None = None
    dataset_indices = np.arange(num_tiles, dtype=np.int64)
    if use_supertiles and num_tiles >= 2:
        st_index = build_supertile_index(tiling_result)
        dataset_indices = st_index.ordered_indices

    collator = TileBatchCollator(
        reader,
        tiling_result,
        tile_encoder.get_transform(),
        st_index,
        num_workers=num_workers,
    )
    if st_index is not None and adaptive_batching:
        groups = _build_sampler_groups(dataset_indices, st_index.tile_to_st)
        sampler = SuperTileBatchSampler(groups, batch_size)
        return DataLoader(
            TileIndexDataset(dataset_indices),
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=False,
        )
    return DataLoader(
        TileIndexDataset(dataset_indices),
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
    adaptive_batching: bool = False,
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
        adaptive_batching=adaptive_batching,
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
    adaptive_batching: bool = False,
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
        adaptive_batching=adaptive_batching,
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


def _build_sampler_groups(
    dataset_indices: np.ndarray,
    tile_to_st: np.ndarray,
) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    current_st = -1
    start = 0

    for pos, tile_idx in enumerate(dataset_indices):
        st_id = int(tile_to_st[tile_idx])
        if st_id != current_st:
            if pos > start:
                groups.append(np.arange(start, pos, dtype=np.int64))
            current_st = st_id
            start = pos

    if start < len(dataset_indices):
        groups.append(np.arange(start, len(dataset_indices), dtype=np.int64))

    return groups


def save_features(
    features: Tensor,
    output_dir: Path,
    slide_id: str,
    *,
    tile_index: Tensor | np.ndarray | None = None,
) -> Path:
    """Atomically save features, normalizing tile rows by tile_index when provided."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_path = output_dir / f"{slide_id}.pt"
    tmp_path = output_dir / f"{slide_id}.pt.tmp"
    if tile_index is not None:
        if features.ndim != 2:
            raise ValueError("tile_index can only be used when saving tile-level features")
        tile_index_t = torch.as_tensor(tile_index, dtype=torch.long)
        if tile_index_t.ndim != 1 or tile_index_t.numel() != features.shape[0]:
            raise ValueError("tile_index must align with the first dimension of features")
        sorted_index, order = torch.sort(tile_index_t)
        expected = torch.arange(features.shape[0], dtype=torch.long)
        if not torch.equal(sorted_index.cpu(), expected):
            raise ValueError("tile_index must be a permutation of 0..num_tiles-1")
        features = features[order]
    torch.save(features, tmp_path)
    os.replace(tmp_path, pt_path)
    return pt_path
