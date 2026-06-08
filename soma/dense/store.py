"""DenseFeatureStore — load cached dense ``(d, h, w)`` feature grids.

Unlike :class:`soma.features.FeatureStore`, which infers a feature's shape from
its tensor *rank* (1-D slide / 2-D bag / 3-D hierarchical), a dense grid is also
3-D ``(channels, grid_h, grid_w)`` and would be mis-read as hierarchical with
``feature_dim`` taken from the last axis (``w``) instead of the channel axis
(``d``). The dense store therefore reads geometry from a **per-sample
``.meta.json`` sidecar**, never from rank.

The sidecar is the single source of truth for shape, written next to every
``.pt`` regardless of whether the feature cache is enabled — so the store behaves
identically on cached, uncached, and (later) live-re-encode paths. ``cache_metadata.json``
is used only by the cache *validator*, not here.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from slide2vec.artifacts import load_array

from soma.dataset import ensure_filename_safe_id
from soma.dense.geometry import DenseGridGeometry

__all__ = [
    "DENSE_SIDECAR_SUFFIX",
    "DENSE_ARTIFACT_TYPE",
    "DENSE_PAYLOAD_SUBDIR",
    "resolve_dense_payload_dir",
    "dense_grid_metadata",
    "write_dense_grid",
    "DenseFeatureStore",
]

DENSE_SIDECAR_SUFFIX = ".meta.json"
DENSE_ARTIFACT_TYPE = "dense_grid"
DENSE_PAYLOAD_SUBDIR = "dense_embeddings"


def resolve_dense_payload_dir(path: Path | str) -> Path:
    """Resolve the directory holding dense ``.pt`` grids.

    Dense-specific on purpose: it descends into ``dense_embeddings/`` if present,
    else treats ``path`` as a plain payload dir. Unlike the generic pooled
    ``resolve_feature_payload_dir``, it never falls through to a sibling
    ``tile_embeddings``/``slide_embeddings`` dir — so a cache root that happens to
    hold both pooled and dense payloads still resolves to the dense grids.
    """
    root = Path(path)
    candidate = root / DENSE_PAYLOAD_SUBDIR
    if candidate.is_dir():
        return candidate
    return root


def dense_grid_metadata(
    geometry: DenseGridGeometry,
    *,
    feature_dim: int,
    pad_mode: str,
    image_pad_value: float,
    mask_pad_value: int,
    dense_input_mode: str = "whole",
    channel_dim: int = 0,
) -> dict:
    """Build the self-describing sidecar payload for one dense grid.

    Carries everything the decoder/head and evaluation need to map the grid back
    to the mask: the channel/grid layout plus the §5 padding+crop geometry.
    """
    return {
        "artifact_type": DENSE_ARTIFACT_TYPE,
        "feature_type": DENSE_ARTIFACT_TYPE,
        "feature_dim": int(feature_dim),
        "channel_dim": int(channel_dim),
        "grid_shape": [int(geometry.grid_shape[0]), int(geometry.grid_shape[1])],
        "target_size": [int(geometry.target_size[0]), int(geometry.target_size[1])],
        "encoded_size": [int(geometry.encoded_size[0]), int(geometry.encoded_size[1])],
        "patch_size": [int(geometry.patch_size[0]), int(geometry.patch_size[1])],
        "pad": [int(geometry.pad[0]), int(geometry.pad[1])],
        "pad_mode": str(pad_mode),
        "crop_box": [int(v) for v in geometry.crop_box],
        "image_pad_value": float(image_pad_value),
        "mask_pad_value": int(mask_pad_value),
        "dense_input_mode": str(dense_input_mode),
    }


def write_dense_grid(
    out_dir: Path | str,
    sample_id: str,
    grid: torch.Tensor,
    metadata: dict,
) -> Path:
    """Write a ``(d, h, w)`` grid plus its sidecar, validating shape vs metadata.

    Fails loud if the tensor's channel/grid axes disagree with the metadata — a
    silent mismatch here would train the decoder on the wrong layout.
    """
    if grid.ndim != 3:
        raise ValueError(
            f"dense grid for '{sample_id}' must be 3-D (channels, grid_h, grid_w), "
            f"got shape {tuple(grid.shape)}"
        )
    channel_dim = int(metadata["channel_dim"])
    if channel_dim != 0:
        raise ValueError(
            f"write_dense_grid only supports channel_dim=0, got {channel_dim}"
        )
    feature_dim = int(metadata["feature_dim"])
    grid_h, grid_w = (int(v) for v in metadata["grid_shape"])
    if tuple(grid.shape) != (feature_dim, grid_h, grid_w):
        raise ValueError(
            f"dense grid for '{sample_id}' has shape {tuple(grid.shape)} but metadata "
            f"declares (feature_dim={feature_dim}, grid_shape=({grid_h}, {grid_w}))"
        )
    sample_id = ensure_filename_safe_id(sample_id)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / f"{sample_id}.pt"
    sidecar_path = out_dir / f"{sample_id}{DENSE_SIDECAR_SUFFIX}"
    torch.save(grid, feature_path)
    sidecar_path.write_text(json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8")
    return feature_path


class DenseFeatureStore:
    """Index and load dense ``(d, h, w)`` grids from a directory of ``.pt`` files.

    Every ``.pt`` must have a matching ``<sample_id>.meta.json`` sidecar; shape is
    read from the sidecar, never inferred from tensor rank.
    """

    def __init__(self, feature_dir: Path | str) -> None:
        # Accept a cache dir and descend into dense_embeddings/, or a plain dir of
        # .pt files as-is. Dense-specific resolver: never falls through to a pooled
        # sibling dir (e.g. tile_embeddings) if both happen to exist.
        self._feature_dir = resolve_dense_payload_dir(feature_dir)
        self._index: dict[str, Path] = {}
        self._meta_cache: dict[str, dict] = {}
        self._feature_dim: int | None = None
        self._grid_shape: tuple[int, int] | None = None
        self._build_index()

    def _build_index(self) -> None:
        for path in sorted(self._feature_dir.glob("*.pt")):
            self._index[path.stem] = path

    def _sidecar_path(self, sample_id: str) -> Path:
        return self._feature_dir / f"{sample_id}{DENSE_SIDECAR_SUFFIX}"

    def metadata(self, sample_id: str) -> dict:
        """Return the sidecar metadata for ``sample_id`` (cached)."""
        if sample_id in self._meta_cache:
            return self._meta_cache[sample_id]
        if sample_id not in self._index:
            raise KeyError(
                f"Sample '{sample_id}' not found in dense feature store. "
                f"Available: {sorted(self._index)}"
            )
        sidecar = self._sidecar_path(sample_id)
        if not sidecar.is_file():
            raise FileNotFoundError(
                f"Dense feature '{sample_id}' is missing its required sidecar {sidecar.name}; "
                "shape cannot be read from rank for dense grids."
            )
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        self._meta_cache[sample_id] = meta
        return meta

    def _ensure_shape(self) -> None:
        if self._feature_dim is not None:
            return
        if not self._index:
            raise ValueError("Cannot determine dense feature_dim: no grids found")
        first_id = next(iter(self._index))
        meta = self.metadata(first_id)
        self._feature_dim = int(meta["feature_dim"])
        gh, gw = (int(v) for v in meta["grid_shape"])
        self._grid_shape = (gh, gw)

    @property
    def available_samples(self) -> list[str]:
        return list(self._index.keys())

    @property
    def feature_dim(self) -> int:
        """Channel dimensionality ``d`` — from the sidecar, not the last axis."""
        self._ensure_shape()
        assert self._feature_dim is not None
        return self._feature_dim

    @property
    def grid_shape(self) -> tuple[int, int]:
        self._ensure_shape()
        assert self._grid_shape is not None
        return self._grid_shape

    @property
    def feature_dir(self) -> Path:
        return self._feature_dir

    def load(self, sample_id: str) -> torch.Tensor:
        """Load the dense grid for ``sample_id`` as a float32 ``(d, h, w)`` tensor."""
        if sample_id not in self._index:
            raise KeyError(
                f"Sample '{sample_id}' not found in dense feature store. "
                f"Available: {sorted(self._index)}"
            )
        tensor = load_array(self._index[sample_id])
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        meta = self.metadata(sample_id)
        feature_dim = int(meta["feature_dim"])
        grid_h, grid_w = (int(v) for v in meta["grid_shape"])
        if tuple(tensor.shape) != (feature_dim, grid_h, grid_w):
            raise ValueError(
                f"Dense grid '{sample_id}' on disk has shape {tuple(tensor.shape)} but its "
                f"sidecar declares (d={feature_dim}, grid=({grid_h}, {grid_w}))."
            )
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            return tensor.float()
        return tensor

    def validate_coverage(self, sample_ids: list[str]) -> None:
        missing = sorted(set(sample_ids) - set(self._index))
        if missing:
            raise ValueError(f"Missing dense features for {len(missing)} samples: {missing}")

    def __len__(self) -> int:
        return len(self._index)
