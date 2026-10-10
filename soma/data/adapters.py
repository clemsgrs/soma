"""Adapters that turn user-held arrays or a directory of files into a :class:`SetSource`."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from soma.data.validation import ensure_filename_safe_id

__all__ = ["ArraySetSource", "DirectorySetSource", "from_arrays", "from_directory"]

_FEATURE_SUFFIXES = (".pt", ".npy", ".npz", ".h5")
_COORDS_SUFFIXES = (".coords.pt", ".coords.npy", ".coords.npz", ".coordinates.npz")


def _as_float_tensor(value) -> Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(np.asarray(value))
    if tensor.is_floating_point() and tensor.dtype != torch.float32:
        tensor = tensor.float()
    return tensor


def _declare_rank(tensor: Tensor, *, sample_id: str) -> tuple[int, int]:
    if tensor.ndim not in (1, 2, 3):
        raise ValueError(
            f"Features for '{sample_id}' have rank {tensor.ndim}; a set source serves rank "
            "1 (D,), 2 (N, D) or 3 (M, K, D) tensors."
        )
    return int(tensor.ndim), int(tensor.shape[-1])


def _check_shape(tensor: Tensor, *, rank: int, feature_dim: int, sample_id: str) -> Tensor:
    if tensor.ndim != rank:
        raise ValueError(
            f"Features for '{sample_id}' have rank {tensor.ndim}, but the source declares "
            f"rank {rank}."
        )
    if int(tensor.shape[-1]) != feature_dim:
        raise ValueError(
            f"Features for '{sample_id}' have feature_dim {tensor.shape[-1]}, but the "
            f"source declares feature_dim {feature_dim}."
        )
    return tensor


class ArraySetSource:
    """In-memory :class:`SetSource` over tensors the caller already holds."""

    def __init__(
        self,
        features: Mapping[str, Tensor | np.ndarray],
        coords: Mapping[str, Tensor | np.ndarray] | None = None,
    ) -> None:
        if not features:
            raise ValueError("from_arrays needs at least one sample.")
        self._features = {str(k): _as_float_tensor(v) for k, v in features.items()}
        first_id, first = next(iter(self._features.items()))
        self._rank, self._feature_dim = _declare_rank(first, sample_id=first_id)
        for sample_id, tensor in self._features.items():
            _check_shape(tensor, rank=self._rank, feature_dim=self._feature_dim, sample_id=sample_id)
        self._coords = {str(k): torch.as_tensor(np.asarray(v)) if not torch.is_tensor(v) else v for k, v in (coords or {}).items()}
        unknown = sorted(set(self._coords) - set(self._features))
        if unknown:
            raise ValueError(f"coords name sample(s) without features: {unknown}")

    @property
    def sample_ids(self) -> list[str]:
        return list(self._features)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def rank(self) -> Literal[1, 2, 3]:
        return self._rank  # type: ignore[return-value]

    def load(self, sample_id: str) -> Tensor:
        try:
            return self._features[sample_id].clone()
        except KeyError:
            raise KeyError(
                f"Sample '{sample_id}' not found. Available: {sorted(self._features)}"
            ) from None

    def coords(self, sample_id: str) -> Tensor | None:
        if sample_id not in self._features:
            raise KeyError(f"Sample '{sample_id}' not found. Available: {sorted(self._features)}")
        value = self._coords.get(sample_id)
        return None if value is None else value.clone()

    def __len__(self) -> int:
        return len(self._features)


def from_arrays(
    features: Mapping[str, Tensor | np.ndarray],
    coords: Mapping[str, Tensor | np.ndarray] | None = None,
) -> ArraySetSource:
    """A :class:`SetSource` over tensors in memory; rank declared from the first and checked on all."""
    return ArraySetSource(features, coords)


def _read_array(path: Path, *, key: str = "features"):
    suffix = path.suffix.lower()
    if suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=True)
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        with np.load(path) as archive:
            if key in archive.files:
                return archive[key]
            if len(archive.files) == 1:
                return archive[archive.files[0]]
            raise ValueError(
                f"{path} holds arrays {archive.files}; expected a single array or one named {key!r}."
            )
    if suffix == ".h5":
        import h5py

        with h5py.File(path, "r") as handle:
            if key in handle:
                return handle[key][()]
            names = list(handle.keys())
            if len(names) == 1:
                return handle[names[0]][()]
            raise ValueError(
                f"{path} holds datasets {names}; expected a single dataset or one named {key!r}."
            )
    raise ValueError(f"Unsupported feature file {path}; expected one of {_FEATURE_SUFFIXES}.")


def _read_coords(path: Path) -> Tensor:
    """Read a sibling coords file as an ``(N, 2)`` level-0 ``(x, y)`` tensor.

    A ``.npz`` holding ``x`` and ``y`` arrays is hs2p's ``<id>.coordinates.npz`` tiling
    artifact (``tile_index``, ``x``, ``y``, ``tissue_fractions``); its rows are returned
    in ``tile_index`` order, the order the features were extracted in. Any other file is
    a single ``(N, 2)`` array (or one named ``coords``).
    """
    if path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            if "x" in archive.files and "y" in archive.files:
                x, y = archive["x"], archive["y"]
                if "tile_index" in archive.files:
                    order = np.argsort(archive["tile_index"], kind="stable")
                    x, y = x[order], y[order]
                return torch.as_tensor(np.stack([x, y], axis=1))
    array = _read_array(path, key="coords")
    return array if torch.is_tensor(array) else torch.as_tensor(np.asarray(array))


class DirectorySetSource:
    """:class:`SetSource` over ``<stem>.pt|.npy|.npz|.h5`` files with optional sibling coords.

    ``ids`` maps each ``sample_id`` to its file stem; without it the stem is the id.
    Ids are never guessed from directory structure. The rank is declared from the first
    file and every ``load`` is checked against it.
    """

    def __init__(self, directory: str | Path, *, ids: Mapping[str, str] | None = None) -> None:
        self._dir = Path(directory)
        if not self._dir.is_dir():
            raise FileNotFoundError(f"{self._dir} is not a directory.")
        if ids is None:
            stems: dict[str, str] = {}
            for path in sorted(self._dir.iterdir()):
                if path.suffix.lower() in _FEATURE_SUFFIXES and not any(
                    path.name.endswith(s) for s in _COORDS_SUFFIXES
                ):
                    stems.setdefault(path.stem, path.stem)
            if not stems:
                raise ValueError(f"No feature files ({_FEATURE_SUFFIXES}) found in {self._dir}.")
        else:
            stems = {str(sample_id): str(stem) for sample_id, stem in ids.items()}
        self._index: dict[str, Path] = {}
        for sample_id, stem in stems.items():
            ensure_filename_safe_id(sample_id)
            path = self._feature_path(stem)
            if path is None:
                raise FileNotFoundError(
                    f"No feature file {stem}{{{','.join(_FEATURE_SUFFIXES)}}} in {self._dir} "
                    f"for sample '{sample_id}'."
                )
            self._index[sample_id] = path
        first_id, first_path = next(iter(self._index.items()))
        self._rank, self._feature_dim = _declare_rank(
            _as_float_tensor(_read_array(first_path)), sample_id=first_id
        )

    def _feature_path(self, stem: str) -> Path | None:
        for suffix in _FEATURE_SUFFIXES:
            candidate = self._dir / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
        return None

    @property
    def directory(self) -> Path:
        return self._dir

    @property
    def sample_ids(self) -> list[str]:
        return list(self._index)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def rank(self) -> Literal[1, 2, 3]:
        return self._rank  # type: ignore[return-value]

    def _path(self, sample_id: str) -> Path:
        try:
            return self._index[sample_id]
        except KeyError:
            raise KeyError(
                f"Sample '{sample_id}' not found. Available: {sorted(self._index)}"
            ) from None

    def load(self, sample_id: str) -> Tensor:
        tensor = _as_float_tensor(_read_array(self._path(sample_id)))
        return _check_shape(tensor, rank=self._rank, feature_dim=self._feature_dim, sample_id=sample_id)

    def coords(self, sample_id: str) -> Tensor | None:
        stem = self._path(sample_id).name
        stem = stem[: stem.rfind(".")]
        for suffix in _COORDS_SUFFIXES:
            candidate = self._dir / f"{stem}{suffix}"
            if candidate.is_file():
                return _read_coords(candidate)
        return None

    def __len__(self) -> int:
        return len(self._index)


def from_directory(path: str | Path, ids: Mapping[str, str] | None = None) -> DirectorySetSource:
    """A :class:`SetSource` over a directory of ``<id>.pt|.npy|.npz|.h5`` files."""
    return DirectorySetSource(path, ids=ids)
