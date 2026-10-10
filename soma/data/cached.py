"""CachedSetSource: the :class:`~soma.data.SetSource` over soma's pooled feature cache."""

from __future__ import annotations

import csv
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from slide2vec.artifacts import load_array

__all__ = ["CachedSetSource"]

_COORDS_SUFFIXES = (".coords.pt", ".coords.npy", ".coords.npz", ".coordinates.npz")


class CachedSetSource:
    """Index and load the embeddings an extraction wrote: one ``.pt``/``.npz`` per sample.

    Each file holds a ``(D,)`` vector (rank 1), an ``(N, D)`` tile bag (rank 2) or an
    ``(M, K, D)`` hierarchical bag (rank 3); ``rank`` is read from the first file.
    Floating-point features are served as ``float32``. Rank-1 features are served from
    an in-memory packed matrix built once per process (and persisted next to the
    per-sample files), so repeat runs read the whole matrix in one shot.

    The feature manifest (``process_list.csv``) statuses stay internal: ``sample_ids``
    lists what is on disk, ``expected_feature_samples`` what extraction succeeded on,
    ``empty_feature_samples`` the slides without tissue.
    """

    def __init__(self, feature_dir: Path | str) -> None:
        from soma.cache import resolve_feature_payload_dir
        from soma.cache._types import PACKED_FILENAME

        self._packed_filename = PACKED_FILENAME
        self._feature_dir = resolve_feature_payload_dir(feature_dir)
        self._index: dict[str, Path] = {}
        self._sample_statuses: dict[str, str] = {}
        self._feature_manifest_path: Path | None = None
        self._feature_dim: int | None = None
        self._rank: int | None = None
        self._packed_matrix: torch.Tensor | None = None
        self._packed_row: dict[str, int] = {}
        self._packed_attempted: bool = False
        self._build_index()
        self._load_feature_manifest()

    def _build_index(self) -> None:
        for path in sorted([*self._feature_dir.glob("*.pt"), *self._feature_dir.glob("*.npz")]):
            if path.name == self._packed_filename or any(
                path.name.endswith(suffix) for suffix in _COORDS_SUFFIXES
            ):
                continue
            self._index[path.stem] = path

    def _load_feature_manifest(self) -> None:
        candidate_paths = [
            self._feature_dir / "process_list.csv",
            self._feature_dir.parent / "process_list.csv",
        ]
        for path in candidate_paths:
            if not path.is_file():
                continue
            self._feature_manifest_path = path
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    sample_id = str(row["sample_id"])
                    if sample_id in self._sample_statuses:
                        raise ValueError(f"Duplicate sample_id in feature manifest: {sample_id}")
                    status = str(row.get("feature_status", "")).strip().lower()
                    if not status:
                        raise ValueError(
                            f"Missing feature_status for sample_id={sample_id} in {path}"
                        )
                    self._sample_statuses[sample_id] = status
                    if sample_id in self._index:
                        continue
                    feature_path_text = str(row.get("feature_path", "")).strip()
                    if not feature_path_text:
                        continue
                    feature_path = Path(feature_path_text)
                    if not feature_path.is_absolute():
                        feature_path = (path.parent / feature_path).resolve()
                    if feature_path.is_file():
                        self._index[sample_id] = feature_path
            return

    # --- SetSource ------------------------------------------------------------------ #

    @property
    def sample_ids(self) -> list[str]:
        return list(self._index.keys())

    @property
    def feature_dim(self) -> int:
        """Feature dimensionality (inferred from the first file)."""
        self._ensure_metadata()
        assert self._feature_dim is not None
        return self._feature_dim

    @property
    def rank(self) -> Literal[1, 2, 3]:
        """Rank of the stored feature tensors: 1 vector, 2 bag, 3 hierarchical bag."""
        self._ensure_metadata()
        assert self._rank is not None
        return self._rank  # type: ignore[return-value]

    def load(self, sample_id: str) -> torch.Tensor:
        if sample_id not in self._index:
            raise KeyError(
                f"Sample '{sample_id}' not found in feature store. Available: {sorted(self._index)}"
            )
        if self._packed_matrix is None and not self._packed_attempted:
            self._maybe_build_packed_cache()
        if self._packed_matrix is not None:
            row = self._packed_row.get(sample_id)
            if row is not None:
                # Clone so a caller can mutate the result in place without corrupting the
                # shared packed matrix; the per-file path also returns a fresh tensor.
                return self._packed_matrix[row].clone()
        return self._read_file(self._index[sample_id])

    def coords(self, sample_id: str) -> torch.Tensor | None:
        """Level-0 tile origins written beside the features, when extraction kept them."""
        path = self._index.get(sample_id)
        if path is None:
            raise KeyError(
                f"Sample '{sample_id}' not found in feature store. Available: {sorted(self._index)}"
            )
        from soma.data.adapters import _read_coords

        for suffix in _COORDS_SUFFIXES:
            candidate = path.with_name(f"{path.stem}{suffix}")
            if candidate.is_file():
                return _read_coords(candidate)
        return None

    # --- feature manifest ----------------------------------------------------------- #

    @property
    def has_feature_manifest(self) -> bool:
        return self._feature_manifest_path is not None

    @property
    def feature_manifest_path(self) -> Path | None:
        return self._feature_manifest_path

    @property
    def feature_statuses(self) -> dict[str, str]:
        return dict(self._sample_statuses)

    @property
    def expected_feature_samples(self) -> list[str]:
        if not self._sample_statuses:
            return self.sample_ids
        return [sid for sid, status in self._sample_statuses.items() if status == "success"]

    @property
    def empty_feature_samples(self) -> list[str]:
        return [sid for sid, status in self._sample_statuses.items() if status == "empty"]

    @property
    def feature_dir(self) -> Path:
        return self._feature_dir

    def __len__(self) -> int:
        return len(self._index)

    # --- internals ------------------------------------------------------------------ #

    def _ensure_metadata(self) -> None:
        if self._feature_dim is not None:
            return
        if not self._index:
            raise ValueError("Cannot determine feature_dim: no features found")
        first_path = next(iter(self._index.values()))
        tensor = load_array(first_path)
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        if tensor.ndim not in {1, 2, 3}:
            raise ValueError(
                f"Unsupported feature tensor rank {tensor.ndim} in {first_path}; "
                "expected 1-D, 2-D, or 3-D tensors."
            )
        self._rank = int(tensor.ndim)
        self._feature_dim = int(tensor.shape[-1])

    @staticmethod
    def _read_file(path: Path) -> torch.Tensor:
        tensor = load_array(path)
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            return tensor.float()
        return tensor

    def _maybe_build_packed_cache(self) -> None:
        """Build (or load) the in-memory packed matrix for rank-1 features.

        Persisted to the packed filename so later runs over the same cache read it in
        one go. Any failure falls back silently to per-file loading.
        """
        self._packed_attempted = True
        self._ensure_metadata()
        if self._rank != 1:
            return

        sample_ids = sorted(self._index)
        packed_path = self._feature_dir / self._packed_filename
        if packed_path.is_file():
            try:
                blob = torch.load(packed_path, map_location="cpu")
                ids = list(blob["sample_ids"])
                feats = blob["features"]
                if set(ids) >= set(self._index) and feats.shape[0] == len(ids):
                    self._packed_matrix = feats.float() if feats.is_floating_point() else feats
                    self._packed_row = {sid: i for i, sid in enumerate(ids)}
                    return
            except Exception:
                pass  # stale/corrupt pack -> rebuild below

        try:
            dim = int(self._feature_dim)
            matrix = torch.empty((len(sample_ids), dim), dtype=torch.float32)
            # Fill a preallocated matrix in bounded chunks: never hold all N tensors at
            # once, and cap concurrent reads.
            max_workers = min(8, (os.cpu_count() or 4))
            chunk = 8192
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for start in range(0, len(sample_ids), chunk):
                    batch = sample_ids[start : start + chunk]
                    for offset, tensor in enumerate(
                        pool.map(lambda sid: self._read_file(self._index[sid]), batch)
                    ):
                        matrix[start + offset] = tensor.reshape(-1)
        except Exception:
            self._packed_matrix = None
            return
        self._packed_matrix = matrix
        self._packed_row = {sid: i for i, sid in enumerate(sample_ids)}
        try:
            tmp = packed_path.with_name(self._packed_filename + ".tmp")
            torch.save({"sample_ids": sample_ids, "features": matrix}, tmp)
            tmp.replace(packed_path)
        except Exception:
            pass  # in-memory pack still valid for this run; persistence is best-effort
