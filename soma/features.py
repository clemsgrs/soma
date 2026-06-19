"""FeatureStore — index and load precomputed tile embeddings from disk."""

from __future__ import annotations

import csv
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from slide2vec.artifacts import load_array

from soma.cache import resolve_feature_payload_dir

# Single-file packed cache of all 1-D (single-vector) features, written next to
# the per-sample files. Lets repeat training runs (e.g. multi-seed sweeps) read
# the whole feature matrix in one shot instead of one tiny file per sample.
PACKED_FILENAME = "packed_features.pt"


class FeatureStore:
    """Indexes and loads precomputed feature embeddings produced by feature extraction.

    Expects a directory of .pt files, one per sample. Each file contains either:
    - a tensor of shape (num_tiles, feature_dim) for tile-level features, or
    - a tensor of shape (num_regions, num_tiles_per_region, feature_dim) for
      hierarchical features, or
    - a tensor of shape (feature_dim,) for slide-level features.

    For single-vector (1-D) features, ``load`` is served from an in-memory packed
    matrix so each file is read from disk at most once per process (and at most
    once ever, once the on-disk packed cache exists), rather than once per epoch.
    """

    def __init__(self, feature_dir: Path | str) -> None:
        self._feature_dir = resolve_feature_payload_dir(feature_dir)
        self._index: dict[str, Path] = {}
        self._sample_statuses: dict[str, str] = {}
        self._feature_manifest_path: Path | None = None
        self._feature_dim: int | None = None
        self._is_slide_level: bool | None = None
        self._is_hierarchical: bool | None = None
        self._feature_rank: int | None = None
        # In-memory packed matrix for 1-D features (built lazily on first load).
        self._packed_matrix: torch.Tensor | None = None
        self._packed_row: dict[str, int] = {}
        self._packed_attempted: bool = False
        self._build_index()
        self._load_feature_manifest()

    def _build_index(self) -> None:
        for path in sorted(
            [
                *self._feature_dir.glob("*.pt"),
                *self._feature_dir.glob("*.npz"),
            ]
        ):
            if path.name == PACKED_FILENAME:
                continue
            sample_id = path.stem
            self._index[sample_id] = path

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
                reader = csv.DictReader(handle)
                for row in reader:
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

    @property
    def available_samples(self) -> list[str]:
        return list(self._index.keys())

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
            return self.available_samples
        return [sample_id for sample_id, status in self._sample_statuses.items() if status == "success"]

    @property
    def empty_feature_samples(self) -> list[str]:
        return [sample_id for sample_id, status in self._sample_statuses.items() if status == "empty"]

    @property
    def is_slide_level(self) -> bool:
        """True if features are slide-level (1-D per sample), False if tile-level (2-D)."""
        self._ensure_metadata()
        return self._is_slide_level

    @property
    def is_hierarchical(self) -> bool:
        """True if features are hierarchical (3-D per sample)."""
        self._ensure_metadata()
        return self._is_hierarchical

    @property
    def feature_rank(self) -> int:
        """Rank of the stored feature tensors."""
        self._ensure_metadata()
        return self._feature_rank

    @property
    def feature_dim(self) -> int:
        """Feature dimensionality (inferred from the first file)."""
        self._ensure_metadata()
        return self._feature_dim

    def _ensure_metadata(self) -> None:
        if self._feature_dim is not None:
            return
        if not self._index:
            msg = "Cannot determine feature_dim: no features found"
            raise ValueError(msg)
        first_path = next(iter(self._index.values()))
        tensor = load_array(first_path)
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        if tensor.ndim not in {1, 2, 3}:
            raise ValueError(
                f"Unsupported feature tensor rank {tensor.ndim} in {first_path}; "
                "expected 1-D, 2-D, or 3-D tensors."
            )
        self._feature_rank = int(tensor.ndim)
        self._is_slide_level = tensor.ndim == 1
        self._is_hierarchical = tensor.ndim == 3
        self._feature_dim = tensor.shape[0] if tensor.ndim == 1 else tensor.shape[-1]

    def load(self, sample_id: str) -> torch.Tensor:
        """Load embeddings for a single sample.

        Floating-point features are normalized to ``float32`` so downstream
        linear layers do not fail when caches were written in fp16/bf16. For
        single-vector (1-D) features the value is served from an in-memory
        packed matrix; other ranks (bags, hierarchical) are read per-file.
        """
        if sample_id not in self._index:
            msg = f"Sample '{sample_id}' not found in feature store. Available: {sorted(self._index)}"
            raise KeyError(msg)
        if self._packed_matrix is None and not self._packed_attempted:
            self._maybe_build_packed_cache()
        if self._packed_matrix is not None:
            row = self._packed_row.get(sample_id)
            if row is not None:
                # Clone so a caller can mutate the result in place (e.g. an
                # augmentation/normalization transform) without corrupting the
                # shared packed matrix; the per-file path also returns a fresh
                # tensor per call. A single feature vector is cheap to copy.
                return self._packed_matrix[row].clone()
        return self._read_file(self._index[sample_id])

    @staticmethod
    def _read_file(path: Path) -> torch.Tensor:
        tensor = load_array(path)
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            return tensor.float()
        return tensor

    def _maybe_build_packed_cache(self) -> None:
        """Build (or load) the in-memory packed matrix for 1-D features.

        Only applies to single-vector features; bags/hierarchical tensors are
        left on the per-file path. The packed matrix is persisted to
        ``PACKED_FILENAME`` so later runs over the same cache read it in one go.
        Any failure falls back silently to per-file loading.
        """
        self._packed_attempted = True
        self._ensure_metadata()
        if self._feature_rank != 1:
            return

        sample_ids = sorted(self._index)
        packed_path = self._feature_dir / PACKED_FILENAME
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
            # Fill a preallocated matrix in bounded chunks: never hold all N
            # tensors at once (accumulating them spiked memory enough to get the
            # process OOM-killed on large datasets), and cap concurrent reads.
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
            tmp = packed_path.with_name(PACKED_FILENAME + ".tmp")
            torch.save({"sample_ids": sample_ids, "features": matrix}, tmp)
            tmp.replace(packed_path)
        except Exception:
            pass  # in-memory pack still valid for this run; persistence is best-effort

    def validate_coverage(self, sample_ids: list[str]) -> None:
        """Check that all requested sample IDs have features on disk."""
        available = set(self.expected_feature_samples)
        missing = sorted(set(sample_ids) - available)
        if missing:
            msg = f"Missing features for {len(missing)} samples: {missing}"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self._index)

    @property
    def feature_dir(self) -> Path:
        return self._feature_dir
