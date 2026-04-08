"""FeatureStore — index and load precomputed tile embeddings from disk."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from soma.cache import resolve_feature_payload_dir

from slide2vec.artifacts import load_array


class FeatureStore:
    """Indexes and loads precomputed feature embeddings produced by feature extraction.

    Expects a directory of .pt files, one per sample. Each file contains either:
    - a tensor of shape (num_tiles, feature_dim) for tile-level features, or
    - a tensor of shape (num_regions, num_tiles_per_region, feature_dim) for
      hierarchical features, or
    - a tensor of shape (feature_dim,) for slide-level features.
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
        self._build_index()
        self._load_feature_manifest()

    def _build_index(self) -> None:
        for path in sorted(
            [
                *self._feature_dir.glob("*.pt"),
                *self._feature_dir.glob("*.npz"),
            ]
        ):
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
        """Load tile embeddings for a single sample.

        Returns the stored feature tensor unchanged.
        """
        if sample_id not in self._index:
            msg = f"Sample '{sample_id}' not found in feature store. Available: {sorted(self._index)}"
            raise KeyError(msg)
        tensor = load_array(self._index[sample_id])
        if torch.is_tensor(tensor):
            return tensor
        return torch.as_tensor(tensor)

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
