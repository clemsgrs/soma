"""FeatureStore — index and load precomputed tile embeddings from disk."""

from __future__ import annotations

from pathlib import Path

import torch

from soma.cache import resolve_feature_payload_dir


class FeatureStore:
    """Indexes and loads precomputed feature embeddings produced by feature extraction.

    Expects a directory of .pt files, one per sample. Each file contains either:
    - a tensor of shape (num_tiles, feature_dim) for tile-level features, or
    - a tensor of shape (feature_dim,) for slide-level features.
    """

    def __init__(self, feature_dir: Path | str) -> None:
        self._feature_dir = resolve_feature_payload_dir(feature_dir)
        self._index: dict[str, Path] = {}
        self._feature_dim: int | None = None
        self._is_slide_level: bool | None = None
        self._build_index()

    def _build_index(self) -> None:
        for path in sorted(self._feature_dir.glob("*.pt")):
            sample_id = path.stem
            self._index[sample_id] = path

    @property
    def available_samples(self) -> list[str]:
        return list(self._index.keys())

    @property
    def is_slide_level(self) -> bool:
        """True if features are slide-level (1-D per sample), False if tile-level (2-D)."""
        self._ensure_metadata()
        return self._is_slide_level

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
        tensor = torch.load(first_path, weights_only=True, map_location="cpu")
        self._is_slide_level = tensor.ndim == 1
        self._feature_dim = tensor.shape[0] if self._is_slide_level else tensor.shape[1]

    def load(self, sample_id: str) -> torch.Tensor:
        """Load tile embeddings for a single sample.

        Returns a tensor of shape (num_tiles, feature_dim).
        """
        if sample_id not in self._index:
            msg = f"Sample '{sample_id}' not found in feature store. Available: {sorted(self._index)}"
            raise KeyError(msg)
        return torch.load(
            self._index[sample_id], weights_only=True, map_location="cpu"
        )

    def validate_coverage(self, sample_ids: list[str]) -> None:
        """Check that all requested sample IDs have features on disk."""
        available = set(self._index)
        missing = sorted(set(sample_ids) - available)
        if missing:
            msg = f"Missing features for {len(missing)} samples: {missing}"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self._index)

    @property
    def feature_dir(self) -> Path:
        return self._feature_dir
