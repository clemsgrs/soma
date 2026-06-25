"""Training-facing dense source contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

from soma.dense.geometry import DenseGridGeometry


@runtime_checkable
class DenseFeatureSource(Protocol):
    """Cache-backed dense source surface consumed by dense training paths."""

    provenance: "DenseSourceProvenance"

    @property
    def available_samples(self) -> list[str]:
        ...

    @property
    def feature_dim(self) -> int:
        ...

    def load(self, sample_id: str) -> torch.Tensor:
        ...

    def metadata(self, sample_id: str) -> dict:
        ...

    def geometry(self, sample_id: str) -> DenseGridGeometry:
        ...

    def spacing_um(self, sample_id: str) -> float | None:
        ...

    def validate_coverage(self, sample_ids: list[str]) -> None:
        ...


@dataclass(frozen=True)
class DenseSourceProvenance:
    """Where a cache-backed dense source came from."""

    kind: str
    feature_dir: Path | str | None = None
    dataset_csv: Path | str | None = None
    splits_csv: Path | str | None = None
    parent_dataset_csv: Path | str | None = None
    parent_splits_csv: Path | str | None = None

    def to_dict(self) -> dict[str, str]:
        data: dict[str, str] = {"kind": str(self.kind)}
        for key in (
            "feature_dir",
            "dataset_csv",
            "splits_csv",
            "parent_dataset_csv",
            "parent_splits_csv",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = str(value)
        return data


class CacheBackedDenseSource:
    """Adapter over dense stores with explicit provenance and spacing access."""

    def __init__(self, store: object, *, provenance: DenseSourceProvenance) -> None:
        self._store = store
        self.provenance = provenance

    @property
    def store(self) -> object:
        return self._store

    @property
    def available_samples(self) -> list[str]:
        return list(self._store.available_samples)

    @property
    def feature_dim(self) -> int:
        return int(self._store.feature_dim)

    @property
    def feature_dir(self) -> Path | None:
        return getattr(self._store, "feature_dir", None)

    @property
    def grid_shape(self) -> tuple[int, int]:
        return tuple(int(v) for v in self._store.grid_shape)

    def load(self, sample_id: str) -> torch.Tensor:
        return self._store.load(sample_id)

    def metadata(self, sample_id: str) -> dict:
        return self._store.metadata(sample_id)

    def geometry(self, sample_id: str) -> DenseGridGeometry:
        return self._store.geometry(sample_id)

    def spacing_um(self, sample_id: str) -> float | None:
        value = self.metadata(sample_id).get("spacing_um")
        return None if value is None else float(value)

    def validate_coverage(self, sample_ids: list[str]) -> None:
        self._store.validate_coverage(sample_ids)

    def __len__(self) -> int:
        return len(self._store)
