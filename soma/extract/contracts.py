"""Contracts of persistent frozen-feature extraction: the loading surface of a feature
source, and what an extraction records in ``extraction_provenance.json``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class FeatureSource(Protocol):
    """Minimal loading surface shared by every persistent feature representation."""

    @property
    def sample_ids(self) -> list[str]: ...

    @property
    def feature_dim(self) -> int: ...

    def load(self, sample_id: str) -> torch.Tensor: ...


@runtime_checkable
class PooledFeatureSource(FeatureSource, Protocol):
    """Source contract for vectors, tile bags, and hierarchical tensors."""

    @property
    def rank(self) -> int: ...


@dataclass(frozen=True)
class FeatureProvenance:
    """Completed extraction facts ``extraction_provenance.json`` records."""

    kind: str
    encoder_name: str | None = None
    zero_sample_ids: tuple[str, ...] = ()

    @property
    def zero_roi_sample_ids(self) -> tuple[str, ...]:
        """Parent slides for which annotation sampling produced no ROI."""
        return self.zero_sample_ids


@dataclass(frozen=True)
class ExtractionArtifacts:
    """Deterministic files and directories published by an extraction."""

    feature_dir: Path
    tiling_dir: Path | None = None
    dataset_csv: Path | None = None
    provenance_json: Path | None = None
