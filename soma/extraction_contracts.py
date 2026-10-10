"""Public contracts for persistent frozen-feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

from soma.data._legacy import LegacySamples

#: Delete in slice 7 (#581): extraction still consumes the legacy fat records.
FeatureDataset = LegacySamples


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
    """Completed extraction facts needed to interpret an extraction result."""

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


@dataclass(frozen=True)
class FeatureExtractionResult:
    """The exact result returned by :meth:`FeatureExtractor.extract`."""

    source: FeatureSource
    dataset: FeatureDataset
    provenance: FeatureProvenance
    artifacts: ExtractionArtifacts
