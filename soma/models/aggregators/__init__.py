"""Aggregator module — MIL bag-level aggregation."""

from soma.models.aggregators.base import Aggregator, AggregatorOutput
from soma.models.aggregators.registry import aggregator_registry

# Import to trigger registration
from soma.models.aggregators.pooling import MaxPool, MeanPool
from soma.models.aggregators.mil import ABMIL, CLAM_MB, CLAM_SB, DSMIL, DTFDMIL, HIPT, TransMIL

__all__ = [
    "Aggregator",
    "AggregatorOutput",
    "aggregator_registry",
    "MeanPool",
    "MaxPool",
    "ABMIL",
    "CLAM_SB",
    "CLAM_MB",
    "DSMIL",
    "DTFDMIL",
    "HIPT",
    "TransMIL",
]
