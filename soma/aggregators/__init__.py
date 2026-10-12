"""Aggregator module — MIL bag-level aggregation."""

from soma.aggregators.base import Aggregator, AggregatorOutput
from soma.aggregators.registry import aggregator_registry

# Import to trigger registration
from soma.aggregators.pooling import MaxPool, MeanPool
from soma.aggregators.mil import ABMIL, CLAM_MB, CLAM_SB, DSMIL, DTFDMIL, HIPT, TransMIL

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
