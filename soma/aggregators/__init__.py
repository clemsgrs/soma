"""Aggregator module — MIL bag-level aggregation."""

from soma.aggregators.base import Aggregator, AggregatorOutput
from soma.aggregators.registry import aggregator_registry

# Import to trigger registration
from soma.aggregators.pooling import MaxPool, MeanPool
from soma.aggregators.mil import ABMIL, CLAM, DSMIL, DTFDMIL, TransMIL

__all__ = [
    "Aggregator",
    "AggregatorOutput",
    "aggregator_registry",
    "MeanPool",
    "MaxPool",
    "ABMIL",
    "CLAM",
    "DSMIL",
    "DTFDMIL",
    "TransMIL",
]
