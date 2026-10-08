"""Aggregator registry."""

from soma.registry import Registry

aggregator_registry = Registry("aggregators", bundled_module="soma.benchmarks")
