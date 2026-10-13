"""Training-batch samplers, by name: the ROI batch strategies and the event-balanced sampler.

Each entry's params are what a user sets; the sample ids, per-ROI class pixel counts or
event indicators, the batch size and the seed come from the run.
"""

from __future__ import annotations

from dataclasses import dataclass

from soma.learn.roi_sampler import ClassConditionedRoiBatchSampler, UniformRoiBatchSampler
from soma._internal.params import check_positive
from soma.registry import Registry
from soma.learn.survival_sampler import EventBalancedBatchSampler

__all__ = [
    "ClassConditionedRoiParams",
    "EventBalancedParams",
    "UniformRoiParams",
    "sampler_registry",
]

sampler_registry = Registry("samplers")


@dataclass
class UniformRoiParams:
    """:class:`~soma.learn.roi_sampler.UniformRoiBatchSampler`'s params; ``None`` draws
    rounds the ROIs down to whole effective batches."""

    draws_per_epoch: int | None = None

    def __post_init__(self) -> None:
        check_positive("draws_per_epoch", self.draws_per_epoch, integer=True, optional=True)


@dataclass
class ClassConditionedRoiParams:
    """:class:`~soma.learn.roi_sampler.ClassConditionedRoiBatchSampler`'s params;
    ``None`` ratios request every class equally."""

    draws_per_epoch: int | None = None
    class_request_ratios: list[float] | None = None

    def __post_init__(self) -> None:
        check_positive("draws_per_epoch", self.draws_per_epoch, integer=True, optional=True)


@dataclass
class EventBalancedParams:
    """:class:`~soma.learn.survival_sampler.EventBalancedBatchSampler`'s params."""

    min_events_per_window: int = 1

    def __post_init__(self) -> None:
        check_positive("min_events_per_window", self.min_events_per_window, integer=True)


sampler_registry.register("uniform", UniformRoiBatchSampler, params=UniformRoiParams)
sampler_registry.register("class_conditioned", ClassConditionedRoiBatchSampler, params=ClassConditionedRoiParams)
sampler_registry.register("event_balanced", EventBalancedBatchSampler, params=EventBalancedParams)
