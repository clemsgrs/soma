"""Event-balanced batch sampling for batched CoxPH survival training.

The Cox partial likelihood is only defined over a risk set containing at least
one event. With random batching over a heavily-censored cohort (common in
pathology) some batches would contain zero events — a degenerate risk set that
contributes no gradient. This sampler instead constructs every batch to hold at
least ``min_events_per_window`` events, drawing fresh each epoch so risk sets
vary across epochs (lower-variance, less-biased Cox gradient). Emitted batches
never exceed ``batch_size``.
"""

from __future__ import annotations

from collections.abc import Iterator
from math import ceil

import numpy as np
from torch.utils.data import Sampler


class EventBalancedBatchSampler(Sampler[list[int]]):
    """Yield batches of dataset indices, each with >= ``min_events_per_window`` events.

    Indices are aligned with the dataset order, so ``events[i]`` must be the
    event indicator (1 = event, 0 = censored) of dataset item ``i``. Each epoch
    (each ``__iter__``) reshuffles both events and censored samples, guarantees
    the per-batch event floor, then fills the remaining slots from the pooled
    leftovers. Every sample is emitted once per epoch, and no emitted batch
    exceeds ``batch_size``.

    Args:
        events: Per-index event indicator (1 = event, 0 = censored), aligned
            with the dataset.
        batch_size: Target batch size (the Cox risk-set size). Must be >= 2.
        min_events_per_window: Minimum events guaranteed per batch (>= 1).
        seed: Base RNG seed; the generator state advances across epochs so each
            epoch draws a different (reproducible) partition.

    Raises:
        ValueError: If ``batch_size < 2``, ``min_events_per_window < 1``,
            ``min_events_per_window > batch_size``, or the cohort has too few
            events to put ``min_events_per_window`` in every batch.
    """

    def __init__(
        self,
        events,
        batch_size: int,
        min_events_per_window: int = 1,
        seed: int = 0,
    ) -> None:
        events = [int(e) for e in events]
        self._n = len(events)
        self._event_idx = [i for i, e in enumerate(events) if e == 1]
        self._cens_idx = [i for i, e in enumerate(events) if e != 1]
        self._batch_size = int(batch_size)
        self._min_events = int(min_events_per_window)

        if self._n < 2:
            raise ValueError("EventBalancedBatchSampler requires at least 2 samples.")
        if self._batch_size < 2:
            raise ValueError(
                f"EventBalancedBatchSampler requires batch_size >= 2, got {self._batch_size}."
            )
        if self._min_events < 1:
            raise ValueError(
                f"min_events_per_window must be >= 1, got {self._min_events}."
            )
        if self._min_events > self._batch_size:
            raise ValueError(
                f"min_events_per_window ({self._min_events}) cannot exceed "
                f"batch_size ({self._batch_size})."
            )

        self._n_batches = max(1, ceil(self._n / self._batch_size))
        self._target_sizes = _balanced_batch_sizes(self._n, self._n_batches)
        if min(self._target_sizes) < 2:
            raise ValueError(
                "Event-balanced sampling cannot cover every sample with capped "
                f"batch_size={self._batch_size} without creating a singleton Cox risk set."
            )
        if min(self._target_sizes) < self._min_events:
            raise ValueError(
                "Event-balanced sampling cannot satisfy min_events_per_window="
                f"{self._min_events} with capped batch sizes {self._target_sizes}."
            )
        n_events = len(self._event_idx)
        required = self._n_batches * self._min_events
        if n_events < required:
            raise ValueError(
                f"Event-balanced sampling needs at least {required} events "
                f"({self._n_batches} batches x {self._min_events} per batch), but the "
                f"training cohort has only {n_events}. Lower batch_size or "
                "min_events_per_window, or provide a cohort with more events."
            )

        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self) -> Iterator[list[int]]:
        events = self._rng.permutation(self._event_idx).tolist()
        censored = self._rng.permutation(self._cens_idx).tolist()

        batches: list[list[int]] = [[] for _ in range(self._n_batches)]

        # 1) Guarantee the event floor in every batch.
        cursor = 0
        for batch in batches:
            for _ in range(self._min_events):
                batch.append(int(events[cursor]))
                cursor += 1

        # 2) Fill remaining slots from the pooled leftovers (extra events + all
        #    censored), shuffled together.
        pool = self._rng.permutation(events[cursor:] + censored).tolist()
        pi = 0
        for batch, target_size in zip(batches, self._target_sizes):
            while len(batch) < target_size and pi < len(pool):
                batch.append(int(pool[pi]))
                pi += 1

        yield from batches


def _balanced_batch_sizes(n_items: int, n_batches: int) -> list[int]:
    base_size, extra = divmod(n_items, n_batches)
    return [base_size + (1 if i < extra else 0) for i in range(n_batches)]
