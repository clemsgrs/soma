"""Training-batch ROI sampling for cached semantic-segmentation grids.

This module chooses already-extracted ROI dataset items for decoder training.  It
does not create ROI coordinates; that preprocessing concern lives in
``PreprocessingConfig.sampling``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from copy import deepcopy

import numpy as np
from torch.utils.data import Sampler

from soma.config import ROI_BATCH_SAMPLING_STRATEGIES, RoiBatchSamplingStrategy


class SegmentationRoiBatchSampler(Sampler[list[int]]):
    """Yield fixed-budget batches of cached ROI indices.

    ``class_conditioned`` batches follow relative request ratios over the ``K``
    columns in ``class_pixel_counts``. Each request draws an eligible ROI with
    probability proportional to that ROI's annotated-pixel count for the
    requested class. The selected ROI remains intact; its full mask is consumed
    by the ordinary segmentation loss.
    """

    def __init__(
        self,
        *,
        sample_ids: Sequence[str],
        class_pixel_counts: Sequence[Sequence[int]],
        batch_size: int,
        draws_per_epoch: int,
        strategy: RoiBatchSamplingStrategy,
        class_request_ratios: Sequence[float] | None = None,
        seed: int = 0,
    ) -> None:
        self._sample_ids = tuple(str(sample_id) for sample_id in sample_ids)
        raw_counts = np.asarray(class_pixel_counts)
        self._batch_size = int(batch_size)
        self._draws_per_epoch = int(draws_per_epoch)
        self._strategy = str(strategy)
        self._seed = int(seed)
        self._epoch = 0
        self._epochs: dict[int, dict[str, object]] = {}

        if self._strategy not in ROI_BATCH_SAMPLING_STRATEGIES:
            raise ValueError(
                "ROI batch sampling strategy must be 'uniform' or "
                f"'class_conditioned', got {self._strategy!r}."
            )
        if raw_counts.ndim != 2 or raw_counts.shape[0] != len(self._sample_ids):
            raise ValueError(
                "class_pixel_counts must be a two-dimensional (N, K) matrix with "
                f"one row per ROI; got shape {raw_counts.shape} for "
                f"{len(self._sample_ids)} ROIs."
            )
        if not self._sample_ids:
            raise ValueError("ROI sampling requires at least one sample_id.")
        if raw_counts.shape[1] < 1:
            raise ValueError("class_pixel_counts must contain at least one class column.")
        if not np.issubdtype(raw_counts.dtype, np.integer):
            raise ValueError("class_pixel_counts must contain integer pixel counts.")
        self._counts = raw_counts.astype(np.int64, copy=False)
        self._num_classes = int(self._counts.shape[1])
        if np.any(self._counts < 0):
            raise ValueError("class_pixel_counts cannot contain negative values.")
        if len(set(self._sample_ids)) != len(self._sample_ids):
            raise ValueError("sample_ids must be unique for an auditable ROI population.")
        if self._batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self._draws_per_epoch < 1:
            raise ValueError("draws_per_epoch must be >= 1.")
        if self._draws_per_epoch % self._batch_size != 0:
            raise ValueError("draws_per_epoch must be divisible by batch_size.")
        if self._strategy == "class_conditioned":
            if class_request_ratios is None:
                ratios = np.ones(self._num_classes, dtype=np.float64)
            else:
                ratios = np.asarray(class_request_ratios, dtype=np.float64)
            if ratios.shape != (self._num_classes,):
                raise ValueError(
                    "class_request_ratios must contain one value per class; got "
                    f"{ratios.shape} for K={self._num_classes}."
                )
            if np.any(~np.isfinite(ratios)) or np.any(ratios < 0) or ratios.sum() <= 0:
                raise ValueError(
                    "class_request_ratios must be finite, non-negative relative "
                    "weights with at least one positive value."
                )
            self._class_request_ratios = ratios
            self._request_probabilities = ratios / ratios.sum()
            missing = [
                class_index
                for class_index in range(self._num_classes)
                if self._class_request_ratios[class_index] > 0
                and self._counts[:, class_index].sum() == 0
            ]
            if missing:
                raise ValueError(
                    "Class-conditioned ROI sampling needs annotated pixels for every "
                    f"requested class; missing classes {missing}."
                )
        else:
            if class_request_ratios is not None:
                raise ValueError(
                    "class_request_ratios are only valid with class_conditioned "
                    "ROI sampling."
                )
            self._class_request_ratios = None
            self._request_probabilities = None
            if self._draws_per_epoch > len(self._sample_ids):
                raise ValueError(
                    "Uniform ROI sampling traverses each ROI at most once per epoch, so "
                    f"draws_per_epoch={self._draws_per_epoch} exceeds "
                    f"{len(self._sample_ids)} ROIs."
                )

    def __len__(self) -> int:
        return self._draws_per_epoch // self._batch_size

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(np.random.SeedSequence([self._seed, self._epoch]))
        if self._strategy == "uniform":
            indices = rng.permutation(len(self._sample_ids))[: self._draws_per_epoch].tolist()
            requests: list[int | None] = [None] * self._draws_per_epoch
        else:
            indices = []
            requests = []
            for batch_requests in self._conditioned_request_batches(rng):
                for raw_class_index in batch_requests:
                    class_index = int(raw_class_index)
                    weights = self._counts[:, class_index].astype(np.float64)
                    selected = int(rng.choice(len(self._sample_ids), p=weights / weights.sum()))
                    requests.append(class_index)
                    indices.append(selected)

        selections = [
            {
                "requested_class": requested_class,
                "selected_roi": self._sample_ids[index],
                "actual_class_pixel_counts": self._counts[index].tolist(),
            }
            for requested_class, index in zip(requests, indices)
        ]
        self._epochs[self._epoch] = {
            "epoch": self._epoch,
            "target_request_counts": [
                sum(requested_class == class_index for requested_class in requests)
                for class_index in range(self._num_classes)
            ],
            "actual_class_pixel_counts": self._counts[indices].sum(axis=0).tolist(),
            "unique_roi_count": len(set(indices)),
            "roi_draw_counts": {
                self._sample_ids[index]: count
                for index, count in sorted(Counter(indices).items())
            },
            "selections": selections,
        }

        for start in range(0, len(indices), self._batch_size):
            yield [int(index) for index in indices[start : start + self._batch_size]]

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic ``seed + epoch`` partition for the next iteration."""
        self._epoch = int(epoch)

    def _conditioned_request_batches(
        self, rng: np.random.Generator
    ) -> Iterator[list[int]]:
        """Plan requests while keeping cumulative exposure close to the ratios."""
        assert self._request_probabilities is not None
        probabilities = self._request_probabilities
        assigned = np.zeros(self._num_classes, dtype=np.int64)
        priority = rng.permutation(self._num_classes)
        rank = np.empty(self._num_classes, dtype=np.int64)
        rank[priority] = np.arange(self._num_classes)
        for _ in range(len(self)):
            batch: list[int] = []
            for _ in range(self._batch_size):
                next_total = int(assigned.sum()) + 1
                deficits = next_total * probabilities - assigned
                best_deficit = deficits.max()
                candidates = np.flatnonzero(
                    np.isclose(deficits, best_deficit, rtol=0.0, atol=1e-12)
                )
                class_index = int(candidates[np.argmin(rank[candidates])])
                assigned[class_index] += 1
                batch.append(class_index)
            rng.shuffle(batch)
            yield batch

    def audit(self) -> dict[str, object]:
        """Return a JSON-serializable audit of every generated epoch."""
        return {
            "schema_version": 2,
            "strategy": self._strategy,
            "batch_size": self._batch_size,
            "draws_per_epoch": self._draws_per_epoch,
            "classes": list(range(self._num_classes)),
            "class_request_ratios": (
                None
                if self._class_request_ratios is None
                else self._class_request_ratios.tolist()
            ),
            "resolved_request_probabilities": (
                None
                if self._request_probabilities is None
                else self._request_probabilities.tolist()
            ),
            "epochs": deepcopy([self._epochs[epoch] for epoch in sorted(self._epochs)]),
        }
