"""Training-batch ROI sampling for cached semantic-segmentation grids.

This module chooses already-extracted ROI dataset items for decoder training.  It
does not create ROI coordinates; that preprocessing concern lives in
``PreprocessingConfig.sampling``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from copy import deepcopy

import numpy as np
from torch.utils.data import Sampler

from soma.config import ROI_BATCH_SAMPLING_STRATEGIES, RoiBatchSamplingStrategy


class SegmentationRoiBatchSampler(Sampler[list[int]]):
    """Yield fixed-budget batches of cached ROI indices.

    ``class_conditioned`` batches contain equal target requests for the four
    segmentation classes.  Each request draws an eligible ROI with probability
    proportional to that ROI's annotated-pixel count for the requested class.
    The selected ROI remains intact; its full mask is consumed by the ordinary
    segmentation loss.
    """

    def __init__(
        self,
        *,
        sample_ids: Sequence[str],
        class_pixel_counts: Sequence[Sequence[int]],
        batch_size: int,
        draws_per_epoch: int,
        strategy: RoiBatchSamplingStrategy,
        seed: int = 0,
    ) -> None:
        self._sample_ids = tuple(str(sample_id) for sample_id in sample_ids)
        self._counts = np.asarray(class_pixel_counts, dtype=np.int64)
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
        if self._counts.shape != (len(self._sample_ids), 4):
            raise ValueError(
                "class_pixel_counts must have one four-class row per ROI; got "
                f"shape {self._counts.shape} for {len(self._sample_ids)} ROIs."
            )
        if np.any(self._counts < 0):
            raise ValueError("class_pixel_counts cannot contain negative values.")
        if self._batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self._draws_per_epoch < 1:
            raise ValueError("draws_per_epoch must be >= 1.")
        if self._draws_per_epoch % self._batch_size != 0:
            raise ValueError("draws_per_epoch must be divisible by batch_size.")
        if self._strategy == "class_conditioned":
            if self._batch_size % 4 != 0:
                raise ValueError("Class-conditioned batch_size must be divisible by four.")
            missing = [
                class_index
                for class_index in range(4)
                if self._counts[:, class_index].sum() == 0
            ]
            if missing:
                raise ValueError(
                    "Class-conditioned ROI sampling needs annotated pixels for every "
                    f"requested class; missing classes {missing}."
                )
        elif self._draws_per_epoch > len(self._sample_ids):
            raise ValueError(
                "Uniform ROI sampling traverses each ROI at most once per epoch, so "
                f"draws_per_epoch={self._draws_per_epoch} exceeds {len(self._sample_ids)} ROIs."
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
            requests_per_class = self._batch_size // 4
            for _ in range(len(self)):
                batch_requests = np.repeat(np.arange(4), requests_per_class)
                rng.shuffle(batch_requests)
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
                for class_index in range(4)
            ],
            "actual_class_pixel_counts": self._counts[indices].sum(axis=0).tolist(),
            "selections": selections,
        }

        for start in range(0, len(indices), self._batch_size):
            yield [int(index) for index in indices[start : start + self._batch_size]]

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic ``seed + epoch`` partition for the next iteration."""
        self._epoch = int(epoch)

    def audit(self) -> dict[str, object]:
        """Return a JSON-serializable audit of every generated epoch."""
        return {
            "strategy": self._strategy,
            "batch_size": self._batch_size,
            "draws_per_epoch": self._draws_per_epoch,
            "classes": [0, 1, 2, 3],
            "epochs": deepcopy([self._epochs[epoch] for epoch in sorted(self._epochs)]),
        }
