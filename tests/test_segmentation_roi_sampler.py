"""Training-batch ROI sampling for cached segmentation grids."""

from __future__ import annotations

from collections import Counter

import pytest

from soma.config import TrainingConfig
from soma.training.segmentation_roi_sampler import SegmentationRoiBatchSampler


def test_class_conditioned_batch_size_must_be_divisible_by_four() -> None:
    with pytest.raises(ValueError, match="divisible by four"):
        TrainingConfig(batch_size=6, roi_batch_sampling="class_conditioned")


def test_roi_draw_budget_must_be_whole_batches() -> None:
    with pytest.raises(ValueError, match="whole batches"):
        TrainingConfig(
            batch_size=4,
            roi_batch_sampling="uniform",
            roi_draws_per_epoch=6,
        )


@pytest.mark.parametrize(
    ("batch_size", "requests_per_class"),
    [(4, 1), (8, 2), (16, 4)],
)
def test_class_conditioned_batches_request_each_class_equally(
    batch_size: int, requests_per_class: int
) -> None:
    sampler = SegmentationRoiBatchSampler(
        sample_ids=["roi-a", "roi-b", "roi-c", "roi-d"],
        class_pixel_counts=[
            [4, 1, 1, 1],
            [1, 4, 1, 1],
            [1, 1, 4, 1],
            [1, 1, 1, 4],
        ],
        batch_size=batch_size,
        draws_per_epoch=batch_size,
        strategy="class_conditioned",
        seed=7,
    )

    batches = list(sampler)
    requested_classes = [
        selection["requested_class"]
        for selection in sampler.audit()["epochs"][0]["selections"]
    ]

    assert [len(batch) for batch in batches] == [batch_size]
    assert Counter(requested_classes) == {
        0: requests_per_class,
        1: requests_per_class,
        2: requests_per_class,
        3: requests_per_class,
    }


def _explicit_sampler(*, seed: int = 13) -> SegmentationRoiBatchSampler:
    return SegmentationRoiBatchSampler(
        sample_ids=["a", "b", "c", "d", "e"],
        class_pixel_counts=[
            [8, 0, 0, 0],
            [2, 5, 0, 0],
            [0, 5, 7, 0],
            [0, 0, 3, 11],
            [0, 0, 0, 1],
        ],
        batch_size=4,
        draws_per_epoch=8,
        strategy="class_conditioned",
        seed=seed,
    )


def test_class_conditioned_selection_is_exact_and_pixel_count_weighted() -> None:
    sampler = _explicit_sampler()

    batches = list(sampler)

    assert batches == [[3, 2, 1, 1], [0, 3, 3, 1]]


def test_sampler_audit_records_requests_selected_rois_and_actual_pixels() -> None:
    sampler = _explicit_sampler()
    list(sampler)

    epoch = sampler.audit()["epochs"][0]

    assert epoch == {
        "epoch": 0,
        "target_request_counts": [2, 2, 2, 2],
        "actual_class_pixel_counts": [14, 20, 16, 33],
        "selections": [
            {"requested_class": 3, "selected_roi": "d", "actual_class_pixel_counts": [0, 0, 3, 11]},
            {"requested_class": 2, "selected_roi": "c", "actual_class_pixel_counts": [0, 5, 7, 0]},
            {"requested_class": 1, "selected_roi": "b", "actual_class_pixel_counts": [2, 5, 0, 0]},
            {"requested_class": 0, "selected_roi": "b", "actual_class_pixel_counts": [2, 5, 0, 0]},
            {"requested_class": 0, "selected_roi": "a", "actual_class_pixel_counts": [8, 0, 0, 0]},
            {"requested_class": 3, "selected_roi": "d", "actual_class_pixel_counts": [0, 0, 3, 11]},
            {"requested_class": 2, "selected_roi": "d", "actual_class_pixel_counts": [0, 0, 3, 11]},
            {"requested_class": 1, "selected_roi": "b", "actual_class_pixel_counts": [2, 5, 0, 0]},
        ],
    }


def test_seed_and_epoch_reproduce_advancing_partitions() -> None:
    first = _explicit_sampler()
    repeated = _explicit_sampler()

    epoch_zero = list(first)
    first.set_epoch(1)
    epoch_one = list(first)
    repeated_epoch_zero = list(repeated)
    repeated.set_epoch(1)
    repeated_epoch_one = list(repeated)

    assert epoch_zero == repeated_epoch_zero
    assert epoch_one == repeated_epoch_one
    assert epoch_zero != epoch_one


def test_uniform_and_conditioned_arms_have_identical_draw_budgets() -> None:
    sample_ids = [f"roi-{index}" for index in range(8)]
    counts = [[1, 1, 1, 1] for _ in sample_ids]
    samplers = [
        SegmentationRoiBatchSampler(
            sample_ids=sample_ids,
            class_pixel_counts=counts,
            batch_size=4,
            draws_per_epoch=8,
            strategy=strategy,
            seed=5,
        )
        for strategy in ("uniform", "class_conditioned")
    ]

    uniform_batches, conditioned_batches = [list(sampler) for sampler in samplers]

    assert [len(batch) for batch in uniform_batches] == [4, 4]
    assert [len(batch) for batch in conditioned_batches] == [4, 4]
    assert sorted(index for batch in uniform_batches for index in batch) == list(range(8))
