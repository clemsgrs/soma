"""``loaders``: one fold's train / tune / test loaders over a source and a cohort."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch.utils.data import DataLoader

from soma.data import Cohort, require_coverage
from soma.learn.bag_dataset import BagDataset, HierarchicalBagDataset
from soma.learn.collate import bag_collate_fn, hierarchical_bag_collate_fn
from soma.learn.sample_dataset import SampleDataset, sample_collate_fn

__all__ = ["Loaders", "loaders"]

_BY_RANK = {
    1: (SampleDataset, sample_collate_fn),
    2: (BagDataset, bag_collate_fn),
    3: (HierarchicalBagDataset, hierarchical_bag_collate_fn),
}


@dataclass(frozen=True)
class Loaders:
    """One fold's loaders: ``train`` shuffles, ``tune`` and each of ``tests`` keep split order."""

    train: DataLoader
    tune: DataLoader | None
    tests: dict[str, DataLoader]


def loaders(
    source,
    cohort: Cohort,
    *,
    head: Any,
    fold: int = 0,
    batch_size: int = 1,
    num_workers: int = 0,
    targets=None,
    augment: Callable | None = None,
    preprocess: Callable | None = None,
    sampler=None,
    images=None,
    **loader_kwargs: Any,
) -> Loaders:
    """Build fold ``fold``'s loaders for :func:`soma.fit` and :func:`soma.predict`.

    The source picks the shape. A set source (it declares ``rank``) yields vectors,
    padded bags or hierarchical bags, with the targets ``head.extract_targets`` reads off
    the cohort's records. A grid source (it declares ``geometry``) yields
    :class:`~soma.learn.grid.GridBatch` es: cached feature grids from a
    :class:`~soma.data.GridSource`, or pixels from an :class:`~soma.data.ImageSource`
    for a model with an encoder stage. Every sample of the fold must be served by
    ``source``. ``tune`` is ``None`` when the fold has no tune split.

    Grid-only arguments:
        targets: The :class:`~soma.data.TargetSource` serving each sample's label map or
            points; it must match the head's ``target_kind``. Required for the grid.
        augment: ``(pixels, target) -> (pixels, target)`` applied to the **train** split
            only (e.g. :class:`~soma.learn.SegmentationAugment`). Refused for cached grids.
            Point targets reach it in the pixel frame and return to level-0.
        preprocess: The encoder stage's itemwise CPU preprocessor
            (``FrozenEncoder.preprocessor()``), run after augmentation inside the loader
            workers. Without it the batch carries raw uint8 pixels and the stage
            preprocesses them itself.
        sampler: A batch sampler over the train split's indices (in split order), used
            as the train loader's ``batch_sampler`` (e.g. the ROI batch sampler).
        images: An :class:`~soma.data.ImageSource` whose uint8 pixels ride along with
            cached grids as an image-prior decoder's side input.

    Extra keyword arguments go to every ``DataLoader``.
    """
    if not 0 <= fold < cohort.num_folds:
        raise ValueError(f"fold {fold} out of range; the cohort has {cohort.num_folds} fold(s)")
    if not hasattr(source, "rank") and hasattr(source, "geometry"):
        from soma.learn.grid import grid_loaders

        split = cohort.folds[fold]
        return grid_loaders(
            source,
            head=head,
            train=split.train,
            tune=split.tune,
            tests=split.tests,
            batch_size=batch_size,
            num_workers=num_workers,
            targets=targets,
            augment=augment,
            preprocess=preprocess,
            sampler=sampler,
            images=images,
            loader_kwargs=loader_kwargs,
        )
    grid_only = {
        "targets": targets,
        "augment": augment,
        "preprocess": preprocess,
        "sampler": sampler,
        "images": images,
    }
    passed = sorted(name for name, value in grid_only.items() if value is not None)
    if passed:
        raise ValueError(f"{passed} apply to a grid source; a set source takes none of them")
    try:
        dataset_cls, collate = _BY_RANK[int(source.rank)]
    except (AttributeError, KeyError):
        raise ValueError(
            f"a set source declares rank 1, 2 or 3 (got {getattr(source, 'rank', None)!r}); "
            "a grid source declares geometry"
        ) from None
    split = cohort.folds[fold]
    require_coverage(source, split.all_sample_ids)
    collate_fn = functools.partial(collate, target_dtypes=head.target_dtypes)

    def make(ids, shuffle: bool) -> DataLoader:
        records = [cohort.record(sample_id) for sample_id in ids]
        return DataLoader(
            dataset_cls(records, source, head.extract_targets),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            **loader_kwargs,
        )

    return Loaders(
        train=make(split.train, True),
        tune=make(split.tune, False) if split.tune else None,
        tests={name: make(ids, False) for name, ids in split.tests.items()},
    )
