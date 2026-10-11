"""``loaders``: one fold's train / tune / test loaders over a set source and a cohort."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

from torch.utils.data import DataLoader

from soma.data import Cohort, SetSource, require_coverage
from soma.training.bag_dataset import BagDataset, HierarchicalBagDataset
from soma.training.collate import bag_collate_fn, hierarchical_bag_collate_fn
from soma.training.sample_dataset import SampleDataset, sample_collate_fn

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
    source: SetSource,
    cohort: Cohort,
    *,
    head: Any,
    fold: int = 0,
    batch_size: int = 1,
    num_workers: int = 0,
    **loader_kwargs: Any,
) -> Loaders:
    """Build fold ``fold``'s loaders for :func:`soma.fit` and :func:`soma.predict`.

    The source's rank picks the batch shape (vectors, padded bags, hierarchical bags);
    ``head`` supplies the targets each sample carries (``extract_targets``) and their
    dtypes. Every sample of the fold must be served by ``source``. ``tune`` is ``None``
    when the fold has no tune split. Extra keyword arguments go to every ``DataLoader``.
    """
    if not 0 <= fold < cohort.num_folds:
        raise ValueError(f"fold {fold} out of range; the cohort has {cohort.num_folds} fold(s)")
    try:
        dataset_cls, collate = _BY_RANK[int(source.rank)]
    except KeyError:
        raise ValueError(f"a set source has rank 1, 2 or 3, got {source.rank!r}") from None
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
