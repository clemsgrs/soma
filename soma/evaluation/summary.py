"""Fold/seed aggregation helpers shared by the CV summary and the leaderboard.

Every spread reported by soma is the *sample* standard deviation (``ddof=1``):
folds and seeds are a sample of the possible splits/initialisations, not the
whole population. A single value has no spread and yields ``nan``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


def sample_std(values: Iterable[float]) -> float:
    """Sample standard deviation (``ddof=1``); ``nan`` for fewer than two values."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2:
        return float("nan")
    return float(np.std(arr, ddof=1))


@dataclass(frozen=True)
class ValueSummary:
    mean: float
    std: float
    n: int
    n_nan: int


def summarize_values(values: Iterable[float]) -> ValueSummary:
    """Mean and sample std over the *finite* values, counting the non-finite ones.

    A fold whose tune split cannot support a threshold-free metric (e.g. AUROC on a
    single-class split) reports ``nan``; it must not drag the mean to ``nan`` but the
    reader has to know it was excluded, hence ``n_nan``.
    """
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    n_nan = int(arr.size - finite.size)
    if finite.size == 0:
        return ValueSummary(mean=math.nan, std=math.nan, n=0, n_nan=n_nan)
    return ValueSummary(
        mean=float(np.mean(finite)),
        std=sample_std(finite),
        n=int(finite.size),
        n_nan=n_nan,
    )
