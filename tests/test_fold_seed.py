"""Tests for per-fold seed independence (Phase 1.1).

The framework runs cross-validation by calling ``seed_everything(training.seed)``
once per fold. Without a per-fold offset, every fold draws the same random
weight init, the same DTFD-MIL pseudo-bag partition, the same dropout pattern,
and the same dataloader shuffle, which silently compresses CV variance.

These tests pin the invariant: ``seed_everything(seed, fold=k)`` must produce
different draws for different ``fold`` values, and fold 0 must remain
backwards-compatible with ``seed_everything(seed)``.
"""

from __future__ import annotations

import numpy as np
import torch

from soma.training.seed import seed_everything


def _first_draws() -> tuple[float, float, float]:
    return (
        float(np.random.rand()),
        float(torch.rand(1).item()),
        float(np.random.rand()),
    )


def test_fold_seed_changes_random_state() -> None:
    """Two folds with the same base seed must produce different draws."""
    seed_everything(0, fold=0)
    fold0 = _first_draws()
    seed_everything(0, fold=1)
    fold1 = _first_draws()
    assert fold0 != fold1, (
        f"fold=0 and fold=1 produced identical draws ({fold0} == {fold1}); "
        "per-fold seeding is missing"
    )


def test_fold_zero_matches_no_fold_arg() -> None:
    """fold=0 (default) must preserve historical seeding behavior."""
    seed_everything(42, fold=0)
    with_fold = _first_draws()
    seed_everything(42)
    without_fold = _first_draws()
    assert with_fold == without_fold


def test_fold_seed_deterministic() -> None:
    """Same (seed, fold) must reproduce the same draws."""
    seed_everything(7, fold=3)
    a = _first_draws()
    seed_everything(7, fold=3)
    b = _first_draws()
    assert a == b
