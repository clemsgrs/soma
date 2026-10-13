"""Deterministic seeding utility."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, *, fold: int = 0) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python's random, NumPy, and PyTorch (CPU + CUDA). The
    ``fold`` argument is added to ``seed`` so that each fold in a
    cross-validation run draws independent random state (weight init,
    dropout pattern, DTFD-MIL pseudo-bag partition, dataloader shuffle).
    ``fold=0`` preserves the historical single-seed behavior.

    Args:
        seed: Base random seed value.
        fold: Cross-validation fold index. Offsets the base seed so that
            different folds get statistically independent draws.
    """
    effective = seed + fold
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)
