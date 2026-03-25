"""Deterministic seeding utility."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python's random, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
