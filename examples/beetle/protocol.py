"""Shared constants for the repository-local BEETLE development protocol."""

ARM_NAMES = ("uniform", "class_conditioned")
NUM_FOLDS = 5
BATCH_SIZE_CANDIDATES = (16, 8, 4)

__all__ = ["ARM_NAMES", "BATCH_SIZE_CANDIDATES", "NUM_FOLDS"]
