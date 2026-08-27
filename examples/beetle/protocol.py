"""Shared constants for the repository-local BEETLE development protocol."""

ARM_NAMES = ("uniform", "class_conditioned")
NUM_FOLDS = 5
BATCH_SIZE_CANDIDATES = (16, 8, 4)
PIXEL_MAPPING = {
    "background": 0,
    "other": 1,
    "non_invasive_epithelium": 2,
    "invasive_epithelium": 3,
    "necrosis": 4,
}
ANNOTATED_LABEL_NAME_BY_VALUE = {
    value: name for name, value in PIXEL_MAPPING.items() if value != 0
}

__all__ = [
    "ANNOTATED_LABEL_NAME_BY_VALUE",
    "ARM_NAMES",
    "BATCH_SIZE_CANDIDATES",
    "NUM_FOLDS",
    "PIXEL_MAPPING",
]
