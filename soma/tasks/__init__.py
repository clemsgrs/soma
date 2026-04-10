"""Task heads — map aggregated representations to predictions."""

from soma.tasks.base import TaskHead
from soma.tasks.registry import task_registry

# Import to trigger registration
from soma.tasks.classification import (
    BinaryClassificationHead,
    BranchAwareClassificationHead,
    MulticlassClassificationHead,
)
from soma.tasks.ordinal_classification import OrdinalClassificationHead
from soma.tasks.regression import RegressionHead

__all__ = [
    "TaskHead",
    "task_registry",
    "BinaryClassificationHead",
    "MulticlassClassificationHead",
    "BranchAwareClassificationHead",
    "OrdinalClassificationHead",
    "RegressionHead",
]
