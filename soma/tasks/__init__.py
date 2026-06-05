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
from soma.tasks.survival import SurvivalHead


def list_task_heads() -> list[str]:
    """Return registered task-head names in a stable order."""
    return sorted(task_registry.list())

__all__ = [
    "TaskHead",
    "task_registry",
    "list_task_heads",
    "BinaryClassificationHead",
    "MulticlassClassificationHead",
    "BranchAwareClassificationHead",
    "OrdinalClassificationHead",
    "RegressionHead",
    "SurvivalHead",
]
