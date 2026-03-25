"""Task heads — map aggregated representations to predictions."""

from soma.tasks.base import TaskHead
from soma.tasks.registry import task_registry

# Import to trigger registration
from soma.tasks.classification import ClassificationHead

__all__ = [
    "TaskHead",
    "task_registry",
    "ClassificationHead",
]
