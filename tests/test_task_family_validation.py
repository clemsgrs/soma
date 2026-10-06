"""Config validation keys on the registered head's task family, not its name."""

from __future__ import annotations

import pytest

from soma.tasks.registry import task_family_of, task_registry
from soma.tasks.segmentation import SegmentationHead


def test_task_family_of_builtin_heads():
    assert task_family_of("segmentation") == "segmentation"
    assert task_family_of("binary_classification") == "binary_classification"
    assert task_family_of("branch_aware_classification") == "multiclass_classification"
    assert task_family_of("survival") == "survival"


def test_task_family_of_unknown_name_lists_registered_heads():
    with pytest.raises(ValueError, match="Unknown task head 'nope'.*segmentation"):
        task_family_of("nope")


def test_registered_subclass_inherits_family():
    class MyHead(SegmentationHead):
        pass

    name = "_test_family_subclass"
    if name not in task_registry:
        task_registry.register(name, MyHead)
    assert task_family_of(name) == "segmentation"
