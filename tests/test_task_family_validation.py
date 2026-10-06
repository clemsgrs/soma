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


def test_registered_subclass_inherits_family(monkeypatch):
    class MyHead(SegmentationHead):
        pass

    monkeypatch.setattr(task_registry, "_entries", dict(task_registry._entries))
    task_registry.register("_test_family_subclass", MyHead)
    assert task_family_of("_test_family_subclass") == "segmentation"


def test_clam_mb_keeps_a_registered_branch_aware_subclass():
    """A user subclass of the branch-aware head survives the clam_mb swap."""
    from soma.pipeline import _bag_head_class
    from soma.tasks.classification import (
        BranchAwareClassificationHead,
        MulticlassClassificationHead,
    )

    class MyBranchHead(BranchAwareClassificationHead):
        pass

    assert (
        _bag_head_class(MyBranchHead, task_name="my_branch", aggregator_name="clam_mb")
        is MyBranchHead
    )
    # The ordinary multiclass head is swapped for the built-in branch-aware one.
    assert (
        _bag_head_class(
            MulticlassClassificationHead,
            task_name="multiclass_classification",
            aggregator_name="clam_mb",
        )
        is BranchAwareClassificationHead
    )
    # Other aggregators keep whatever is registered.
    assert (
        _bag_head_class(
            MulticlassClassificationHead,
            task_name="multiclass_classification",
            aggregator_name="abmil",
        )
        is MulticlassClassificationHead
    )
    with pytest.raises(ValueError, match="clam_mb does not support task 'segmentation'"):
        _bag_head_class(SegmentationHead, task_name="segmentation", aggregator_name="clam_mb")
