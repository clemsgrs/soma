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


def test_survival_head_class_keeps_a_registered_subclass_and_infers_its_loss():
    from soma.pipeline import _survival_head_class
    from soma.tasks.survival import CoxSurvivalHead, SurvivalHead

    class MyNll(SurvivalHead):
        pass

    class MyCox(CoxSurvivalHead):
        pass

    assert _survival_head_class(SurvivalHead, task_name="survival", task_params={}) == (
        SurvivalHead,
        "nll",
    )
    assert _survival_head_class(
        SurvivalHead, task_name="survival", task_params={"loss": "cox"}
    ) == (CoxSurvivalHead, "cox")
    assert _survival_head_class(MyNll, task_name="my_nll", task_params={}) == (MyNll, "nll")
    assert _survival_head_class(MyCox, task_name="my_cox", task_params={}) == (MyCox, "cox")


def test_registered_survival_subclass_still_validates_the_dataset(tmp_path, monkeypatch):
    """An unchanged subclass registered under another name hits the survival checks."""
    import torch

    from soma.config import PipelineConfig, TaskConfig, TrainingConfig
    from soma.pipeline import Pipeline
    from soma.tasks.survival import SurvivalHead

    class MySurvival(SurvivalHead):
        pass

    monkeypatch.setattr(task_registry, "_entries", dict(task_registry._entries))
    task_registry.register("_test_my_survival", MySurvival)

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,label,event,bin\n"
        "a,/a.svs,1,2,0\nb,/b.svs,2,0,0\nc,/c.svs,3,1,0\nd,/d.svs,4,0,0\n"
    )
    splits_csv = tmp_path / "splits.csv"
    splits_csv.write_text("sample_id,split\na,train\nb,train\nc,test\nd,test\n")
    features = tmp_path / "features"
    features.mkdir()
    for sid in "abcd":
        torch.save(torch.zeros(4), features / f"{sid}.pt")
    config = PipelineConfig(
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        output_root=tmp_path / "out",
        run_id="custom-survival",
        dataset_type="slide",
        aggregator=None,
        task=TaskConfig(name="_test_my_survival", params={"num_bins": 1}),
        training=TrainingConfig(
            epochs=1, batch_size=2, seed=0, checkpoint_selection="last",
            patience=None, allow_missing_tune=True,
        ),
    )
    with pytest.raises(ValueError, match="event"):
        Pipeline(config, feature_dir=features).run()


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
