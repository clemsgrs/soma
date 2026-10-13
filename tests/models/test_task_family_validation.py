"""Config validation keys on the registered head's task family, not its name."""

from __future__ import annotations

import pytest

from soma.data import CachedSetSource
from soma.models.heads.registry import task_family_of, task_registry
from soma.models.heads.segmentation import SegmentationHead
from tests.experiment_configs import experiment


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


def test_survival_task_keeps_a_registered_subclass_and_infers_its_loss(monkeypatch):
    from soma.models.heads.survival import CoxSurvivalHead, SurvivalHead, resolve_survival_task

    class MyNll(SurvivalHead):
        pass

    class MyCox(CoxSurvivalHead):
        pass

    monkeypatch.setattr(task_registry, "_entries", dict(task_registry._entries))
    task_registry.register("my_nll", MyNll)
    task_registry.register("my_cox", MyCox)
    assert resolve_survival_task("survival") == (SurvivalHead, "nll")
    assert resolve_survival_task("cox_survival") == (CoxSurvivalHead, "cox")
    assert resolve_survival_task("my_nll") == (MyNll, "nll")
    assert resolve_survival_task("my_cox") == (MyCox, "cox")


def test_registered_survival_subclass_still_validates_the_dataset(tmp_path, monkeypatch):
    """An unchanged subclass registered under another name hits the survival checks."""
    import torch

    from soma.pipeline import Pipeline
    from soma.models.heads.survival import SurvivalHead, SurvivalParams

    class MySurvival(SurvivalHead):
        pass

    monkeypatch.setattr(task_registry, "_entries", dict(task_registry._entries))
    task_registry.register("_test_my_survival", MySurvival, params=SurvivalParams)

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
    config = experiment(
        dataset_csv,
        splits_csv,
        output_dir=tmp_path / "out",
        run={"run_id": "custom-survival"},
        head="_test_my_survival",
        head_params={"num_bins": 1},
        cohort={"allow_missing_tune": True},
        loaders={"batch_size": 2},
        trainer={"max_epochs": 1, "callbacks": [{"name": "checkpoint", "params": {"selection": "last"}}]},
    )
    with pytest.raises(ValueError, match="event"):
        Pipeline(config, source=CachedSetSource(features)).run()


def test_registered_cox_head_rejects_wrong_window_batch_size(tmp_path, monkeypatch):
    """A Cox subclass inherits window constraints without a loss selector."""
    from soma.plan import check
    from soma.models.heads.survival import CoxSurvivalHead, CoxSurvivalParams

    class MyCox(CoxSurvivalHead):
        pass

    monkeypatch.setattr(task_registry, "_entries", dict(task_registry._entries))
    task_registry.register("_test_my_cox", MyCox, params=CoxSurvivalParams)
    with pytest.raises(ValueError, match="Cox accumulation mode .*batch_size = 1"):
        check(
            experiment(
                head="_test_my_cox", head_params={"cox_window": 2}, aggregator="abmil", loaders={"batch_size": 2}
            )
        )


def test_clam_mb_rejects_custom_heads_without_branch_representations(tmp_path, monkeypatch):
    """CLAM-MB must not silently replace a registered head's custom objective."""
    import torch

    from soma.pipeline import Pipeline
    from soma.models.heads.classification import ClassificationParams, MulticlassClassificationHead, infer_label_map

    class SevenLossHead(MulticlassClassificationHead):
        def compute_loss(self, predictions, targets):
            return predictions.sum() * 0 + 7

    monkeypatch.setattr(task_registry, "_entries", dict(task_registry._entries))
    task_registry.register("_test_seven_loss", SevenLossHead, params=ClassificationParams, infer=infer_label_map)
    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,label\n"
        "a,/a.svs,0\nb,/b.svs,1\nc,/c.svs,0\nd,/d.svs,1\ne,/e.svs,0\nf,/f.svs,1\n"
    )
    splits_csv = tmp_path / "splits.csv"
    splits_csv.write_text(
        "sample_id,split\na,train\nb,train\nc,tune\nd,tune\ne,test\nf,test\n"
    )
    features = tmp_path / "features"
    features.mkdir()
    for sid in "abcdef":
        torch.save(torch.zeros(2, 4), features / f"{sid}.pt")
    config = experiment(
        dataset_csv,
        splits_csv,
        output_dir=tmp_path / "out",
        aggregator={"name": "clam_mb", "params": {"hidden_dim": 4, "attn_dim": 2}},
        head="_test_seven_loss",
        evaluation={"metrics": ["accuracy"]},
        loaders={"batch_size": 2},
        trainer={"max_epochs": 1},
    )
    with pytest.raises(ValueError, match="BranchAwareClassificationHead"):
        Pipeline(config, source=CachedSetSource(features)).run()
