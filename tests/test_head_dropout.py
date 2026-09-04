"""Tests for the optional input dropout on the classification and survival task heads.

The knob defaults to ``0.0``, and at that default nothing is built: no module, no
extra ``state_dict`` entry, no random number drawn. A run that does not ask for
dropout must be bit-identical to one from before the knob existed.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from soma.aggregators.pooling import MeanPool
from soma.tasks.classification import (
    BinaryClassificationHead,
    BranchAwareClassificationHead,
    MulticlassClassificationHead,
)
from soma.tasks.survival import CoxSurvivalHead, SurvivalHead
from soma.training.model import EmbeddingModel, MILModel

from tests.test_pipeline import D as PIPELINE_D, NUM_SAMPLES, _setup_synthetic_data

D = 8

# (head factory, input shape) for every head the knob covers.
HEADS = {
    "binary": (lambda **kw: BinaryClassificationHead(input_dim=D, num_classes=2, **kw), (4, D)),
    "multiclass": (
        lambda **kw: MulticlassClassificationHead(input_dim=D, num_classes=3, **kw),
        (4, D),
    ),
    "branch_aware": (
        lambda **kw: BranchAwareClassificationHead(input_dim=D, num_classes=3, **kw),
        (4, 3, D),
    ),
    "survival": (lambda **kw: SurvivalHead(input_dim=D, num_bins=4, **kw), (4, D)),
    "cox": (lambda **kw: CoxSurvivalHead(input_dim=D, **kw), (4, D)),
}


@pytest.fixture(params=sorted(HEADS))
def head_case(request):
    build, shape = HEADS[request.param]
    return build, shape


class TestDefaultIsInert:
    """At ``dropout=0.0`` the head is exactly the head that existed before the knob."""

    def test_default_builds_no_dropout_module(self, head_case):
        build, _ = head_case
        head = build()
        assert head.dropout is None
        assert "dropout" not in dict(head.named_modules())
        assert "dropout" not in head._modules

    def test_default_leaves_the_state_dict_untouched(self, head_case):
        build, _ = head_case
        assert not any("dropout" in key for key in build().state_dict())

    def test_default_draws_no_random_numbers_in_forward(self, head_case):
        """A train-mode forward at the default must not touch the random stream.

        This is the bit-identity claim: a constructed-but-zero-probability
        ``nn.Dropout`` in the forward would advance the generator and desynchronise
        every subsequent draw of the run.
        """
        build, shape = head_case
        X = torch.randn(*shape)
        head = build()
        head.train()

        before = torch.get_rng_state()
        head(X)
        assert torch.equal(torch.get_rng_state(), before)

    def test_default_forward_is_deterministic(self, head_case):
        build, shape = head_case
        X = torch.randn(*shape)
        torch.manual_seed(0)
        head = build()
        head.train()
        assert torch.equal(head(X), head(X))


class TestDropoutRequested:
    def test_module_is_built(self, head_case):
        build, _ = head_case
        head = build(dropout=0.3)
        assert isinstance(head.dropout, nn.Dropout)
        assert head.dropout.p == 0.3

    def test_dropout_adds_no_checkpoint_state(self, head_case):
        """Dropout is parameter-free: a checkpoint stays loadable across the knob."""
        build, _ = head_case
        assert list(build(dropout=0.3).state_dict()) == list(build().state_dict())

    def test_applied_to_the_head_input(self, head_case):
        """The mask lands on ``X``, before the head's linear layer."""
        build, shape = head_case
        X = torch.randn(*shape)
        torch.manual_seed(0)
        head = build(dropout=0.5)
        head.train()
        plain = build()
        plain.load_state_dict(head.state_dict())

        torch.manual_seed(7)
        got = head(X)
        torch.manual_seed(7)
        expected = plain(F.dropout(X, p=0.5, training=True))

        assert torch.equal(got, expected)

    def test_train_mode_is_stochastic(self, head_case):
        build, shape = head_case
        X = torch.randn(*shape)
        torch.manual_seed(0)
        head = build(dropout=0.5)
        head.train()
        assert not torch.equal(head(X), head(X))

    def test_eval_mode_is_a_no_op(self, head_case):
        build, shape = head_case
        X = torch.randn(*shape)
        torch.manual_seed(0)
        head = build(dropout=0.5)
        head.eval()
        plain = build()
        plain.load_state_dict(head.state_dict())
        assert torch.equal(head(X), plain(X))

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_rejects_probabilities_outside_the_unit_interval(self, head_case, bad):
        build, _ = head_case
        with pytest.raises(ValueError, match="dropout"):
            build(dropout=bad)


class TestBothModelingPaths:
    """The knob must work behind an aggregator and on a frozen slide embedding."""

    def test_single_embedding_path_applies_dropout(self):
        X = torch.randn(4, D)
        torch.manual_seed(0)
        model = EmbeddingModel(
            task_head=BinaryClassificationHead(input_dim=D, num_classes=2, dropout=0.5)
        )
        model.train()
        assert not torch.equal(model(X).logits, model(X).logits)

    def test_single_embedding_path_default_is_deterministic(self):
        X = torch.randn(4, D)
        torch.manual_seed(0)
        model = EmbeddingModel(task_head=BinaryClassificationHead(input_dim=D, num_classes=2))
        model.train()
        assert torch.equal(model(X).logits, model(X).logits)

    def test_aggregated_path_applies_dropout(self):
        X = torch.randn(2, 6, D)
        torch.manual_seed(0)
        model = MILModel(
            aggregator=MeanPool(input_dim=D),
            task_head=SurvivalHead(input_dim=D, num_bins=4, dropout=0.5),
        )
        model.train()
        assert not torch.equal(model(X).logits, model(X).logits)

    def test_aggregated_path_default_is_deterministic(self):
        X = torch.randn(2, 6, D)
        torch.manual_seed(0)
        model = MILModel(
            aggregator=MeanPool(input_dim=D),
            task_head=SurvivalHead(input_dim=D, num_bins=4),
        )
        model.train()
        assert torch.equal(model(X).logits, model(X).logits)


class TestReachesTheHeadThroughTheConfig:
    """``task.params.dropout`` must survive the pipeline's head construction."""

    def _train(self, tmp_path, *, slide_level):
        from soma import Dataset, Splits
        from soma.config import AggregatorConfig, TaskConfig, TrainingConfig
        from soma.features import FeatureStore
        from soma.pipeline import train_one_fold

        dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
        if slide_level:
            # One (D,) vector per sample — the aggregation: null path.
            feature_dir = tmp_path / "slide_feats"
            feature_dir.mkdir()
            for i in range(NUM_SAMPLES):
                torch.save(torch.randn(PIPELINE_D), feature_dir / f"s{i}.pt")

        dataset = Dataset(dataset_csv)
        splits = Splits(splits_csv, dataset)
        return train_one_fold(
            feature_store=FeatureStore(feature_dir),
            dataset=dataset,
            fold_split=splits.folds[0],
            aggregator=None if slide_level else AggregatorConfig(name="mean_pool"),
            task=TaskConfig(name="binary_classification", params={"dropout": 0.2}),
            training=TrainingConfig(epochs=2, patience=10, batch_size=2),
            fold_dir=tmp_path / "fold",
        )

    def test_single_embedding_path(self, tmp_path):
        result = self._train(tmp_path, slide_level=True)
        assert "auroc" in result.test_reports["test"].metrics

    def test_aggregated_path(self, tmp_path):
        result = self._train(tmp_path, slide_level=False)
        assert "auroc" in result.test_reports["test"].metrics
