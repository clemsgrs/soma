"""``soma.testing.check_learner`` on soma's own learners: every aggregator, seg and det
(cached grids and the live encoder stage)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from soma import components
from soma.aggregators import aggregator_registry
from soma.decoders.registry import decoder_registry
from soma.dense.geometry import compute_dense_geometry
from soma.learn import Budget, DenseLearner, GridBatch, MILLearner
from soma.tasks.classification import BinaryClassificationHead, BranchAwareClassificationHead
from soma.tasks.detection import DetectionHead
from soma.tasks.segmentation import SegmentationHead
from soma.testing import ConformanceError, FakeEncoder, check_learner
from soma.training.collate import BagBatch, HierarchicalBagBatch
from soma.training.model import DenseModel, MILModel

D = 16
BUDGET = Budget(max_epochs=2, max_steps=None, updates_per_epoch=2)

AGGREGATOR_PARAMS = {
    "abmil": {"hidden_dim": 8},
    "clam_sb": {"hidden_dim": 8, "attn_dim": 4, "k_sample": 2},
    "clam_mb": {"hidden_dim": 8, "attn_dim": 4, "k_sample": 2},
    "dsmil": {"att_dim": 8},
    "dtfdmil": {"embed_dim": D, "hidden_dim": 8, "n_groups": 2},
    "transmil": {"att_dim": 16, "n_layers": 2, "n_heads": 2, "n_landmarks": 4, "dropout": 0.0},
    "hipt": {"region_size": 8, "patch_size": 4, "embed_dim_region": 12, "embed_dim_slide": 12},
}


def _mil_learner(name: str) -> tuple[MILLearner, object]:
    torch.manual_seed(0)
    agg = aggregator_registry.get(name)(input_dim=D, **AGGREGATOR_PARAMS.get(name, {}))
    if name == "clam_mb":
        head = BranchAwareClassificationHead(input_dim=agg.output_dim, num_classes=2)
    else:
        head = BinaryClassificationHead(input_dim=agg.output_dim, num_classes=2)
    learner = MILLearner(MILModel(aggregator=agg, task_head=head))
    targets = {"label": torch.tensor([0, 1, 1, 0])}
    if name == "hipt":
        batch = HierarchicalBagBatch(
            features=torch.randn(4, 3, 4, D),
            mask=torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=torch.bool),
            targets=targets,
            sample_ids=("a", "b", "c", "d"),
        )
    else:
        mask = torch.ones(4, 12, dtype=torch.bool)
        mask[1, 9:] = False
        batch = BagBatch(features=torch.randn(4, 12, D), mask=mask, targets=targets, sample_ids=("a", "b", "c", "d"))
    return learner, batch


@pytest.mark.parametrize("name", [entry["name"] for entry in components("aggregators")])
def test_mil_learner_conforms_for_every_aggregator(name):
    learner, batch = _mil_learner(name)
    check_learner(learner, batch, budget=BUDGET)


@pytest.mark.parametrize("name", [entry["name"] for entry in components("aggregators")])
def test_mil_learner_conforms_for_every_aggregator_under_bf16_autocast(name):
    """``Trainer(precision="bf16")`` wraps the steps in autocast with fp32 bag features.

    The padded batch above (a mask, as ``bag_collate_fn`` always emits) is what every
    aggregator sees in training; an aggregator that scatters autocast outputs into an
    fp32 buffer only fails there.
    """
    learner, batch = _mil_learner(name)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        check_learner(learner, batch, budget=BUDGET)


def _grid_batch(targets: dict[str, torch.Tensor]) -> GridBatch:
    return GridBatch(features=torch.randn(2, D, 8, 8), pixels=None, targets=targets, sample_ids=("a", "b"))


def test_dense_learner_conforms_for_segmentation():
    torch.manual_seed(0)
    geometry = compute_dense_geometry(target_size=32, patch_size=4)
    head = SegmentationHead(num_classes=3, geometry=geometry, metrics=["mean_dice"])
    decoder = decoder_registry.get("linear")(input_dim=D, num_classes=3)
    learner = DenseLearner(DenseModel(decoder, head))
    batch = _grid_batch({"mask": torch.randint(0, 3, (2, 32, 32))})
    check_learner(learner, batch, budget=BUDGET)


def _points() -> torch.Tensor:
    """Two level-0 points per image (one NaN-padded row in the second)."""
    first = torch.tensor([[10.0, 12.0, 0.0], [20.0, 8.0, 1.0]])
    second = torch.tensor([[5.0, 25.0, 1.0], [float("nan")] * 3])
    return torch.stack([first, second])


def test_dense_learner_conforms_for_detection():
    torch.manual_seed(0)
    geometry = compute_dense_geometry(target_size=32, patch_size=4)
    head = DetectionHead(
        num_classes=2, geometry=geometry, spacing_um=1.0, match_distance=3.0, sigma=1.5,
        metrics=["mean_f1"],
    )
    decoder = decoder_registry.get("linear")(input_dim=D, num_classes=2)
    learner = DenseLearner(DenseModel(decoder, head))
    ignore = torch.zeros(2, 32, 32, dtype=torch.bool)
    ignore[1, :4] = True
    check_learner(learner, _grid_batch({"points": _points(), "ignore": ignore}), budget=BUDGET)


@pytest.mark.parametrize("raw", [True, False], ids=["uint8-pixels", "preprocessed-pixels"])
def test_dense_learner_conforms_with_a_live_encoder_stage(raw):
    """The parameter-free stage sits inside the model: nothing of it is optimized or
    checkpointed, and pixels stay on the stage's input device."""
    torch.manual_seed(0)
    stage = FakeEncoder(target_size=32, patch_size=4)
    head = SegmentationHead(num_classes=3, geometry=stage.geometry, metrics=["mean_dice"])
    decoder = decoder_registry.get("linear")(input_dim=stage.feature_dim, num_classes=3)
    learner = DenseLearner(DenseModel(decoder, head, encoder=stage))
    pixels = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
    if not raw:
        pixels = torch.stack([stage.preprocessor()(item) for item in pixels])
    batch = GridBatch(features=None, pixels=pixels, targets={"mask": torch.randint(0, 3, (2, 32, 32))}, sample_ids=("a", "b"))
    check_learner(learner, batch, budget=BUDGET)
    assert not any(key.startswith("encoder") for key in learner.trainable_state_dict())


def test_check_learner_names_the_broken_method():
    class Forgetful(MILLearner):
        def validation_summary(self, outputs):
            return {"auroc": 0.5}  # no "loss"

    learner, batch = _mil_learner("mean_pool")
    broken = Forgetful(learner.model)
    with pytest.raises(ConformanceError, match="validation_summary"):
        check_learner(broken, batch, budget=BUDGET)


def test_check_learner_catches_a_parameter_outside_every_param_group():
    class Partial(MILLearner):
        def configure_optimizers(self, budget):
            return torch.optim.SGD(self.model.task_head.parameters(), lr=0.1)

    learner, batch = _mil_learner("abmil")
    with pytest.raises(ConformanceError, match="configure_optimizers"):
        check_learner(Partial(learner.model), batch, budget=BUDGET)


def test_check_learner_catches_an_optimized_parameter_outside_the_trainable_state():
    class WithAux(MILLearner):
        def __init__(self, model):
            super().__init__(model)
            self.aux = torch.nn.Linear(D, 1)

        def training_step(self, batch):
            return super().training_step(batch) + self.aux(batch.features).mean()

    learner, batch = _mil_learner("mean_pool")
    with pytest.raises(ConformanceError, match=r"trainable_state_dict.*aux\.weight"):
        check_learner(WithAux(learner.model), batch, budget=BUDGET)
