"""``soma.testing.check_learner`` on soma's own learners: every aggregator, seg and det."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from soma.aggregators import aggregator_registry, list_aggregators
from soma.decoders.registry import decoder_registry
from soma.dense import DenseSampleSpacing
from soma.dense.geometry import compute_dense_geometry
from soma.detection.encode import render_peak_heatmap
from soma.learn import Budget, DenseLearner, MILLearner
from soma.tasks.classification import BinaryClassificationHead, BranchAwareClassificationHead
from soma.tasks.detection import DetectionHead
from soma.tasks.segmentation import SegmentationHead
from soma.testing import ConformanceError, check_learner
from soma.training.collate import BagBatch, HierarchicalBagBatch
from soma.training.model import MILModel, SegmentationModel
from soma.training.segmentation_dataset import SegmentationBatch

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


@pytest.mark.parametrize("name", list_aggregators())
def test_mil_learner_conforms_for_every_aggregator(name):
    learner, batch = _mil_learner(name)
    check_learner(learner, batch, budget=BUDGET)


def _grid_batch(targets: dict[str, torch.Tensor]) -> SegmentationBatch:
    return SegmentationBatch(features=torch.randn(2, D, 8, 8), targets=targets, sample_ids=("a", "b"))


def test_dense_learner_conforms_for_segmentation():
    torch.manual_seed(0)
    geometry = compute_dense_geometry(target_size=32, patch_size=4)
    head = SegmentationHead(num_classes=3, geometry=geometry, metrics=["mean_dice"])
    decoder = decoder_registry.get("linear")(input_dim=D, num_classes=3)
    learner = DenseLearner(SegmentationModel(decoder=decoder, task_head=head))
    batch = _grid_batch({"mask": torch.randint(0, 3, (2, 32, 32))})
    check_learner(learner, batch, budget=BUDGET)


def test_dense_learner_conforms_for_detection():
    torch.manual_seed(0)
    geometry = compute_dense_geometry(target_size=32, patch_size=4)
    head = DetectionHead(
        num_classes=2,
        geometry=geometry,
        delta_px=3.0,
        sigma_px=1.5,
        score_threshold=0.5,
        sample_spacings={"a": DenseSampleSpacing(1.0, 1.0), "b": DenseSampleSpacing(1.0, 1.0)},
        metrics=["mean_f1"],
    )
    decoder = decoder_registry.get("linear")(input_dim=D, num_classes=2)
    learner = DenseLearner(SegmentationModel(decoder=decoder, task_head=head))
    xy, cls = np.array([[10.0, 12.0], [20.0, 8.0]]), np.array([0, 1])
    heatmap = render_peak_heatmap(xy, cls, target_size=(32, 32), num_classes=2, sigma=1.5)
    gt = torch.tensor([[10.0, 12.0, 0.0], [20.0, 8.0, 1.0]])
    batch = _grid_batch({"heatmap": torch.stack([heatmap, heatmap]), "gt_points": torch.stack([gt, gt])})
    check_learner(learner, batch, budget=BUDGET)


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
