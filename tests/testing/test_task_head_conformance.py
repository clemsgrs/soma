"""``soma.testing.check_task_head`` on every built-in set-shape task head."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import soma.benchmark  # noqa: F401  (registers the benchmark-private heads)
from soma.models.heads.classification import BinaryClassificationHead
from soma.models.heads.registry import task_registry
from soma.models.heads.regression import RegressionHead
from soma.models.heads.survival import CoxSurvivalHead
from soma.testing import ConformanceError, check_task_head
from soma.learn.sample_dataset import SampleBatch

D = 8
B = 12
GRID_FAMILIES = {"segmentation", "detection"}


def _set_heads() -> list[str]:
    return sorted(
        name
        for name in task_registry.list()
        if task_registry.get(name).task_family not in GRID_FAMILIES
    )


def _targets(family: str, num_classes: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    if family in {"binary_classification", "multiclass_classification", "ordinal_classification"}:
        return {"label": torch.arange(B) % num_classes}
    if family == "regression":
        return {"value": torch.randn(B, generator=generator)}
    time = torch.rand(B, generator=generator) * 10
    return {
        "time": time,
        "event": (torch.arange(B) % 3 != 0).float(),
        "bin": (time / 3).long().clamp(max=3),
    }


def _head_and_batch(head_cls: type, **params):
    torch.manual_seed(0)
    family = head_cls.task_family
    if family in {"binary_classification"}:
        params.setdefault("num_classes", 2)
    elif family in {"multiclass_classification", "ordinal_classification"}:
        params.setdefault("num_classes", 3)
    elif family == "survival" and head_cls is not CoxSurvivalHead:
        params.setdefault("num_bins", 4)
    head = head_cls(input_dim=D, **params)
    num_classes = int(getattr(head, "num_classes", 0) or 0)
    shape = (B, num_classes, D) if getattr(head, "supports_branch_representation", False) else (B, D)
    batch = SampleBatch(
        features=torch.randn(*shape),
        targets=_targets(family, num_classes),
        sample_ids=tuple(f"s{i:02d}" for i in range(B)),
    )
    return head, batch


@pytest.mark.parametrize("name", _set_heads())
def test_every_builtin_set_head_conforms(name):
    head, batch = _head_and_batch(task_registry.get(name))
    check_task_head(head, batch)


def test_cox_survival_head_conforms():
    head, batch = _head_and_batch(CoxSurvivalHead)
    check_task_head(head, batch)


def test_a_multi_output_regression_head_conforms():
    # A gene panel: one (B, G) ``value`` tensor, one value_<k> column per output.
    torch.manual_seed(0)
    head = RegressionHead(input_dim=D, num_outputs=2)
    batch = SampleBatch(
        features=torch.randn(B, D),
        targets={"value": torch.randn(B, 2, generator=torch.Generator().manual_seed(1))},
        sample_ids=tuple(f"s{i:02d}" for i in range(B)),
    )
    check_task_head(head, batch)


def test_a_head_built_with_a_label_map_conforms_on_its_raw_labels():
    head, batch = _head_and_batch(
        BinaryClassificationHead, label_map={"benign": 0, "tumor": 1}
    )
    assert head.classes == ("benign", "tumor")
    check_task_head(head, batch)


class _UndeclaredColumnHead(BinaryClassificationHead):
    def to_columns(self, processed):
        columns = super().to_columns(processed)
        columns["confidence"] = np.max(processed["probabilities"], axis=1)
        return columns


def test_a_head_emitting_an_undeclared_column_is_refused():
    head, batch = _head_and_batch(_UndeclaredColumnHead)
    with pytest.raises(ConformanceError, match="to_columns"):
        check_task_head(head, batch)


class _TextLabelHead(BinaryClassificationHead):
    def to_columns(self, processed):
        columns = super().to_columns(processed)
        columns["predicted_label"] = np.array(["neg", "pos"])[columns["predicted_label"]]
        return columns


def test_a_head_emitting_a_non_numeric_column_is_refused():
    head, batch = _head_and_batch(_TextLabelHead)
    with pytest.raises(ConformanceError, match="numeric"):
        check_task_head(head, batch)
