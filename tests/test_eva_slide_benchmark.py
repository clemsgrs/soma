"""EVA slide-level sub-benchmarks and their benchmark-private components (issue #535)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from soma.aggregators.registry import aggregator_registry
from soma.benchmarks import Benchmark, get_benchmark, list_benchmarks
from soma.benchmarks import eva_slide as slide
from soma.tasks.registry import task_family_of
from tests import eva_reference as eva

ROOT = Path(__file__).resolve().parents[1]


# --- registry ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dataset", "head"),
    [("camelyon16_small", "eva_mil_binary"), ("panda_small", "eva_mil_multiclass")],
)
def test_slide_subbenchmarks_join_the_eva_family(dataset, head):
    bench = get_benchmark(f"eva/{dataset}")
    assert f"eva/{dataset}" in list_benchmarks()
    assert isinstance(bench, Benchmark)
    assert bench.primary_metric == "test/balanced_accuracy"
    assert bench.canonical_seeds == tuple(range(20))
    assert bench.facet.varied == ("encoder",)
    assert bench.facet.fixed == {"dataset": dataset, "task": head, "protocol": "eva-slide-abmil"}


def test_private_components_are_registered_under_eva_names():
    assert aggregator_registry.get("eva_abmil") is slide.EvaABMIL
    assert task_family_of("eva_mil_binary") == "binary_classification"
    assert task_family_of("eva_mil_multiclass") == "multiclass_classification"


def test_a_fresh_process_resolves_the_private_aggregator():
    """A saved benchmark config names ``eva_abmil``; loading it must not need an import."""
    code = (
        "from soma.aggregators.registry import aggregator_registry\n"
        "from soma.tasks.registry import task_registry\n"
        "print(aggregator_registry.get('eva_abmil').__name__, "
        "task_registry.get('eva_mil_binary').__name__)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["EvaABMIL", "EvaMILBinaryHead"]


@pytest.mark.parametrize(
    ("dataset", "encoder", "expected"),
    [
        ("camelyon16_small", "uni2", 0.849),
        ("camelyon16_small", "virchow2", 0.861),
        ("panda_small", "uni2", 0.657),
        ("panda_small", "virchow2", 0.646),
    ],
)
def test_reference_rows_are_eva_test_balanced_accuracy(dataset, encoder, expected):
    (row,) = get_benchmark(f"eva/{dataset}").expected(encoder=encoder)
    assert row.metric == "test/balanced_accuracy"
    assert row.expected == pytest.approx(expected)
    assert row.tolerance == pytest.approx(0.02)
    assert row.relative is True
    assert row.tolerance_band() == pytest.approx(0.02 * expected)


# --- config -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dataset", "target_mpp", "head", "task_params"),
    [
        ("camelyon16_small", 0.25, "eva_mil_binary", {}),
        ("panda_small", 0.5, "eva_mil_multiclass", {"num_classes": 6}),
    ],
)
def test_config_transcribes_the_eva_offline_protocol(dataset, target_mpp, head, task_params):
    config = get_benchmark(f"eva/{dataset}").build_config(
        encoder="virchow2", dataset_csv="d.csv", splits_csv="s.csv", output_root="out", seed=7
    )
    assert config.dataset_type == "slide"
    assert config.preprocessing.requested_spacing_um == target_mpp
    assert config.preprocessing.requested_tile_size_px == 224
    assert config.encoder.output_variant == "cls"  # virchow2 CLS-only, as EVA
    assert config.aggregator.name == "eva_abmil" and config.aggregator.params == {}
    assert config.task.name == head and dict(config.task.params) == task_params
    training = config.training
    assert (training.epochs, training.max_steps) == (100, None)
    assert training.learning_rate == pytest.approx(1e-3)
    assert training.weight_decay == pytest.approx(0.01)
    assert (training.optimizer, training.scheduler) == ("adamw", "none")
    assert training.lr_warmup_epochs == 5
    assert training.lr_warmup_factor == pytest.approx(1 / 3)
    assert training.patience == 20
    # EVA's Lightning EarlyStopping: plain patience, no hold on an initial plateau.
    assert training.hold_patience_until_monitor_moves is False
    assert (training.monitor, training.monitor_mode) == ("balanced_accuracy", "max")
    assert training.batch_size == 32
    assert training.tune_is_test is False
    assert training.checkpoint_selection == "best"
    assert training.seed == 7
    assert "balanced_accuracy" in config.evaluation.metrics


def test_uni2_keeps_its_default_output():
    config = get_benchmark("eva/panda_small").build_config(encoder="uni2")
    assert config.encoder.output_variant is None


# --- ABMIL oracle -----------------------------------------------------------------------


def _copy_eva_weights(reference: eva.ABMIL, aggregator: slide.EvaABMIL, head) -> None:
    with torch.no_grad():
        aggregator.projector.load_state_dict(reference.projector[0].state_dict())
        aggregator.pool.fc1.load_state_dict(reference.gated_attention.attention_a[0].state_dict())
        aggregator.pool.fc_gate.load_state_dict(reference.gated_attention.attention_b[0].state_dict())
        # EVA's attention_c bias shifts every attention logit equally: softmax ignores it.
        aggregator.pool.fc2.weight.copy_(reference.gated_attention.attention_c.weight)
        reference_linears = [m for m in reference.classifier._network if isinstance(m, torch.nn.Linear)]
        head_linears = [m for m in head.mlp if isinstance(m, torch.nn.Linear)]
        assert len(reference_linears) == len(head_linears) == 3
        for theirs, ours in zip(reference_linears, head_linears):
            ours.load_state_dict(theirs.state_dict())


@pytest.mark.parametrize(("head_cls", "output_size"), [(slide.EvaMILBinaryHead, 1), (slide.EvaMILMulticlassHead, 6)])
def test_eva_abmil_and_head_match_eva_abmil(head_cls, output_size):
    torch.manual_seed(0)
    input_dim, max_tiles = 48, 20
    reference = eva.ABMIL(input_size=input_dim, output_size=output_size, projected_input_size=128)
    aggregator = slide.EvaABMIL(input_dim)
    head = head_cls(128) if output_size == 1 else head_cls(128, num_classes=6)
    _copy_eva_weights(reference, aggregator, head)
    # Same parameter count as EVA except the softmax-invariant attention_c bias.
    ours = sum(p.numel() for p in [*aggregator.parameters(), *head.parameters()])
    assert ours == sum(p.numel() for p in reference.parameters()) - 1

    sizes = [20, 7, 1, 13]
    features = torch.randn(len(sizes), max_tiles, input_dim) * 3.0
    mask = torch.arange(max_tiles)[None, :] < torch.tensor(sizes)[:, None]
    padded = features.masked_fill(~mask[..., None], float("-inf"))  # EVA's Pad2DTensor
    garbage = features.masked_fill(~mask[..., None], 1e3)  # soma ignores padded values

    reference.eval(), aggregator.eval(), head.eval()
    expected = reference(padded).squeeze(-1)  # EVA's HeadModule squeezes the last dim
    out = aggregator(garbage, mask=mask)
    logits = head(out.bag_representation).squeeze(-1)
    torch.testing.assert_close(logits, expected, rtol=1e-5, atol=1e-6)
    assert out.bag_representation.shape == (len(sizes), 128)
    assert out.tile_attention.shape == (len(sizes), max_tiles)


def test_eva_abmil_has_no_dropout():
    modules = [*slide.EvaABMIL(16).modules(), *slide.EvaMILBinaryHead(128).modules()]
    assert not any(isinstance(m, torch.nn.Dropout) for m in modules)


# --- heads ------------------------------------------------------------------------------


def _record(label, sample_id="s"):
    return SimpleNamespace(label=label, sample_id=sample_id)


def test_binary_head_is_one_logit_with_bce():
    head = slide.EvaMILBinaryHead(128, metrics=["balanced_accuracy", "accuracy"])
    logits = head(torch.randn(5, 128))
    assert logits.shape == (5, 1)
    targets = {"label": torch.tensor([0, 1, 1, 0, 1])}
    expected = torch.nn.BCEWithLogitsLoss()(logits.squeeze(-1), targets["label"].float())
    torch.testing.assert_close(head.compute_loss(logits, targets), expected)


def test_binary_head_thresholds_the_logit_at_zero():
    head = slide.EvaMILBinaryHead(128, metrics=["balanced_accuracy", "accuracy"])
    raw = torch.tensor([[-2.0], [0.0], [1e-4], [3.0]])
    processed = head.postprocess(raw)
    assert processed["predicted_labels"].tolist() == [0, 0, 1, 1]  # torchmetrics: sigmoid > 0.5
    np.testing.assert_allclose(processed["probabilities"].sum(axis=1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(processed["probabilities"][:, 1], torch.sigmoid(raw).squeeze(-1))
    metrics = head.compute_metrics(raw, {"label": torch.tensor([0, 0, 1, 1])})
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    # Sensitivity 2/3, specificity 1/1.
    metrics = head.compute_metrics(raw, {"label": torch.tensor([0, 1, 1, 1])})
    assert metrics["balanced_accuracy"] == pytest.approx((2 / 3 + 1) / 2)
    assert metrics["accuracy"] == pytest.approx(0.75)


def test_multiclass_head_is_cross_entropy_over_class_indices():
    head = slide.EvaMILMulticlassHead(128, num_classes=6, metrics=["balanced_accuracy"])
    logits = head(torch.randn(8, 128))
    assert logits.shape == (8, 6)
    targets = {"label": torch.arange(8) % 6}
    torch.testing.assert_close(head.compute_loss(logits, targets), F.cross_entropy(logits, targets["label"]))
    assert head.postprocess(logits)["probabilities"].shape == (8, 6)
    assert head.extract_targets(_record(5)) == {"label": 5}
    with pytest.raises(ValueError, match="class index"):
        head.extract_targets(_record(6))


def test_multiclass_auto_params_reads_the_class_indices():
    dataset = SimpleNamespace(samples={i: _record(label) for i, label in enumerate([0, 3, 2])})
    assert slide.EvaMILMulticlassHead.auto_params(dataset) == {"num_classes": 4}


def test_binary_head_refuses_a_non_binary_label():
    with pytest.raises(ValueError, match="class index"):
        slide.EvaMILBinaryHead(128).extract_targets(_record(2))
