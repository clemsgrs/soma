"""EVA segmentation sub-benchmarks and their benchmark-private components (issue #522)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from soma.benchmarks import Benchmark, get_benchmark, list_benchmarks
from soma.benchmarks import eva_segmentation as seg
from soma.benchmarks.eva import EvaTileClassificationBenchmark
from soma.curation.manifest import CuratedManifest
from soma.decoders.registry import decoder_registry
from soma.dense import compute_dense_geometry
from soma.tasks.dense_metrics import dense_confusion_counts, reduce_foreground_dice
from soma.tasks.registry import task_family_of, task_registry


def _logits_from_pred(pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(pred, num_classes).permute(0, 3, 1, 2).float() * 10.0


# --- registry ---------------------------------------------------------------------------


def test_segmentation_subbenchmarks_join_the_eva_family():
    names = list_benchmarks()
    for dataset in ("consep", "monusac"):
        bench = get_benchmark(f"eva/{dataset}")
        assert f"eva/{dataset}" in names
        assert isinstance(bench, Benchmark)
        assert bench.primary_metric == "test/foreground_mean_dice"
        assert bench.canonical_seeds == (0, 1, 2, 3, 4)
        assert bench.facet.varied == ("encoder",)
        assert bench.facet.fixed["task"] == "eva_segmentation"
    # The classification members keep their own class; both share one family prefix.
    assert isinstance(get_benchmark("eva/bach"), EvaTileClassificationBenchmark)
    assert get_benchmark("eva/bach").primary_metric == "test/balanced_accuracy"


def test_private_components_are_registered_under_eva_names():
    assert "eva_conv_ms" in decoder_registry
    assert "eva_conv_with_image" in decoder_registry
    assert "eva_segmentation" in task_registry
    assert task_family_of("eva_segmentation") == "segmentation"


# --- reference rows ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dataset", "encoder", "expected"),
    [
        ("consep", "uni2", 0.630),
        ("consep", "virchow2", 0.640),
        ("monusac", "uni2", 0.642),
        ("monusac", "virchow2", 0.669),
    ],
)
def test_reference_rows_carry_the_leaderboard_dice(dataset, encoder, expected):
    rows = get_benchmark(f"eva/{dataset}").expected(encoder=encoder)
    assert [(r.metric, r.expected, r.tolerance, r.relative) for r in rows] == [
        ("test/foreground_mean_dice", expected, 0.02, True)
    ]


def test_classification_reference_rows_are_untouched():
    rows = get_benchmark("eva/bach").expected(encoder="uni2")
    assert [(r.metric, r.expected) for r in rows] == [("test/balanced_accuracy", 0.915)]


# --- config -----------------------------------------------------------------------------


def test_build_config_encodes_the_eva_segmentation_protocol(tmp_path: Path):
    config = get_benchmark("eva/consep").build_config(
        encoder="uni2",
        dataset_csv=tmp_path / "d.csv",
        splits_csv=tmp_path / "s.csv",
        output_root=tmp_path / "out",
        seed=3,
    )
    assert config.dataset_type == "segmentation"
    assert config.decoder.name == "eva_conv_with_image" and config.decoder.params == {}
    assert config.task.name == "eva_segmentation"
    assert config.task.params == {"num_classes": 5}
    assert config.preprocessing.requested_tile_size_px == 224
    assert config.preprocessing.feature_kind == "patch_features_prenorm"  # EVA's timm tap
    assert config.evaluation.metrics[0] == "foreground_mean_dice"
    training = config.training
    assert (training.optimizer, training.scheduler) == ("adamw", "none")
    # EVA's SemanticSegmentationModule defaults ``lr_scheduler`` to torch ConstantLR
    # (factor 1/3 for 5 epochs), which its YAML configs never override.
    assert training.lr_warmup_epochs == 5
    assert training.lr_warmup_factor == pytest.approx(1 / 3)
    assert training.learning_rate == pytest.approx(2e-3)
    assert training.weight_decay == pytest.approx(0.01)
    assert training.batch_size == 64
    assert training.epochs is None and training.max_steps == 2000
    assert training.patience == 200
    assert training.monitor == "foreground_mean_dice" and training.monitor_mode == "max"
    assert training.tune_is_test is True
    assert training.seed == 3
    assert config.encoder.output_variant is None
    assert config.tags == ["eva", "consep", "uni2", "segmentation"]


def test_monusac_config_ignores_the_ambiguous_class(tmp_path: Path):
    config = get_benchmark("eva/monusac").build_config(
        dataset_csv=tmp_path / "d.csv", splits_csv=tmp_path / "s.csv", output_root=tmp_path
    )
    assert config.task.params == {"num_classes": 5, "ignore_index": 5}
    assert config.training.patience == 100
    assert config.encoder.name == "uni2"  # DEFAULT_ENCODER shared with the family


def test_build_config_honours_smoke_overrides_and_cache(tmp_path: Path):
    config = get_benchmark("eva/consep").build_config(
        dataset_csv="d.csv",
        splits_csv="s.csv",
        output_root=tmp_path,
        max_steps=5,
        patience=1,
        overrides={"cache": {"enabled": True, "root_dir": str(tmp_path / "cache")}},
    )
    assert config.training.max_steps == 5 and config.training.patience == 1
    assert str(config.cache.root_dir) == str(tmp_path / "cache")


def test_saved_benchmark_config_loads_in_a_fresh_process(tmp_path: Path):
    """The private head/decoder resolve without the caller importing ``soma.benchmarks``."""
    import os
    import subprocess
    import sys

    import soma
    from soma.config import save_config

    config = get_benchmark("eva/monusac").build_config(
        dataset_csv=tmp_path / "d.csv", splits_csv=tmp_path / "s.csv", output_root=tmp_path
    )
    path = tmp_path / "config.yaml"
    save_config(config, path)
    script = (
        "import sys\n"
        "from soma.config import load_config\n"
        f"config = load_config({str(path)!r})\n"
        "assert 'soma.benchmarks' in sys.modules, 'registries import the bundled module'\n"
        "print(config.task.name, config.decoder.name)\n"
    )
    # Import the same soma as this process (the checkout may shadow an installed copy).
    env = {**os.environ, "PYTHONPATH": str(Path(soma.__file__).resolve().parents[1])}
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, cwd=tmp_path, env=env
    )
    assert result.stdout.strip() == "eva_segmentation eva_conv_with_image"


def test_curate_delegates_to_the_segmentation_curator(monkeypatch, tmp_path):
    calls = {}

    def _fake(name, raw_root, output_dir):
        calls["args"] = (name, str(raw_root), str(output_dir))
        return CuratedManifest(
            dataset_csv=Path(output_dir) / "dataset.csv", splits_csv=Path(output_dir) / "splits.csv"
        )

    monkeypatch.setattr(seg, "curate_eva_segmentation_dataset", _fake)
    manifest = get_benchmark("eva/monusac").curate(tmp_path / "raw", tmp_path / "out")
    assert calls["args"] == ("monusac", str(tmp_path / "raw"), str(tmp_path / "out"))
    assert manifest.dataset_csv == tmp_path / "out" / "dataset.csv"


# --- components -------------------------------------------------------------------------


def test_conv_ms_decoder_matches_eva_shape_and_layers():
    decoder = seg.EvaConvMSDecoder(input_dim=7, num_classes=5)
    out = decoder(torch.randn(2, 7, 16, 16))
    assert tuple(out.shape) == (2, 5, 64, 64)  # two nearest x2 upsamples
    assert decoder.num_classes == 5
    kinds = [type(m).__name__ for m in decoder.layers]
    assert kinds == ["Upsample", "Conv2d", "Upsample", "Conv2d"]
    assert decoder.layers[1].out_channels == 64 and decoder.layers[1].kernel_size == (3, 3)
    assert all(m.mode == "nearest" for m in decoder.layers if isinstance(m, torch.nn.Upsample))


def test_conv_ms_decoder_builds_through_the_registry_for_a_grid():
    from soma.decoders.registry import build_decoder_for_grid

    geometry = compute_dense_geometry(target_size=224, patch_size=14)
    decoder = build_decoder_for_grid("eva_conv_ms", None, geometry=geometry, input_dim=3, num_classes=5)
    assert isinstance(decoder, seg.EvaConvMSDecoder)
    assert tuple(decoder(torch.zeros(1, 3, 16, 16)).shape) == (1, 5, 64, 64)


def test_eva_dice_loss_is_monai_batch_dice_without_cross_entropy():
    # Perfect sharp prediction -> loss ~ 0; uniform logits -> 1 - mean Dice of 1/C-probs.
    mask = torch.tensor([[[0, 1], [1, 0]]])
    perfect = _logits_from_pred(mask, num_classes=2) * 10
    assert seg.eva_dice_loss(perfect, mask, num_classes=2, ignore_index=255) == pytest.approx(0.0, abs=1e-4)
    uniform = torch.zeros(1, 2, 2, 2)
    # probs = 0.5 everywhere: intersection_c = 0.5 * 2 = 1, denominator_c = 2 + 2 = 4.
    expected = 1.0 - (2 * 1 + 1e-5) / (4 + 1e-5)
    assert seg.eva_dice_loss(uniform, mask, num_classes=2, ignore_index=255) == pytest.approx(expected)
    # The built-in objective adds cross-entropy, so it must differ on the uniform case.
    from soma.tasks.dense_metrics import segmentation_loss

    assert segmentation_loss(uniform, mask, num_classes=2, ignore_index=255) > expected + 0.1


def test_eva_dice_loss_drops_ignored_pixels_and_keeps_gradients():
    mask = torch.tensor([[[0, 5], [1, 5]]])
    logits = torch.randn(1, 2, 2, 2, requires_grad=True)
    loss = seg.eva_dice_loss(logits, mask, num_classes=2, ignore_index=5)
    loss.backward()
    assert torch.isfinite(loss)
    # Ignored pixels receive no gradient.
    assert logits.grad[0, :, 0, 1].abs().sum() == 0 and logits.grad[0, :, 1, 1].abs().sum() == 0
    assert logits.grad[0, :, 0, 0].abs().sum() > 0


def test_eva_confusion_counts_keep_predicted_area_on_ignored_pixels():
    # EVA's wrapper masks by the predicted ignore channel, which is never set, so an
    # ignored pixel still counts as predicted area for the class predicted there.
    mask = torch.tensor([[[0, 1], [1, 5]]])
    pred = torch.tensor([[[0, 1], [1, 1]]])
    logits = _logits_from_pred(pred, num_classes=2)
    eva_counts = seg.eva_confusion_counts(logits, mask, num_classes=2, ignore_index=5)
    soma_counts = dense_confusion_counts(logits, mask, num_classes=2, ignore_index=5)
    assert eva_counts.tolist() == [[[1, 1, 1], [2, 3, 2]]]
    assert soma_counts.tolist() == [[[1, 1, 1], [2, 2, 2]]]
    assert reduce_foreground_dice(eva_counts, num_classes=2) == pytest.approx(4 / 5)
    assert reduce_foreground_dice(soma_counts, num_classes=2) == pytest.approx(1.0)
    # Without an ignored class the two countings agree.
    clean = torch.tensor([[[0, 1], [1, 0]]])
    assert seg.eva_confusion_counts(logits, clean, num_classes=2, ignore_index=5).tolist() == (
        dense_confusion_counts(logits, clean, num_classes=2, ignore_index=5).tolist()
    )


def test_eva_head_reports_foreground_mean_dice_through_the_built_in_reduction():
    geometry = compute_dense_geometry(target_size=4, patch_size=1)
    head = task_registry.get("eva_segmentation")(
        num_classes=3,
        geometry=geometry,
        ignore_index=5,
        metrics=["foreground_mean_dice", "mean_dice"],
    )
    assert isinstance(head, seg.EvaSegmentationHead)
    mask = torch.zeros(1, 4, 4, dtype=torch.long)
    mask[0, :2, :] = 1  # class 2 absent from the target
    pred = mask.clone()
    pred[0, 3, 3] = 2  # a false positive on the absent class
    logits = _logits_from_pred(pred, num_classes=3)
    metrics = head.compute_metrics(logits, {"mask": mask})
    assert set(metrics) == {"foreground_mean_dice", "mean_dice"}
    # Foreground: class 1 perfect (1.0), class 2 has no target -> skipped.
    assert metrics["foreground_mean_dice"] == pytest.approx(1.0)
    # mean_dice scores the false-positive-only class 2 as 0 and includes background.
    assert metrics["mean_dice"] < 1.0
    assert head.compute_loss(logits, {"mask": mask}) == pytest.approx(
        seg.eva_dice_loss(logits, mask, num_classes=3, ignore_index=5)
    )


def test_conv_with_image_decoder_matches_eva_online_shape_and_layers():
    decoder = seg.EvaConvWithImageDecoder(input_dim=7, num_classes=5)
    grid = torch.randn(2, 7, 16, 16)
    image = torch.rand(2, 3, 224, 224)
    out = decoder(grid, image=image)
    assert tuple(out.shape) == (2, 5, 224, 224)  # logits at the image resolution
    assert decoder.num_classes == 5 and decoder.consumes_image is True
    assert [type(m).__name__ for m in decoder.layers] == ["Upsample", "Conv2dBnReLU"]
    assert [type(m).__name__ for m in decoder.image_block] == ["Conv2dBnReLU", "Conv2dBnReLU"]
    assert decoder.image_block[0][0].in_channels == 64 + 3  # features + RGB
    assert decoder.classifier.kernel_size == (1, 1) and decoder.classifier.out_channels == 5
    with pytest.raises(ValueError, match="image"):
        decoder(grid, image=torch.rand(2, 1, 224, 224))


def test_conv_with_image_decoder_normalises_the_image_like_eva():
    decoder = seg.EvaConvWithImageDecoder(input_dim=2, num_classes=3).eval()
    grid = torch.zeros(1, 2, 2, 2)
    mean = torch.tensor(seg.IMAGE_MEAN).view(1, 3, 1, 1).expand(1, 3, 8, 8)
    seen = {}

    def spy(module, inputs):
        seen["image"] = inputs[0][:, 64:]

    decoder.image_block.register_forward_pre_hook(spy)
    decoder(grid, image=mean)
    assert torch.allclose(seen["image"], torch.zeros(1, 3, 8, 8), atol=1e-6)


def test_conv_with_image_decoder_builds_through_the_registry_for_a_grid():
    from soma.decoders.registry import build_decoder_for_grid

    geometry = compute_dense_geometry(target_size=224, patch_size=14)
    decoder = build_decoder_for_grid(
        "eva_conv_with_image", None, geometry=geometry, input_dim=3, num_classes=5
    )
    assert isinstance(decoder, seg.EvaConvWithImageDecoder)
