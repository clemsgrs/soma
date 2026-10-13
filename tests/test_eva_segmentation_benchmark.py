"""EVA segmentation sub-benchmarks and their benchmark-private components (issue #522)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from soma.config import Component
from soma.plan import head_family
from tests.experiment_configs import callback
from soma.benchmark import Benchmark, benchmark_registry, get_benchmark
from soma.benchmark.eva import segmentation as seg
from soma.benchmark.eva import EvaTileClassificationBenchmark
from soma.benchmark.curation import CuratedManifest
from soma.models.decoders.registry import decoder_registry
from soma.data.geometry import compute_dense_geometry
from soma.models.heads.dense_metrics import dense_confusion_counts, reduce_foreground_dice
from soma.models.heads.registry import task_family_of, task_registry


def _logits_from_pred(pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(pred, num_classes).permute(0, 3, 1, 2).float() * 10.0


# --- registry ---------------------------------------------------------------------------


def test_segmentation_subbenchmarks_join_the_eva_family():
    names = sorted(benchmark_registry.list())
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
    params = config.learner.params
    assert config.learner.name == "dense" and head_family(config) == "segmentation"
    assert params["decoder"] == Component("eva_conv_with_image")
    assert params["head"] == Component("eva_segmentation", {"num_classes": 5})
    assert config.features.extraction.tiling.requested_tile_size_px == 224
    assert config.features.dense.feature_kind == "patch_features_prenorm"  # EVA's timm tap
    assert config.evaluation.metrics[0] == "foreground_mean_dice"
    optimizer = params["optimizer"]
    assert optimizer.name == "adamw"
    # EVA's warm-up (ConstantLR, factor 1/3 for 5 epochs) is the only scheduler leaf.
    (scheduler,) = params["scheduler"]
    assert scheduler.name == "constant"
    assert scheduler.params["total_iters"] == 5
    assert scheduler.params["factor"] == pytest.approx(1 / 3)
    assert optimizer.params["lr"] == pytest.approx(2e-3)
    assert optimizer.params["weight_decay"] == pytest.approx(0.01)
    assert config.loaders.batch_size == 64
    assert config.trainer.max_epochs is None and config.trainer.max_steps == 2000
    monitor = {"monitor": "foreground_mean_dice", "mode": "max"}
    assert callback(config, "checkpoint") == monitor
    assert callback(config, "early_stopping") == {**monitor, "patience": 200}
    assert config.cohort.tune_is_test is True
    assert config.run.seed == 3
    assert config.features.encoder.output_variant is None
    # CoNSeP's online transforms are deterministic: the cached grids are EVA's samples.
    assert config.features.kind == "cached"
    assert config.loaders.augment is None


def test_monusac_config_ignores_the_ambiguous_class(tmp_path: Path):
    config = get_benchmark("eva/monusac").build_config(
        dataset_csv=tmp_path / "d.csv", splits_csv=tmp_path / "s.csv", output_root=tmp_path
    )
    assert config.learner.params["head"].params == {"num_classes": 5, "ignore_index": 5}
    assert callback(config, "early_stopping")["patience"] == 100
    assert config.features.encoder.name == "uni2"  # DEFAULT_ENCODER shared with the family


def test_monusac_config_random_resized_crops_whole_train_images_live(tmp_path: Path):
    """EVA's online MoNuSAC config trains on torchvision ``RandomResizedCrop(224)``
    (default scale (0.08, 1.0)) drawn every step, so the run re-encodes live."""
    config = get_benchmark("eva/monusac").build_config(
        dataset_csv=tmp_path / "d.csv", splits_csv=tmp_path / "s.csv", output_root=tmp_path
    )
    assert config.features.kind == "live"
    assert config.loaders.augment == Component("segmentation", {"crop_scale": [0.08, 1.0]})


def test_build_config_honours_smoke_overrides_and_cache(tmp_path: Path):
    config = get_benchmark("eva/consep").build_config(
        dataset_csv="d.csv",
        splits_csv="s.csv",
        output_root=tmp_path,
        max_steps=5,
        patience=1,
        overrides={"cache": {"enabled": True, "root_dir": str(tmp_path / "cache")}},
    )
    assert config.trainer.max_steps == 5 and callback(config, "early_stopping")["patience"] == 1
    assert str(config.features.cache.root_dir) == str(tmp_path / "cache")


def test_saved_benchmark_config_loads_in_a_fresh_process(tmp_path: Path):
    """The private head/decoder resolve without the caller importing ``soma.benchmark``."""
    import os
    import subprocess
    import sys

    import soma

    config = get_benchmark("eva/monusac").build_config(
        dataset_csv=tmp_path / "d.csv", splits_csv=tmp_path / "s.csv", output_root=tmp_path
    )
    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    script = (
        "import sys\n"
        "from soma.config import ExperimentConfig\n"
        "from soma.plan import check\n"
        f"config = ExperimentConfig.from_yaml({str(path)!r})\n"
        "check(config)\n"
        "assert 'soma.benchmark' in sys.modules, 'registries import the bundled module'\n"
        "params = config.learner.params\n"
        "print(params['head'].name, params['decoder'].name)\n"
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
    from soma.models.decoders.registry import build_decoder_for_grid

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
    from soma.models.heads.dense_metrics import segmentation_loss

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
    from soma.models.decoders.registry import build_decoder_for_grid

    geometry = compute_dense_geometry(target_size=224, patch_size=14)
    decoder = build_decoder_for_grid(
        "eva_conv_with_image", None, geometry=geometry, input_dim=3, num_classes=5
    )
    assert isinstance(decoder, seg.EvaConvWithImageDecoder)


# --- scoring ----------------------------------------------------------------------------


def _finished_run(tmp_path: Path, dataset: str, masks: dict, predicted: dict) -> tuple[Path, Path]:
    """A finished single-fold EVA run: flat masks, its saved config and ``predictions_test/``.

    Returns ``(output_root, dataset_csv)``.
    """
    import numpy as np
    import pandas as pd
    from PIL import Image

    from soma.evaluation import LabelMapPredictions

    rows = []
    for sample_id, mask in masks.items():
        mask = np.asarray(mask, dtype=np.uint8)
        image_path, mask_path = tmp_path / f"{sample_id}.png", tmp_path / f"{sample_id}_mask.png"
        Image.fromarray(np.zeros((*mask.shape, 3), dtype=np.uint8)).save(image_path)
        Image.fromarray(mask).save(mask_path)
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": str(image_path),
                "label_mask_path": str(mask_path),
                "spacing_at_level_0": 0.25,
            }
        )
    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(dataset_csv, index=False)
    config = get_benchmark(f"eva/{dataset}").build_config(
        dataset_csv=dataset_csv, splits_csv=tmp_path / "splits.csv", output_root=tmp_path
    )
    run_dir = tmp_path / "experiments" / "exp" / "runs" / "run"
    run_dir.mkdir(parents=True)
    config.to_yaml(run_dir / "config.yaml")
    spec = seg.DATASETS[dataset]
    LabelMapPredictions(
        {sid: np.asarray(m, dtype=np.uint8) for sid, m in predicted.items()},
        classes=spec.class_names,
        ignore_index=5 if spec.ignore_index is None else spec.ignore_index,
    ).save(run_dir / "predictions_test")
    return tmp_path, dataset_csv


def _core_evaluation(output_root: Path, dataset_csv: Path) -> dict[str, float]:
    from soma.data import AnnotationManifest, LabelMapSource
    from soma.evaluation import LabelMapPredictions, evaluate

    predictions = LabelMapPredictions.load(
        output_root / "experiments" / "exp" / "runs" / "run" / "predictions_test"
    )
    shape = predictions[predictions.sample_ids[0]].shape
    targets = LabelMapSource.from_manifests(AnnotationManifest.from_csv(dataset_csv), size=shape)
    return evaluate(predictions, targets, metrics=seg.METRICS).metrics


def test_monusac_score_counts_ignored_pixels_as_predicted_area_like_eva(tmp_path: Path):
    # One nucleus pixel, one Ambiguous (ignored) pixel predicted as that nucleus class.
    # EVA counts the ignored pixel as predicted area: Dice 2*1 / (2 + 1) = 2/3.
    output_root, dataset_csv = _finished_run(
        tmp_path, "monusac", masks={"a": [[1, 5]]}, predicted={"a": [[1, 1]]}
    )
    scores = get_benchmark("eva/monusac").score(output_root)
    assert set(scores) == {f"test/{metric}" for metric in seg.METRICS}
    assert scores["test/foreground_mean_dice"] == pytest.approx(2 / 3)
    # Core evaluate drops the ignored pixel from both sides (the standard counts).
    assert _core_evaluation(output_root, dataset_csv)["foreground_mean_dice"] == pytest.approx(1.0)


def test_monusac_score_equals_core_evaluate_without_ignored_pixels(tmp_path: Path):
    output_root, dataset_csv = _finished_run(
        tmp_path,
        "monusac",
        masks={"a": [[0, 1], [2, 2]], "b": [[3, 3], [0, 4]]},
        predicted={"a": [[0, 1], [2, 1]], "b": [[3, 0], [0, 4]]},
    )
    scores = get_benchmark("eva/monusac").score(output_root)
    core = _core_evaluation(output_root, dataset_csv)
    assert scores == pytest.approx({f"test/{metric}": core[metric] for metric in seg.METRICS})


def test_eva_label_map_counts_match_the_training_time_eva_counts():
    mask = torch.tensor([[[0, 1], [1, 5]]])
    pred = torch.tensor([[[0, 1], [1, 1]]])
    expected = seg.eva_confusion_counts(
        _logits_from_pred(pred, num_classes=2), mask, num_classes=2, ignore_index=5
    )
    counts = seg.eva_label_map_counts(pred[0].numpy(), mask[0], num_classes=2, ignore_index=5)
    assert counts.tolist() == expected[0].tolist()
