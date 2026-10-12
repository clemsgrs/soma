"""Dense prediction: ROIs -> dense token grids -> decoder -> segmentation / detection.

The cached scenarios extract the grids once (through the weight-free literal encoder);
the live scenarios re-encode the ROIs every step through the run's frozen encoder stage,
here ``soma.testing.FakeEncoder`` standing in for slide2vec's kit, so augmentation and the
encoder stage ride inside the model.
"""

from __future__ import annotations

import pytest

from soma.evaluation import LabelMapPredictions, PointPredictions
from soma.testing import FakeEncoder
from tests.e2e.harness import CPU_EXECUTION, CPU_LOADER, run_soma
from tests.e2e.synthetic import SPACING_UM, make_dense_dataset


def _config(
    tmp_path, encoder_name, kind, *, roi_px, task, metrics, epochs, live=False, augmentation=None,
    learning_rate=3e-3,
):
    dataset_csv, splits_csv = make_dense_dataset(tmp_path / "data", kind, size=roi_px)
    return {
        "run": {"output_root": str(tmp_path / "runs"), "seed": 0},
        "data": {
            "dataset_csv": str(dataset_csv),
            "splits_csv": str(splits_csv),
            "dataset_type": kind,
            "feature_mode": "live" if live else "cached",
        },
        "preprocessing": {
            "backend": "openslide",
            "requested_tile_size_px": roi_px,
            "requested_spacing_um": SPACING_UM,
        },
        "execution": CPU_EXECUTION,
        "cache": {"enabled": True, "root_dir": str(tmp_path / "cache")},
        "encoder": {"name": encoder_name, "precision": "fp32", "batch_size": 8},
        "decoder": {"name": "lightweight_conv"},
        "task": task,
        "evaluation": {"metrics": metrics},
        "normalization": {"method": "none" if live else "zscore"},
        "augmentation": augmentation,
        "training": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": 4,
            "patience": epochs,
            **CPU_LOADER,
        },
    }


@pytest.fixture
def fake_live_encoder(monkeypatch):
    """Serve every live run a ``FakeEncoder`` instead of a slide2vec kit; record the calls."""
    import soma.extract

    built: list[dict] = []

    def live_encoder(encoder, *, target_size, **kwargs):
        built.append({"encoder": encoder, "target_size": target_size, **kwargs})
        return FakeEncoder(target_size=target_size, patch_size=8)

    monkeypatch.setattr(soma.extract, "live_encoder", live_encoder)
    return built


def test_segmentation_is_learned(tmp_path, encoder_name, encoded_images, new_artifact):
    artifact = new_artifact("dense_segmentation")
    config = _config(
        tmp_path,
        encoder_name,
        "segmentation",
        roi_px=96,
        task={"name": "segmentation", "params": {"num_classes": 3}},
        metrics=["mean_dice", "mean_iou"],
        epochs=20,
    )
    run = run_soma(config, tmp_path / "config.yaml")

    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_equal("rois_encoded", sum(encoded_images), 12)
    artifact.check_at_least("test/mean_dice", run.summary["test/mean_dice"], 0.8)
    artifact.check_at_least("test/mean_iou", run.summary["test/mean_iou"], 0.7)
    predictions = LabelMapPredictions.load(run.run_dir / "predictions_test")
    artifact.check_equal("predictions_test/ids", sorted(predictions.sample_ids), ["roi10", "roi11"])
    artifact.check_equal("predictions_tune/written", (run.run_dir / "predictions_tune" / "index.parquet").is_file(), True)
    artifact.assert_passed()


def test_detection_is_learned(tmp_path, encoder_name, encoded_images, new_artifact):
    artifact = new_artifact("dense_detection")
    config = _config(
        tmp_path,
        encoder_name,
        "detection",
        roi_px=96,
        task={
            "name": "detection",
            "params": {"num_classes": 2, "match_distance": 3.0, "sigma": 1.5},
        },
        metrics=["mean_f1", "f1_per_class"],
        epochs=30,
    )
    run = run_soma(config, tmp_path / "config.yaml")

    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_equal("rois_encoded", sum(encoded_images), 12)
    artifact.check_at_least("test/mean_f1", run.summary["test/mean_f1"], 0.6)
    predictions = PointPredictions.from_parquet(run.run_dir / "predictions_test.parquet")
    artifact.check_equal("predictions_test/ids", sorted(predictions.sample_ids), ["roi10", "roi11"])
    artifact.check_equal("thresholds_file/absent", (run.run_dir / "detection_thresholds.json").exists(), False)
    artifact.assert_passed()


def test_live_segmentation_with_augmentation_is_learned(
    tmp_path, encoder_name, encoded_images, fake_live_encoder, new_artifact
):
    """``feature_mode: live``: the run builds one encoder stage, re-encodes augmented train
    ROIs every step, and never extracts a cache."""
    artifact = new_artifact("dense_live_segmentation")
    config = _config(
        tmp_path,
        encoder_name,
        "segmentation",
        roi_px=96,
        task={"name": "segmentation", "params": {"num_classes": 3}},
        metrics=["mean_dice", "mean_iou"],
        epochs=30,
        live=True,
        augmentation={"name": "segmentation", "params": {"horizontal_flip": 0.5, "vertical_flip": 0.5}},
    )
    run = run_soma(config, tmp_path / "config.yaml")

    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_equal("encoder_stages_built", len(fake_live_encoder), 1)
    artifact.check_equal("rois_extracted", sum(encoded_images), 0)
    artifact.check_at_least("test/mean_dice", run.summary["test/mean_dice"], 0.8)
    artifact.assert_passed()


def test_live_detection_is_learned(tmp_path, encoder_name, encoded_images, fake_live_encoder, new_artifact):
    """Live detection works by construction: pixels in, level-0 points as targets."""
    artifact = new_artifact("dense_live_detection")
    config = _config(
        tmp_path,
        encoder_name,
        "detection",
        roi_px=96,
        task={
            "name": "detection",
            "params": {"num_classes": 2, "match_distance": 3.0, "sigma": 1.5},
        },
        metrics=["mean_f1", "f1_per_class"],
        epochs=30,
        live=True,
    )
    run = run_soma(config, tmp_path / "config.yaml")

    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_equal("encoder_stages_built", len(fake_live_encoder), 1)
    artifact.check_equal("rois_extracted", sum(encoded_images), 0)
    artifact.check_at_least("test/mean_f1", run.summary["test/mean_f1"], 0.6)
    artifact.assert_passed()


def test_live_run_re_encodes_a_supplied_image_source(tmp_path, encoder_name, encoded_images, fake_live_encoder):
    """A caller-supplied ``ImageSource`` under ``feature_mode: live`` gets the run's frozen
    encoder stage, exactly like the image source the run would build itself."""
    import yaml

    from soma.config import load_config
    from soma.data import TileImageSource
    from soma.pipeline import Pipeline
    from tests.e2e.synthetic import SPACING_UM

    raw = _config(
        tmp_path,
        encoder_name,
        "segmentation",
        roi_px=96,
        task={"name": "segmentation", "params": {"num_classes": 3}},
        metrics=["mean_dice"],
        epochs=1,
        live=True,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    config = load_config(config_path)
    pipeline = Pipeline(config)
    images = TileImageSource.from_manifest(
        pipeline.image_manifest, size=96, spacing_um=SPACING_UM, backend="openslide"
    )

    result = Pipeline(config, source=images).run()

    assert len(fake_live_encoder) == 1
    assert sum(encoded_images) == 0
    assert "test/mean_dice" in result.summary
    assert (result.run_dir / "bundle" / "model.json").is_file()
