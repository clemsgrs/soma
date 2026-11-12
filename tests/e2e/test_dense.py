"""Dense prediction: ROIs -> dense token grids -> decoder -> segmentation / detection.

The cached scenarios extract the grids once (through the weight-free ``fake`` encoder);
the live scenarios re-encode the ROIs every step through the run's frozen encoder stage,
here ``soma.testing.FakeEncoder`` standing in for slide2vec's kit, so augmentation and the
encoder stage ride inside the model.
"""

from __future__ import annotations

import json

import pytest
import yaml

import soma
from soma.cli import main as soma_main
from soma.evaluation import LabelMapPredictions, PointPredictions
from soma.testing import FakeEncoder
from tests.e2e.harness import CPU_EXECUTION, CPU_LOADER, Run, learner_section, run_soma, trainer_section
from tests.e2e.synthetic import SPACING_UM, make_dense_dataset

#: The dense grid cache keys of the cached scenarios (pinned in slice 6c; moved once in
#: slice 8, when the scenarios took soma.testing's ``fake`` encoder).
PINNED_SEGMENTATION_CACHE_KEYS = ["dense_image/37fe3481540fca89"]
PINNED_DETECTION_CACHE_KEYS = ["dense_image/37fe3481540fca89"]


def _config(
    tmp_path, encoder_name, kind, *, roi_px, head, metrics, epochs, live=False, augmentation=None,
    learning_rate=3e-3,
):
    dataset_csv, splits_csv = make_dense_dataset(tmp_path / "data", kind, size=roi_px)
    return {
        "run": {"output_dir": str(tmp_path / "runs"), "seed": 0},
        "cohort": {"records": str(dataset_csv), "splits": str(splits_csv)},
        "features": {
            "kind": "live" if live else "cached",
            "encoder": {"name": encoder_name},
            "extraction": {
                "tiling": {
                    "requested_spacing_um": SPACING_UM,
                    "requested_tile_size_px": roi_px,
                    "tolerance": 0.05,
                    "overlap": 0.0,
                    "backend": "openslide",
                },
            },
            "execution": {**CPU_EXECUTION, "precision": "fp32", "batch_size": 8},
            "cache": {"enabled": True, "root_dir": str(tmp_path / "cache")},
        },
        "loaders": {"batch_size": 4, **CPU_LOADER, "augment": augmentation},
        "learner": learner_section(
            "dense",
            {"decoder": {"name": "lightweight_conv"}, "head": head},
            learning_rate=learning_rate,
            normalization={"method": "none" if live else "zscore"},
        ),
        "trainer": trainer_section(epochs),
        "evaluation": {"metrics": metrics},
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
        head={"name": "segmentation", "params": {"num_classes": 3}},
        metrics=["mean_dice", "mean_iou"],
        epochs=20,
    )
    run = run_soma(config, tmp_path / "config.yaml")

    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_equal("rois_encoded", sum(encoded_images), 12)
    # Cache keys derive from the features config alone; a config-layout change must not move them.
    cache_keys = sorted(p.relative_to(tmp_path / "cache").as_posix() for p in (tmp_path / "cache").glob("*/*"))
    artifact.check_equal("cache_keys", cache_keys, PINNED_SEGMENTATION_CACHE_KEYS)
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
        head={
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
    cache_keys = sorted(p.relative_to(tmp_path / "cache").as_posix() for p in (tmp_path / "cache").glob("*/*"))
    artifact.check_equal("cache_keys", cache_keys, PINNED_DETECTION_CACHE_KEYS)
    artifact.check_at_least("test/mean_f1", run.summary["test/mean_f1"], 0.6)
    predictions = PointPredictions.from_parquet(run.run_dir / "predictions_test.parquet")
    artifact.check_equal("predictions_test/ids", sorted(predictions.sample_ids), ["roi10", "roi11"])
    artifact.check_equal("thresholds_file/absent", (run.run_dir / "detection_thresholds.json").exists(), False)
    artifact.assert_passed()


def test_live_segmentation_with_augmentation_is_learned(
    tmp_path, encoder_name, encoded_images, fake_live_encoder, new_artifact
):
    """``features.kind: live``: the run builds one encoder stage, re-encodes augmented train
    ROIs every step, and never extracts a cache."""
    artifact = new_artifact("dense_live_segmentation")
    config = _config(
        tmp_path,
        encoder_name,
        "segmentation",
        roi_px=96,
        head={"name": "segmentation", "params": {"num_classes": 3}},
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
        head={
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


def test_live_detection_with_augmentation_is_learned(
    tmp_path, encoder_name, encoded_images, fake_live_encoder, new_artifact
):
    """Live detection with ``loaders.augment``: the train points move with the augmented
    pixels (points cropped out are dropped) and the run still learns."""
    artifact = new_artifact("dense_live_detection_augmented")
    config = _config(
        tmp_path,
        encoder_name,
        "detection",
        roi_px=96,
        head={
            "name": "detection",
            "params": {"num_classes": 2, "match_distance": 3.0, "sigma": 1.5},
        },
        metrics=["mean_f1", "f1_per_class"],
        epochs=30,
        live=True,
        augmentation={
            "name": "segmentation",
            "params": {
                "crop_scale": [0.6, 1.0],
                "horizontal_flip": 0.5,
                "vertical_flip": 0.5,
                "rotation_degrees": 15.0,
            },
        },
    )
    run = run_soma(config, tmp_path / "config.yaml")

    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_equal("encoder_stages_built", len(fake_live_encoder), 1)
    artifact.check_equal("rois_extracted", sum(encoded_images), 0)
    artifact.check_at_least("test/mean_f1", run.summary["test/mean_f1"], 0.6)
    artifact.assert_passed()


def test_live_run_re_encodes_a_supplied_image_source(tmp_path, encoder_name, encoded_images, fake_live_encoder):
    """A caller-supplied ``ImageSource`` under ``features.kind: live`` gets the run's frozen
    encoder stage, exactly like the image source the run would build itself."""
    import yaml

    from soma.config import ExperimentConfig
    from soma.data import TileImageSource
    from soma.pipeline import Pipeline
    from tests.e2e.synthetic import SPACING_UM

    raw = _config(
        tmp_path,
        encoder_name,
        "segmentation",
        roi_px=96,
        head={"name": "segmentation", "params": {"num_classes": 3}},
        metrics=["mean_dice"],
        epochs=1,
        live=True,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    config = ExperimentConfig.from_yaml(config_path)
    pipeline = Pipeline(config)
    images = TileImageSource.from_manifest(
        pipeline.image_manifest, size=96, spacing_um=SPACING_UM, backend="openslide"
    )

    result = Pipeline(config, source=images).run()

    assert len(fake_live_encoder) == 1
    assert sum(encoded_images) == 0
    assert "test/mean_dice" in result.summary
    assert (result.run_dir / "bundle" / "model.json").is_file()


def _live_segmentation_over_cached_grids(tmp_path, encoder_name, *, augmentation=None):
    """A ``features.kind: live`` segmentation config and a grid cache over its ROIs."""
    from tests.e2e.synthetic import write_cached_grids

    config = _config(
        tmp_path,
        encoder_name,
        "segmentation",
        roi_px=96,
        head={"name": "segmentation", "params": {"num_classes": 3}},
        metrics=["mean_dice"],
        epochs=1,
        live=True,
        augmentation=augmentation,
    )
    grid_dir = write_cached_grids(config["cohort"]["records"], tmp_path / "grids")
    return config, grid_dir


def test_live_config_trains_on_supplied_cached_grids(tmp_path, encoder_name, encoded_images, fake_live_encoder):
    """``--feature-dir`` on a ``features.kind: live`` config supplies cached grids: the run
    attaches them on the cached path and never builds an encoder stage."""
    config, grid_dir = _live_segmentation_over_cached_grids(tmp_path, encoder_name)

    run = run_soma(config, tmp_path / "config.yaml", "--feature-dir", str(grid_dir))

    assert fake_live_encoder == []
    assert sum(encoded_images) == 0
    assert "test/mean_dice" in run.summary


def test_augmentation_refuses_supplied_cached_grids(tmp_path, encoder_name, fake_live_encoder):
    """Cached grids cannot be augmented, so a supplied grid source with loaders.augment set fails."""
    import yaml

    from soma.config import ExperimentConfig
    from soma.data import CachedGridSource
    from soma.pipeline import Pipeline

    config, grid_dir = _live_segmentation_over_cached_grids(
        tmp_path, encoder_name, augmentation={"name": "segmentation", "params": {"horizontal_flip": 0.5}}
    )
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    with pytest.raises(ValueError, match="loaders.augment needs pixels"):
        Pipeline(ExperimentConfig.from_yaml(tmp_path / "config.yaml"), source=CachedGridSource(grid_dir)).run()
    assert fake_live_encoder == []


# A grid bundle names its encoder, so the bundle alone can predict from pixels.


def _cached_segmentation_run(tmp_path, encoder_name) -> Run:
    """One epoch of cached grid segmentation through the CLI: soma.extract encodes the
    ROIs with the ``fake`` encoder and the decoder trains on the cached grids."""
    config = _config(
        tmp_path,
        encoder_name,
        "segmentation",
        roi_px=96,
        head={"name": "segmentation", "params": {"num_classes": 3}},
        metrics=["mean_dice"],
        epochs=1,
    )
    return run_soma(config, tmp_path / "config.yaml")


@pytest.fixture
def presets_built(monkeypatch) -> list[str]:
    """The slide2vec presets built from here on (``Model.from_preset`` calls)."""
    import slide2vec

    built: list[str] = []
    from_preset = slide2vec.Model.from_preset

    def counting(name, *args, **kwargs):
        built.append(name)
        return from_preset(name, *args, **kwargs)

    monkeypatch.setattr(slide2vec.Model, "from_preset", counting)
    return built


def test_a_cached_grid_bundle_copied_out_of_its_run_yields_its_encoder_stage(tmp_path, encoder_name):
    """A bundle trained on cached grids records the encoder that extracted them. Copied
    to an empty directory, with no plan and no run directory, it gives back that frozen
    encoder stage, whose grids have the geometry the bundle was trained on."""
    import shutil

    import torch

    from soma.pipeline.build import bundle_encoder_stage

    run = _cached_segmentation_run(tmp_path, encoder_name)
    copied = shutil.copytree(run.run_dir / "bundle", tmp_path / "elsewhere" / "bundle")

    bundle = soma.load(copied)
    stage = bundle_encoder_stage(bundle)

    pixels = torch.randint(0, 256, (2, 3, 96, 96), dtype=torch.uint8)
    with torch.inference_mode():
        grid = stage(pixels)
    assert tuple(grid.shape[-2:]) == tuple(bundle.preprocessing["geometry"]["grid_shape"])
    assert stage.geometry.layout == bundle.model.head.geometry.layout


def test_loading_a_cached_grid_bundle_builds_no_encoder(tmp_path, encoder_name, presets_built):
    """``soma.load`` of a bundle trained on cached grids stays cheap: the encoder is built
    only when its stage is asked for, once for every bundle with the same recipe."""
    import gc

    from soma.pipeline.build import bundle_encoder_stage

    run = _cached_segmentation_run(tmp_path, encoder_name)
    gc.collect()  # no stage of an earlier scenario is left to share
    presets_built.clear()  # the run's own extraction

    first, second = soma.load(run.run_dir / "bundle"), soma.load(run.run_dir / "bundle")
    assert presets_built == []
    assert first.model.encoder is None

    stage = bundle_encoder_stage(first)
    assert bundle_encoder_stage(second) is stage
    assert presets_built == [encoder_name]


def _as_written_before_the_recipe(bundle_dir) -> None:
    """Make ``bundle_dir`` look like a cached grid bundle written before soma recorded the
    encoder recipe: its ``preprocessing.json`` has no ``encoder``."""
    path = bundle_dir / "preprocessing.json"
    preprocessing = json.loads(path.read_text())
    del preprocessing["encoder"]
    path.write_text(json.dumps(preprocessing))


def test_a_live_grid_bundle_hands_back_the_stage_it_carries(tmp_path, encoder_name, fake_live_encoder):
    """A bundle trained live builds its frozen encoder stage when it loads; that is the
    stage it gives back, and it records the same recipe a cached bundle does."""
    from soma.pipeline.build import bundle_encoder_stage

    config = _config(
        tmp_path,
        encoder_name,
        "segmentation",
        roi_px=96,
        head={"name": "segmentation", "params": {"num_classes": 3}},
        metrics=["mean_dice"],
        epochs=1,
        live=True,
    )
    run = run_soma(config, tmp_path / "config.yaml")

    bundle = soma.load(run.run_dir / "bundle")
    assert bundle.model.encoder is not None
    assert bundle_encoder_stage(bundle) is bundle.model.encoder
    assert bundle.preprocessing["encoder"]["encoder"] == encoder_name
    assert bundle.preprocessing["encoder"]["target_size"] == fake_live_encoder[-1]["target_size"] == 96


def test_an_old_cached_grid_bundle_takes_its_encoder_from_its_run_directory(tmp_path, encoder_name):
    """A cached grid bundle with no recorded recipe is refused outside its run directory,
    and the message names the fix. Inside its run directory, or given that directory, the
    run's ``plan.yaml`` supplies the recipe."""
    import shutil

    from soma.pipeline.build import bundle_encoder_stage

    run = _cached_segmentation_run(tmp_path, encoder_name)
    _as_written_before_the_recipe(run.run_dir / "bundle")
    copied = shutil.copytree(run.run_dir / "bundle", tmp_path / "elsewhere" / "bundle")

    with pytest.raises(ValueError, match="records no encoder recipe.*run directory.*plan.yaml"):
        bundle_encoder_stage(soma.load(copied))

    inside = bundle_encoder_stage(soma.load(run.run_dir / "bundle"))
    given = bundle_encoder_stage(soma.load(copied), run_dir=run.run_dir)
    assert given is inside
    assert inside.geometry.layout == soma.load(copied).model.head.geometry.layout


@pytest.mark.parametrize("composite", ["configured", "supplied"])
def test_a_composite_grid_bundle_has_no_single_encoder_stage(tmp_path, encoder_name, composite):
    """A run over a composite of several encoders' grids, configured as the ``composite``
    encoder or supplied as a ``CompositeDenseFeatureStore``, records that fact in its
    bundle, and asking that bundle for its encoder stage is refused."""
    from soma.config import ExperimentConfig
    from soma.extract.dense import CachedGridSource
    from soma.extract.dense.composite import CompositeDenseFeatureStore
    from soma.pipeline import Pipeline
    from soma.pipeline.build import bundle_encoder_stage
    from tests.e2e.synthetic import write_cached_grids

    config = _config(
        tmp_path,
        encoder_name,
        "segmentation",
        roi_px=96,
        head={"name": "segmentation", "params": {"num_classes": 3}},
        metrics=["mean_dice"],
        epochs=1,
    )
    config["learner"]["params"]["normalization"] = {"method": "none"}  # refused over a composite
    if composite == "configured":
        members = [{"name": encoder_name}, {"name": encoder_name, "dense": {"overlap": 0.25}}]
        config["features"]["encoder"] = {"name": "composite", "params": {"members": members}}
        run_dir = run_soma(config, tmp_path / "config.yaml").run_dir
    else:
        del config["features"]["encoder"]
        records = config["cohort"]["records"]
        grids = [CachedGridSource(write_cached_grids(records, tmp_path / f"grids_{k}")) for k in range(2)]
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        source = CompositeDenseFeatureStore(grids, concat_resolution="grid")
        run_dir = Pipeline(ExperimentConfig.from_yaml(tmp_path / "config.yaml"), source=source).run().run_dir

    with pytest.raises(ValueError, match="composite encoder"):
        bundle_encoder_stage(soma.load(run_dir / "bundle"))


def test_a_bundle_over_supplied_grids_of_no_named_encoder_has_no_encoder_stage(tmp_path, encoder_name):
    """Grids supplied with ``--feature-dir`` to a config that names no encoder leave the
    bundle nothing to record: asking for its encoder stage is refused."""
    from soma.pipeline.build import bundle_encoder_stage
    from tests.e2e.synthetic import write_cached_grids

    config = _config(
        tmp_path,
        encoder_name,
        "segmentation",
        roi_px=96,
        head={"name": "segmentation", "params": {"num_classes": 3}},
        metrics=["mean_dice"],
        epochs=1,
    )
    del config["features"]["encoder"]
    grid_dir = write_cached_grids(config["cohort"]["records"], tmp_path / "grids")
    run = run_soma(config, tmp_path / "config.yaml", "--feature-dir", str(grid_dir))

    with pytest.raises(ValueError, match="names no encoder"):
        bundle_encoder_stage(soma.load(run.run_dir / "bundle"))


def test_a_recipe_whose_frame_differs_from_the_trained_grids_is_refused(tmp_path, encoder_name):
    """Grids supplied with ``--feature-dir`` but pooled over 16 px patches, while the
    config's ``fake`` encoder pools 8 px ones: the recorded recipe would not rebuild the
    grids the bundle was trained on, so its stage is refused rather than returned."""
    from soma.pipeline.build import bundle_encoder_stage
    from tests.e2e.synthetic import write_cached_grids

    config = _config(
        tmp_path,
        encoder_name,
        "segmentation",
        roi_px=96,
        head={"name": "segmentation", "params": {"num_classes": 3}},
        metrics=["mean_dice"],
        epochs=1,
    )
    grid_dir = write_cached_grids(config["cohort"]["records"], tmp_path / "grids", patch_size=16)
    run = run_soma(config, tmp_path / "config.yaml", "--feature-dir", str(grid_dir))

    with pytest.raises(ValueError, match="frame"):
        bundle_encoder_stage(soma.load(run.run_dir / "bundle"))


def test_sliding_window_prediction_needs_no_encoder_for_a_cached_grid_bundle(tmp_path, encoder_name):
    """``SlidingWindowSegmentationPredictor.from_bundles`` takes the encoder stage a cached
    grid bundle records, so the bundle alone predicts a whole image, exactly as it does
    behind the stage passed as ``encoder=``."""
    import numpy as np

    from soma.evaluation.sliding_window import SlidingWindowSegmentationPredictor
    from soma.pipeline.build import bundle_encoder_stage

    run = _cached_segmentation_run(tmp_path, encoder_name)
    bundle = soma.load(run.run_dir / "bundle")
    image = np.random.default_rng(0).integers(0, 256, (150, 200, 3), dtype=np.uint8)

    alone = SlidingWindowSegmentationPredictor.from_bundles([bundle], device="cpu")
    given = SlidingWindowSegmentationPredictor.from_bundles(
        [bundle], encoder=bundle_encoder_stage(bundle), device="cpu"
    )

    labels = alone.predict_array(image, overlap=0.5).labels
    assert labels.shape == (150, 200)
    assert np.array_equal(labels, given.predict_array(image, overlap=0.5).labels)


@pytest.mark.parametrize(
    ("kind", "flags", "head_params", "metrics"),
    [
        (
            "segmentation",
            ["--num-classes", "3"],
            {"num_classes": 3},
            ["test/mean_dice", "test/mean_iou"],
        ),
        (
            "detection",
            ["--num-classes", "2", "--match-distance", "4.5"],
            {"num_classes": 2, "match_distance": 4.5},
            ["test/mean_f1"],
        ),
    ],
)
def test_a_dense_preset_runs_from_the_command_line(
    kind, flags, head_params, metrics, tmp_path, encoder_name, new_artifact
):
    """``soma run --preset segmentation|detection`` with the preset's own flags only: the
    head's class scheme (and the matching distance) are preset arguments, so the run
    builds its head, trains and is scored. ROIs are the encoder's recommended tile size."""
    artifact = new_artifact(f"dense_preset_{kind}")
    tile_px = {entry["name"]: entry for entry in soma.components("encoders")}[encoder_name]["input_size"]
    dataset_csv, splits_csv = make_dense_dataset(tmp_path / "data", kind, size=tile_px)
    soma_main(
        [
            "run", "--preset", kind, "--encoder", encoder_name, "--decoder", "lightweight_conv", *flags,
            "--cohort", str(dataset_csv), "--splits", str(splits_csv), "--out", str(tmp_path / "runs"),
        ]
    )
    (summary,) = (tmp_path / "runs").glob("experiments/*/runs/*/summary.json")
    run = Run(run_dir=summary.parent, summary=json.loads(summary.read_text()), config_path=summary.parent / "config.yaml")

    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    plan = yaml.safe_load((run.run_dir / "plan.yaml").read_text())
    head = plan["learner"]["params"]["head"]["params"]
    artifact.check_equal("plan/head", {key: head[key] for key in head_params}, head_params)
    artifact.check_equal("metrics/scored", sorted(set(metrics) - set(run.summary)), [])
    artifact.check_equal("predictions_test/written", any(run.run_dir.glob("predictions_test*")), True)
    artifact.assert_passed()
