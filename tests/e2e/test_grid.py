"""The grid shape through the public verbs: sources, ``loaders``, ``fit``, ``predict``,
``sweep_threshold`` and ``evaluate`` composed by hand, no config and no run directory.

The live scenarios re-encode the synthetic ROIs every step through
``soma.testing.FakeEncoder`` (a parameter-free pooling stage that stands in for a
``FrozenEncoder``), so pixels, augmentation and the encoder stage all sit inside the
model. Labels are a function of colour, so only a model that receives the right pixels
and the right supervision can score well.
"""

from __future__ import annotations

import numpy as np
import torch

import soma
from soma.data import (
    AnnotationManifest,
    Cohort,
    ImageManifest,
    LabelMapSource,
    PointSource,
    TileImageSource,
)
from soma.decoders.registry import build_decoder_for_grid
from soma.evaluation import LabelMapPredictions, PointPredictions, sweep_threshold
from soma.learn import SegmentationAugment, loaders
from soma.tasks.detection import DetectionHead
from soma.tasks.segmentation import SegmentationHead
from soma.testing import FakeEncoder, check_encoder_stage, check_image_source
from soma.training.model import DenseModel
from tests.e2e.synthetic import SPACING_UM, make_dense_dataset

ROI_PX = 96


def _sources(tmp_path, kind):
    dataset_csv, splits_csv = make_dense_dataset(tmp_path / "data", kind, size=ROI_PX)
    cohort = Cohort.from_csv(dataset_csv, splits_csv, targets=[])
    images = TileImageSource.from_manifest(
        ImageManifest.from_csv(dataset_csv), size=ROI_PX, spacing_um=SPACING_UM, backend="openslide"
    )
    return cohort, images, AnnotationManifest.from_csv(dataset_csv)


def _decoder(stage, num_classes):
    return build_decoder_for_grid(
        "lightweight_conv",
        {},
        geometry=stage.geometry,
        input_dim=stage.feature_dim,
        num_classes=num_classes,
    )


def _loss_ratio(history) -> float:
    losses = history["train_loss"].to_numpy(dtype=float)
    return float(losses[-1] / losses[0])


def test_live_segmentation_through_the_verbs(tmp_path, new_artifact):
    artifact = new_artifact("grid_live_segmentation")
    cohort, images, annotations = _sources(tmp_path, "segmentation")
    check_image_source(images)
    stage = FakeEncoder(target_size=ROI_PX, patch_size=8)
    check_encoder_stage(stage)
    labels = LabelMapSource.from_manifests(annotations, size=(ROI_PX, ROI_PX))

    torch.manual_seed(0)
    head = SegmentationHead(num_classes=3, geometry=stage.geometry, metrics=["mean_dice", "mean_iou"])
    model = DenseModel(_decoder(stage, 3), head, encoder=stage)
    fold = loaders(
        images,
        cohort,
        head=head,
        targets=labels,
        batch_size=4,
        preprocess=stage.preprocessor(),
        augment=SegmentationAugment(horizontal_flip=0.5, vertical_flip=0.5),
    )
    result = soma.fit(
        model,
        fold.train,
        fold.tune,
        optimizer={"name": "adam", "params": {"lr": 3e-3}},
        max_epochs=20,
        patience=None,
        progress=False,
        device="cpu",
    )
    predictions = soma.predict(result.learner, fold.tests["test"])
    test_ids = list(cohort.folds[0].tests["test"])
    evaluation = soma.evaluate(predictions, labels.subset(test_ids), metrics=["mean_dice", "mean_iou"])

    predictions.save(tmp_path / "label_maps")
    reread = LabelMapPredictions.load(tmp_path / "label_maps")
    again = soma.evaluate(reread, labels.subset(test_ids), metrics=["mean_dice", "mean_iou"])

    artifact.check_equal("predictions/type", type(predictions).__name__, "LabelMapPredictions")
    artifact.check_equal("predictions/ids", sorted(predictions.sample_ids), sorted(test_ids))
    artifact.check_equal("encoder/parameters", sum(p.numel() for p in stage.parameters()), 0)
    artifact.check_at_most("train_loss_ratio", _loss_ratio(result.history), 0.9)
    artifact.check_at_least("test/mean_dice", evaluation.metrics["mean_dice"], 0.8)
    artifact.check_at_least("test/mean_iou", evaluation.metrics["mean_iou"], 0.7)
    artifact.check_equal("round_trip/metrics", again.metrics, evaluation.metrics)
    artifact.assert_passed()


def test_live_detection_through_the_verbs(tmp_path, new_artifact):
    """Live detection works by construction: pixels in, level-0 points as targets, the
    head renders its heatmap at loss time; the threshold is swept on tune as a frame
    operation and applied unchanged to test."""
    artifact = new_artifact("grid_live_detection")
    cohort, images, annotations = _sources(tmp_path, "detection")
    stage = FakeEncoder(target_size=ROI_PX, patch_size=8)
    points = PointSource.from_manifest(annotations)

    torch.manual_seed(0)
    head = DetectionHead(
        num_classes=2,
        geometry=stage.geometry,
        spacing_um=SPACING_UM,
        match_distance=3.0,
        sigma=1.5,
        metrics=["mean_f1", "f1_per_class"],
    )
    model = DenseModel(_decoder(stage, 2), head, encoder=stage)
    fold = loaders(images, cohort, head=head, targets=points, batch_size=4, preprocess=stage.preprocessor())
    result = soma.fit(
        model,
        fold.train,
        fold.tune,
        optimizer={"name": "adam", "params": {"lr": 3e-3}},
        max_epochs=30,
        patience=None,
        progress=False,
        device="cpu",
    )
    split = cohort.folds[0]
    tune = soma.predict(result.learner, fold.tune)
    threshold, tune_evaluation = sweep_threshold(tune, points.subset(split.tune), per_class=True)
    test = soma.predict(result.learner, fold.tests["test"])
    evaluation = soma.evaluate(test.with_threshold(threshold), points.subset(split.tests["test"]))

    test.to_parquet(tmp_path / "points.parquet")
    reread = PointPredictions.from_parquet(tmp_path / "points.parquet")
    again = soma.evaluate(reread.with_threshold(threshold), points.subset(split.tests["test"]))

    artifact.check_equal("predictions/type", type(test).__name__, "PointPredictions")
    artifact.check_equal("predictions/ids", sorted(test.sample_ids), sorted(split.tests["test"]))
    artifact.check_equal("threshold/classes", sorted(threshold), [0, 1])
    artifact.check_equal(
        "predictions/match_distance_px", test.match_distance_px, 3.0 / SPACING_UM
    )
    artifact.check_at_most("train_loss_ratio", _loss_ratio(result.history), 0.9)
    artifact.check_at_least("tune/mean_f1", tune_evaluation.metrics["mean_f1"], 0.6)
    artifact.check_at_least("test/mean_f1", evaluation.metrics["mean_f1"], 0.6)
    artifact.check_equal("round_trip/metrics", again.metrics, evaluation.metrics)
    artifact.check_equal(
        "scores_above_floor", bool((test.frame["score"].to_numpy() >= head.min_score).all()), True
    )
    artifact.assert_passed()


def test_point_augmentation_is_refused_until_keypoints_land(tmp_path):
    cohort, images, annotations = _sources(tmp_path, "detection")
    stage = FakeEncoder(target_size=ROI_PX, patch_size=8)
    head = DetectionHead(num_classes=2, geometry=stage.geometry, spacing_um=SPACING_UM, match_distance=3.0)
    try:
        loaders(
            images,
            cohort,
            head=head,
            targets=PointSource.from_manifest(annotations),
            augment=SegmentationAugment(horizontal_flip=0.5),
        )
    except ValueError as error:
        assert "point targets cannot be augmented" in str(error)
    else:  # pragma: no cover - the refusal is the behaviour under test
        raise AssertionError("loaders accepted an augmentation over point targets")
    _ = np  # keep numpy imported for the artifact helpers
