"""Pipeline parity: ``soma run <config.yaml>`` is the public verbs in the canonical order.

Each scenario runs one fold twice over the same cached features: once through the CLI
(``--feature-dir``), once composed by hand from plain constructors carrying the YAML's
values, without importing ``soma.pipeline``. The hand composition follows the canonical
fold order (ADR 0017):

1. ``soma.seed_everything(seed, fold=0)``
2. fit the feature adaptor on the train split
3. build the model
4. build the loaders
5. ``Baseline.fit`` on the train targets (set shape)
6. ``soma.fit`` with ``Checkpoint`` and ``EarlyStopping``
7. the tune-time step (detection: ``sweep_threshold`` on tune)
8. per split: ``soma.predict`` (+ baseline rows) -> ``soma.evaluate`` -> write predictions
9. ``soma.save``

Every predictions file must be byte-identical, the metrics equal and
``bundle/weights.safetensors`` byte-identical. A failure here is fixed in the pipeline,
never by loosening the comparison: byte identity depends on the RNG consumption order.
"""

from __future__ import annotations

import json
from pathlib import Path

import soma
from soma.models.aggregators import ABMIL
from soma.data import (
    AnnotationManifest,
    CachedSetSource,
    Cohort,
    FoldSplit,
    ImageManifest,
    LabelMapSource,
    PointSource,
)
from soma.models.decoders.registry import build_decoder_for_grid
from soma.extract.dense import CachedGridSource
from soma.evaluation import Baseline, Predictions, sweep_threshold
from soma.learn import Checkpoint, DenseLearner, EarlyStopping, MILLearner, loaders
from soma.models.heads.classification import BinaryClassificationHead
from soma.models.heads.detection import DetectionHead
from soma.models.heads.segmentation import SegmentationHead
from soma.learn.feature_adaptor import FeatureAdaptor
from soma.models.model import DenseModel, MILModel
from tests.e2e.harness import CPU_EXECUTION, CPU_LOADER, run_soma, sha256
from tests.e2e.synthetic import (
    SPACING_UM,
    make_dense_dataset,
    write_cached_bags,
    write_cached_grids,
)

SEED = 0
ROI_PX = 64


def _prediction_files(root: Path) -> dict[str, str]:
    """sha256 of every predictions file: parquet frames and label-map directories."""
    files = {}
    for path in sorted(root.glob("predictions_*")):
        members = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for member in members:
            files[member.relative_to(root).as_posix()] = sha256(member)
    return files


def _check_parity(artifact, run, hand_dir: Path, hand_metrics: dict) -> None:
    artifact.record_run("pipeline", run)
    artifact.check_training_reduced_loss("pipeline", run)
    pipeline_files = _prediction_files(run.run_dir)
    artifact.check_equal("predictions/files", sorted(pipeline_files), sorted(_prediction_files(hand_dir)))
    artifact.check_equal("predictions/sha256", pipeline_files, _prediction_files(hand_dir))
    artifact.check_equal("metrics", json.loads((run.run_dir / "metrics.json").read_text()), hand_metrics)
    pipeline_weights = (run.run_dir / "bundle" / "weights.safetensors").read_bytes()
    hand_weights = (hand_dir / "bundle" / "weights.safetensors").read_bytes()
    artifact.check_equal("weights/identical", pipeline_weights == hand_weights, True)
    artifact.check_equal("best_model_pt/absent", (run.run_dir / "best_model.pt").exists(), False)


def _fit(learner, fold, *, epochs: int):
    """Step 6 with the callbacks the YAML's trainer section describes."""
    return soma.fit(
        learner,
        fold.train,
        fold.tune,
        callbacks=[
            Checkpoint(monitor="loss", mode="min", selection="best"),
            EarlyStopping(monitor="loss", mode="min", patience=epochs, hold_until_moves=True),
        ],
        progress=False,
        max_epochs=epochs,
        device="cpu",
    )


def _optimizer(learning_rate: float) -> dict:
    return {"name": "adam", "params": {"lr": learning_rate, "weight_decay": 1e-5}}


def _learner_and_trainer(name: str, model: dict, *, epochs: int, learning_rate: float) -> dict:
    """The YAML's ``learner`` and ``trainer`` sections: ``model`` leaves, a z-score adaptor,
    Adam with cosine annealing, checkpoint selection and early stopping at the budget."""
    return {
        "learner": {
            "name": name,
            "params": {
                **model,
                "normalization": {"method": "zscore"},
                "optimizer": _optimizer(learning_rate),
                "scheduler": "cosine",
            },
        },
        "trainer": {
            "max_epochs": epochs,
            "callbacks": [{"name": "checkpoint"}, {"name": "early_stopping", "params": {"patience": epochs}}],
        },
    }


def test_set_mil_classification_pipeline_equals_hand_composition(tmp_path, new_artifact):
    """Set shape: ABMIL over cached bags; one test sample is tile-less, so the pipeline
    and the hand composition must both score it with the training ``Baseline``."""
    artifact = new_artifact("parity_set_mil_classification")
    dataset_csv, splits_csv, feature_dir = write_cached_bags(tmp_path / "data")
    epochs, learning_rate, batch_size = 6, 5e-3, 2
    config = {
        "run": {"output_dir": str(tmp_path / "runs"), "seed": SEED},
        "cohort": {"records": str(dataset_csv), "splits": str(splits_csv)},
        "features": {"execution": CPU_EXECUTION},
        "loaders": {"batch_size": batch_size, **CPU_LOADER},
        **_learner_and_trainer(
            "mil",
            {"aggregator": {"name": "abmil"}, "head": {"name": "binary_classification"}},
            epochs=epochs,
            learning_rate=learning_rate,
        ),
        "evaluation": {"metrics": ["auroc", "balanced_accuracy"]},
    }
    run = run_soma(config, tmp_path / "config.yaml", "--feature-dir", str(feature_dir))

    hand = tmp_path / "hand"
    hand.mkdir()
    cohort = Cohort.from_csv(dataset_csv, splits_csv, targets={"label": "label"}, dtypes={"label": object})
    source = CachedSetSource(feature_dir)
    split = cohort.folds[0]

    def covered(ids):
        return tuple(sid for sid in ids if sid in source.sample_ids)

    soma.seed_everything(SEED, fold=0)  # 1
    adaptor = FeatureAdaptor(method="zscore", num_features=source.feature_dim)  # 2
    adaptor.fit([source.load(sid) for sid in covered(split.train)])
    aggregator = ABMIL(input_dim=source.feature_dim)  # 3
    head = BinaryClassificationHead(
        input_dim=aggregator.output_dim,
        num_classes=2,
        label_map={"normal": 0, "tumor": 1},
        metrics=["auroc", "balanced_accuracy"],
    )
    learner = MILLearner(
        MILModel(aggregator, head, feature_adaptor=adaptor),
        optimizer=_optimizer(learning_rate),
        schedulers=("cosine",),
    )
    with_features = Cohort(
        cohort.records,
        [FoldSplit(train=covered(split.train), tune=covered(split.tune), tests={"test": covered(split.tests["test"])})],
    )
    fold = loaders(source, with_features, head=head, batch_size=batch_size, num_workers=0, pin_memory=False)  # 4
    baseline = Baseline.fit(  # 5
        with_features.targets("train"),
        task_family=head.task_family,
        classes=head.classes,
        target_keys=head.target_keys,
    )
    result = _fit(learner, fold, epochs=epochs)  # 6
    metrics = {}
    for name, loader in (("tune", fold.tune), ("test", fold.tests["test"])):  # 8
        ids = split.tune if name == "tune" else split.tests["test"]
        tileless = [sid for sid in ids if sid not in source.sample_ids]
        predictions = Predictions.concat([soma.predict(result.learner, loader), baseline.predict(tileless)])
        metrics[name] = soma.evaluate(predictions, cohort.targets(name), metrics=["auroc", "balanced_accuracy"]).metrics
        if name == "test":
            predictions.to_parquet(hand / "predictions_test.parquet")
    soma.save(result, hand / "bundle")  # 9

    artifact.check_equal("tileless/placeholder_rows", metrics["test"]["num_placeholder_samples"], 1)
    _check_parity(artifact, run, hand, metrics)
    artifact.assert_passed()


def _grid_config(tmp_path, dataset_csv, splits_csv, *, head, metrics, epochs, learning_rate):
    return {
        "run": {"output_dir": str(tmp_path / "runs"), "seed": SEED},
        "cohort": {"records": str(dataset_csv), "splits": str(splits_csv)},
        "features": {
            "kind": "cached",
            "extraction": {
                "tiling": {
                    "requested_spacing_um": SPACING_UM,
                    "requested_tile_size_px": ROI_PX,
                    "tolerance": 0.05,
                    "overlap": 0.0,
                    "backend": "openslide",
                },
            },
            "execution": CPU_EXECUTION,
        },
        "loaders": {"batch_size": 4, **CPU_LOADER},
        **_learner_and_trainer(
            "dense",
            {"decoder": {"name": "lightweight_conv"}, "head": head},
            epochs=epochs,
            learning_rate=learning_rate,
        ),
        "evaluation": {"metrics": metrics},
    }


def _grid_model(source, head, split, *, learning_rate):
    """Steps 2-3 of the grid shape: the adaptor is fit on every train grid position."""
    adaptor = FeatureAdaptor(method="zscore", num_features=source.feature_dim)
    adaptor.fit([source.load(sid).movedim(0, -1) for sid in split.train])
    decoder = build_decoder_for_grid(
        "lightweight_conv", {}, geometry=head.geometry, input_dim=source.feature_dim, num_classes=head.num_classes
    )
    return DenseLearner(
        DenseModel(decoder, head, adaptor=adaptor), optimizer=_optimizer(learning_rate), schedulers=("cosine",)
    )


def test_grid_segmentation_pipeline_equals_hand_composition(tmp_path, new_artifact):
    artifact = new_artifact("parity_grid_segmentation")
    dataset_csv, splits_csv = make_dense_dataset(tmp_path / "data", "segmentation", size=ROI_PX)
    grid_dir = write_cached_grids(dataset_csv, tmp_path / "grids")
    epochs, learning_rate = 5, 3e-3
    metric_names = ["mean_dice", "mean_iou"]
    config = _grid_config(
        tmp_path,
        dataset_csv,
        splits_csv,
        head={"name": "segmentation", "params": {"num_classes": 3}},
        metrics=metric_names,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    run = run_soma(config, tmp_path / "config.yaml", "--feature-dir", str(grid_dir))

    hand = tmp_path / "hand"
    hand.mkdir()
    cohort = Cohort.from_csv(dataset_csv, splits_csv)
    source = CachedGridSource(grid_dir)
    split = cohort.folds[0]
    soma.seed_everything(SEED, fold=0)  # 1
    head = SegmentationHead(
        num_classes=3,
        geometry=source.geometry(split.train[0]),
        metrics=metric_names,
        spacing_um=SPACING_UM,
        image_backend="openslide",
        class_names=("class_0", "class_1", "class_2"),
    )
    learner = _grid_model(source, head, split, learning_rate=learning_rate)  # 2-3
    label_maps = LabelMapSource.from_manifests(
        AnnotationManifest.from_csv(dataset_csv), ImageManifest.from_csv(dataset_csv), **head.label_map_read_params
    )
    fold = loaders(source, cohort, head=head, targets=label_maps, batch_size=4, pin_memory=False)  # 4
    result = _fit(learner, fold, epochs=epochs)  # 6
    metrics = {}
    for name, loader, ids in (("tune", fold.tune, split.tune), ("test", fold.tests["test"], split.tests["test"])):
        predictions = soma.predict(result.learner, loader)  # 8
        metrics[name] = soma.evaluate(predictions, label_maps.subset(ids), metrics=metric_names).metrics
        predictions.save(hand / f"predictions_{name}")
    soma.save(result, hand / "bundle")  # 9

    _check_parity(artifact, run, hand, metrics)
    artifact.assert_passed()


def test_grid_detection_pipeline_equals_hand_composition(tmp_path, new_artifact):
    """Grid detection: the per-class threshold is swept on tune (step 7) and applied
    unchanged to every split before scoring; the written predictions keep every peak."""
    artifact = new_artifact("parity_grid_detection")
    dataset_csv, splits_csv = make_dense_dataset(tmp_path / "data", "detection", size=ROI_PX)
    grid_dir = write_cached_grids(dataset_csv, tmp_path / "grids")
    epochs, learning_rate = 6, 3e-3
    metric_names = ["mean_f1", "f1_per_class"]
    config = _grid_config(
        tmp_path,
        dataset_csv,
        splits_csv,
        head={"name": "detection", "params": {"num_classes": 2, "match_distance": 3.0, "sigma": 1.5}},
        metrics=metric_names,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    run = run_soma(config, tmp_path / "config.yaml", "--feature-dir", str(grid_dir))

    hand = tmp_path / "hand"
    hand.mkdir()
    cohort = Cohort.from_csv(dataset_csv, splits_csv)
    source = CachedGridSource(grid_dir)
    split = cohort.folds[0]
    soma.seed_everything(SEED, fold=0)  # 1
    head = DetectionHead(
        num_classes=2,
        geometry=source.geometry(split.train[0]),
        spacing_um=SPACING_UM,
        match_distance=3.0,
        sigma=1.5,
        metrics=metric_names,
        class_names=("class_0", "class_1"),
    )
    learner = _grid_model(source, head, split, learning_rate=learning_rate)  # 2-3
    points = PointSource.from_manifest(AnnotationManifest.from_csv(dataset_csv))
    fold = loaders(source, cohort, head=head, targets=points, batch_size=4, pin_memory=False)  # 4
    result = _fit(learner, fold, epochs=epochs)  # 6
    tune_predictions = soma.predict(result.learner, fold.tune)  # 7
    threshold, _ = sweep_threshold(tune_predictions, points.subset(split.tune), per_class=True)
    leaves = [{"name": name, "params": {"matching": "hungarian"}} for name in metric_names]
    metrics = {}
    for name, ids in (("tune", split.tune), ("test", split.tests["test"])):  # 8
        predictions = tune_predictions if name == "tune" else soma.predict(result.learner, fold.tests["test"])
        scored = predictions.with_threshold(threshold)
        metrics[name] = soma.evaluate(scored, points.subset(ids), metrics=leaves).metrics
        predictions.to_parquet(hand / f"predictions_{name}.parquet")
    soma.save(result, hand / "bundle")  # 9

    # The bundle records the swept cut keyed by class name, so it applies as recorded.
    recorded = soma.load(run.run_dir / "bundle").preprocessing["score_threshold"]
    artifact.check_equal(
        "bundle/score_threshold",
        recorded,
        {head.class_names[k]: float(v) for k, v in threshold.items()},
    )
    artifact.check_equal(
        "bundle/score_threshold_applies",
        tune_predictions.with_threshold(recorded).threshold,
        tune_predictions.with_threshold(threshold).threshold,
    )
    _check_parity(artifact, run, hand, metrics)
    artifact.assert_passed()
