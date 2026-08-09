"""Headline reporting in the greedy (OCELOT-official) eval path (#146, oracle dropped #152).

``examples/ocelot/eval_greedy.build_greedy_report`` is the testable seam: it emits, per
test split, the leakage-free **headline** (per-class thresholds frozen on the tune split,
applied once to test). The oracle ceiling that #146 added for the health gate is gone
(#152), so no test-side threshold sweep happens here. This builds a tiny on-disk dense
detection setup + an (untrained) DetectionHead model — no encoder/GPU needed — and
asserts the report shape, including the tune-only form used by holdout_test sweeps.
"""

from __future__ import annotations

import functools
import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

from soma.dense.geometry import compute_dense_geometry
from soma.dense.store import dense_grid_metadata, write_dense_grid
from soma.dataset import DetectionManifest, Splits
from soma.dense import DenseFeatureStore
from soma.decoders.registry import decoder_registry
from soma.pipeline import _make_loaders, _resolve_detection_px
from soma.tasks.detection import DetectionHead
from soma.training.detection_dataset import DetectionDataset, detection_collate_fn
from soma.training.model import SegmentationModel

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_GREEDY = REPO_ROOT / "examples" / "ocelot" / "eval_greedy.py"

NUM_CLASSES = 2
TARGET = 16
PATCH = 4
FEATURE_DIM = 4
SPACING = 0.2  # µm/px; 0.6 µm -> 3 px (δ), 0.3 µm -> 1.5 px (σ)


def _load_eval_greedy():
    spec = importlib.util.spec_from_file_location("ocelot_eval_greedy", EVAL_GREEDY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_detection_run(root: Path, sample_ids: list[str]):
    dense_dir = root / "dense"
    points_dir = root / "points"
    dense_dir.mkdir()
    points_dir.mkdir()

    geom = compute_dense_geometry(target_size=TARGET, patch_size=PATCH)
    meta = dense_grid_metadata(geom, feature_dim=FEATURE_DIM, pad_mode="reflect", spacing_um=SPACING)
    meta.update(source_spacing_um=SPACING, effective_spacing_um=SPACING)
    for sid in sample_ids:
        write_dense_grid(dense_dir, sid, torch.randn(FEATURE_DIM, *geom.grid_shape), meta)
        (points_dir / f"{sid}.csv").write_text("x,y,class\n4,4,0\n11,11,1\n")

    (root / "manifest.csv").write_text(
        "sample_id,image_path,points_path,spacing_at_level_0\n"
        + "\n".join(
            f"{sid},{sid}.jpg,{points_dir / f'{sid}.csv'},{SPACING}"
            for sid in sample_ids
        )
        + "\n"
    )
    # Two test splits so per-split headline reporting is exercised.
    assign = {
        sample_ids[0]: "train", sample_ids[1]: "train", sample_ids[2]: "tune",
        sample_ids[3]: "test", sample_ids[4]: "test_2",
    }
    (root / "splits.csv").write_text(
        "sample_id,split,fold\n" + "\n".join(f"{sid},{s},0" for sid, s in assign.items()) + "\n"
    )

    manifest = DetectionManifest(root / "manifest.csv")
    splits = Splits(root / "splits.csv", manifest)
    store = DenseFeatureStore(dense_dir)
    return manifest, splits, store


def _build_model_and_loaders(manifest, splits, store):
    fold_split = splits.folds[0]
    train_records = [manifest.samples[s] for s in fold_split.train]
    tune_records = [manifest.samples[s] for s in fold_split.tune]
    test_by_split = {n: [manifest.samples[s] for s in ids] for n, ids in fold_split.tests.items()}

    grid_spacing = store.spacing(train_records[0].sample_id).effective_spacing_um
    delta_px = _resolve_detection_px(0.6, grid_spacing, "match_distance")
    sigma_px = _resolve_detection_px(0.3, grid_spacing, "sigma")
    geometry = store.geometry(train_records[0].sample_id)

    head = DetectionHead(
        num_classes=NUM_CLASSES, geometry=geometry, delta_px=delta_px, sigma_px=sigma_px,
        nms_distance_px=delta_px,
        matching="greedy",
        sample_spacings={sid: store.spacing(sid) for sid in store.available_samples},
        metrics=["mean_f1", "f1_per_class"],
    )

    decoder_cls = decoder_registry.get("lightweight_conv")
    rh = geometry.encoded_size[0] / geometry.grid_shape[0]
    rw = geometry.encoded_size[1] / geometry.grid_shape[1]
    num_upsample_blocks = max(0, math.ceil(math.log2(max(rh, rw))))
    decoder_obj = decoder_cls(
        input_dim=store.feature_dim, num_classes=NUM_CLASSES, num_upsample_blocks=num_upsample_blocks
    )
    model = SegmentationModel(decoder=decoder_obj, task_head=head)
    model.eval()

    collate = functools.partial(detection_collate_fn, target_dtypes=head.target_dtypes)
    from soma.config import TrainingConfig

    _, tune_loader, test_loaders = _make_loaders(
        DetectionDataset, collate, train_records, tune_records, test_by_split,
        TrainingConfig(epochs=1, batch_size=2), store, head.extract_targets,
    )
    return model, head, tune_loader, test_loaders


def test_build_greedy_report_emits_headline_only(tmp_path: Path):
    np.random.seed(0)
    torch.manual_seed(0)
    sample_ids = ["s0", "s1", "s2", "s3", "s4"]
    manifest, splits, store = _build_detection_run(tmp_path, sample_ids)
    model, head, tune_loader, test_loaders = _build_model_and_loaders(manifest, splits, store)

    module = _load_eval_greedy()
    report = module.build_greedy_report(
        model=model, head=head, device=torch.device("cpu"),
        tune_loader=tune_loader, test_loaders=test_loaders,
        dataset=manifest, matching="greedy",
    )

    # Top-level headline thresholds (tune-frozen) and the tune metrics survive.
    headline_thresholds = report["score_threshold_per_class"]
    assert len(headline_thresholds) == NUM_CLASSES
    assert "mean_f1" in report["tune"]

    # Each test split carries a headline block and no oracle (dropped in #152).
    assert set(test_loaders) == {"test", "test_2"}
    for name in test_loaders:
        block = report[name]
        assert set(block) == {"headline"}

        # A tune-frozen headline mF1 per split, with per-class thresholds.
        assert "mean_f1" in block["headline"]["metrics"]
        assert len(block["headline"]["score_threshold_per_class"]) == NUM_CLASSES

        # The headline uses exactly the tune-frozen thresholds (unchanged behaviour).
        assert block["headline"]["score_threshold_per_class"] == headline_thresholds
        assert "ceiling" not in block["headline"]["note"].lower()


def test_build_greedy_report_tune_only(tmp_path: Path):
    """With no test loaders (the holdout_test model-selection form), the report carries
    only matching + tune-frozen thresholds + tune metrics — no per-split blocks."""
    np.random.seed(0)
    torch.manual_seed(0)
    sample_ids = ["s0", "s1", "s2", "s3", "s4"]
    manifest, splits, store = _build_detection_run(tmp_path, sample_ids)
    model, head, tune_loader, _ = _build_model_and_loaders(manifest, splits, store)

    module = _load_eval_greedy()
    report = module.build_greedy_report(
        model=model, head=head, device=torch.device("cpu"),
        tune_loader=tune_loader, test_loaders={},
        dataset=manifest, matching="greedy",
    )
    assert set(report) == {"matching", "score_threshold_per_class", "tune"}
    assert "mean_f1" in report["tune"]


def test_build_greedy_report_headline_matches_frozen_eval(tmp_path: Path):
    """AC4: the headline number is the tune-frozen sweep applied once to test — i.e. the
    same value the old single-number path produced. Re-deriving it independently (sweep on
    tune, eval on test with that threshold) must reproduce the reported headline exactly.
    """
    np.random.seed(1)
    torch.manual_seed(1)
    sample_ids = ["s0", "s1", "s2", "s3", "s4"]
    manifest, splits, store = _build_detection_run(tmp_path, sample_ids)
    model, head, tune_loader, test_loaders = _build_model_and_loaders(manifest, splits, store)

    from soma.pipeline import _evaluate_detection, _sweep_detection_thresholds

    frozen = _sweep_detection_thresholds(model, tune_loader, torch.device("cpu"), head)
    head.score_threshold = frozen
    expected = {
        name: _evaluate_detection(
            model, loader, name, torch.device("cpu"), head=head, dataset=manifest
        ).metrics
        for name, loader in test_loaders.items()
    }

    module = _load_eval_greedy()
    report = module.build_greedy_report(
        model=model, head=head, device=torch.device("cpu"),
        tune_loader=tune_loader, test_loaders=test_loaders,
        dataset=manifest, matching="greedy",
    )
    for name in test_loaders:
        assert report[name]["headline"]["metrics"] == expected[name]
        assert report["score_threshold_per_class"] == frozen
