"""``soma reproduce`` end to end for the EVA benchmarks on synthetic raw roots.

``eva/consep`` (issue #522) runs on a synthetic HoVer-Net layout; ``eva/camelyon16_small``
and ``eva/panda_small`` (issue #535) on tiny pyramidal-TIFF cohorts.

The CLI curates the raw root with the EVA geometry, trains the benchmark-private
components with the weight-free literal encoder, and scores the primary metric from the
run's ``summary.json``. The protocol's training budget and patience are shrunk in-process
so the scenarios stay CPU-sized; everything else is the registered benchmark's own config.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy.io import savemat

from soma.benchmarks import eva_segmentation as seg
from soma.benchmarks import eva_slide as eva_slide_bench
from soma.cli import main as soma_main
from soma.curation.eva_segmentation import CONSEP_TILE_PX, OUTPUT_PX
from tests.e2e.harness import Run
from tests.eva_slide_raw import write_fake_camelyon16, write_fake_panda
from tests.e2e.synthetic import _BACKGROUND, _CLASS_COLOURS, _noisy

#: Synthetic CoNSeP images: 2 x 2 grid tiles of 250 px (EVA's 1000 px images give 4 x 4).
IMAGE_PX = 2 * CONSEP_TILE_PX


def write_fake_consep(root: Path, *, train: int = 3, test: int = 1, seed: int = 0) -> None:
    """HoVer-Net layout with coloured discs whose ``type_map`` uses raw classes 1..7."""
    rng = np.random.default_rng(seed)
    yy, xx = np.ogrid[:IMAGE_PX, :IMAGE_PX]
    for split_dir, count, prefix in (("Train", train, "train"), ("Test", test, "test")):
        (root / split_dir / "Images").mkdir(parents=True)
        (root / split_dir / "Labels").mkdir(parents=True)
        for i in range(1, count + 1):
            image = np.empty((IMAGE_PX, IMAGE_PX, 3), dtype=np.int16)
            image[:] = _BACKGROUND
            type_map = np.zeros((IMAGE_PX, IMAGE_PX), dtype=np.uint8)
            # Two colours map onto two merged classes: raw 1/2 -> "other"/"inflammatory"
            # share a colour (merged class 1 and 2), raw 3..7 -> epithelial/spindle colours.
            for raw, colour_key in ((1, 1), (4, 2), (6, 2)):
                for _ in range(6):
                    x, y = (int(v) for v in rng.integers(30, IMAGE_PX - 30, 2))
                    radius = int(rng.integers(18, 34))
                    disc = (xx - x) ** 2 + (yy - y) ** 2 < radius**2
                    type_map[disc] = raw
                    image[disc] = _CLASS_COLOURS[colour_key]
            Image.fromarray(_noisy(image, rng, 8)).save(root / split_dir / "Images" / f"{prefix}_{i}.png")
            savemat(
                root / split_dir / "Labels" / f"{prefix}_{i}.mat",
                {"type_map": type_map, "inst_map": type_map},
            )


def test_reproduce_eva_consep_curates_trains_and_scores(
    tmp_path, monkeypatch, encoder_name, encoded_images, new_artifact
):
    artifact = new_artifact("reproduce_eva_consep")
    raw_root = tmp_path / "raw"
    write_fake_consep(raw_root)
    # Shrink the fixed step budget and patience; the recipe is otherwise untouched.
    monkeypatch.setattr(seg, "MAX_STEPS", 30)
    monkeypatch.setitem(seg.DATASETS, "consep", replace(seg.DATASETS["consep"], patience=30))

    out_dir = tmp_path / "curated"
    output_root = tmp_path / "runs"
    code = 0
    try:
        soma_main(
            [
                "reproduce",
                "eva/consep",
                "--raw-root",
                str(raw_root),
                "--out-dir",
                str(out_dir),
                "--output-root",
                str(output_root),
                "--encoder",
                encoder_name,
                "--seeds",
                "1",
            ]
        )
    except SystemExit as exc:
        code = int(exc.code or 0)
    artifact.check_equal("exit_code", code, 0)

    # Curation followed the EVA geometry: 4 tiles per 500 px image, 224 px outputs.
    dataset = pd.read_csv(out_dir / "dataset.csv")
    splits = pd.read_csv(out_dir / "splits.csv")
    artifact.check_equal("curated_samples", len(dataset), 4 * 4)
    artifact.check_equal("train_samples", int((splits["split"] == "train").sum()), 12)
    artifact.check_equal("test_samples", int((splits["split"] == "test").sum()), 4)
    tile = np.asarray(Image.open(dataset.image_path.iloc[0]))
    artifact.check_equal("tile_shape", list(tile.shape), [OUTPUT_PX, OUTPUT_PX, 3])
    artifact.check_equal("rois_encoded", sum(encoded_images), 16)

    # One seed ran under the family output layout and scored the primary metric.
    summaries = sorted((output_root / "seed_0").glob("**/summary.json"))
    artifact.check_equal("summaries", len(summaries), 1)
    run_dir = summaries[0].parent
    run = Run(run_dir=run_dir, summary=json.loads(summaries[0].read_text()), config_path=run_dir / "config.yaml")
    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_at_least("test/foreground_mean_dice", run.summary["test/foreground_mean_dice"], 0.0)
    artifact.check_equal("has_mean_dice", "test/mean_dice" in run.summary, True)
    text = run.config_path.read_text()
    artifact.check_equal("decoder_is_eva_conv_with_image", "eva_conv_with_image" in text, True)
    artifact.check_equal("head_is_eva_segmentation", "eva_segmentation" in text, True)
    artifact.check_equal("prenorm_feature_tap", "patch_features_prenorm" in text, True)
    artifact.assert_passed()


def _reproduce(name: str, raw_root: Path, out_dir: Path, output_root: Path, encoder: str) -> int:
    try:
        soma_main(
            [
                "reproduce",
                name,
                "--raw-root",
                str(raw_root),
                "--out-dir",
                str(out_dir),
                "--output-root",
                str(output_root),
                "--encoder",
                encoder,
                "--seeds",
                "1",
            ]
        )
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def _single_run(output_root: Path) -> Run:
    (summary_path,) = sorted((output_root / "seed_0").glob("**/summary.json"))
    run_dir = summary_path.parent
    return Run(
        run_dir=run_dir,
        summary=json.loads(summary_path.read_text()),
        config_path=run_dir / "config.yaml",
    )


@pytest.mark.parametrize(
    ("dataset", "write_raw", "splits", "head", "epochs"),
    [
        (
            "camelyon16_small",
            write_fake_camelyon16,
            {"train": 7, "tune": 4, "test": 4},
            "eva_mil_binary",
            60,
        ),
        # Six grades from one training slide each: more steps to separate them.
        (
            "panda_small",
            write_fake_panda,
            {"train": 6, "tune": 6, "test": 6},
            "eva_mil_multiclass",
            300,
        ),
    ],
)
def test_reproduce_eva_slide_benchmark_curates_trains_and_scores(
    dataset,
    write_raw,
    splits,
    head,
    epochs,
    tmp_path,
    monkeypatch,
    encoder_name,
    encoded_images,
    new_artifact,
):
    """``soma reproduce eva/<slide dataset>`` on a tiny synthetic raw root.

    The curator samples EVA's tiles into one hs2p artifact per slide; soma skips tiling,
    embeds exactly those tiles, and trains ``eva_abmil`` + the EVA head. Only the epoch
    budget and patience are shrunk; the rest is the registered benchmark's own config.
    """
    artifact = new_artifact(f"reproduce_eva_{dataset}")
    raw_root = write_raw(tmp_path / "raw")
    monkeypatch.setattr(eva_slide_bench, "EPOCHS", epochs)
    monkeypatch.setattr(eva_slide_bench, "PATIENCE", epochs)

    out_dir, output_root = tmp_path / "curated", tmp_path / "runs"
    code = _reproduce(f"eva/{dataset}", raw_root, out_dir, output_root, encoder_name)
    artifact.check_equal("exit_code", code, 0)

    manifest = pd.read_csv(out_dir / "dataset.csv")
    split = pd.read_csv(out_dir / "splits.csv")["split"].value_counts().to_dict()
    artifact.check_equal("curated_splits", split, splits)
    artifact.check_equal(
        "every_row_supplies_coordinates", bool(manifest["coordinates_path"].notna().all()), True
    )
    # slide2vec embedded exactly the curated tiles and nothing else.
    artifact.check_equal("tiles_encoded", sum(encoded_images), int(manifest["num_tiles"].sum()))
    artifact.check_equal(
        "no_hs2p_tiling", sorted(p.name for p in tmp_path.glob("runs/**/previews/*/*.jpg")), []
    )

    run = _single_run(output_root)
    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_at_least("test/balanced_accuracy", run.summary["test/balanced_accuracy"], 0.75)
    text = run.config_path.read_text()
    artifact.check_equal("aggregator_is_eva_abmil", "eva_abmil" in text, True)
    artifact.check_equal(f"head_is_{head}", head in text, True)
    artifact.assert_passed()
