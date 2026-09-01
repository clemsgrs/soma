"""Show that manifest identity is portable across storage roots.

Since soma 1.12.0, path columns (``image_path``, ``mask_path``,
``label_mask_path``, ``points_path``) are excluded from the semantic manifest
digest. Moving a dataset to a different storage root therefore keeps
``experiment_id`` — and the leaderboard triple
``(dataset_checksum, splits_checksum, task)`` — unchanged, so runs stay
comparable and checkpoints stay reusable after relocation.

Content still counts: adding an explicit ``image_path_sha256`` column declares
the *bytes* behind the paths as part of the experiment, and does change
identity.

No pipeline runs and no encoder weights are involved — identity is computed
from the manifests and the config alone.

Run with no arguments:

    python examples/portable_identity.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from soma import (
    EncoderConfig,
    AggregatorConfig,
    PipelineConfig,
    PreprocessingConfig,
    TaskConfig,
    TrainingConfig,
)
from soma.output_layout import build_experiment_spec

WORK = Path(tempfile.mkdtemp(prefix="soma-portable-identity-"))

SAMPLES = ["s00", "s01", "s02", "s03"]
LABELS = [0, 1, 0, 1]
SPLITS = ["train", "train", "tune", "tune"]


def write_manifests(
    root_name: str, image_root: str, *, path_checksums: list[str] | None = None
) -> tuple[Path, Path]:
    """Write the same dataset content with image paths under ``image_root``."""
    root = WORK / root_name
    root.mkdir()
    frame = {
        "sample_id": SAMPLES,
        "image_path": [f"{image_root}/{s}.tif" for s in SAMPLES],
        "label": LABELS,
    }
    if path_checksums is not None:
        frame["image_path_sha256"] = path_checksums
    dataset_csv = root / "dataset.csv"
    splits_csv = root / "splits.csv"
    pd.DataFrame(frame).to_csv(dataset_csv, index=False)
    pd.DataFrame({"sample_id": SAMPLES, "split": SPLITS}).to_csv(splits_csv, index=False)
    return dataset_csv, splits_csv


def identity(dataset_csv: Path, splits_csv: Path):
    """Build the experiment spec for one fixed model config over these manifests."""
    config = PipelineConfig(
        dataset_csv=str(dataset_csv),
        splits_csv=str(splits_csv),
        output_root=dataset_csv.parent / "outputs",
        dataset_type="slide",
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=224, requested_spacing_um=0.5
        ),
        encoder=EncoderConfig(name="phikon"),
        aggregator=AggregatorConfig(name="abmil"),
        task=TaskConfig(name="binary_classification"),
        training=TrainingConfig(epochs=5, seed=0),
    )
    return build_experiment_spec(config)


def report(title: str, spec) -> None:
    print(f"{title}")
    print(f"  experiment_id:     {spec.experiment_id}")
    print("  leaderboard triple:")
    print(f"    dataset_checksum: {spec.dataset_checksum}")
    print(f"    splits_checksum:  {spec.splits_checksum}")
    print(f"    task:             {spec.canonical_spec['task']['name']}")


# --- 1. The same dataset under two different storage roots ------------------------
local = identity(*write_manifests("local", "/local/images"))
archive = identity(*write_manifests("archive", "/archive/deep-storage/images"))

report("manifests under /local/images:", local)
print()
report("same content relocated to /archive/deep-storage/images:", archive)
print()
same = (
    local.experiment_id == archive.experiment_id
    and local.dataset_checksum == archive.dataset_checksum
    and local.splits_checksum == archive.splits_checksum
)
print(f"identical after relocation: {same}")
assert same

# --- 2. Declaring image bytes changes identity ------------------------------------
checksums = [f"{'ab' * 8}{i}" for i in range(len(SAMPLES))]
with_bytes = identity(
    *write_manifests("with-checksums", "/local/images", path_checksums=checksums)
)
print()
report("same paths plus an image_path_sha256 column:", with_bytes)
print()
print(f"experiment_id changed: {with_bytes.experiment_id != local.experiment_id}")
assert with_bytes.experiment_id != local.experiment_id
assert with_bytes.dataset_checksum != local.dataset_checksum
