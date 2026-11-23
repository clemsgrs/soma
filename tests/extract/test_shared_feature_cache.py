"""A feature cache entry shared by several datasets describes every slide it holds.

Feature cache keys ignore the dataset, so datasets extracted with the same settings fill
one entry. The entry's own ``process_list.csv`` is what ``CachedSetSource(<entry dir>)``
and any check of the entry read: a slide whose features the entry holds must be listed
``success`` there, whichever dataset extracted it (issue #766). These scenarios run real
tiling and extraction of synthetic slides with the weight-free ``fake`` encoder.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from slide2vec import ExecutionOptions
from slide2vec.encoders.registry import encoder_registry

import soma
from soma.config import CacheOptions
from soma.data import CachedSetSource, ImageManifest
from soma.testing import synthetic_cohort
from tests.extraction_configs import extraction

SLIDES = ("slide00", "slide01", "slide02", "slide03")
#: Every synthetic slide has 144 tiles, so one batch holds one slide.
BATCH_SIZE = 256


@pytest.fixture(scope="module")
def cohort(tmp_path_factory) -> pd.DataFrame:
    records, _ = synthetic_cohort(tmp_path_factory.mktemp("cohort"), n=len(SLIDES))
    return pd.read_csv(records)


def _dataset(cohort: pd.DataFrame, root: Path, name: str, sample_ids) -> ImageManifest:
    path = root / f"{name}.csv"
    cohort[cohort["sample_id"].isin(sample_ids)].to_csv(path, index=False)
    return ImageManifest.from_csv(path)


def _extract(dataset: ImageManifest, root: Path, run: str, kind: str = "tile"):
    """Extract ``kind`` features into the cache shared by every run under ``root``."""
    return soma.extract(
        dataset,
        encoder="fake",
        extraction=extraction(0.5, 32, region_tile_multiple=2 if kind == "hierarchical" else None),
        execution=ExecutionOptions(
            num_gpus=1, num_workers_per_gpu=0, num_preprocessing_workers=0, batch_size=BATCH_SIZE
        ),
        cache=CacheOptions(enabled=True, root_dir=root / "cache"),
        output_dir=root / run,
    )


def _entry(root: Path, kind: str = "tile") -> Path:
    (entry,) = (root / "cache" / kind).iterdir()
    return entry


def _entry_statuses(root: Path, kind: str = "tile") -> dict[str, str]:
    process_list = pd.read_csv(_entry(root, kind) / "process_list.csv")
    return dict(zip(process_list["sample_id"], process_list["feature_status"]))


@pytest.mark.parametrize("kind", ["tile", "hierarchical"])
@pytest.mark.parametrize(
    "first, second",
    [
        pytest.param(SLIDES[:2], SLIDES[2:], id="different-slides"),
        pytest.param(SLIDES[:2], SLIDES[1:3], id="one-slide-in-both"),
    ],
)
def test_an_entry_shared_by_two_datasets_lists_every_slide_it_holds_as_success(
    tmp_path, cohort, first, second, kind
):
    _extract(_dataset(cohort, tmp_path, "first", first), tmp_path, "first", kind)
    _extract(_dataset(cohort, tmp_path, "second", second), tmp_path, "second", kind)

    held = sorted({*first, *second})
    assert _entry_statuses(tmp_path, kind) == {sample_id: "success" for sample_id in held}
    assert sorted(CachedSetSource(_entry(tmp_path, kind)).sample_ids) == held


@pytest.mark.parametrize("encoded_slides", [0, 1], ids=["fails-at-first-slide", "fails-at-second-slide"])
def test_after_a_failed_run_the_entry_lists_every_slide_it_holds_and_the_rerun_all(
    tmp_path, cohort, monkeypatch, encoded_slides
):
    _extract(_dataset(cohort, tmp_path, "first", SLIDES[:2]), tmp_path, "first")
    second = _dataset(cohort, tmp_path, "second", SLIDES[2:])
    encoder = encoder_registry.require("fake")
    encode = encoder.encode_tiles
    batches = 0

    def fail_later(self, batch):
        nonlocal batches
        batches += 1
        if batches > encoded_slides:
            raise RuntimeError("interrupted")
        return encode(self, batch)

    with monkeypatch.context() as patch:
        patch.setattr(encoder, "encode_tiles", fail_later)
        with pytest.raises(Exception, match="interrupted"):
            _extract(second, tmp_path, "second")

    # slide2vec embeds the slides in its own order: read which ones the failed run wrote.
    held = sorted(path.stem for path in (_entry(tmp_path) / "tile_embeddings").glob("*.pt"))
    assert len(held) == 2 + encoded_slides and set(SLIDES[:2]) <= set(held)
    assert _entry_statuses(tmp_path) == {sample_id: "success" for sample_id in held}
    assert sorted(CachedSetSource(_entry(tmp_path)).sample_ids) == held

    _extract(second, tmp_path, "second")

    assert _entry_statuses(tmp_path) == {sample_id: "success" for sample_id in SLIDES}
    assert sorted(CachedSetSource(_entry(tmp_path)).sample_ids) == list(SLIDES)
