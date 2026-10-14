"""``soma.testing.synthetic_cohort``: slides whose labels are a function of tissue colour,
written in the shape a user's cohort takes (``cohort.csv`` + ``splits.csv``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from soma import Cohort, ImageManifest
from soma.testing import synthetic_cohort


def _tissue_brightness(image_path: str) -> float:
    """Mean intensity of the slide's central block, which lies inside the tissue disc."""
    import tifffile

    image = tifffile.imread(image_path).astype(np.float64)
    h, w = image.shape[:2]
    return float(image[h // 2 - 16 : h // 2 + 16, w // 2 - 16 : w // 2 + 16].mean())


def test_a_synthetic_cohort_is_a_cohort_whose_labels_follow_the_tissue_colour(tmp_path: Path):
    records, splits = synthetic_cohort(tmp_path / "data", n=8, seed=0)

    frame = pd.read_csv(records)
    assert records == tmp_path / "data" / "cohort.csv"
    assert splits == tmp_path / "data" / "splits.csv"
    assert list(frame.columns) == ["sample_id", "image_path", "patient_id", "label", "severity"]
    assert list(pd.read_csv(splits).columns) == ["sample_id", "split"]

    # Darker tissue is more severe, and the label is "severity above one half".
    brightness = frame["image_path"].map(_tissue_brightness)
    assert brightness.is_unique
    assert list(frame.sort_values("severity")["sample_id"]) == list(
        frame.assign(b=brightness).sort_values("b", ascending=False)["sample_id"]
    )
    assert (frame["label"] == (frame["severity"] > 0.5).astype(int)).all()
    assert brightness[frame["label"] == 1].max() < brightness[frame["label"] == 0].min()

    # The files enter soma through the data contracts; every split holds both labels, and
    # a patient's two slides share a split.
    cohort = Cohort.from_csv(records, splits, targets=["label"])
    assert cohort.unit == "patient_id"
    assert ImageManifest.from_csv(records).sample_ids == cohort.sample_ids
    fold = cohort.folds[0]
    for ids in (fold.train, fold.tune, fold.tests["test"]):
        assert set(frame.set_index("sample_id").loc[list(ids), "label"]) == {0, 1}
    dealt = frame.merge(pd.read_csv(splits), on="sample_id")
    assert (dealt.groupby("patient_id")["split"].nunique() == 1).all()
    assert (dealt.groupby("patient_id").size() == 2).all()


def test_a_synthetic_cohort_is_the_same_cohort_for_the_same_seed(tmp_path: Path):
    first = synthetic_cohort(tmp_path / "a", n=4, seed=3)
    second = synthetic_cohort(tmp_path / "b", n=4, seed=3)
    other = synthetic_cohort(tmp_path / "c", n=4, seed=4)

    def content(records: Path) -> tuple:
        frame = pd.read_csv(records)
        return (
            frame.drop(columns="image_path").to_csv(index=False),
            tuple(Path(p).read_bytes() for p in frame["image_path"]),
        )

    assert content(first[0]) == content(second[0])
    assert first[1].read_bytes() == second[1].read_bytes()
    assert content(first[0]) != content(other[0])
