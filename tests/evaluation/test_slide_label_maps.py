"""``soma.evaluate`` on the label TIFFs ``soma.predict_slides`` writes, against the slides'
annotation rasters.

Both rasters are read window by window on the map's pixel grid. The slides are synthetic
TIFFs, the bundles carry the weight-free ``fake`` encoder's recipe, and everything runs on
the CPU. The class scheme is a dataset's own: raw annotation values 1 to 4 form three
classes, and 0 is ignored.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import soma
from soma.data import AnnotationManifest, ImageManifest, LabelMapSource, build_label_remap
from soma.evaluation import LabelMapPredictions, SlideLabelMaps
from soma.evaluation.grid import label_map_counts
from tests.whole_slides import (
    MEMORY_PRELUDE,
    SPACING_UM,
    TIFFFILE,
    blocky,
    peak_rss_of,
    read_map,
    save_bundle,
    write_large_slide,
    write_mask,
    write_pyramid,
    write_records,
    write_slide,
)

CLASSES = {"stroma": [1], "tumour": [2, 3], "necrosis": [4]}
IGNORE = [0]
#: The head of a run trained on these annotation rasters.
SCHEME = {
    "class_names": list(CLASSES),
    "label_remap": build_label_remap(CLASSES, ignore=IGNORE),
    "pixel_mapping": {"unannotated": 0, "stroma": 1, "tumour": [2, 3], "necrosis": 4},
}
#: Each annotation raw value's class index; the ignored value 0 is 255.
CLASS_OF_VALUE = np.array([255, 0, 1, 1, 2], dtype=np.int64)
METRICS = ["mean_dice", "dataset_global_mean_dice", "foreground_mean_dice", "mean_iou", "dice_per_class"]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


class ArrayTargets:
    """A target source over label maps already in memory."""

    target_kind = "label_map"

    def __init__(self, maps):
        self._maps = dict(maps)

    @property
    def sample_ids(self):
        return list(self._maps)

    def load(self, sample_id):
        return torch.from_numpy(self._maps[sample_id])


def raw_annotation(height: int, width: int, *, seed: int, block: int = 4) -> np.ndarray:
    """Raw annotation values 0 to 4 in ``block`` px squares."""
    rng = np.random.default_rng(seed)
    cells = rng.integers(0, 5, size=(-(-height // block), -(-width // block)), dtype=np.uint8)
    return np.ascontiguousarray(cells.repeat(block, 0).repeat(block, 1)[:height, :width])


def slide_targets(records: str) -> LabelMapSource:
    """The cohort's annotation rasters, aligned to its slides; read with ``tifffile``."""
    return LabelMapSource.from_manifests(
        AnnotationManifest.from_csv(records),
        ImageManifest.from_csv(records),
        size=(0, 0),
        backend="tifffile",
        image_backend="tifffile",
    )


def predict(tmp_path, records, bundle, *, overlap=0.5, min_coverage=None):
    extraction = {"tiling": {"overlap": overlap, **TIFFFILE}}
    if min_coverage is not None:
        extraction["masks"] = {"min_coverage": {"tissue": min_coverage}}
    return soma.predict_slides(
        output_dir=tmp_path / "predictions", bundles=bundle, records=records, extraction=extraction, batch_size=16
    )


def counts_of(row, num_classes: int = 3) -> np.ndarray:
    """A per-slide row's ``(C, 3)`` (tp, fp, fn)."""
    return np.array([[row[f"{kind}_class_{c}"] for kind in ("tp", "fp", "fn")] for c in range(num_classes)])


def tp_fp_fn(counts: np.ndarray) -> np.ndarray:
    """``(C, 3)`` (intersection, predicted, target) as (tp, fp, fn)."""
    inter, predicted, target = counts[:, 0], counts[:, 1], counts[:, 2]
    return np.stack([inter, predicted - inter, target - inter], axis=1)


# --- streamed equals in-memory -------------------------------------------------------- #


def single_level(tmp_path, height, width, mask_downsample):
    """A slide at 0.5 um/px and its annotation raster ``mask_downsample`` times coarser;
    returns the slide row and the annotation on the map's grid (the slide's)."""
    slide = write_slide(tmp_path / "s1.tif", blocky(height, width))
    tissue = write_mask(tmp_path / "s1_tissue.tif", np.ones((height, width)))
    raw = raw_annotation(height // mask_downsample, width // mask_downsample, seed=1)
    annotation = write_mask(tmp_path / "s1_annotation.tif", raw, spacing_um=0.5 * mask_downsample)
    row = {"sample_id": "s1", "image_path": str(slide), "mask_path": str(tissue), "label_mask_path": str(annotation)}
    return row, raw.repeat(mask_downsample, 0).repeat(mask_downsample, 1)


def read_at_a_pyramid_level(tmp_path, height, width, _):
    """A slide whose 0.5 um/px read level is level 1 of a 0.25 um/px pyramid, annotated at
    level 0: the map takes every other annotation pixel."""
    slide, (full_height, full_width), base_spacing = write_pyramid(
        tmp_path / "s1.tif", blocky(height, width), downsample=2
    )
    tissue = write_mask(tmp_path / "s1_tissue.tif", np.ones((height, width)))
    raw = raw_annotation(full_height, full_width, seed=1, block=3)
    annotation = write_mask(tmp_path / "s1_annotation.tif", raw, spacing_um=base_spacing)
    row = {"sample_id": "s1", "image_path": str(slide), "mask_path": str(tissue), "label_mask_path": str(annotation)}
    return row, raw[::2, ::2]


@pytest.mark.parametrize(
    ("cohort", "mask_downsample"),
    [(single_level, 1), (single_level, 4), (read_at_a_pyramid_level, None)],
    ids=["mask-at-the-slide-resolution", "lower-resolution-mask", "slide-read-at-a-pyramid-level"],
)
def test_streamed_counts_equal_scoring_the_whole_arrays_in_memory(tmp_path, cohort, mask_downsample):
    """The map is read back whole and the annotation is put on the map's grid by hand;
    ``evaluate`` over those arrays and over the streamed directory agree on every count
    and every metric. The 48 px windows (three 16 px strides) split the slide unevenly."""
    height, width = 200, 264
    row, annotation = cohort(tmp_path, height, width, mask_downsample)
    records = write_records(tmp_path / "dataset.csv", [row])
    bundle = save_bundle(tmp_path / "run" / "bundle", head=SCHEME)
    (map_row,) = predict(tmp_path, records, bundle).itertuples()
    label_map = read_map(tmp_path / "predictions" / map_row.path)
    assert label_map.shape == (height, width) and label_map.min() > 0  # every pixel predicted
    prediction, target = label_map.astype(np.int64) - 1, CLASS_OF_VALUE[annotation]

    streamed = soma.evaluate(
        SlideLabelMaps(tmp_path / "predictions" / "run", window_px=48), slide_targets(records), metrics=METRICS
    )
    in_memory = soma.evaluate(
        LabelMapPredictions({"s1": prediction}, classes=list(CLASSES)), ArrayTargets({"s1": target}), metrics=METRICS
    )

    assert streamed.metrics == in_memory.metrics
    assert streamed.num_samples == 1
    (scored,) = streamed.per_sample
    assert scored["sample_id"] == "s1"
    expected = label_map_counts(prediction, target, num_classes=3, ignore_index=255)
    np.testing.assert_array_equal(counts_of(scored), tp_fp_fn(expected))
    assert scored["annotated_px"] == int((target != 255).sum())
    assert scored["not_predicted_fraction"] == 0.0


# --- the not-predicted rule ------------------------------------------------------------ #


def half_tissue_slide(tmp_path, annotation: np.ndarray, *, name: str = "s1"):
    """A slide the size of ``annotation`` whose tissue mask keeps only the tiles of its
    left half."""
    height, width = annotation.shape
    slide = write_slide(tmp_path / f"{name}.tif", blocky(height, width))
    tissue = np.zeros((height, width))
    tissue[:, : width // 2] = 1
    return {
        "sample_id": name,
        "image_path": str(slide),
        "mask_path": str(write_mask(tmp_path / f"{name}_tissue.tif", tissue)),
        "label_mask_path": str(write_mask(tmp_path / f"{name}_annotation.tif", annotation)),
    }


def hand_counts(label_map: np.ndarray, target: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """``(C, 3)`` (tp, fp, fn) by the rules: ignore (255) is never counted, and an
    annotated pixel with value 0 (not predicted) is a false negative only."""
    annotated = target != 255
    return np.array(
        [
            [
                int(((target == c) & (label_map == c + 1)).sum()),
                int((annotated & (target != c) & (label_map == c + 1)).sum()),
                int(((target == c) & (label_map != c + 1)).sum()),
            ]
            for c in range(num_classes)
        ]
    )


def test_annotated_pixels_outside_the_kept_tiles_are_false_negatives(tmp_path):
    """Only the left half is predicted. Rows 0-15 are ignored; below them, stroma
    everywhere but tumour in columns 48-79 (half of it predicted) and necrosis in columns
    96-111 (none of it predicted)."""
    annotation = np.zeros((96, 128), dtype=np.uint8)  # raw 0: ignored
    annotation[16:] = 1  # stroma
    annotation[16:, 48:80] = 2  # tumour
    annotation[16:, 96:112] = 4  # necrosis
    records = write_records(tmp_path / "dataset.csv", [half_tissue_slide(tmp_path, annotation)])
    bundle = save_bundle(tmp_path / "run" / "bundle", head=SCHEME)
    (map_row,) = predict(tmp_path, records, bundle, overlap=0.0, min_coverage=0.5).itertuples()
    label_map = read_map(tmp_path / "predictions" / map_row.path)
    assert (label_map[:, :64] > 0).all() and (label_map[:, 64:] == 0).all()

    evaluation = soma.evaluate(SlideLabelMaps(tmp_path / "predictions"), slide_targets(records), metrics=METRICS)

    (scored,) = evaluation.per_sample
    assert scored["annotated_px"] == 80 * 128
    assert scored["not_predicted_px"] == 80 * 64
    assert scored["not_predicted_fraction"] == 0.5
    counts = counts_of(scored)
    np.testing.assert_array_equal(counts, hand_counts(label_map, CLASS_OF_VALUE[annotation]))
    assert (counts[2, 0], counts[2, 2]) == (0, 80 * 16)  # necrosis is never predicted: all FN
    assert counts[:, 0].sum() + counts[:, 1].sum() == 80 * 64  # only predicted pixels are TP or FP
    assert counts[:, 0].sum() + counts[:, 2].sum() == 80 * 128  # every annotated pixel is TP or FN


# --- ignore ---------------------------------------------------------------------------- #


def rewrite_map(path, label_map: np.ndarray):
    """Write ``label_map`` over the label TIFF at ``path``, with the same metadata."""
    import json

    import tifffile
    from hs2p import write_label_tiff

    with tifffile.TiffFile(path) as tiff:
        metadata = json.loads(tiff.pages[0].description)
    height, width = label_map.shape
    tiles = (label_map[y : y + 16, x : x + 16] for y in range(0, height, 16) for x in range(0, width, 16))
    write_label_tiff(
        path=path, tiles=tiles, dimensions=(width, height), tile_size=16, spacing_um=0.5, metadata=metadata
    )


def test_predictions_on_ignored_pixels_change_no_count(tmp_path):
    height, width = 96, 128
    row, annotation = single_level(tmp_path, height, width, 1)
    records = write_records(tmp_path / "dataset.csv", [row])
    bundle = save_bundle(tmp_path / "run" / "bundle", head=SCHEME)
    (map_row,) = predict(tmp_path, records, bundle).itertuples()
    maps = SlideLabelMaps(tmp_path / "predictions" / "run", window_px=32)
    before = soma.evaluate(maps, slide_targets(records), metrics=METRICS)

    path = tmp_path / "predictions" / map_row.path
    label_map = read_map(path)
    ignored = CLASS_OF_VALUE[annotation] == 255
    changed = label_map.copy()
    changed[ignored] = (label_map[ignored] + 1) % 4  # another class, or not predicted
    rewrite_map(path, changed)
    assert ignored.any() and (read_map(path) != label_map).sum() == ignored.sum()
    after = soma.evaluate(maps, slide_targets(records), metrics=METRICS)

    assert after.per_sample == before.per_sample
    assert after.metrics == before.metrics


# --- ids -------------------------------------------------------------------------------- #


def test_a_slide_without_a_prediction_or_without_a_target_is_refused_with_the_count(tmp_path):
    annotation = np.ones((96, 128), dtype=np.uint8)
    rows = [half_tissue_slide(tmp_path, annotation, name=name) for name in ("s1", "s2", "s3")]
    bundle = save_bundle(tmp_path / "run" / "bundle", head=SCHEME)
    predict(tmp_path, write_records(tmp_path / "two.csv", rows[:2]), bundle)
    maps = SlideLabelMaps(tmp_path / "predictions")
    assert maps.sample_ids == ["s1", "s2"]
    targets = slide_targets(write_records(tmp_path / "three.csv", rows))

    with pytest.raises(ValueError, match=r"1 target\(s\) have no prediction: \['s3'\]"):
        soma.evaluate(maps, targets)
    with pytest.raises(ValueError, match=r"1 prediction id\(s\) have no target: \['s2'\]"):
        soma.evaluate(maps, targets.subset(["s1"]))


# --- the prediction directory ------------------------------------------------------------ #


def test_an_output_directory_is_scored_through_its_one_bundle_and_refused_with_several(tmp_path):
    rows = [half_tissue_slide(tmp_path, np.ones((96, 128), dtype=np.uint8))]
    records = write_records(tmp_path / "dataset.csv", rows)
    bundles = [save_bundle(tmp_path / name / "bundle", seed=seed, head=SCHEME) for name, seed in (("a", 0), ("b", 1))]
    predict(tmp_path, records, bundles)

    with pytest.raises(ValueError, match=r"holds the maps of 2 bundles \['a', 'b'\]; score one bundle's directory"):
        SlideLabelMaps(tmp_path / "predictions")
    maps = SlideLabelMaps(tmp_path / "predictions" / "b")
    assert (maps.sample_ids, maps.classes) == (["s1"], tuple(CLASSES))
    assert (maps.backend, maps.mask_backend) == ("tifffile", "tifffile")  # the plan's readers
    with pytest.raises(ValueError, match="is not a soma predict output"):
        SlideLabelMaps(tmp_path / "a")


def test_a_map_without_a_recorded_class_scheme_is_refused(tmp_path):
    import json

    import tifffile
    from hs2p import write_label_tiff

    records = write_records(tmp_path / "dataset.csv", [half_tissue_slide(tmp_path, np.ones((96, 128), dtype=np.uint8))])
    (map_row,) = predict(tmp_path, records, save_bundle(tmp_path / "run" / "bundle", head=SCHEME)).itertuples()
    path = tmp_path / "predictions" / map_row.path
    with tifffile.TiffFile(path) as tiff:
        metadata = json.loads(tiff.pages[0].description)
    label_map = read_map(path)
    old = {key: value for key, value in metadata.items() if key not in ("classes", "ignore", "pixel_mapping")}
    write_label_tiff(path=path, tiles=[label_map], dimensions=(128, 96), tile_size=128, spacing_um=0.5, metadata=old)

    with pytest.raises(ValueError, match="records no class scheme .*Predict the slides again"):
        SlideLabelMaps(tmp_path / "predictions")


# --- memory ------------------------------------------------------------------------------ #

#: The child process of the memory test: it scores a small slide first (imports, the
#: readers), resets the peak RSS, scores the large slide and prints the RSS before it, the
#: peak during it and the slide's row.
MEMORY_CHILD = MEMORY_PRELUDE + r"""
from soma.data import AnnotationManifest, ImageManifest, LabelMapSource
from soma.evaluation import SlideLabelMaps

tmp = Path(sys.argv[1])

def score(name):
    records = str(tmp / f"{name}.csv")
    targets = LabelMapSource.from_manifests(
        AnnotationManifest.from_csv(records), ImageManifest.from_csv(records),
        size=(0, 0), backend="tifffile", image_backend="tifffile",
    )
    return soma.evaluate(SlideLabelMaps(tmp / name), targets)

score("small")
reset_peak()
before = status("VmRSS")
(row,) = score("large").per_sample
print(json.dumps({"before": before, "peak": status("VmHWM"), "row": row}))
"""


def test_peak_memory_stays_far_below_the_raw_rasters(tmp_path):
    """A 32768 x 32768 px map (1 GiB as raw uint8, as is its annotation on the map's
    grid) with three predicted tissue patches, scored against a 16 um/px annotation
    raster. Both are read in 1024 px windows: the peak RSS while scoring stays below an
    eighth of one raw raster (it is about 50 MiB here)."""
    side, coarse = 32768, 32  # the masks are at 16 um/px: one mask pixel is 32 map pixels
    write_large_slide(tmp_path / "large.tif", side)
    tissue = np.zeros((side // coarse, side // coarse), dtype=np.uint8)
    annotation = np.zeros_like(tissue)  # raw 0: ignored
    for row, col in ((128, 128), (512, 512), (896, 768)):
        tissue[row : row + 4, col : col + 4] = 1
    annotation[128:132, 128:132] = 1  # stroma, predicted
    annotation[512:516, 512:516] = 2  # tumour, predicted
    annotation[300:302, 300:302] = 4  # necrosis, not predicted
    write_mask(tmp_path / "large_tissue.tif", tissue, spacing_um=SPACING_UM * coarse)
    write_mask(tmp_path / "large_annotation.tif", annotation, spacing_um=SPACING_UM * coarse)
    write_slide(tmp_path / "small.tif", blocky(96, 96))
    write_mask(tmp_path / "small_tissue.tif", np.ones((96, 96)))
    write_mask(tmp_path / "small_annotation.tif", np.ones((96, 96)))
    bundle = save_bundle(tmp_path / "run" / "bundle", head=SCHEME)
    for name in ("large", "small"):
        records = write_records(
            tmp_path / f"{name}.csv",
            [
                {
                    "sample_id": name,
                    "image_path": str(tmp_path / f"{name}.tif"),
                    "mask_path": str(tmp_path / f"{name}_tissue.tif"),
                    "label_mask_path": str(tmp_path / f"{name}_annotation.tif"),
                }
            ],
        )
        soma.predict_slides(
            output_dir=tmp_path / name,
            bundles=bundle,
            records=records,
            extraction={"tiling": {"overlap": 0.0, **TIFFFILE}, "masks": {"min_coverage": {"tissue": 0.5}}},
            batch_size=64,
        )

    rss = peak_rss_of(MEMORY_CHILD, str(tmp_path))

    pixel = coarse * coarse  # map pixels per annotation pixel
    assert rss["row"]["annotated_px"] == (16 + 16 + 4) * pixel
    assert rss["row"]["not_predicted_px"] == 4 * pixel  # the necrosis patch
    assert rss["peak"] - rss["before"] < side * side / 8
