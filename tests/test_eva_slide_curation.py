"""EVA Cam16Small / PANDASmall curation: oracles against EVA's own code (issue #535).

EVA is not importable here, so ``tests/eva_reference.py`` holds its code verbatim
(kaiko-ai/eva @ f5d80152). The curator must pick the same tiles, levels and read sizes as
EVA's ``PatchCoordinates.from_file`` + ``ForegroundGridSampler``, and the same slides and
splits as EVA's ``Camelyon16`` / ``PANDASmall`` datasets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from soma.curation import eva_slide
from tests import eva_reference as eva
from tests.eva_slide_raw import write_fake_camelyon16, write_fake_panda, write_pyramid

openslide = pytest.importorskip("openslide")

PANDA_RAW = Path("/data/pathology/projects/fmtf-benchmark/raw/eva/panda")


def eva_coordinates(path: Path, *, target_mpp: float, max_samples: int):
    sampler = eva.ForegroundGridSampler(max_samples=max_samples)
    return eva.PatchCoordinates.from_file(
        wsi_path=str(path), width=224, height=224, sampler=sampler, target_mpp=target_mpp
    )


# --- spacing ----------------------------------------------------------------------------


class _Properties(dict):
    """openslide-style properties for the reference ``WsiOpenslide.mpp``."""


def _reference_mpp(properties: dict[str, str]) -> float:
    wsi = eva.WsiOpenslide.__new__(eva.WsiOpenslide)
    wsi._wsi = type("Slide", (), {"properties": _Properties(properties)})()
    return wsi.mpp


@pytest.mark.parametrize(
    "properties",
    [
        {"openslide.mpp-x": "0.24309399999999998", "openslide.mpp-y": "0.24309399999999998"},
        {"openslide.mpp-x": "0.22632099999999999", "openslide.mpp-y": "0.22631600000000002"},
        {"tiff.XResolution": "20000", "tiff.YResolution": "19800", "tiff.ResolutionUnit": "centimeter"},
        {"tiff.XResolution": "51000", "tiff.YResolution": "51000", "tiff.ResolutionUnit": "inch"},
        {
            "openslide.mpp-x": "0.5",
            "openslide.mpp-y": "",
            "tiff.XResolution": "4.0",
            "tiff.YResolution": "4.0",
            "tiff.ResolutionUnit": "millimeter",
        },
    ],
)
def test_slide_mpp_follows_eva_openslide_rule(properties):
    if properties.get("tiff.ResolutionUnit") == "inch":
        # EVA's conversion table has no inch entry: it refuses the slide.
        with pytest.raises(ValueError, match="not supported"):
            _reference_mpp(properties)
        with pytest.raises(ValueError, match="not supported"):
            eva_slide.eva_slide_mpp(properties)
        return
    assert eva_slide.eva_slide_mpp(properties) == _reference_mpp(properties)


def test_slide_mpp_refuses_a_slide_without_spacing():
    with pytest.raises(ValueError, match="mpp"):
        eva_slide.eva_slide_mpp({})


# --- sampler, mask and read plan --------------------------------------------------------


@pytest.mark.parametrize(
    ("mpp", "target_mpp", "max_samples", "size"),
    [
        # Cam16-like native spacings at target 0.25: level-0 reads of 230 / 247 px.
        (0.2431, 0.25, 1000, (2400, 1700)),
        (0.2265, 0.25, 12, (2400, 1700)),
        # PANDA-like native spacings at target 0.5; 0.5032 is coarser than the target, so
        # EVA reads level 0 (its closest-level rule falls back to the finest level).
        (0.4862, 0.5, 200, (1600, 2600)),
        (0.452, 0.5, 9, (1600, 2600)),
        (0.5032, 0.5, 200, (1600, 2600)),
        # A fine slide read from a coarser pyramid level (level 2 at 0.48 µm/px).
        (0.12, 0.5, 40, (4096, 3072)),
    ],
)
def test_sampler_matches_eva_patch_coordinates(tmp_path, mpp, target_mpp, max_samples, size):
    path = write_pyramid(tmp_path / "slide.tif", size=size, mpp=mpp, seed=int(mpp * 1e4))
    reference = eva_coordinates(path, target_mpp=target_mpp, max_samples=max_samples)

    ours = eva_slide.sample_eva_coordinates(path, target_mpp=target_mpp, max_samples=max_samples)

    assert ours.mpp == eva.WsiOpenslide(str(path)).mpp
    assert list(zip(ours.x.tolist(), ours.y.tolist())) == [tuple(c) for c in reference.x_y]
    assert ours.read_level == reference.level_idx
    assert ours.read_tile_size_px == reference.width == reference.height
    assert ours.mask_level == reference.mask.mask_level_idx
    assert ours.tile_size_lv0 == int(target_mpp / reference_mpp(path) * 224)
    assert 0 < len(ours.x) <= max_samples
    assert np.all(ours.foreground_fractions >= eva_slide.MIN_FOREGROUND_RATIO)


def reference_mpp(path: Path) -> float:
    return eva.WsiOpenslide(str(path)).mpp


def test_foreground_mask_matches_eva_get_mask(tmp_path):
    path = write_pyramid(tmp_path / "slide.tif", size=(2048, 1536), mpp=0.25)
    wsi = eva.WsiOpenslide(str(path))
    level = eva.get_mask_level(wsi, 224, 224, 0.5)
    reference = eva.get_mask(wsi, level)

    slide = openslide.OpenSlide(str(path))
    try:
        rgba = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))
    finally:
        slide.close()
    np.testing.assert_array_equal(eva_slide.eva_foreground_mask(rgba), reference.mask_array)


@pytest.mark.parametrize("target_mpp", [0.25, 0.5, 1.0, 2.0, 0.1])
def test_level_rules_match_eva(target_mpp):
    downsamples = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    for mpp in (0.2431, 0.2265, 0.4862, 0.452, 0.5032, 0.12):
        wsi = eva.WsiOpenslide.__new__(eva.WsiOpenslide)
        wsi._wsi = type(
            "Slide",
            (),
            {
                "properties": {"openslide.mpp-x": str(mpp), "openslide.mpp-y": str(mpp)},
                "level_downsamples": downsamples,
                "level_dimensions": [(10000 // d, 8000 // d) for d in (1, 2, 4, 8, 16, 32)],
            },
        )()
        assert eva_slide.eva_closest_level(mpp, downsamples, target_mpp) == wsi.get_closest_level(
            target_mpp
        )
        assert eva_slide.eva_mask_level(mpp, downsamples, target_mpp) == eva.get_mask_level(
            wsi, 224, 224, target_mpp
        )


REAL_SLIDES = [
    (Path("/data/pathology/projects/fmtf-benchmark/raw/eva/panda/train_images/0005f7aaab2800f6170c399693a96917.tiff"), 0.5, 200),
    (Path("/data/pathology/projects/fmtf-benchmark/raw/eva/camelyon16/images/test_001.tif"), 0.25, 1000),
]


@pytest.mark.parametrize(("path", "target_mpp", "max_samples"), REAL_SLIDES, ids=["panda", "camelyon16"])
def test_sampler_matches_eva_on_a_real_slide(path, target_mpp, max_samples):
    if not path.is_file():
        pytest.skip("raw data not mounted")
    reference = eva_coordinates(path, target_mpp=target_mpp, max_samples=max_samples)
    ours = eva_slide.sample_eva_coordinates(path, target_mpp=target_mpp, max_samples=max_samples)
    assert list(zip(ours.x.tolist(), ours.y.tolist())) == [tuple(c) for c in reference.x_y]
    assert (ours.read_level, ours.read_tile_size_px) == (reference.level_idx, reference.width)
    assert ours.mpp == reference_mpp(path)
    assert ours.read_level == 0
    assert 222 <= ours.read_tile_size_px <= 247


# --- splits -----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize(
    "ratios", [(0.1, 0.05, 0.05), (0.7, 0.15, 0.15), (0.6, 0.4, 0.0), (0.5, 0.2, 0.0)]
)
def test_stratified_split_is_eva_stratified_split(seed, ratios):
    rng = np.random.default_rng(seed)
    targets = rng.integers(0, 6, 300).tolist()
    samples = [f"s{i}" for i in range(300)]
    ours = eva_slide.eva_stratified_split(samples, targets, *ratios, seed=seed)
    reference = eva.stratified_split(samples, targets, *ratios, seed=seed)
    assert [list(map(int, part)) if part is not None else None for part in ours] == [
        list(map(int, part)) if part is not None else None for part in reference
    ]


def write_panda_labels(path: Path, grades: list[int], noise: list[int], seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    image_ids = [rng.bytes(16).hex() for _ in grades]
    pd.DataFrame(
        {
            "image_id": image_ids,
            "data_provider": ["karolinska"] * len(grades),
            "isup_grade": grades,
            "noise_ratio_10": noise,
        }
    ).to_csv(path)
    return image_ids


def _eva_panda_small(root: Path, labels_csv: Path) -> dict[str, list[str]]:
    files = eva.PANDASmallFiles(str(root), str(labels_csv))
    return {
        ours: [Path(p).stem for p in files._load_file_paths(theirs)]
        for ours, theirs in (("train", "train"), ("tune", "val"), ("test", "test"))
    }


def test_panda_small_split_is_eva_panda_small(tmp_path):
    rng = np.random.default_rng(3)
    grades = rng.integers(0, 6, 400).tolist()
    noise = (rng.random(400) > 0.1).astype(int).tolist()
    labels_csv = tmp_path / "labels.csv"
    image_ids = write_panda_labels(labels_csv, grades, noise)
    (tmp_path / "train_images").mkdir()
    for image_id in image_ids:
        (tmp_path / "train_images" / f"{image_id}.tiff").touch()

    ours = eva_slide.panda_small_split(pd.read_csv(labels_csv, index_col="image_id"))
    assert ours == _eva_panda_small(tmp_path, labels_csv)
    # EVA's "non noisy" filter keeps noise_ratio_10 != 0.
    dropped = {i for i, n in zip(image_ids, noise) if n == 0}
    assert dropped and not dropped & {i for part in ours.values() for i in part}


def test_panda_small_split_sizes_follow_the_per_grade_floor():
    """The 9555 slides EVA keeps, per ISUP grade, floor to 952 / 475 / 475.

    EVA's documentation quotes 955 / 477 / 477 (the ratios times 9555), but its code takes
    ``floor(ratio * n)`` per grade.
    """
    per_grade = {0: 2603, 1: 2399, 2: 1209, 3: 1118, 4: 1124, 5: 1102}
    targets = [grade for grade, n in per_grade.items() for _ in range(n)]
    train, val, test = eva_slide.eva_stratified_split(
        list(range(len(targets))), targets, *eva_slide.PANDA_SMALL_RATIOS, seed=42
    )
    assert len(targets) == 9555
    assert (len(train), len(val), len(test)) == (952, 475, 475)


#: sha256 of the sorted PANDASmall test image ids, newline-joined (EVA's split, verified
#: against the noisy-label CSV with md5 5e4bfc78bda9603d2e2faf3ed4b21dfa).
PANDA_SMALL_TEST_IDS_SHA256 = "edd41eb23992ebce2f1b293de9115b049c878f1384ae55bfdf942df3311b1e63"


@pytest.mark.skipif(not PANDA_RAW.is_dir(), reason="PANDA raw data not mounted")
def test_panda_small_split_on_the_real_label_csv(tmp_path):
    labels_csv = PANDA_RAW / eva_slide.PANDA_LABELS_CSV
    labels = pd.read_csv(labels_csv, index_col="image_id")
    (tmp_path / "train_images").mkdir()
    for image_id in labels.index:
        (tmp_path / "train_images" / f"{image_id}.tiff").touch()

    ours = eva_slide.panda_small_split(labels)
    assert ours == _eva_panda_small(tmp_path, labels_csv)
    assert {k: len(v) for k, v in ours.items()} == {"train": 952, "tune": 475, "test": 475}
    digest = hashlib.sha256("\n".join(sorted(ours["test"])).encode()).hexdigest()
    assert digest == PANDA_SMALL_TEST_IDS_SHA256


def test_camelyon16_small_split_is_eva_camelyon16(tmp_path):
    ids = (
        # normal_086 is not in the official release: EVA's 216 training slides lack it.
        [f"normal_{i:03d}" for i in range(1, 161) if i != 86]
        + [f"tumor_{i:03d}" for i in range(1, 112)]
        + [f"test_{i:03d}" for i in range(1, 131) if i != 49]
    )
    for slide_id in ids:
        group = "testing/images" if slide_id.startswith("test") else f"training/{slide_id.split('_')[0]}"
        (tmp_path / group).mkdir(parents=True, exist_ok=True)
        (tmp_path / group / f"{slide_id}.tif").touch()
    files = eva.Camelyon16Files(str(tmp_path))
    reference = {
        Path(p).stem: ours
        for ours, theirs in (("train", "train"), ("tune", "val"), ("test", "test"))
        for p in files._load_file_paths(theirs)
    }
    ours = eva_slide.camelyon16_small_split(ids)
    assert ours == reference
    counts = {s: sum(v == s for v in ours.values()) for s in ("train", "tune", "test")}
    assert counts == eva_slide.CAMELYON16_SPLIT_SIZES
    assert tuple(eva.Camelyon16Files._val_slides) == eva_slide.CAMELYON16_VAL_SLIDES


CAMELYON16_RAW = Path("/data/pathology/projects/fmtf-benchmark/raw/eva/camelyon16")


@pytest.mark.skipif(not CAMELYON16_RAW.is_dir(), reason="Camelyon16 raw data not mounted")
def test_camelyon16_small_rows_on_the_real_reference_table():
    rows = eva_slide.camelyon16_small_rows(CAMELYON16_RAW)
    counts = {s: sum(r.split == s for r in rows) for s in ("train", "tune", "test")}
    assert counts == {"train": 216, "tune": 54, "test": 129}
    assert sum(r.label for r in rows) == 160  # 111 training + 49 test tumor slides
    assert {r.sample_id for r in rows if r.split == "tune"} == set(eva_slide.CAMELYON16_VAL_SLIDES)


# --- curators ---------------------------------------------------------------------------


def _stage(manifest, target_mpp: float, tmp_path: Path):
    from soma.config import PreprocessingConfig
    from soma.data._legacy import legacy_samples_from_csv
    from soma.preprocessing.supplied_coordinates import stage_supplied_coordinates

    dataset = legacy_samples_from_csv(manifest.dataset_csv)
    preprocessing = PreprocessingConfig(requested_spacing_um=target_mpp, requested_tile_size_px=224)
    return stage_supplied_coordinates(dataset, tmp_path / "tiling", preprocessing)


def test_camelyon16_small_curator_writes_eva_tiles_and_a_slide_manifest(tmp_path):
    raw = write_fake_camelyon16(tmp_path / "raw")
    manifest = eva_slide.curate_camelyon16_small(raw, tmp_path / "curated", workers=2)

    dataset = pd.read_csv(manifest.dataset_csv)
    splits = pd.read_csv(manifest.splits_csv).set_index("sample_id")["split"]
    assert dataset["sample_id"].tolist() == sorted(dataset["sample_id"])
    assert dict(splits.value_counts()) == {"train": 7, "tune": 4, "test": 4}
    assert splits["tumor_001"] == "tune" and splits["test_002"] == "test"
    labels = dataset.set_index("sample_id")["label"]
    assert labels["tumor_003"] == 1 and labels["normal_002"] == 0 and labels["test_004"] == 1

    for row in dataset.itertuples():
        reference = eva_coordinates(Path(row.image_path), target_mpp=0.25, max_samples=1000)
        result = eva_slide_artifact(row.coordinates_path)
        assert sorted(zip(result.x.tolist(), result.y.tolist())) == sorted(map(tuple, reference.x_y))
        assert (result.read_level, result.read_tile_size_px) == (reference.level_idx, reference.width)
        assert result.requested_tile_size_px == 224 and result.requested_spacing_um == 0.25
        assert row.spacing_at_level_0 == pytest.approx(reference_mpp(Path(row.image_path)), rel=1e-12)
        assert result.spacing_at_level_0 == row.spacing_at_level_0
        assert row.num_tiles == len(reference.x_y)

    summary = json.loads(manifest.summary_json.read_text())
    assert summary["split_counts"] == {"train": 7, "tune": 4, "test": 4}
    assert summary["read_tile_size_px_counts"] == {"230": 15}
    # soma's bring-your-own-coordinates staging accepts every artifact as is.
    staged = _stage(manifest, 0.25, tmp_path)
    assert len(staged.samples) == 15


def eva_slide_artifact(coordinates_path: str):
    from hs2p.artifacts import load_tiling_result

    from soma.preprocessing.supplied_coordinates import coordinates_meta_path

    return load_tiling_result(Path(coordinates_path), coordinates_meta_path(Path(coordinates_path)))


def test_panda_small_curator_follows_eva_split_and_sampling(tmp_path):
    raw = write_fake_panda(tmp_path / "raw")
    manifest = eva_slide.curate_panda_small(raw, tmp_path / "curated", workers=1)

    dataset = pd.read_csv(manifest.dataset_csv)
    splits = pd.read_csv(manifest.splits_csv).set_index("sample_id")["split"]
    expected = _eva_panda_small(raw, raw / eva_slide.PANDA_LABELS_CSV)
    assert {s: sorted(splits.index[splits == s]) for s in expected} == {
        s: sorted(ids) for s, ids in expected.items()
    }
    assert {s: len(ids) for s, ids in expected.items()} == {"train": 6, "tune": 6, "test": 6}
    labels = pd.read_csv(raw / eva_slide.PANDA_LABELS_CSV, index_col="image_id")
    assert all(labels.loc[r.sample_id, "isup_grade"] == r.label for r in dataset.itertuples())
    row = dataset.iloc[0]
    reference = eva_coordinates(Path(row.image_path), target_mpp=0.5, max_samples=200)
    result = eva_slide_artifact(row.coordinates_path)
    assert sorted(zip(result.x.tolist(), result.y.tolist())) == sorted(map(tuple, reference.x_y))

    summary = json.loads(manifest.summary_json.read_text())
    assert summary["labels_csv_matches_eva"] is False
    assert len(_stage(manifest, 0.5, tmp_path).samples) == 18


def test_panda_small_curator_refuses_a_missing_image(tmp_path):
    raw = write_fake_panda(tmp_path / "raw", per_grade=1)
    next((raw / "train_images").glob("*.tiff")).unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        eva_slide.curate_panda_small(raw, tmp_path / "curated", workers=1)


def test_curation_is_byte_identical_on_rerun(tmp_path):
    raw = write_fake_camelyon16(tmp_path / "raw")
    first = eva_slide.curate_camelyon16_small(raw, tmp_path / "a", workers=1)
    before = {p: p.read_bytes() for p in (first.dataset_csv, first.splits_csv, first.summary_json)}
    again = eva_slide.curate_camelyon16_small(raw, tmp_path / "a", workers=3)
    assert {p: p.read_bytes() for p in (again.dataset_csv, again.splits_csv, again.summary_json)} == before


#: Native spacings of real slides that pandas' default CSV parser does not read back
#: exactly (Camelyon16 normal_001, a PANDA slide).
UNSTABLE_SPACINGS = [0.24309399999999998, 0.45201826153776614]


@pytest.mark.parametrize("mpp", UNSTABLE_SPACINGS)
def test_curated_spacing_survives_the_manifest_round_trip(tmp_path, monkeypatch, mpp):
    """The manifest and the artifacts must declare the same spacing after ``Dataset`` reads it.

    soma's staging compares the two exactly, and ``Dataset`` parses the manifest with
    pandas' default (not round-trip exact) float parser.
    """
    raw = write_fake_camelyon16(tmp_path / "raw")
    monkeypatch.setattr(eva_slide, "eva_slide_mpp", lambda properties: mpp)
    manifest = eva_slide.curate_camelyon16_small(raw, tmp_path / "curated", workers=1)
    staged = _stage(manifest, 0.25, tmp_path)
    record = next(iter(staged.samples.values()))
    assert record.spacing_at_level_0 == pytest.approx(mpp, rel=1e-12)
    # The tiles are still EVA's, sampled at EVA's exact spacing.
    assert eva_slide_artifact(str(record.coordinates_path)).tile_size_lv0 == int(0.25 / mpp * 224)


def test_manifest_spacing_is_a_fixed_point_of_the_csv_round_trip():
    import io

    rng = np.random.default_rng(0)
    for value in [*UNSTABLE_SPACINGS, *rng.uniform(0.1, 1.0, 2000).tolist()]:
        declared = eva_slide.manifest_spacing(value)
        assert declared == pytest.approx(value, rel=1e-12)
        text = pd.DataFrame({"spacing_at_level_0": [declared]}).to_csv(index=False)
        assert pd.read_csv(io.StringIO(text))["spacing_at_level_0"].iloc[0] == declared
