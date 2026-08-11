from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from soma.curation.pathorob import curate_pathorob_ri_views


COHORTS = {
    "camelyon": {
        "metadata": "camelyon.csv",
        "labels": 2,
        "centers": 2,
        "per_cell": 5_100,
        "dataset_repository": "bifold-pathomics/PathoROB-camelyon",
        "dataset_revision": "b2e762542abbf85dde3f23ec70a2bf1848dcf5c8",
    },
    "tcga-4x4": {
        "metadata": "tcga_4x4.csv",
        "labels": 4,
        "centers": 4,
        "per_cell": 360,
        "dataset_repository": "bifold-pathomics/PathoROB-tcga",
        "dataset_revision": "6e1dbd4306ebee9759b32503914523e84bddabd0",
    },
    "tolkach-esca": {
        "metadata": "tolkach_esca_reduced.csv",
        "labels": 6,
        "centers": 3,
        "per_cell": 500,
        "dataset_repository": "bifold-pathomics/PathoROB-tolkach_esca",
        "dataset_revision": "c42219a2c168c5995e44487f5747bdacaf4bc2da",
    },
}
METADATA_REVISION = "6583cf0b0d902c8cc032308262fa3a3befdc0687"


def _write_prepared_tree(raw_root: Path) -> None:
    for cohort, spec in COHORTS.items():
        cohort_dir = raw_root / cohort
        cohort_dir.mkdir(parents=True)
        image = cohort_dir / "images" / "tile.png"
        image.parent.mkdir()
        image.touch()

        metadata_rows: list[dict[str, str]] = []
        index_rows: list[dict[str, str]] = []
        for label_index in range(spec["labels"]):
            for center_index in range(spec["centers"]):
                for row_index in range(spec["per_cell"]):
                    sample_id = (
                        f"{cohort}-l{label_index}-c{center_index}-r{row_index}"
                    )
                    metadata_rows.append(
                        {
                            "slide_id": sample_id,
                            "patch_id": "0",
                            "biological_class": f"label-{label_index}",
                            "medical_center": f"center-{center_index}",
                            "subset": "ID",
                        }
                    )
                    index_rows.append(
                        {
                            "slide_id": sample_id,
                            "patch_id": "0",
                            "sample_id": sample_id,
                            "image_path": "images/tile.png",
                        }
                    )

        if cohort == "camelyon":
            for row_index in range(2_002):
                sample_id = f"camelyon-ood-{row_index}"
                metadata_rows.append(
                    {
                        "slide_id": sample_id,
                        "patch_id": "0",
                        "biological_class": "ood-label",
                        "medical_center": "ood-center",
                        "subset": "OOD",
                    }
                )
                index_rows.append(
                    {
                        "slide_id": sample_id,
                        "patch_id": "0",
                        "sample_id": sample_id,
                        "image_path": "images/tile.png",
                    }
                )

        pd.DataFrame(metadata_rows).to_csv(cohort_dir / spec["metadata"], index=False)
        pd.DataFrame(index_rows).to_csv(cohort_dir / "source_index.csv", index=False)
        (cohort_dir / "provenance.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cohort": cohort,
                    "sources": {
                        "dataset": {
                            "repository": spec["dataset_repository"],
                            "revision": spec["dataset_revision"],
                        },
                        "metadata": {
                            "repository": "bifold-pathomics/PathoROB",
                            "revision": METADATA_REVISION,
                        },
                    },
                }
            )
        )


def test_curate_pathorob_ri_views_emits_exact_balanced_manifests(tmp_path: Path):
    raw_root = tmp_path / "raw"
    _write_prepared_tree(raw_root)

    manifests = curate_pathorob_ri_views(raw_root, tmp_path / "curated")

    assert set(manifests) == set(COHORTS)
    for cohort, spec in COHORTS.items():
        dataset = pd.read_csv(manifests[cohort].dataset_csv)
        splits = pd.read_csv(manifests[cohort].splits_csv)
        expected_total = spec["labels"] * spec["centers"] * spec["per_cell"]

        assert list(dataset.columns) == [
            "sample_id",
            "image_path",
            "label",
            "medical_center",
            "group_id",
        ]
        assert len(dataset) == expected_total
        assert (
            dataset.groupby(["label", "medical_center"], sort=False)
            .size()
            .tolist()
            == [spec["per_cell"]] * (spec["labels"] * spec["centers"])
        )
        assert "slide_id" not in dataset.columns
        assert dataset["group_id"].tolist() == dataset["sample_id"].tolist()

        assert list(splits.columns) == ["sample_id", "split", "fold"]
        assert len(splits) == expected_total
        assert splits["sample_id"].tolist() == dataset["sample_id"].tolist()
        assert splits["split"].eq("test").all()
        assert splits["fold"].eq(0).all()

    camelyon = pd.read_csv(manifests["camelyon"].dataset_csv)
    assert not camelyon["sample_id"].str.contains("-ood-").any()


def test_curator_rejects_row_without_each_required_typed_neighbour(tmp_path: Path):
    raw_root = tmp_path / "raw"
    _write_prepared_tree(raw_root)
    cohort_root = raw_root / "camelyon"
    metadata = pd.read_csv(cohort_root / "camelyon.csv", dtype=str)
    source_index = pd.read_csv(cohort_root / "source_index.csv", dtype=str)
    target_id = "camelyon-l0-c0-r0"

    unsupported = (
        ((metadata["biological_class"] == "label-0") & (metadata["medical_center"] == "center-1"))
        | ((metadata["biological_class"] == "label-1") & (metadata["medical_center"] == "center-0"))
        | (source_index["sample_id"] == target_id)
    )
    affected_ids = set(source_index.loc[unsupported, "sample_id"])
    metadata_rows = metadata["slide_id"].isin(affected_ids)
    index_rows = source_index["sample_id"].isin(affected_ids)
    metadata.loc[metadata_rows, "patch_id"] = metadata.loc[metadata_rows, "slide_id"]
    source_index.loc[index_rows, "patch_id"] = source_index.loc[index_rows, "sample_id"]
    metadata.loc[metadata_rows, "slide_id"] = "blocked-group"
    source_index.loc[index_rows, "slide_id"] = "blocked-group"
    metadata.to_csv(cohort_root / "camelyon.csv", index=False)
    source_index.to_csv(cohort_root / "source_index.csv", index=False)

    with pytest.raises(
        ValueError,
        match=f"{target_id}.*same_label_other_center",
    ):
        curate_pathorob_ri_views(raw_root, tmp_path / "curated")


def test_curator_rejects_metadata_key_missing_from_source_index(tmp_path: Path):
    raw_root = tmp_path / "raw"
    _write_prepared_tree(raw_root)
    index_path = raw_root / "camelyon" / "source_index.csv"
    source_index = pd.read_csv(index_path, dtype=str)
    source_index.iloc[1:].to_csv(index_path, index=False)

    with pytest.raises(
        ValueError,
        match=r"missing.*\(slide_id, patch_id\).*camelyon-l0-c0-r0",
    ):
        curate_pathorob_ri_views(raw_root, tmp_path / "curated")


def test_curator_rejects_source_index_key_that_matches_multiple_rows(tmp_path: Path):
    raw_root = tmp_path / "raw"
    _write_prepared_tree(raw_root)
    index_path = raw_root / "camelyon" / "source_index.csv"
    source_index = pd.read_csv(index_path, dtype=str)
    source_index.loc[1, ["slide_id", "patch_id"]] = source_index.loc[
        0, ["slide_id", "patch_id"]
    ]
    source_index.to_csv(index_path, index=False)

    with pytest.raises(
        ValueError,
        match=r"source_index.csv.*duplicate.*\(slide_id, patch_id\).*camelyon-l0-c0-r0",
    ):
        curate_pathorob_ri_views(raw_root, tmp_path / "curated")


def test_curator_rejects_duplicate_metadata_join_key(tmp_path: Path):
    raw_root = tmp_path / "raw"
    _write_prepared_tree(raw_root)
    metadata_path = raw_root / "camelyon" / "camelyon.csv"
    metadata = pd.read_csv(metadata_path, dtype=str)
    metadata.loc[1, ["slide_id", "patch_id"]] = metadata.loc[
        0, ["slide_id", "patch_id"]
    ]
    metadata.to_csv(metadata_path, index=False)

    with pytest.raises(
        ValueError,
        match=r"camelyon.csv.*duplicate.*\(slide_id, patch_id\).*camelyon-l0-c0-r0",
    ):
        curate_pathorob_ri_views(raw_root, tmp_path / "curated")


def test_curator_rejects_missing_prepared_image(tmp_path: Path):
    raw_root = tmp_path / "raw"
    _write_prepared_tree(raw_root)
    index_path = raw_root / "camelyon" / "source_index.csv"
    source_index = pd.read_csv(index_path, dtype=str)
    source_index.loc[0, "image_path"] = "images/missing.png"
    source_index.to_csv(index_path, index=False)

    with pytest.raises(
        ValueError,
        match=r"camelyon-l0-c0-r0.*image.*does not exist",
    ):
        curate_pathorob_ri_views(raw_root, tmp_path / "curated")


def test_curator_rejects_sample_id_reused_across_cohorts(tmp_path: Path):
    raw_root = tmp_path / "raw"
    _write_prepared_tree(raw_root)
    index_path = raw_root / "tcga-4x4" / "source_index.csv"
    source_index = pd.read_csv(index_path, dtype=str)
    source_index.loc[0, "sample_id"] = "camelyon-l0-c0-r0"
    source_index.to_csv(index_path, index=False)

    with pytest.raises(
        ValueError,
        match=r"sample_id.*camelyon-l0-c0-r0.*both camelyon and tcga-4x4",
    ):
        curate_pathorob_ri_views(raw_root, tmp_path / "curated")


@pytest.mark.parametrize("field", ["sample_id", "slide_id", "patch_id"])
def test_curator_rejects_unsafe_identifier(field: str, tmp_path: Path):
    raw_root = tmp_path / "raw"
    _write_prepared_tree(raw_root)
    cohort_root = raw_root / "camelyon"
    index_path = cohort_root / "source_index.csv"
    metadata_path = cohort_root / "camelyon.csv"
    source_index = pd.read_csv(index_path, dtype=str)
    metadata = pd.read_csv(metadata_path, dtype=str)
    source_index.loc[0, field] = "../unsafe"
    if field in {"slide_id", "patch_id"}:
        metadata.loc[0, field] = "../unsafe"
    source_index.to_csv(index_path, index=False)
    metadata.to_csv(metadata_path, index=False)

    with pytest.raises(
        ValueError,
        match=rf"unsafe {field}.*\.\./unsafe",
    ):
        curate_pathorob_ri_views(raw_root, tmp_path / "curated")


@pytest.mark.parametrize(
    ("source_field", "manifest_field"),
    [
        ("biological_class", "label"),
        ("medical_center", "medical_center"),
        ("slide_id", "group_id"),
    ],
)
def test_curator_rejects_empty_required_metadata_value(
    source_field: str,
    manifest_field: str,
    tmp_path: Path,
):
    raw_root = tmp_path / "raw"
    _write_prepared_tree(raw_root)
    metadata_path = raw_root / "camelyon" / "camelyon.csv"
    metadata = pd.read_csv(metadata_path, dtype=str)
    metadata.loc[0, source_field] = ""
    metadata.to_csv(metadata_path, index=False)

    with pytest.raises(
        ValueError,
        match=rf"empty {manifest_field}",
    ):
        curate_pathorob_ri_views(raw_root, tmp_path / "curated")


def test_curator_rejects_mismatched_prepared_data_revision(tmp_path: Path):
    raw_root = tmp_path / "raw"
    _write_prepared_tree(raw_root)
    provenance_path = raw_root / "camelyon" / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["sources"]["dataset"]["revision"] = "unknown-revision"
    provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(
        ValueError,
        match=r"camelyon.*dataset revision.*unknown-revision.*expected b2e762",
    ):
        curate_pathorob_ri_views(raw_root, tmp_path / "curated")


def test_recuration_is_byte_identical_and_preserves_metadata_order(tmp_path: Path):
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "curated"
    _write_prepared_tree(raw_root)
    for cohort in COHORTS:
        index_path = raw_root / cohort / "source_index.csv"
        source_index = pd.read_csv(index_path, dtype=str)
        source_index.iloc[::-1].to_csv(index_path, index=False)

    curate_pathorob_ri_views(raw_root, output_root)
    first = {
        path.relative_to(output_root): path.read_bytes()
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
    }
    curate_pathorob_ri_views(raw_root, output_root)
    second = {
        path.relative_to(output_root): path.read_bytes()
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
    }

    assert first == second
    camelyon = pd.read_csv(output_root / "camelyon" / "dataset.csv")
    assert camelyon["sample_id"].iloc[:2].tolist() == [
        "camelyon-l0-c0-r0",
        "camelyon-l0-c0-r1",
    ]


def test_curator_rejects_extra_center(tmp_path: Path):
    raw_root = tmp_path / "raw"
    _write_prepared_tree(raw_root)
    metadata_path = raw_root / "camelyon" / "camelyon.csv"
    metadata = pd.read_csv(metadata_path, dtype=str)
    metadata.loc[0, "medical_center"] = "extra-center"
    metadata.to_csv(metadata_path, index=False)

    with pytest.raises(
        ValueError,
        match=r"2 label x 2 center grid with 5100 rows per cell",
    ):
        curate_pathorob_ri_views(raw_root, tmp_path / "curated")


def test_curator_rejects_duplicate_sample_id_within_cohort(tmp_path: Path):
    raw_root = tmp_path / "raw"
    _write_prepared_tree(raw_root)
    index_path = raw_root / "camelyon" / "source_index.csv"
    source_index = pd.read_csv(index_path, dtype=str)
    source_index.loc[1, "sample_id"] = source_index.loc[0, "sample_id"]
    source_index.to_csv(index_path, index=False)

    with pytest.raises(
        ValueError,
        match=r"camelyon.*duplicate sample_id.*camelyon-l0-c0-r0",
    ):
        curate_pathorob_ri_views(raw_root, tmp_path / "curated")
