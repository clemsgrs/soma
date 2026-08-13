"""Deterministic curators for the three CRoMa benchmark cohorts (PathoROB tile data)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from soma.curation.manifest import CuratedManifest, write_manifest


@dataclass(frozen=True)
class _View:
    metadata_filename: str
    labels: int
    centers: int
    rows_per_cell: int
    select_id_subset: bool
    dataset_repository: str
    dataset_revision: str

    @property
    def total_rows(self) -> int:
        return self.labels * self.centers * self.rows_per_cell


@dataclass(frozen=True)
class _PreparedView:
    cohort: str
    dataset_rows: list[dict[str, str]]
    provenance: dict[str, Any]


_VIEWS = {
    "camelyon": _View(
        "camelyon.csv",
        2,
        2,
        5_100,
        True,
        "bifold-pathomics/PathoROB-camelyon",
        "b2e762542abbf85dde3f23ec70a2bf1848dcf5c8",
    ),
    "tcga-4x4": _View(
        "tcga_4x4.csv",
        4,
        4,
        360,
        True,
        "bifold-pathomics/PathoROB-tcga",
        "6e1dbd4306ebee9759b32503914523e84bddabd0",
    ),
    "tolkach-esca": _View(
        "tolkach_esca_reduced.csv",
        6,
        3,
        500,
        False,
        "bifold-pathomics/PathoROB-tolkach_esca",
        "c42219a2c168c5995e44487f5747bdacaf4bc2da",
    ),
}
_METADATA_REPOSITORY = "bifold-pathomics/PathoROB"
_METADATA_REVISION = "6583cf0b0d902c8cc032308262fa3a3befdc0687"
_MIN_TYPED_NEIGHBOURS = 5


def curate_croma_view(
    raw_root: str | Path,
    output_dir: str | Path,
    *,
    cohort: str,
) -> CuratedManifest:
    """Curate one prepared cohort into one Soma Manifest."""
    view = _get_view(cohort)
    prepared = _prepare_view(Path(raw_root), cohort, view, sample_id_owners={})
    return _write_view(prepared, Path(output_dir))


def curate_croma_views(
    raw_root: str | Path,
    output_root: str | Path,
) -> dict[str, CuratedManifest]:
    """Orchestrate all cohort curators with family-wide ID validation."""
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    sample_id_owners: dict[str, str] = {}
    prepared_views = [
        _prepare_view(raw_root, cohort, view, sample_id_owners)
        for cohort, view in _VIEWS.items()
    ]
    return {
        prepared.cohort: _write_view(
            prepared,
            output_root / prepared.cohort,
        )
        for prepared in prepared_views
    }


def _get_view(cohort: str) -> _View:
    try:
        return _VIEWS[cohort]
    except KeyError:
        raise ValueError(
            f"Unknown CRoMa cohort {cohort!r}; expected one of "
            f"{list(_VIEWS)}."
        ) from None


def _prepare_view(
    raw_root: Path,
    cohort: str,
    view: _View,
    sample_id_owners: dict[str, str],
) -> _PreparedView:
    cohort_root = raw_root / cohort
    provenance = _load_and_validate_provenance(cohort_root, cohort, view)
    metadata = pd.read_csv(
        cohort_root / view.metadata_filename,
        dtype=str,
        keep_default_na=False,
    )
    _validate_required_metadata_values(metadata, cohort, view.metadata_filename)
    _validate_identifiers(
        metadata, cohort, view.metadata_filename, ("slide_id", "patch_id")
    )
    _validate_unique_keys(metadata, cohort, view.metadata_filename)
    if view.select_id_subset:
        metadata = metadata.loc[metadata["subset"] == "ID"].copy()

    source_index = pd.read_csv(
        cohort_root / "source_index.csv",
        dtype=str,
        keep_default_na=False,
    )
    _validate_identifiers(
        source_index,
        cohort,
        "source_index.csv",
        ("slide_id", "patch_id", "sample_id"),
    )
    _validate_unique_keys(source_index, cohort, "source_index.csv")
    joined = metadata.merge(
        source_index,
        on=["slide_id", "patch_id"],
        how="left",
        sort=False,
    )
    _validate_join_matches(joined, cohort)
    _validate_sample_id_uniqueness(joined, cohort, sample_id_owners)
    _validate_balance(joined, cohort, view)
    _validate_typed_neighbour_support(joined, cohort)
    _validate_images(joined, cohort, cohort_root)
    dataset_rows = [
        {
            "sample_id": row.sample_id,
            "image_path": str(_resolve_image_path(cohort_root, row.image_path)),
            "label": row.biological_class,
            "medical_center": row.medical_center,
            "group_id": row.slide_id,
        }
        for row in joined.itertuples(index=False)
    ]
    return _PreparedView(
        cohort=cohort,
        dataset_rows=dataset_rows,
        provenance=provenance,
    )


def _write_view(prepared: _PreparedView, output_dir: Path) -> CuratedManifest:
    split_rows = [
        {"sample_id": row["sample_id"], "split": "test", "fold": 0}
        for row in prepared.dataset_rows
    ]
    return write_manifest(
        output_dir,
        dataset_type="tile",
        dataset_rows=prepared.dataset_rows,
        split_rows=split_rows,
        summary={
            "cohort": prepared.cohort,
            "dataset_type": "tile",
            "num_samples": len(prepared.dataset_rows),
            "prepared_data_provenance": prepared.provenance,
            "view": "robustness_index",
        },
    )


def _resolve_image_path(cohort_root: Path, image_path: Any) -> Path:
    path = Path(str(image_path))
    if not path.is_absolute():
        path = cohort_root / path
    return path.resolve()


def _load_and_validate_provenance(
    cohort_root: Path,
    cohort: str,
    view: _View,
) -> dict[str, Any]:
    provenance = json.loads((cohort_root / "provenance.json").read_text())
    schema_version = provenance.get("schema_version")
    if schema_version != 1:
        raise ValueError(
            f"PathoROB {cohort} provenance schema_version {schema_version!r} "
            "does not match expected 1."
        )
    declared_cohort = provenance.get("cohort")
    if declared_cohort != cohort:
        raise ValueError(
            f"PathoROB {cohort} provenance declares cohort {declared_cohort!r}; "
            f"expected {cohort!r}."
        )
    expected_sources = {
        "dataset": (view.dataset_repository, view.dataset_revision),
        "metadata": (_METADATA_REPOSITORY, _METADATA_REVISION),
    }
    for source_name, (expected_repository, expected_revision) in expected_sources.items():
        try:
            source = provenance["sources"][source_name]
            repository = source["repository"]
            revision = source["revision"]
        except (KeyError, TypeError):
            raise ValueError(
                f"PathoROB {cohort} provenance.json is missing the prepared "
                f"{source_name} source repository or revision."
            ) from None
        if repository != expected_repository:
            raise ValueError(
                f"PathoROB {cohort} prepared {source_name} repository "
                f"{repository!r} does not match expected {expected_repository!r}."
            )
        if revision != expected_revision:
            raise ValueError(
                f"PathoROB {cohort} prepared {source_name} revision {revision!r} "
                f"does not match expected {expected_revision}."
            )
    return provenance


def _validate_join_matches(joined: pd.DataFrame, cohort: str) -> None:
    missing = joined["sample_id"].isna()
    if not missing.any():
        return
    row = joined.loc[missing].iloc[0]
    raise ValueError(
        f"PathoROB {cohort} source_index.csv is missing (slide_id, patch_id) key "
        f"({row['slide_id']!r}, {row['patch_id']!r}) required by metadata."
    )


def _validate_unique_keys(frame: pd.DataFrame, cohort: str, source: str) -> None:
    duplicated = frame.duplicated(["slide_id", "patch_id"], keep=False)
    if not duplicated.any():
        return
    row = frame.loc[duplicated].iloc[0]
    raise ValueError(
        f"PathoROB {cohort} {source} has duplicate (slide_id, patch_id) key "
        f"({row['slide_id']!r}, {row['patch_id']!r}); the join would match multiple rows."
    )


def _validate_images(joined: pd.DataFrame, cohort: str, cohort_root: Path) -> None:
    for image_path in dict.fromkeys(joined["image_path"]):
        resolved = _resolve_image_path(cohort_root, image_path)
        if resolved.is_file():
            continue
        sample_id = joined.loc[joined["image_path"] == image_path, "sample_id"].iloc[0]
        raise ValueError(
            f"PathoROB {cohort} sample_id {sample_id!r} image {str(resolved)!r} "
            "does not exist or is not a file."
        )


def _validate_sample_id_uniqueness(
    joined: pd.DataFrame,
    cohort: str,
    owners: dict[str, str],
) -> None:
    duplicated = joined["sample_id"].duplicated(keep=False)
    if duplicated.any():
        sample_id = joined.loc[duplicated, "sample_id"].iloc[0]
        raise ValueError(
            f"PathoROB {cohort} has duplicate sample_id {sample_id!r}."
        )

    for sample_id in joined["sample_id"]:
        previous_cohort = owners.get(sample_id)
        if previous_cohort is not None:
            raise ValueError(
                f"PathoROB sample_id {sample_id!r} occurs in both "
                f"{previous_cohort} and {cohort}; IDs must be unique across the family."
            )
        owners[sample_id] = cohort


_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _validate_identifiers(
    frame: pd.DataFrame,
    cohort: str,
    source: str,
    fields: tuple[str, ...],
) -> None:
    for field in fields:
        for value in frame[field]:
            if _SAFE_IDENTIFIER.fullmatch(value):
                continue
            raise ValueError(
                f"PathoROB {cohort} {source} has unsafe {field} {value!r}; "
                "identifiers must be non-empty and filename-safe."
            )


def _validate_required_metadata_values(
    metadata: pd.DataFrame,
    cohort: str,
    source: str,
) -> None:
    output_fields = {
        "biological_class": "label",
        "medical_center": "medical_center",
        "slide_id": "group_id",
    }
    for source_field, output_field in output_fields.items():
        empty = metadata[source_field].astype(str).str.strip().eq("")
        if not empty.any():
            continue
        row_index = int(empty.to_numpy().nonzero()[0][0])
        raise ValueError(
            f"PathoROB {cohort} {source} row {row_index} has empty {output_field} "
            f"(source column {source_field!r})."
        )


def _validate_balance(joined: pd.DataFrame, cohort: str, view: _View) -> None:
    if len(joined) != view.total_rows:
        raise ValueError(
            f"CRoMa {cohort} cohort requires exactly {view.total_rows} rows; "
            f"found {len(joined)}."
        )

    num_labels = joined["biological_class"].nunique()
    num_centers = joined["medical_center"].nunique()
    if num_labels != view.labels or num_centers != view.centers:
        raise ValueError(
            f"CRoMa {cohort} cohort requires exactly {view.labels} labels and "
            f"{view.centers} centers; found {num_labels} labels and "
            f"{num_centers} centers."
        )

    cell_counts = joined.groupby(
        ["biological_class", "medical_center"], sort=False, dropna=False
    ).size()
    expected_cells = view.labels * view.centers
    if len(cell_counts) != expected_cells or not cell_counts.eq(view.rows_per_cell).all():
        raise ValueError(
            f"CRoMa {cohort} cohort requires a {view.labels} label x "
            f"{view.centers} center grid with {view.rows_per_cell} rows per cell."
        )


def _validate_typed_neighbour_support(
    joined: pd.DataFrame,
    cohort: str,
) -> None:
    label = joined["biological_class"]
    center = joined["medical_center"]
    group = joined["slide_id"]

    label_count = joined.groupby(label, sort=False)["sample_id"].transform("size")
    center_count = joined.groupby(center, sort=False)["sample_id"].transform("size")
    cell_count = joined.groupby([label, center], sort=False)["sample_id"].transform(
        "size"
    )
    group_label_count = joined.groupby([group, label], sort=False)[
        "sample_id"
    ].transform("size")
    group_center_count = joined.groupby([group, center], sort=False)[
        "sample_id"
    ].transform("size")
    group_cell_count = joined.groupby([group, label, center], sort=False)[
        "sample_id"
    ].transform("size")

    eligible = {
        "same_label_other_center": (
            label_count - cell_count - group_label_count + group_cell_count
        ),
        "other_label_same_center": (
            center_count - cell_count - group_center_count + group_cell_count
        ),
    }
    unsupported = pd.DataFrame(
        {name: counts < _MIN_TYPED_NEIGHBOURS for name, counts in eligible.items()}
    )
    if not unsupported.any(axis=None):
        return

    row_index = int(unsupported.any(axis=1).to_numpy().nonzero()[0][0])
    missing_types = [name for name in eligible if unsupported.iloc[row_index][name]]
    sample_id = joined.iloc[row_index]["sample_id"]
    raise ValueError(
        f"PathoROB {cohort} sample_id {sample_id!r} lacks at least "
        f"{_MIN_TYPED_NEIGHBOURS} "
        "eligible non-same-group neighbour(s) for: "
        f"{', '.join(missing_types)}."
    )
