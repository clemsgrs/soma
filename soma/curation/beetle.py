"""Curator for the BEETLE breast-cancer segmentation challenge (slide-manifest path).

Like the EVA and OCELOT curators, this emits Soma's unified Manifest
(``dataset.csv`` + ``splits.csv`` + ``summary.json``) from locally prepared raw data and
**does not download anything** (see ``data/beetle/download.sh``).

BEETLE is a *segmentation* dataset, so the supervision column is ``label_mask_path``: one row per
development WSI pairs the slide (``image_path``) with its multiresolution annotation raster
(``label_mask_path``). No tiles are materialized here — soma runs hs2p annotation sampling over
these slides at train time (``masks:`` / ``sampling:`` in ``examples/segmentation_beetle.yaml``)
to derive ROIs, so the cached slide-manifest path is the sole BEETLE recipe.

Splits preserve BEETLE's predefined CV folds (soma never partitions; the curator verifies that
all slides from one patient share a fold, and sampled ROIs inherit their parent slide's split).
For fold ``k``: a slide whose
``validation_fold == k`` is ``test``, ``== (k+1) % n_folds`` is ``tune``, else ``train``.

Run as ``python -m soma.curation.beetle`` (or via ``examples/make_beetle_manifest.py``).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from soma.curation.manifest import CuratedManifest, write_manifest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BEETLE_ROOT = _REPO_ROOT / "data" / "beetle"

# BEETLE pixel vocabulary (label_map.json): name -> raw mask pixel value. This is the
# masks.pixel_mapping the soma config carries; "background" (0, unannotated) is the ignore
# label. Kept here so the manifest's coverage scan uses the same class scheme as training.
PIXEL_MAPPING = {
    "background": 0,
    "other": 1,
    "non_invasive_epithelium": 2,
    "invasive_epithelium": 3,
    "necrosis": 4,
}
ANNOTATED_FRACTION_MIN = 0.05  # >=5% annotated to keep a tile
CANONICAL_MPP = 0.5
CROP_SIZE = 512
IGNORE_INDEX = 255
FULL_COHORT_SLIDES = 587
FULL_COHORT_PATIENTS = 527

_TIGER_TCGA_PREFIX = "wsirois/wsi-level-annotations/images"
_TIGER_BASE_URL = "https://tiger-training.s3.amazonaws.com"
_TIGER_DATA_URL = "https://tiger.grand-challenge.org/Data/"
_SOURCE_RULES = {
    "tcga": {
        "pattern": re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-"),
        "provenance": "derived_tcga_barcode",
        "rule": "use the first three hyphen-separated TCGA barcode fields",
        "source_url": _TIGER_DATA_URL,
        "reconstruct_public_wsi_path": True,
    },
    "rumc": {
        "pattern": re.compile(r"^(TC_S\d+_P\d+)_C\d+_B\d+$"),
        "provenance": "derived_rumc_name",
        "rule": "use the TC_S##_P###### prefix before case/block fields",
        "source_url": _TIGER_DATA_URL,
        "reconstruct_public_wsi_path": True,
    },
    "jb": {
        "pattern": re.compile(r"^(\d+)[BS]$"),
        "provenance": "derived_jb_name",
        "rule": "use the numeric patient prefix before the B/S specimen suffix",
        "source_url": _TIGER_DATA_URL,
        "reconstruct_public_wsi_path": True,
    },
}
_NATIVE_LEVEL_0_EXCEPTIONS = {
    "TCGA-OL-A66I-01Z-00-DX1.8CE9DCAB-98D3-4163-94AC-1557D86C1E25": {
        "patient_id": "TCGA-OL-A66I",
        "organizer_fold": "fold0",
    },
    "TCGA-OL-A66P-01Z-00-DX1.5ADD0D6D-37C6-4BC9-8C2B-64DB18BE99B3": {
        "patient_id": "TCGA-OL-A66P",
        "organizer_fold": "fold2",
    },
    "TCGA-OL-A6VO-01Z-00-DX1.291D54D6-EBAF-4622-BD42-97AA5997F014": {
        "patient_id": "TCGA-OL-A6VO",
        "organizer_fold": "fold1",
    },
}
_EXPECTED_NATIVE_LEVEL_0_SPACING = 0.657476464
_NATIVE_SPACING_ABS_TOLERANCE = 1e-6
_NATIVE_LEVEL_0_READ_POLICY = "native_level_0_no_upsample"
_PATIENT_ID_RULES = {
    "released_data_overview": {
        "rule": "preserve each non-empty released patient_id verbatim",
        "source_url": "https://zenodo.org/records/16812932/files/data_overview.csv",
    },
    **{
        rule["provenance"]: {
            "rule": rule["rule"],
            "source_url": rule["source_url"],
        }
        for rule in _SOURCE_RULES.values()
    },
}


def read_dev_rows(overview_csv: Path) -> list[dict]:
    """Read all released dev rows and reconstruct documented public WSI paths.

    Local-file validation is deliberately separate: publication validates all rows, while the
    explicit smoke mode first selects its bounded rows and validates only those files.
    """
    rows = [dict(r) for r in csv.DictReader(overview_csv.open()) if r["split"] == "development"]
    if not rows:
        raise RuntimeError(f"No development rows found in {overview_csv}")

    for row in rows:
        source_rule = _SOURCE_RULES.get(row["source"].strip().lower())
        if (
            not row["wsi_path"].strip()
            and source_rule is not None
            and source_rule["reconstruct_public_wsi_path"]
        ):
            row["wsi_path"] = f"images/development/wsis/{row['name']}.tif"
    return rows


def _validate_local_files(rows: list[dict], beetle_root: Path, *, publication: bool) -> None:
    """Require every selected WSI + mask pairing before any manifest is written."""

    def has(rel: str) -> bool:
        return bool(rel.strip()) and (beetle_root / rel).is_file()

    missing = [
        row["name"]
        for row in rows
        if not has(row["wsi_path"]) or not has(row["annotation_mask_path"])
    ]
    if missing:
        cohort = "primary cohort" if publication else "selected non-publication smoke subset"
        raise ValueError(
            "BEETLE development rows are missing local WSI or annotation file(s): "
            f"{missing}. The {cohort} must be locally complete."
        )


def _resolve_patient_identity(row: dict) -> tuple[str, str]:
    released = row["patient_id"].strip()
    if released:
        return released, "released_data_overview"

    source = row["source"].strip().lower()
    source_rule = _SOURCE_RULES.get(source)
    match = (
        source_rule["pattern"].match(row["name"].strip())
        if source_rule is not None
        else None
    )
    if match is None:
        raise ValueError(
            f"Cannot recover missing patient_id for BEETLE slide {row['name']!r} "
            f"from source {row['source']!r}."
        )
    return match.group(1), source_rule["provenance"]


def select_subset(rows: list[dict], n: int | None) -> list[dict]:
    """Deterministically pick ~n dev WSIs balanced across folds."""
    if n is None:
        return rows
    by_fold: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_fold[r["validation_fold"]].append(r)
    folds = sorted(by_fold)
    per_fold = max(1, round(n / len(folds)))
    selected: list[dict] = []
    for fold in folds:
        bucket = sorted(by_fold[fold], key=lambda r: (r["source"], r["name"]))
        selected.extend(bucket[:per_fold])
    return selected[:n]


def build_dataset_rows(
    rows: list[dict],
    beetle_root: Path,
    *,
    native_spacings: dict[str, float] | None = None,
) -> list[dict]:
    """One unified-schema dataset row per slide (supervision column = ``label_mask_path``)."""
    dataset_rows: list[dict] = []
    for r in rows:
        patient_id, patient_id_provenance = _resolve_patient_identity(r)
        measured_spacing = (native_spacings or {}).get(r["name"])
        wsi = (beetle_root / r["wsi_path"]).resolve()
        mask = (beetle_root / r["annotation_mask_path"]).resolve()
        dataset_rows.append(
            {
                "sample_id": r["name"],
                "image_path": str(wsi),
                "label_mask_path": str(mask),
                "patient_id": patient_id,
                "patient_id_provenance": patient_id_provenance,
                "source": r["source"],
                "specimen_type": r["specimen_type"],
                "validation_fold": r["validation_fold"],
                "spacing_at_level_0": measured_spacing if measured_spacing is not None else "",
                "read_policy": (
                    _NATIVE_LEVEL_0_READ_POLICY
                    if measured_spacing is not None
                    else "spacing_aware"
                ),
                "spacing_provenance": (
                    "local_tiff_level_0_resolution_tags"
                    if measured_spacing is not None
                    else ""
                ),
                "in_native_spacing_sensitivity_subset": measured_spacing is None,
            }
        )
    return dataset_rows


def build_split_rows(dataset_rows: list[dict]) -> list[dict]:
    """Slide-level CV splits from BEETLE's fold rotation (test/tune/train)."""
    fold_ids = sorted({r["validation_fold"] for r in dataset_rows})
    fold_nums = sorted(int(f.replace("fold", "")) for f in fold_ids)
    n_folds = len(fold_nums)
    split_rows: list[dict] = []
    for k in fold_nums:
        tune_fold = (k + 1) % n_folds
        for r in dataset_rows:
            wf = int(r["validation_fold"].replace("fold", ""))
            if wf == k:
                split = "test"
            elif wf == tune_fold:
                split = "tune"
            else:
                split = "train"
            split_rows.append({"sample_id": r["sample_id"], "split": split, "fold": k})
    return split_rows


def _validate_full_cohort(dataset_rows: list[dict]) -> None:
    provenance_by_patient: dict[str, set[str]] = defaultdict(set)
    for row in dataset_rows:
        provenance_by_patient[row["patient_id"]].add(row["patient_id_provenance"])
    collisions = {
        patient_id: sorted(provenance)
        for patient_id, provenance in provenance_by_patient.items()
        if len(provenance) != 1
    }
    if collisions:
        details = "; ".join(
            f"{patient_id}: {', '.join(provenance)}"
            for patient_id, provenance in sorted(collisions.items())
        )
        raise ValueError(f"BEETLE patient_id recovery collision(s): {details}.")

    num_slides = len(dataset_rows)
    num_patients = len({row["patient_id"] for row in dataset_rows})
    if (num_slides, num_patients) != (FULL_COHORT_SLIDES, FULL_COHORT_PATIENTS):
        raise ValueError(
            "The primary BEETLE cohort must contain exactly "
            f"{FULL_COHORT_SLIDES} slides / {FULL_COHORT_PATIENTS} patients before "
            f"training; resolved {num_slides} slides / {num_patients} patients."
        )

    observed_folds = {row["validation_fold"] for row in dataset_rows}
    expected_folds = {f"fold{fold}" for fold in range(5)}
    if observed_folds != expected_folds:
        raise ValueError(
            "The full BEETLE cohort must preserve organizer folds fold0 through fold4; "
            f"resolved {sorted(observed_folds)}."
        )

    folds_by_patient: dict[str, set[str]] = defaultdict(set)
    for row in dataset_rows:
        folds_by_patient[row["patient_id"]].add(row["validation_fold"])
    leaking = {
        patient_id: sorted(folds)
        for patient_id, folds in folds_by_patient.items()
        if len(folds) != 1
    }
    if leaking:
        details = "; ".join(
            f"{patient_id}: {', '.join(folds)}"
            for patient_id, folds in sorted(leaking.items())
        )
        raise ValueError(f"BEETLE patient(s) cross organizer folds: {details}.")


def _read_level_0_tiff_spacing(path: Path) -> float:
    """Read isotropic level-0 spacing from TIFF resolution tags, in µm/px."""
    previous_pixel_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as image:
            x_resolution = float(image.tag_v2[282])
            y_resolution = float(image.tag_v2[283])
            resolution_unit = int(image.tag_v2[296])
    except (KeyError, OSError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(
            f"Cannot read level-0 TIFF resolution metadata for BEETLE slide {path.name!r}."
        ) from error
    finally:
        Image.MAX_IMAGE_PIXELS = previous_pixel_limit

    micrometres_per_unit = {2: 25_400.0, 3: 10_000.0}.get(resolution_unit)
    if micrometres_per_unit is None or x_resolution <= 0 or y_resolution <= 0:
        raise ValueError(
            f"Invalid level-0 TIFF resolution metadata for BEETLE slide {path.name!r}: "
            f"XResolution={x_resolution}, YResolution={y_resolution}, "
            f"ResolutionUnit={resolution_unit}."
        )
    x_spacing = micrometres_per_unit / x_resolution
    y_spacing = micrometres_per_unit / y_resolution
    if not math.isclose(x_spacing, y_spacing, rel_tol=0, abs_tol=1e-9):
        raise ValueError(
            f"BEETLE native-spacing exception {path.name!r} is anisotropic: "
            f"{x_spacing} x {y_spacing} µm/px."
        )
    return round((x_spacing + y_spacing) / 2, 9)


def _measure_native_spacing_exceptions(rows: list[dict], beetle_root: Path) -> dict[str, float]:
    """Validate authoritative exception identities and measure their local level-0 spacing."""
    rows_by_id = {row["name"]: row for row in rows}
    if len(rows_by_id) != len(rows):
        raise ValueError("BEETLE development slide names must be unique.")
    missing = sorted(set(_NATIVE_LEVEL_0_EXCEPTIONS) - set(rows_by_id))
    if missing:
        raise ValueError(f"BEETLE native-spacing exception slide(s) are missing: {missing}.")

    for sample_id, expected in _NATIVE_LEVEL_0_EXCEPTIONS.items():
        row = rows_by_id[sample_id]
        patient_id, _ = _resolve_patient_identity(row)
        observed = {
            "patient_id": patient_id,
            "organizer_fold": row["validation_fold"],
            "source": row["source"],
            "image_name": Path(row["wsi_path"]).name,
        }
        wanted = {
            **expected,
            "source": "tcga",
            "image_name": f"{sample_id}.tif",
        }
        if observed != wanted:
            raise ValueError(
                f"BEETLE native-spacing exception {sample_id!r} disagrees with "
                f"authoritative TIGER/organizer identity: expected {wanted}, got {observed}."
            )

    measured = {
        sample_id: _read_level_0_tiff_spacing(beetle_root / rows_by_id[sample_id]["wsi_path"])
        for sample_id in _NATIVE_LEVEL_0_EXCEPTIONS
    }
    unexpected = {
        sample_id: spacing
        for sample_id, spacing in measured.items()
        if not math.isclose(
            spacing,
            _EXPECTED_NATIVE_LEVEL_0_SPACING,
            rel_tol=0,
            abs_tol=_NATIVE_SPACING_ABS_TOLERANCE,
        )
    }
    if unexpected:
        raise ValueError(
            "BEETLE native-spacing exception TIFF metadata disagrees with the released "
            f"0.657476-µm/px cohort: {unexpected}."
        )
    return measured


def _validate_native_spacing_policy(dataset_rows: list[dict]) -> None:
    """Require exactly the three measured exceptions to carry the native-level-0 policy."""
    exception_rows = {
        row["sample_id"]: row
        for row in dataset_rows
        if row["read_policy"] == _NATIVE_LEVEL_0_READ_POLICY
    }
    if set(exception_rows) != set(_NATIVE_LEVEL_0_EXCEPTIONS):
        raise ValueError(
            "BEETLE native-level-0 read policy must apply to exactly the three authoritative "
            f"exceptions; got {sorted(exception_rows)}."
        )
    for sample_id, row in exception_rows.items():
        if (
            row["spacing_provenance"] != "local_tiff_level_0_resolution_tags"
            or not math.isclose(
                float(row["spacing_at_level_0"]),
                _EXPECTED_NATIVE_LEVEL_0_SPACING,
                rel_tol=0,
                abs_tol=_NATIVE_SPACING_ABS_TOLERANCE,
            )
            or row["in_native_spacing_sensitivity_subset"]
        ):
            raise ValueError(f"Invalid native-spacing read policy metadata for {sample_id!r}.")


def _validate_sensitivity_subset(dataset_rows: list[dict]) -> None:
    sensitivity_rows = [
        row for row in dataset_rows if row["in_native_spacing_sensitivity_subset"]
    ]
    observed = (
        len(sensitivity_rows),
        len({row["patient_id"] for row in sensitivity_rows}),
    )
    if observed != (584, 524):
        raise ValueError(
            "The derived native-spacing evaluation sensitivity subset must contain exactly "
            f"584 slides / 524 patients; resolved {observed[0]} slides / "
            f"{observed[1]} patients."
        )


def _write_coverage(dataset_csv: Path, out_path: Path) -> None:
    """Per-slide per-class annotation coverage (soma.curation.segmentation_coverage)."""
    from soma.curation.segmentation_coverage import summarize_coverage, write_coverage_csv

    min_cov = {k: ANNOTATED_FRACTION_MIN for k in PIXEL_MAPPING if k != "background"}
    coverage = summarize_coverage(
        dataset_csv,
        pixel_mapping=PIXEL_MAPPING,
        min_coverage=min_cov,
        tile_size_px=CROP_SIZE,
        spacing_um=CANONICAL_MPP,
    )
    write_coverage_csv(out_path, coverage)
    print(f"Wrote coverage for {len(coverage)} slides to {out_path}")


def curate_beetle_slide_manifest(
    overview_csv: str | Path,
    beetle_root: str | Path,
    output_dir: str | Path,
    *,
    slides: int | None = None,
    coverage: bool = False,
) -> CuratedManifest:
    """Curate the BEETLE development slides into Soma's unified segmentation Manifest.

    Args:
        overview_csv: ``data_overview.csv`` listing every BEETLE slide + annotation.
        beetle_root: Root the CSV's relative ``wsi_path`` / ``annotation_mask_path`` resolve against.
        output_dir: Directory where ``dataset.csv``, ``splits.csv``, ``summary.json`` (and,
            when ``coverage`` is set, ``coverage.csv``) are written. A smoke subset may not
            use the canonical ``<beetle_root>/curated_slide_manifest`` publication path.
        slides: Explicitly create a non-publication smoke subset balanced across folds. The
            default ``None`` is the only publication path and requires exactly 587 slides / 527
            patients; smoke output is marked ``non_publication_smoke_subset`` in ``summary.json``.
        coverage: When ``True``, also scan per-slide per-class annotation coverage
            (requires hs2p + readable WSIs); informs split authoring only.

    Returns:
        A :class:`~soma.curation.manifest.CuratedManifest` for the generated files.
    """
    overview_csv = Path(overview_csv)
    beetle_root = Path(beetle_root)
    output_dir = Path(output_dir)
    is_full_cohort = slides is None
    canonical_output = beetle_root / "curated_slide_manifest"
    if not is_full_cohort and output_dir.resolve() == canonical_output.resolve():
        raise ValueError(
            "A non-publication smoke subset cannot use the canonical BEETLE manifest path; "
            "choose an explicit smoke output directory."
        )

    rows = read_dev_rows(overview_csv)
    chosen = select_subset(rows, slides)
    _validate_local_files(chosen, beetle_root, publication=is_full_cohort)
    native_spacings = (
        _measure_native_spacing_exceptions(chosen, beetle_root) if is_full_cohort else {}
    )
    dataset_rows = build_dataset_rows(
        chosen,
        beetle_root,
        native_spacings=native_spacings,
    )
    if is_full_cohort:
        _validate_native_spacing_policy(dataset_rows)
        _validate_full_cohort(dataset_rows)
        _validate_sensitivity_subset(dataset_rows)
    split_rows = build_split_rows(dataset_rows)

    exception_rows = {
        row["sample_id"]: row
        for row in dataset_rows
        if row["read_policy"] == _NATIVE_LEVEL_0_READ_POLICY
    }
    excluded_sample_ids = [
        sample_id for sample_id in _NATIVE_LEVEL_0_EXCEPTIONS if sample_id in exception_rows
    ]
    excluded_patient_ids = [exception_rows[sample_id]["patient_id"] for sample_id in excluded_sample_ids]
    sensitivity_rows = [
        row for row in dataset_rows if row["in_native_spacing_sensitivity_subset"]
    ]
    patient_id_method_counts = Counter(
        row["patient_id_provenance"] for row in dataset_rows
    )
    summary = {
        "dataset": "BEETLE (breast-cancer segmentation, slide manifest)",
        "dataset_type": "segmentation",
        "num_slides": len(dataset_rows),
        "num_patients": len({row["patient_id"] for row in dataset_rows}),
        "cohort": {
            "kind": (
                "primary_full_cohort" if is_full_cohort else "non_publication_smoke_subset"
            ),
            "num_slides": len(dataset_rows),
            "num_patients": len({row["patient_id"] for row in dataset_rows}),
        },
        "patient_id_resolution": {
            method: {**provenance, "num_slides": patient_id_method_counts[method]}
            for method, provenance in _PATIENT_ID_RULES.items()
            if patient_id_method_counts[method]
        },
        "spacing_exceptions": [
            {
                "sample_id": sample_id,
                "patient_id": exception_rows[sample_id]["patient_id"],
                "organizer_fold": exception_rows[sample_id]["validation_fold"],
                "measured_spacing_um_per_px": exception_rows[sample_id][
                    "spacing_at_level_0"
                ],
                "read_policy": _NATIVE_LEVEL_0_READ_POLICY,
                "source_object": f"{_TIGER_TCGA_PREFIX}/{sample_id}.tif",
                "source_url": f"{_TIGER_BASE_URL}/{_TIGER_TCGA_PREFIX}/{sample_id}.tif",
                "measurement": (
                    "local level-0 TIFF XResolution/YResolution + ResolutionUnit tags"
                ),
            }
            for sample_id in excluded_sample_ids
        ],
        "derived_evaluation_subsets": (
            {
                "native_spacing_sensitivity": {
                    "intended_use": "evaluation_only",
                    "num_slides": len(sensitivity_rows),
                    "num_patients": len({row["patient_id"] for row in sensitivity_rows}),
                    "excluded_sample_ids": excluded_sample_ids,
                    "excluded_patient_ids": excluded_patient_ids,
                }
            }
            if is_full_cohort
            else {}
        ),
        "num_classes": len(PIXEL_MAPPING) - 1,
        "ignore_index": IGNORE_INDEX,
        "pixel_mapping": PIXEL_MAPPING,
        "annotated_fraction_min": ANNOTATED_FRACTION_MIN,
        "canonical_mpp": CANONICAL_MPP,
        "crop_size": CROP_SIZE,
        "cv_folds": sorted({int(r["validation_fold"].replace("fold", "")) for r in dataset_rows}),
        "slides_per_fold": dict(Counter(r["validation_fold"] for r in dataset_rows)),
        "slides_per_source": dict(Counter(r["source"] for r in dataset_rows)),
    }

    manifest = write_manifest(
        output_dir,
        dataset_type="segmentation",
        dataset_rows=dataset_rows,
        split_rows=split_rows,
        summary=summary,
    )
    print(f"Wrote slide manifest ({len(dataset_rows)} slides) + splits to {output_dir}")

    if coverage:
        _write_coverage(manifest.dataset_csv, Path(output_dir) / "coverage.csv")

    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m soma.curation.beetle",
        description="Curate the BEETLE development slides into a Soma segmentation Manifest.",
    )
    parser.add_argument(
        "--beetle-root",
        type=Path,
        default=_DEFAULT_BEETLE_ROOT,
        help="root the overview CSV's relative paths resolve against",
    )
    parser.add_argument(
        "--overview-csv",
        type=Path,
        default=None,
        help="data_overview.csv (default: <beetle-root>/data_overview.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "output directory (default: <beetle-root>/curated_slide_manifest; "
            "with --slides: <beetle-root>/non_publication_smoke_manifest)"
        ),
    )
    parser.add_argument("--slides", type=int, default=None, help="cap dev WSIs (balanced across folds)")
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="also run the (slow) per-slide annotation coverage scan",
    )
    args = parser.parse_args(argv)

    overview_csv = args.overview_csv or (args.beetle_root / "data_overview.csv")
    out = args.out or (
        args.beetle_root
        / (
            "curated_slide_manifest"
            if args.slides is None
            else "non_publication_smoke_manifest"
        )
    )
    curate_beetle_slide_manifest(
        overview_csv,
        args.beetle_root,
        out,
        slides=args.slides,
        coverage=args.coverage,
    )


if __name__ == "__main__":
    main()
