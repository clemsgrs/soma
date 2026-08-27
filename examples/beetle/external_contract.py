"""BEETLE External cohort, ROI sidecar, PNG, and archive contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence
import zipfile

import numpy as np
from PIL import Image

from examples.beetle.protocol import ANNOTATED_LABEL_NAME_BY_VALUE

EXTERNAL_ROI_COUNT = 170
EXTERNAL_PATIENT_COUNT = 54
LABEL_NAME_BY_VALUE = ANNOTATED_LABEL_NAME_BY_VALUE
SUBMISSION_LABELS = frozenset(LABEL_NAME_BY_VALUE)
CLASS_VOCABULARY = tuple(LABEL_NAME_BY_VALUE.values())
NUM_CLASSES = len(LABEL_NAME_BY_VALUE)
MODEL_INDEX_TO_SUBMISSION_LABEL = np.asarray(sorted(SUBMISSION_LABELS), dtype=np.uint8)


@dataclass(frozen=True)
class ExternalRoi:
    roi_filename: str
    patient_id: str
    source_wsi: str
    native_spacing_um: float
    width: int
    height: int


@dataclass(frozen=True)
class ExternalCohort:
    roi_count: int = EXTERNAL_ROI_COUNT
    patient_count: int = EXTERNAL_PATIENT_COUNT

    def __post_init__(self) -> None:
        if self.roi_count <= 0 or self.patient_count <= 0:
            raise ValueError("BEETLE External cohort counts must be positive")


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"BEETLE sidecar {field} must be a positive integer")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"BEETLE sidecar {field} must be a positive integer") from exc
    if parsed <= 0 or parsed != value:
        raise ValueError(f"BEETLE sidecar {field} must be a positive integer")
    return parsed


def load_roi_sidecar(
    path: str | Path, *, expected_rois: int = EXTERNAL_ROI_COUNT
) -> tuple[ExternalRoi, ...]:
    """Load the authoritative flat-ROI filename, source, spacing, and dimension map."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("rois"), list):
        raise ValueError(
            "BEETLE ROI-to-WSI sidecar must use schema_version 1 with a rois list"
        )
    rows = payload["rois"]
    if len(rows) != expected_rois:
        raise ValueError(
            f"BEETLE ROI-to-WSI sidecar requires exactly {expected_rois} ROIs, got {len(rows)}"
        )
    records: list[ExternalRoi] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"BEETLE sidecar ROI row {index} must be an object")
        filename = str(row.get("roi_filename", "")).strip()
        if (
            not filename
            or Path(filename).name != filename
            or Path(filename).suffix.lower() != ".png"
        ):
            raise ValueError(
                f"BEETLE sidecar ROI row {index} requires a flat .png roi_filename"
            )
        patient_id = str(row.get("patient_id", "")).strip()
        source_wsi = str(row.get("source_wsi", "")).strip()
        if not patient_id or not source_wsi:
            raise ValueError(
                f"BEETLE sidecar ROI {filename!r} requires patient_id and source_wsi"
            )
        spacing = row.get("native_spacing_um")
        if (
            isinstance(spacing, bool)
            or not isinstance(spacing, (int, float))
            or not math.isfinite(float(spacing))
            or float(spacing) <= 0
        ):
            raise ValueError(
                f"BEETLE sidecar ROI {filename!r} requires finite positive native_spacing_um"
            )
        records.append(
            ExternalRoi(
                roi_filename=filename,
                patient_id=patient_id,
                source_wsi=source_wsi,
                native_spacing_um=float(spacing),
                width=_positive_int(row.get("width"), field=f"{filename} width"),
                height=_positive_int(row.get("height"), field=f"{filename} height"),
            )
        )
    filenames = [record.roi_filename for record in records]
    if len(set(filenames)) != len(filenames):
        raise ValueError("BEETLE ROI-to-WSI sidecar repeats roi_filename")
    return tuple(sorted(records, key=lambda record: record.roi_filename))


def exact_directory_paths(
    directory: str | Path,
    expected_names: Sequence[str],
    *,
    artifact_label: str,
) -> tuple[Path, ...]:
    """Require a flat directory to contain exactly the declared basenames."""
    directory = Path(directory)
    expected = set(expected_names)
    observed = {path.name for path in directory.iterdir()}
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(
            f"{artifact_label} failed exact coverage; missing={missing}, extra={extra}"
        )
    paths = tuple(directory / name for name in sorted(expected))
    non_files = [path.name for path in paths if not path.is_file()]
    if non_files:
        raise ValueError(f"{artifact_label} entries must be files: {non_files}")
    return paths


def validate_roi_inputs(roi_dir: str | Path, records: Sequence[ExternalRoi]) -> None:
    """Validate External ROI basenames and sidecar-declared native dimensions."""
    paths = exact_directory_paths(
        roi_dir,
        [record.roi_filename for record in records],
        artifact_label="BEETLE External ROI filenames",
    )
    by_name = {record.roi_filename: record for record in records}
    for path in paths:
        record = by_name[path.name]
        with Image.open(path) as image:
            if image.size != (record.width, record.height):
                raise ValueError(
                    f"BEETLE ROI {record.roi_filename!r} dimensions {image.size} disagree "
                    f"with sidecar {(record.width, record.height)}"
                )


def validate_submission_pngs(
    output_dir: str | Path,
    records: Sequence[ExternalRoi],
) -> tuple[Path, ...]:
    """Require exact coverage and BEETLE's PNG dimensions/type/label vocabulary."""
    paths = exact_directory_paths(
        output_dir,
        [record.roi_filename for record in records],
        artifact_label="BEETLE submission filenames",
    )
    by_name = {record.roi_filename: record for record in records}
    for path in paths:
        record = by_name[path.name]
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "L":
                raise ValueError(
                    f"BEETLE submission {path.name!r} must be single-channel grayscale PNG"
                )
            if image.size != (record.width, record.height):
                raise ValueError(
                    f"BEETLE submission {path.name!r} dimensions {image.size} != "
                    f"{(record.width, record.height)}"
                )
            labels = set(int(value) for value in np.unique(np.asarray(image)))
            if not labels <= SUBMISSION_LABELS:
                raise ValueError(
                    f"BEETLE submission {path.name!r} contains invalid labels {sorted(labels)}"
                )
    return paths


def write_flat_submission_zip(paths: Sequence[Path], zip_path: str | Path) -> Path:
    """Write a deterministic flat archive with no directory prefix."""
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(paths, key=lambda value: value.name):
            info = zipfile.ZipInfo(path.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    return zip_path


__all__ = [
    "CLASS_VOCABULARY",
    "EXTERNAL_PATIENT_COUNT",
    "EXTERNAL_ROI_COUNT",
    "ExternalCohort",
    "ExternalRoi",
    "MODEL_INDEX_TO_SUBMISSION_LABEL",
    "NUM_CLASSES",
    "SUBMISSION_LABELS",
    "exact_directory_paths",
    "load_roi_sidecar",
    "validate_roi_inputs",
    "validate_submission_pngs",
    "write_flat_submission_zip",
]
