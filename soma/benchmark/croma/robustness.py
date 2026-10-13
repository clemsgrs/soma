"""Acquire and validate the pinned PathoROB tile data used by Soma benchmarks."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

from PIL import Image

METADATA_REPOSITORY = "bifold-pathomics/PathoROB"
METADATA_URL = "https://github.com/bifold-pathomics/PathoROB"
METADATA_REVISION = "6583cf0b0d902c8cc032308262fa3a3befdc0687"
SOURCE_INDEX_COLUMNS = ("slide_id", "patch_id", "sample_id", "image_path")
_IMAGE_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tiff"}
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class SourceFile:
    """One immutable parquet object in a pinned Hugging Face dataset revision."""

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class PathoROBCohortSource:
    """Pinned acquisition and metadata inputs for one PathoROB cohort."""

    name: str
    repository: str
    revision: str
    license: str
    files: tuple[SourceFile, ...]
    metadata_path: str
    metadata_sha256: str
    metadata_revision: str = METADATA_REVISION

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repository}"

    @property
    def metadata_filename(self) -> str:
        return Path(self.metadata_path).name


PATHOROB_COHORTS = (
    PathoROBCohortSource(
        name="camelyon",
        repository="bifold-pathomics/PathoROB-camelyon",
        revision="b2e762542abbf85dde3f23ec70a2bf1848dcf5c8",
        license="CC0-1.0",
        files=(
            SourceFile(
                path="data/train-00000-of-00001.parquet",
                sha256="ba6a65b97c0c4d98d4da4f303d0060c7e877ee103508b1b0d8b1950e4d79d098",
                size_bytes=466_600_116,
            ),
        ),
        metadata_path="data/metadata/camelyon.csv",
        metadata_sha256="9cad199ae0329b685cdcd5187812b406d533a737c3b58d4eca79c46d95b749ef",
    ),
    PathoROBCohortSource(
        name="tcga-4x4",
        repository="bifold-pathomics/PathoROB-tcga",
        revision="6e1dbd4306ebee9759b32503914523e84bddabd0",
        license="CC-BY-NC-SA-4.0",
        files=(
            SourceFile(
                path="data/train-00000-of-00003.parquet",
                sha256="e8cd2e8a70b6976d75c603c3508fed2cc0688886b3eeccb17ebc7ac329a1b051",
                size_bytes=370_100_844,
            ),
            SourceFile(
                path="data/train-00001-of-00003.parquet",
                sha256="9a6440b36b9e5a75580c86378d3db8dde96c0dc9260e2224fea53ce42c946da4",
                size_bytes=372_049_191,
            ),
            SourceFile(
                path="data/train-00002-of-00003.parquet",
                sha256="d6ea1afe34bf9963ccf278c56d92a9cf60b99ecae09a76e263966424a84c23a0",
                size_bytes=378_240_960,
            ),
        ),
        metadata_path="data/metadata/tcga_4x4.csv",
        metadata_sha256="1a50d09177001907c5793a7e89ccf1b3640bd55704cb9aae0d3bb33a4bcfca5c",
    ),
    PathoROBCohortSource(
        name="tolkach-esca",
        repository="bifold-pathomics/PathoROB-tolkach_esca",
        revision="c42219a2c168c5995e44487f5747bdacaf4bc2da",
        license="CC-BY-SA-4.0",
        files=(
            SourceFile(
                path="data/train-00000-of-00001.parquet",
                sha256="8d05ee072ff48d9aa1744201cd983040236ce9d87ee9071d9bf85bc85e225a90",
                size_bytes=315_677_989,
            ),
        ),
        metadata_path="data/metadata/tolkach_esca_reduced.csv",
        metadata_sha256="da183b32e05ed6d7c791c8a949952597f230d1e9f4a334bbac82e5622388d142",
    ),
)


@dataclass(frozen=True)
class PreparedPathoROBCohort:
    """Validated paths for one prepared cohort."""

    name: str
    root: Path
    source_index: Path
    metadata: Path
    provenance: Path
    rows: int


def prepare_croma(
    raw_root: str | Path,
    *,
    rebuild: bool = False,
) -> tuple[PreparedPathoROBCohort, ...]:
    """Prepare the pinned PathoROB tile sources under ``raw_root``."""
    root = Path(raw_root)
    if root.resolve() == Path(root.anchor):
        raise ValueError("PathoROB raw_root must not be a filesystem root")
    if root.exists() and not root.is_dir():
        raise ValueError(f"PathoROB destination {root} is not a directory")
    if root.exists() and any(root.iterdir()):
        try:
            return validate_prepared_croma(root)
        except (FileNotFoundError, ValueError) as exc:
            if not rebuild:
                raise ValueError(
                    f"PathoROB destination {root} is not a complete matching tree; "
                    "pass rebuild=True to replace it deliberately."
                ) from exc
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}-preparing-", dir=root.parent))
    try:
        for spec in PATHOROB_COHORTS:
            _prepare_cohort(staging / spec.name, spec)
        validate_prepared_croma(staging)
        if root.exists() and not any(root.iterdir()):
            root.rmdir()
        if root.exists():
            backup = Path(
                tempfile.mkdtemp(prefix=f".{root.name}-replaced-", dir=root.parent)
            )
            backup.rmdir()
            root.replace(backup)
            try:
                staging.replace(root)
            except Exception:
                backup.replace(root)
                raise
            shutil.rmtree(backup)
        else:
            staging.replace(root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_prepared_croma(root)


def _prepare_cohort(cohort_root: Path, spec: PathoROBCohortSource) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError(
            "Preparing PathoROB data requires pyarrow; install "
            "soma-pathology[croma]."
        ) from None

    from huggingface_hub import hf_hub_download
    from soma import __version__

    images_dir = cohort_root / "images"
    images_dir.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    seen_sample_ids: set[str] = set()
    for source_file in spec.files:
        parquet_path = Path(
            hf_hub_download(
                repo_id=spec.repository,
                filename=source_file.path,
                repo_type="dataset",
                revision=spec.revision,
            )
        )
        _verify_download(parquet_path, source_file)
        for batch in pq.ParquetFile(parquet_path).iter_batches(
            columns=["image", "slide_id", "patch_id"]
        ):
            for source_row in batch.to_pylist():
                slide_id = str(source_row["slide_id"])
                patch_id = str(source_row["patch_id"])
                for field, value in (("slide_id", slide_id), ("patch_id", patch_id)):
                    if not _SAFE_IDENTIFIER.fullmatch(value):
                        raise ValueError(
                            f"PathoROB cohort {spec.name!r} source has unsafe "
                            f"{field} {value!r}"
                        )
                key = (slide_id, patch_id)
                if key in seen_keys:
                    raise ValueError(
                        f"PathoROB cohort {spec.name!r} source has duplicate "
                        f"(slide_id, patch_id) key {key!r}"
                    )
                seen_keys.add(key)
                sample_id = f"{spec.name}-{slide_id}-{patch_id}"
                if sample_id in seen_sample_ids:
                    raise ValueError(
                        f"PathoROB cohort {spec.name!r} source keys produce duplicate "
                        f"sample_id {sample_id!r}"
                    )
                seen_sample_ids.add(sample_id)
                image_bytes = _image_bytes(source_row["image"], spec.name, key)
                extension = _image_extension(image_bytes, spec.name, key)
                image_path = Path("images") / f"{sample_id}{extension}"
                (cohort_root / image_path).write_bytes(image_bytes)
                rows.append(
                    {
                        "slide_id": slide_id,
                        "patch_id": patch_id,
                        "sample_id": sample_id,
                        "image_path": image_path.as_posix(),
                    }
                )

    index_path = cohort_root / "source_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SOURCE_INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    metadata_url = (
        f"https://raw.githubusercontent.com/{METADATA_REPOSITORY}/"
        f"{spec.metadata_revision}/{spec.metadata_path}"
    )
    with urllib.request.urlopen(metadata_url) as response:
        metadata_bytes = response.read()
    actual_metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    if actual_metadata_sha256 != spec.metadata_sha256:
        raise ValueError(
            f"PathoROB cohort {spec.name!r} metadata checksum mismatch: expected "
            f"{spec.metadata_sha256}, got {actual_metadata_sha256}"
        )
    (cohort_root / spec.metadata_filename).write_bytes(metadata_bytes)

    provenance = _provenance_contract(
        spec,
        rows=len(rows),
        metadata_size=len(metadata_bytes),
        prepared_at=_preparation_timestamp(),
        tool_version=__version__,
    )
    (cohort_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _preparation_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _provenance_contract(
    spec: PathoROBCohortSource,
    *,
    rows: int,
    metadata_size: int,
    prepared_at: str,
    tool_version: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "complete",
        "cohort": spec.name,
        "prepared_at": prepared_at,
        "preparation_tool": {
            "name": "soma.robustness.prepare_croma",
            "version": tool_version,
        },
        "sources": {
            "dataset": {
                "repository": spec.repository,
                "url": spec.url,
                "revision": spec.revision,
                "license": spec.license,
                "files": [
                    {
                        "path": source_file.path,
                        "sha256": source_file.sha256,
                        "size_bytes": source_file.size_bytes,
                    }
                    for source_file in spec.files
                ],
            },
            "metadata": {
                "repository": METADATA_REPOSITORY,
                "url": METADATA_URL,
                "revision": spec.metadata_revision,
                "license": "BSD-3-Clause",
                "files": [
                    {
                        "path": spec.metadata_path,
                        "sha256": spec.metadata_sha256,
                        "size_bytes": metadata_size,
                    }
                ],
            },
        },
        "output": {
            "images": "images",
            "source_index": "source_index.csv",
            "metadata": spec.metadata_filename,
            "index_columns": list(SOURCE_INDEX_COLUMNS),
            "rows": rows,
        },
    }


def _verify_download(path: Path, source_file: SourceFile) -> None:
    size = path.stat().st_size
    digest = _sha256_file(path)
    if size != source_file.size_bytes or digest != source_file.sha256:
        raise ValueError(
            f"PathoROB source file {source_file.path!r} checksum or size mismatch: "
            f"expected sha256={source_file.sha256}, size={source_file.size_bytes}; "
            f"got sha256={digest}, size={size}"
        )


def _sha256_file(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _image_bytes(image: object, cohort: str, key: tuple[str, str]) -> bytes:
    if not isinstance(image, dict) or not isinstance(image.get("bytes"), bytes):
        raise ValueError(
            f"PathoROB cohort {cohort!r} source image for key {key!r} "
            "does not contain encoded bytes"
        )
    return image["bytes"]


def _image_extension(image_bytes: bytes, cohort: str, key: tuple[str, str]) -> str:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_format = image.format
            image.verify()
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
    except Exception as exc:
        raise ValueError(
            f"PathoROB cohort {cohort!r} source image for key {key!r} is invalid"
        ) from exc
    try:
        return _IMAGE_EXTENSIONS[image_format]
    except KeyError:
        raise ValueError(
            f"PathoROB cohort {cohort!r} source image for key {key!r} has "
            f"unsupported format {image_format!r}"
        ) from None


def validate_prepared_croma(
    raw_root: str | Path,
) -> tuple[PreparedPathoROBCohort, ...]:
    """Validate and return the three pinned prepared PathoROB cohorts."""
    root = Path(raw_root)
    prepared: list[PreparedPathoROBCohort] = []
    for spec in PATHOROB_COHORTS:
        cohort_root = root / spec.name
        source_index = cohort_root / "source_index.csv"
        metadata = cohort_root / spec.metadata_filename
        provenance_path = cohort_root / "provenance.json"
        if not cohort_root.is_dir():
            raise ValueError(
                f"PathoROB prepared tree is incomplete: missing {cohort_root}"
            )
        for required in (
            source_index,
            metadata,
            provenance_path,
            cohort_root / "images",
        ):
            if not required.exists():
                raise ValueError(
                    f"PathoROB cohort {spec.name!r} is incomplete: missing {required}"
                )
        with source_index.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if tuple(reader.fieldnames or ()) != SOURCE_INDEX_COLUMNS:
                raise ValueError(
                    f"PathoROB cohort {spec.name!r} source index columns must be "
                    f"{SOURCE_INDEX_COLUMNS!r}; got {tuple(reader.fieldnames or ())!r}"
                )
            rows = list(reader)
        seen_keys: set[tuple[str, str]] = set()
        for row in rows:
            key = (row["slide_id"], row["patch_id"])
            if key in seen_keys:
                raise ValueError(
                    f"PathoROB cohort {spec.name!r} source index has duplicate "
                    f"(slide_id, patch_id) key {key!r}"
                )
            seen_keys.add(key)
            image_path = Path(row["image_path"])
            image = (cohort_root / image_path).resolve()
            if image_path.is_absolute() or not image.is_relative_to(
                cohort_root.resolve()
            ):
                raise ValueError(
                    f"PathoROB cohort {spec.name!r} source index has unsafe image_path "
                    f"{row['image_path']!r}"
                )
            if not image.is_file():
                raise FileNotFoundError(
                    f"PathoROB cohort {spec.name!r} source index references missing image: "
                    f"{row['image_path']!r}"
                )
            _validate_decoded_image(image, spec.name, row["image_path"])
        actual_metadata_sha256 = _sha256_file(metadata)
        if actual_metadata_sha256 != spec.metadata_sha256:
            raise ValueError(
                f"PathoROB cohort {spec.name!r} metadata checksum mismatch: expected "
                f"{spec.metadata_sha256}, got {actual_metadata_sha256}"
            )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        _validate_provenance(provenance, spec, len(rows), metadata.stat().st_size)
        prepared.append(
            PreparedPathoROBCohort(
                name=spec.name,
                root=cohort_root,
                source_index=source_index,
                metadata=metadata,
                provenance=provenance_path,
                rows=len(rows),
            )
        )
    return tuple(prepared)


def _validate_decoded_image(path: Path, cohort: str, declared_path: str) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
    except Exception as exc:
        raise ValueError(
            f"PathoROB cohort {cohort!r} has invalid decoded image "
            f"{declared_path!r}"
        ) from exc


def _validate_provenance(
    provenance: object,
    spec: PathoROBCohortSource,
    rows: int,
    metadata_size: int,
) -> None:
    if not isinstance(provenance, dict):
        raise ValueError(f"PathoROB cohort {spec.name!r} provenance must be an object")
    schema_version = provenance.get("schema_version")
    if schema_version != 1:
        raise ValueError(
            f"PathoROB cohort {spec.name!r} provenance schema_version "
            f"{schema_version!r}; expected 1"
        )
    if provenance.get("status") != "complete":
        raise ValueError(f"PathoROB cohort {spec.name!r} provenance is not complete")
    if provenance.get("cohort") != spec.name:
        raise ValueError(
            f"PathoROB cohort {spec.name!r} provenance cohort mismatch: "
            f"got {provenance.get('cohort')!r}"
        )

    sources = provenance.get("sources")
    if not isinstance(sources, dict):
        raise ValueError(
            f"PathoROB cohort {spec.name!r} provenance sources are missing"
        )
    expected_source_ids = {
        "dataset": (spec.repository, spec.revision),
        "metadata": (METADATA_REPOSITORY, spec.metadata_revision),
    }
    for source_name, (
        expected_repository,
        expected_revision,
    ) in expected_source_ids.items():
        source = sources.get(source_name)
        if not isinstance(source, dict):
            raise ValueError(
                f"PathoROB cohort {spec.name!r} provenance {source_name} source is missing"
            )
        actual_repository = source.get("repository")
        if actual_repository != expected_repository:
            raise ValueError(
                f"PathoROB cohort {spec.name!r} {source_name} repository mismatch: "
                f"expected {expected_repository!r}, got {actual_repository!r}"
            )
        actual_revision = source.get("revision")
        if actual_revision != expected_revision:
            raise ValueError(
                f"PathoROB cohort {spec.name!r} {source_name} revision mismatch: "
                f"expected {expected_revision!r}, got {actual_revision!r}"
            )

    prepared_at = provenance.get("prepared_at")
    tool = provenance.get("preparation_tool")
    if not isinstance(prepared_at, str) or not prepared_at:
        raise ValueError(
            f"PathoROB cohort {spec.name!r} provenance prepared_at is missing"
        )
    if (
        not isinstance(tool, dict)
        or tool.get("name") != "soma.robustness.prepare_croma"
        or not isinstance(tool.get("version"), str)
        or not tool["version"]
    ):
        raise ValueError(
            f"PathoROB cohort {spec.name!r} preparation tool provenance is invalid"
        )

    expected = _provenance_contract(
        spec,
        rows=rows,
        metadata_size=metadata_size,
        prepared_at=prepared_at,
        tool_version=tool["version"],
    )
    if provenance != expected:
        raise ValueError(
            f"PathoROB cohort {spec.name!r} provenance does not match the expected "
            "prepared-data contract"
        )


__all__ = [
    "PATHOROB_COHORTS",
    "PathoROBCohortSource",
    "PreparedPathoROBCohort",
    "SOURCE_INDEX_COLUMNS",
    "SourceFile",
    "prepare_croma",
    "validate_prepared_croma",
]
