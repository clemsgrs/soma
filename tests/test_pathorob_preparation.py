"""Network-free tests for the prepared PathoROB raw-tree contract."""

from __future__ import annotations

import csv
import base64
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
import urllib.request

import huggingface_hub
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import soma
import soma.pathorob as pathorob
from soma.pathorob import (
    PATHOROB_COHORTS,
    PathoROBCohortSource,
    SourceFile,
    prepare_pathorob,
    validate_prepared_pathorob,
)

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ"
    "/pLvAAAAAElFTkSuQmCC"
)
PINNED_SOURCES = (
    {
        "name": "camelyon",
        "repository": "bifold-pathomics/PathoROB-camelyon",
        "revision": "b2e762542abbf85dde3f23ec70a2bf1848dcf5c8",
        "license": "CC0-1.0",
        "files": (
            (
                "data/train-00000-of-00001.parquet",
                "ba6a65b97c0c4d98d4da4f303d0060c7e877ee103508b1b0d8b1950e4d79d098",
                466_600_116,
            ),
        ),
        "metadata_path": "data/metadata/camelyon.csv",
        "metadata_sha256": (
            "9cad199ae0329b685cdcd5187812b406d533a737c3b58d4eca79c46d95b749ef"
        ),
        "metadata_revision": "6583cf0b0d902c8cc032308262fa3a3befdc0687",
    },
    {
        "name": "tcga-4x4",
        "repository": "bifold-pathomics/PathoROB-tcga",
        "revision": "6e1dbd4306ebee9759b32503914523e84bddabd0",
        "license": "CC-BY-NC-SA-4.0",
        "files": (
            (
                "data/train-00000-of-00003.parquet",
                "e8cd2e8a70b6976d75c603c3508fed2cc0688886b3eeccb17ebc7ac329a1b051",
                370_100_844,
            ),
            (
                "data/train-00001-of-00003.parquet",
                "9a6440b36b9e5a75580c86378d3db8dde96c0dc9260e2224fea53ce42c946da4",
                372_049_191,
            ),
            (
                "data/train-00002-of-00003.parquet",
                "d6ea1afe34bf9963ccf278c56d92a9cf60b99ecae09a76e263966424a84c23a0",
                378_240_960,
            ),
        ),
        "metadata_path": "data/metadata/tcga_4x4.csv",
        "metadata_sha256": (
            "1a50d09177001907c5793a7e89ccf1b3640bd55704cb9aae0d3bb33a4bcfca5c"
        ),
        "metadata_revision": "6583cf0b0d902c8cc032308262fa3a3befdc0687",
    },
    {
        "name": "tolkach-esca",
        "repository": "bifold-pathomics/PathoROB-tolkach_esca",
        "revision": "c42219a2c168c5995e44487f5747bdacaf4bc2da",
        "license": "CC-BY-SA-4.0",
        "files": (
            (
                "data/train-00000-of-00001.parquet",
                "8d05ee072ff48d9aa1744201cd983040236ce9d87ee9071d9bf85bc85e225a90",
                315_677_989,
            ),
        ),
        "metadata_path": "data/metadata/tolkach_esca_reduced.csv",
        "metadata_sha256": (
            "da183b32e05ed6d7c791c8a949952597f230d1e9f4a334bbac82e5622388d142"
        ),
        "metadata_revision": "6583cf0b0d902c8cc032308262fa3a3befdc0687",
    },
)
VALIDATION_METADATA = (
    b"subset,slide_id,patch_id,biological_class,medical_center\n"
    b"ID,slide-1,patch-1,label,center\n"
)
VALIDATION_METADATA_SHA256 = (
    "ccb85f7244af50685b996aa48973c5d46efcdd0a89899c5b6e148530104077d7"
)
VALIDATION_SOURCES = (
    PathoROBCohortSource(
        name="camelyon",
        repository="test/tiles-camelyon",
        revision="camelyon-revision",
        license="Test-Camelyon-License",
        files=(SourceFile("camelyon.parquet", "a" * 64, 101),),
        metadata_path="metadata/camelyon.csv",
        metadata_sha256=VALIDATION_METADATA_SHA256,
        metadata_revision="metadata-revision",
    ),
    PathoROBCohortSource(
        name="tcga-4x4",
        repository="test/tiles-tcga",
        revision="tcga-revision",
        license="Test-TCGA-License",
        files=(SourceFile("tcga.parquet", "b" * 64, 102),),
        metadata_path="metadata/tcga_4x4.csv",
        metadata_sha256=VALIDATION_METADATA_SHA256,
        metadata_revision="metadata-revision",
    ),
    PathoROBCohortSource(
        name="tolkach-esca",
        repository="test/tiles-tolkach",
        revision="tolkach-revision",
        license="Test-Tolkach-License",
        files=(SourceFile("tolkach.parquet", "c" * 64, 103),),
        metadata_path="metadata/tolkach_esca_reduced.csv",
        metadata_sha256=VALIDATION_METADATA_SHA256,
        metadata_revision="metadata-revision",
    ),
)


def _literal_validation_provenance(name: str) -> dict:
    source_literals = {
        "camelyon": {
            "repository": "test/tiles-camelyon",
            "url": "https://huggingface.co/datasets/test/tiles-camelyon",
            "revision": "camelyon-revision",
            "license": "Test-Camelyon-License",
            "files": [
                {"path": "camelyon.parquet", "sha256": "a" * 64, "size_bytes": 101}
            ],
            "metadata": "metadata/camelyon.csv",
        },
        "tcga-4x4": {
            "repository": "test/tiles-tcga",
            "url": "https://huggingface.co/datasets/test/tiles-tcga",
            "revision": "tcga-revision",
            "license": "Test-TCGA-License",
            "files": [{"path": "tcga.parquet", "sha256": "b" * 64, "size_bytes": 102}],
            "metadata": "metadata/tcga_4x4.csv",
        },
        "tolkach-esca": {
            "repository": "test/tiles-tolkach",
            "url": "https://huggingface.co/datasets/test/tiles-tolkach",
            "revision": "tolkach-revision",
            "license": "Test-Tolkach-License",
            "files": [
                {
                    "path": "tolkach.parquet",
                    "sha256": "c" * 64,
                    "size_bytes": 103,
                }
            ],
            "metadata": "metadata/tolkach_esca_reduced.csv",
        },
    }
    source = source_literals[name]
    metadata_path = source.pop("metadata")
    return {
        "schema_version": 1,
        "status": "complete",
        "cohort": name,
        "prepared_at": "2026-08-11T12:00:00+00:00",
        "preparation_tool": {
            "name": "soma.pathorob.prepare_pathorob",
            "version": "1.9.0",
        },
        "sources": {
            "dataset": source,
            "metadata": {
                "repository": "bifold-pathomics/PathoROB",
                "url": "https://github.com/bifold-pathomics/PathoROB",
                "revision": "metadata-revision",
                "license": "BSD-3-Clause",
                "files": [
                    {
                        "path": metadata_path,
                        "sha256": VALIDATION_METADATA_SHA256,
                        "size_bytes": 89,
                    }
                ],
            },
        },
        "output": {
            "images": "images",
            "source_index": "source_index.csv",
            "metadata": Path(metadata_path).name,
            "index_columns": ["slide_id", "patch_id", "sample_id", "image_path"],
            "rows": 1,
        },
    }


def _write_complete_tree(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pathorob, "PATHOROB_COHORTS", VALIDATION_SOURCES)
    for spec in VALIDATION_SOURCES:
        name = spec.name
        cohort_dir = root / name
        images_dir = cohort_dir / "images"
        images_dir.mkdir(parents=True)
        image_path = images_dir / f"{name}-sample.png"
        image_path.write_bytes(PNG_1X1)

        with (cohort_dir / "source_index.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["slide_id", "patch_id", "sample_id", "image_path"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "slide_id": f"{name}-slide",
                    "patch_id": "patch-1",
                    "sample_id": f"{name}-sample",
                    "image_path": f"images/{name}-sample.png",
                }
            )

        (cohort_dir / spec.metadata_filename).write_bytes(VALIDATION_METADATA)
        provenance = _literal_validation_provenance(name)
        (cohort_dir / "provenance.json").write_text(
            json.dumps(provenance), encoding="utf-8"
        )


def _install_synthetic_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloads: dict[str, Path] = {}
    metadata: dict[str, bytes] = {}
    specs = [
        PathoROBCohortSource(
            name="camelyon",
            repository="test/tiles-camelyon",
            revision="camelyon-revision",
            license="Test-Camelyon-License",
            files=(SourceFile("camelyon.parquet", "pending", 0),),
            metadata_path="metadata/camelyon.csv",
            metadata_sha256="pending",
            metadata_revision="metadata-revision",
        ),
        PathoROBCohortSource(
            name="tcga-4x4",
            repository="test/tiles-tcga",
            revision="tcga-revision",
            license="Test-TCGA-License",
            files=(SourceFile("tcga.parquet", "pending", 0),),
            metadata_path="metadata/tcga_4x4.csv",
            metadata_sha256="pending",
            metadata_revision="metadata-revision",
        ),
        PathoROBCohortSource(
            name="tolkach-esca",
            repository="test/tiles-tolkach",
            revision="tolkach-revision",
            license="Test-Tolkach-License",
            files=(SourceFile("tolkach.parquet", "pending", 0),),
            metadata_path="metadata/tolkach_esca_reduced.csv",
            metadata_sha256="pending",
            metadata_revision="metadata-revision",
        ),
    ]
    resolved_specs = []
    for source in specs:
        source_path = tmp_path / "downloads" / f"{source.name}.parquet"
        source_path.parent.mkdir(exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "image": {"bytes": PNG_1X1, "path": None},
                        "slide_id": f"{source.name}-slide",
                        "patch_id": "patch-1",
                    }
                ]
            ),
            source_path,
        )
        metadata_bytes = (
            "subset,slide_id,patch_id,biological_class,medical_center\n"
            f"ID,{source.name}-slide,patch-1,label,center\n"
        ).encode()
        metadata[source.metadata_path] = metadata_bytes
        downloads[source.repository] = source_path
        resolved_specs.append(
            replace(
                source,
                files=(
                    replace(
                        source.files[0],
                        sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                        size_bytes=source_path.stat().st_size,
                    ),
                ),
                metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
            )
        )

    monkeypatch.setattr(pathorob, "PATHOROB_COHORTS", tuple(resolved_specs))
    monkeypatch.setattr(
        pathorob,
        "_preparation_timestamp",
        lambda: "2026-08-11T12:00:00+00:00",
    )
    monkeypatch.setattr(soma, "__version__", "9.9.9-test")

    def fake_hf_hub_download(*, repo_id, filename, repo_type, revision):
        assert repo_type == "dataset"
        spec = next(item for item in resolved_specs if item.repository == repo_id)
        assert filename == spec.files[0].path
        assert revision == spec.revision
        return str(downloads[repo_id])

    def fake_urlopen(url):
        metadata_path = next(path for path in metadata if path in str(url))
        return io.BytesIO(metadata[metadata_path])

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_pathorob_sources_are_exactly_pinned():
    actual = tuple(
        {
            "name": source.name,
            "repository": source.repository,
            "revision": source.revision,
            "license": source.license,
            "files": tuple(
                (item.path, item.sha256, item.size_bytes) for item in source.files
            ),
            "metadata_path": source.metadata_path,
            "metadata_sha256": source.metadata_sha256,
            "metadata_revision": source.metadata_revision,
        }
        for source in PATHOROB_COHORTS
    )

    assert actual == PINNED_SOURCES


def test_complete_prepared_tree_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_complete_tree(tmp_path, monkeypatch)

    prepared = validate_prepared_pathorob(tmp_path)

    assert [cohort.name for cohort in prepared] == [
        "camelyon",
        "tcga-4x4",
        "tolkach-esca",
    ]


def test_duplicate_slide_patch_key_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_complete_tree(tmp_path, monkeypatch)
    index = tmp_path / "camelyon" / "source_index.csv"
    with index.open("a", encoding="utf-8") as fh:
        fh.write("camelyon-slide,patch-1,second-sample,images/camelyon-sample.png\n")

    with pytest.raises(ValueError, match=r"duplicate.*slide_id.*patch_id"):
        validate_prepared_pathorob(tmp_path)


def test_source_index_row_with_missing_image_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_complete_tree(tmp_path, monkeypatch)
    (tmp_path / "tcga-4x4" / "images" / "tcga-4x4-sample.png").unlink()

    with pytest.raises(FileNotFoundError, match=r"tcga-4x4.*missing image"):
        validate_prepared_pathorob(tmp_path)


def test_source_index_row_with_invalid_image_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_complete_tree(tmp_path, monkeypatch)
    (tmp_path / "camelyon" / "images" / "camelyon-sample.png").write_bytes(
        b"not an image"
    )

    with pytest.raises(ValueError, match=r"camelyon.*invalid decoded image"):
        validate_prepared_pathorob(tmp_path)


def test_source_index_requires_literal_contract_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_complete_tree(tmp_path, monkeypatch)
    index = tmp_path / "tolkach-esca" / "source_index.csv"
    index.write_text(
        "slide,patch,sample,path\nslide-1,patch-1,sample-1,images/sample-1.png\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"tolkach-esca.*source index columns"):
        validate_prepared_pathorob(tmp_path)


def test_provenance_revision_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_complete_tree(tmp_path, monkeypatch)
    provenance_path = tmp_path / "camelyon" / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["sources"]["dataset"]["revision"] = "different-revision"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match=r"camelyon.*revision mismatch"):
        validate_prepared_pathorob(tmp_path)


def test_provenance_schema_version_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_complete_tree(tmp_path, monkeypatch)
    provenance_path = tmp_path / "tcga-4x4" / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["schema_version"] = 2
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match=r"tcga-4x4.*schema_version.*expected 1"):
        validate_prepared_pathorob(tmp_path)


def test_metadata_revision_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_complete_tree(tmp_path, monkeypatch)
    provenance_path = tmp_path / "tolkach-esca" / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["sources"]["metadata"]["revision"] = "different-revision"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match=r"tolkach-esca.*metadata revision mismatch"):
        validate_prepared_pathorob(tmp_path)


def test_metadata_content_checksum_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_complete_tree(tmp_path, monkeypatch)
    (tmp_path / "camelyon" / "camelyon.csv").write_text("tampered metadata\n")

    with pytest.raises(ValueError, match=r"camelyon.*metadata checksum mismatch"):
        validate_prepared_pathorob(tmp_path)


def test_preparation_reuses_complete_matching_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_complete_tree(tmp_path, monkeypatch)
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    prepared = prepare_pathorob(tmp_path)

    assert [cohort.name for cohort in prepared] == [
        "camelyon",
        "tcga-4x4",
        "tolkach-esca",
    ]
    assert {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == before


def test_preparation_acquires_and_decodes_all_cohorts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install_synthetic_sources(tmp_path, monkeypatch)
    raw_root = tmp_path / "prepared"

    prepared = prepare_pathorob(raw_root)

    assert [cohort.name for cohort in prepared] == [
        "camelyon",
        "tcga-4x4",
        "tolkach-esca",
    ]
    expected = {
        "camelyon": {
            "repository": "test/tiles-camelyon",
            "url": "https://huggingface.co/datasets/test/tiles-camelyon",
            "revision": "camelyon-revision",
            "license": "Test-Camelyon-License",
            "metadata": "camelyon.csv",
        },
        "tcga-4x4": {
            "repository": "test/tiles-tcga",
            "url": "https://huggingface.co/datasets/test/tiles-tcga",
            "revision": "tcga-revision",
            "license": "Test-TCGA-License",
            "metadata": "tcga_4x4.csv",
        },
        "tolkach-esca": {
            "repository": "test/tiles-tolkach",
            "url": "https://huggingface.co/datasets/test/tiles-tolkach",
            "revision": "tolkach-revision",
            "license": "Test-Tolkach-License",
            "metadata": "tolkach_esca_reduced.csv",
        },
    }
    for name, source_expected in expected.items():
        cohort_root = raw_root / name
        with (cohort_root / "source_index.csv").open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows == [
            {
                "slide_id": f"{name}-slide",
                "patch_id": "patch-1",
                "sample_id": f"{name}-{name}-slide-patch-1",
                "image_path": f"images/{name}-{name}-slide-patch-1.png",
            }
        ]
        assert (cohort_root / rows[0]["image_path"]).read_bytes() == PNG_1X1
        assert (cohort_root / source_expected["metadata"]).is_file()
        provenance = json.loads((cohort_root / "provenance.json").read_text())
        assert provenance["schema_version"] == 1
        assert provenance["status"] == "complete"
        assert provenance["cohort"] == name
        assert provenance["prepared_at"] == "2026-08-11T12:00:00+00:00"
        assert provenance["preparation_tool"] == {
            "name": "soma.pathorob.prepare_pathorob",
            "version": "9.9.9-test",
        }
        dataset_source = provenance["sources"]["dataset"]
        assert {
            key: dataset_source[key] for key in source_expected if key != "metadata"
        } == {
            "repository": source_expected["repository"],
            "url": source_expected["url"],
            "revision": source_expected["revision"],
            "license": source_expected["license"],
        }
        metadata_source = provenance["sources"]["metadata"]
        assert {
            key: metadata_source[key]
            for key in ("repository", "url", "revision", "license")
        } == {
            "repository": "bifold-pathomics/PathoROB",
            "url": "https://github.com/bifold-pathomics/PathoROB",
            "revision": "metadata-revision",
            "license": "BSD-3-Clause",
        }


def test_partial_tree_is_rejected_and_preserved(tmp_path: Path):
    raw_root = tmp_path / "prepared"
    partial = raw_root / "camelyon" / "partial.txt"
    partial.parent.mkdir(parents=True)
    partial.write_text("interrupted")

    with pytest.raises(ValueError, match=r"not a complete matching tree.*rebuild"):
        prepare_pathorob(raw_root)
    assert partial.read_text() == "interrupted"


def test_explicit_rebuild_replaces_partial_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install_synthetic_sources(tmp_path, monkeypatch)
    raw_root = tmp_path / "prepared"
    partial = raw_root / "camelyon" / "partial.txt"
    partial.parent.mkdir(parents=True)
    partial.write_text("interrupted")

    prepared = prepare_pathorob(raw_root, rebuild=True)

    assert [cohort.name for cohort in prepared] == [
        "camelyon",
        "tcga-4x4",
        "tolkach-esca",
    ]
    assert not partial.exists()


def test_prepare_pathorob_cli_validates_existing_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    _write_complete_tree(tmp_path, monkeypatch)
    from soma.cli import main

    main(["prepare-pathorob", str(tmp_path)])

    assert capsys.readouterr().out == (
        f"Prepared PathoROB data under {tmp_path} "
        "(camelyon: 1, tcga-4x4: 1, tolkach-esca: 1).\n"
    )
