"""Tests for PyPI packaging metadata."""

from pathlib import Path
import re
import tomllib


def test_pyproject_has_publish_ready_metadata():
    pyproject = Path("pyproject.toml")
    data = tomllib.loads(pyproject.read_text())

    project = data["project"]
    assert project["name"] == "soma-pathology"
    assert project["scripts"]["soma"] == "soma.__main__:entrypoint"
    assert re.fullmatch(r"\d+\.\d+\.\d+", project["version"]) is not None
    assert project["license"] == {"file": "LICENSE"}
    assert project["authors"] == [
        {"name": "Clément Grisi", "email": "clement.grisi@radboudumc.nl"}
    ]
    assert "classifiers" in project
    assert any("slide2vec" in dep for dep in project["dependencies"])

    urls = project["urls"]
    assert urls["Homepage"] == "https://github.com/clemsgrs/soma"
    assert urls["Source"] == "https://github.com/clemsgrs/soma"
    assert urls["Issues"] == "https://github.com/clemsgrs/soma/issues"

    wheel_targets = data["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel_targets["packages"] == ["soma"]

    sdist_targets = data["tool"]["hatch"]["build"]["targets"]["sdist"]
    assert sdist_targets["only-include"] == ["LICENSE", "README.md", "pyproject.toml", "soma"]


def test_slide2vec_minimum_is_aligned_with_hs2p_6():
    # slide2vec 7.0.0 requires hs2p 6.0.0, the same floor soma declares. It resumes an
    # artifact only when its sidecar records the full feature identity, and
    # ``Model.pooled_identity_differences`` (6.3.2) reports a required field the record
    # lacks as ``MISSING_FIELD`` instead of accepting it, which soma's cache check relies
    # on. Earlier floors: 6.3.3 tiles for a patient-level model without the slides'
    # patient ids; 6.3.4 adds the ``patch_features_prenorm`` dense feature kind the EVA
    # segmentation benchmarks tap.
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "slide2vec[fm]>=7.0.0" in data["project"]["dependencies"]


def test_hs2p_minimum_provides_registered_roi_mask_reads():
    # hs2p 5.0 removed the label-read helpers soma's dense reader used; soma reads masks
    # through hs2p.Mask, aligned to the slide's level-0 grid. 5.0.1 sizes the coverage
    # estimate's tiles like tiling (tolerance); 5.0.2 samples a region at the grid's
    # recorded spacing without a float-noise shift and reports the in-canvas part of an
    # overhanging ROI (AlignedMask.dimensions_within_canvas). 5.1.0 renders previews of
    # flat PNG/JPEG slides with their manifest spacing. 6.0.0 rejects unknown config keys
    # and values of the wrong type (soma composes typed hs2p configs, so none reach it),
    # takes ``resolve_tissue_mask`` settings as one ``SegmentationConfig``, records
    # ``null`` segmentation thresholds when no segmentation ran, and lets ``auto`` skip a
    # backend that opens a file but cannot decode it.
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "hs2p>=6.0.0" in data["project"]["dependencies"]


def test_release_metadata_matches_license_and_verified_python_support():
    pyproject = Path("pyproject.toml")
    data = tomllib.loads(pyproject.read_text())

    license_text = Path("LICENSE").read_text(encoding="utf-8")
    classifiers = set(data["project"]["classifiers"])

    assert "Apache License" in license_text
    assert "License :: OSI Approved :: Apache Software License" in classifiers

    python_classifiers = {
        classifier.rsplit("::", maxsplit=1)[-1].strip()
        for classifier in classifiers
        if classifier.startswith("Programming Language :: Python :: 3.")
    }
    assert python_classifiers == {"3.11"}
    assert 'python-version: "3.11"' in Path(".github/workflows/release.yaml").read_text(
        encoding="utf-8"
    )
    assert 'python-version: "3.11"' in Path(".github/workflows/docs.yaml").read_text(
        encoding="utf-8"
    )
    assert "ARG PYTHON_VERSION=3.11" in Path("Dockerfile.ci").read_text(encoding="utf-8")


def test_public_import_surface_dependencies_are_declared():
    data = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = {
        re.split(r"[<>=!~;\\[]", dependency, maxsplit=1)[0].strip().lower()
        for dependency in data["project"]["dependencies"]
    }

    # These packages are imported by modules exposed from `import soma`, so a fresh
    # install must not rely on them arriving only as transitive dependencies.
    assert {"pillow", "rich"}.issubset(dependencies)
    # torchvision is imported by the dense augmentation / segmentation loaders.
    assert "torchvision" in dependencies

    # The dense pixel-classifier adapter stays optional and reports its own missing
    # dependency only when the xgboost classifier is selected.
    assert "xgboost" not in dependencies
    extras = data["project"]["optional-dependencies"]
    assert any(dep.lower().startswith("xgboost") for dep in extras["pixel"])

    import soma

    assert soma.Pipeline is not None
    assert soma.FeatureExtractor is not None
    assert soma.FeatureExtractionResult is not None
    assert not hasattr(soma, "TileFeatureExtractor")
    assert not hasattr(soma, "DenseTileFeatureExtractor")


def test_top_level_package_exports_dense_configuration_and_discovery_surface():
    import soma
    from soma.config import (
        AttentionConfig,
        CompositeConfig,
        EncoderMemberConfig,
        PixelClassifierConfig,
    )

    expected_exports = {
        "AttentionConfig": AttentionConfig,
        "CompositeConfig": CompositeConfig,
        "EncoderMemberConfig": EncoderMemberConfig,
        "PixelClassifierConfig": PixelClassifierConfig,
        "list_decoders": soma.list_decoders,
        "list_pixel_classifiers": soma.list_pixel_classifiers,
    }

    for name, expected in expected_exports.items():
        assert name in soma.__all__
        assert getattr(soma, name) is expected
