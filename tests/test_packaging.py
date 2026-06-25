"""Tests for PyPI packaging metadata."""

from pathlib import Path
import re
import tomllib


def test_pyproject_has_publish_ready_metadata():
    pyproject = Path("pyproject.toml")
    data = tomllib.loads(pyproject.read_text())

    project = data["project"]
    assert project["name"] == "soma-pathology"
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


def test_pypi_distribution_name_keeps_soma_import_package():
    pyproject = Path("pyproject.toml")
    data = tomllib.loads(pyproject.read_text())

    assert data["project"]["name"] == "soma-pathology"
    assert data["project"]["scripts"]["soma"] == "soma.__main__:entrypoint"
    assert data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["soma"]


def test_release_metadata_matches_license_and_verified_python_support():
    pyproject = Path("pyproject.toml")
    data = tomllib.loads(pyproject.read_text())

    license_text = Path("LICENSE").read_text(encoding="utf-8")
    classifiers = set(data["project"]["classifiers"])

    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in license_text
    assert "License :: OSI Approved :: GNU Affero General Public License v3" in classifiers

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

    # The dense pixel-classifier adapter stays optional and reports its own missing
    # dependency only when the xgboost classifier is selected.
    assert "xgboost" not in dependencies

    import soma

    assert soma.Pipeline is not None
    assert soma.TileFeatureExtractor is not None


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
