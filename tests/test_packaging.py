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
