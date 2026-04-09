"""Tests for the Soma release helper."""

from pathlib import Path

import release


def test_bump_version_string_supports_patch_minor_and_major():
    assert release.bump_version_string("0.1.0", "patch") == "0.1.1"
    assert release.bump_version_string("0.1.0", "minor") == "0.2.0"
    assert release.bump_version_string("0.1.0", "major") == "1.0.0"


def test_update_pyproject_version_rewrites_project_version(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "soma"
version = "1.2.3"
description = "demo"
""".lstrip()
    )

    bumped = release.update_pyproject_version(pyproject, "minor")

    assert bumped == "1.3.0"
    assert 'version = "1.3.0"' in pyproject.read_text()


def test_release_branch_and_tag_follow_version():
    assert release.release_branch("1.2.3") == "release-1.2.3"
    assert release.release_tag("1.2.3") == "1.2.3"


def test_open_release_draft_prints_release_url(monkeypatch, capsys):
    monkeypatch.setattr(
        release,
        "run",
        lambda cmd, check=True: "git@github.com:clemsgrs/soma.git",
    )

    release.open_release_draft("1.2.3")

    out = capsys.readouterr().out
    assert "https://github.com/clemsgrs/soma/releases/new?tag=1.2.3&title=1.2.3" in out
