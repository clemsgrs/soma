"""Package-ownership boundaries for repository-local project protocols."""

from __future__ import annotations

import importlib.util


def test_beetle_curator_is_project_owned() -> None:
    assert importlib.util.find_spec("soma.curation.beetle") is None
    assert importlib.util.find_spec("examples.beetle.curate") is not None
