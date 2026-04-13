"""Sphinx configuration for soma documentation."""

from __future__ import annotations

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for sibling in (ROOT.parent / "slide2vec", ROOT.parent / "hs2p"):
    if sibling.exists():
        sys.path.insert(0, str(sibling))

project = "soma"
author = "Clement Grisi"
copyright = "2026, Clement Grisi"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": False,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
always_use_bars_union = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

html_theme = "furo"
html_static_path = ["_static"]
html_title = "soma documentation"
html_theme_options = {
    "source_repository": "https://github.com/clemsgrs/soma",
    "source_branch": "main",
    "source_directory": "docs/",
}
