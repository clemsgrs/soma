"""Sphinx configuration for soma documentation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tomllib


def _optional_extension(module_name: str) -> list[str]:
    try:
        __import__(module_name)
    except ImportError:
        return []
    return [module_name]


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for sibling in (ROOT.parent / "slide2vec", ROOT.parent / "hs2p"):
    if sibling.exists():
        sys.path.insert(0, str(sibling))

import soma

project = "soma"
author = "Clément Grisi"
copyright = "2026, Clément Grisi"
release = soma.__version__
if release == "0.0.0+unknown":
    release = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]
html_context = {
    "github_latest_release_tag": release,
    "github_latest_release_url": (
        f"https://github.com/clemsgrs/soma/releases/tag/{release}"
    ),
}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
] + _optional_extension("sphinx_copybutton") + _optional_extension(
    "sphinx_autodoc_typehints"
) + _optional_extension("nbsphinx")

# Notebooks ship with their outputs already committed: never re-execute them at
# build time (the docs build has no GPU, no gated HF token, and no slides). The
# tutorials are refreshed out-of-band via scripts/execute_tutorials.sh.
nbsphinx_execute = "never"
# nbsphinx renders a "source" download link per notebook; keep it (handy for the
# copy-and-run workflow the tutorials are built around).
nbsphinx_allow_errors = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
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

# Keep the docs build self-contained in offline environments.
intersphinx_mapping = {}

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["sidebar.css"]
html_title = "soma"
html_show_sourcelink = False
if importlib.util.find_spec("furo") is not None:
    html_theme = "furo"
    _sidebar = [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/github.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        "sidebar/variant-selector.html",
    ]
    html_sidebars = {"**": _sidebar}
    html_theme_options = {
        "source_repository": "https://github.com/clemsgrs/soma",
        "source_branch": "main",
        "source_directory": "docs/",
        "top_of_page_buttons": ["view"],
    }
else:
    html_theme = "alabaster"
