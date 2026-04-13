"""Sphinx configuration for soma documentation."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
import sys


def _fetch_latest_release(repo: str) -> tuple[str, str]:
    """Return (tag_name, html_url) for the latest GitHub release, or ('', '') on failure."""
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "sphinx-build"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("tag_name", ""), data.get("html_url", "")
    except Exception:
        return "", ""


def _inject_github_release(app):
    tag, url = _fetch_latest_release("clemsgrs/soma")
    app.config.html_context["github_latest_release_tag"] = tag
    app.config.html_context["github_latest_release_url"] = url


def setup(app):
    app.connect("builder-inited", _inject_github_release)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for sibling in (ROOT.parent / "slide2vec", ROOT.parent / "hs2p"):
    if sibling.exists():
        sys.path.insert(0, str(sibling))

project = "soma"
author = "Clément Grisi"
copyright = "2026, Clément Grisi"
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
html_css_files = ["sidebar.css"]
html_title = "soma"
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
}
