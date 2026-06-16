from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

TUTORIALS = [
    "walkthrough-slide-level.ipynb",
    "walkthrough-dense.ipynb",
]


def _load_reference_generator():
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    module_path = docs_dir / "_generate_reference.py"
    spec = importlib.util.spec_from_file_location("_generate_reference", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, docs_dir


def test_cli_generator_matches_checked_in_file() -> None:
    generator, docs_dir = _load_reference_generator()
    generated = generator.build_cli_rst().strip()
    checked_in = (docs_dir / "cli.rst").read_text(encoding="utf-8").strip()

    assert generated == checked_in
    assert "CLI Guide" in generated
    assert "Available commands" in generated
    assert "What the CLI expects" in generated


def test_sphinx_docs_build(tmp_path: Path) -> None:
    pytest.importorskip("sphinx")
    from sphinx.cmd.build import build_main

    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    out_dir = tmp_path / "html"
    status = build_main(["-W", "-b", "html", str(docs_dir), str(out_dir)])

    assert status == 0
    index_html = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "Made with" not in index_html
    assert "@pradyunsg" not in index_html
    assert (out_dir / "index.html").exists()
    assert (out_dir / "cli.html").exists()


@pytest.mark.parametrize("name", TUTORIALS)
def test_tutorial_notebook_is_executed(name: str) -> None:
    """Tutorials ship with committed outputs (the docs build never executes them).

    Guards against committing an empty / un-executed notebook: each must be valid
    JSON, contain code cells, and every code cell must carry stored outputs with
    no error output. Refresh via ``scripts/execute_tutorials.sh``.
    """
    path = Path(__file__).resolve().parents[1] / "docs" / "tutorials" / name
    nb = json.loads(path.read_text(encoding="utf-8"))

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, f"{name} has no code cells"

    for i, cell in enumerate(code_cells):
        outputs = cell.get("outputs", [])
        assert outputs, f"{name} code cell {i} has no stored outputs (re-execute it)"
        for out in outputs:
            assert out.get("output_type") != "error", (
                f"{name} code cell {i} stored an error output: {out.get('ename')}"
            )
