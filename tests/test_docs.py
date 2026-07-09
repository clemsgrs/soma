from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from soma.config import PipelineConfig, load_config

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
    assert generated.startswith("CLI\n===")
    assert "Available commands" in generated
    assert "What the CLI expects" in generated


def test_ocelot_benchmark_page_matches_registry() -> None:
    generator, docs_dir = _load_reference_generator()
    generated = generator.build_ocelot_benchmark_rst().strip()
    checked_in = (
        docs_dir / "ocelot-detection-benchmark.rst"
    ).read_text(encoding="utf-8").strip()

    assert generated == checked_in
    assert "TBD" not in generated
    assert "soma reproduce ocelot" in generated


def test_ocelot_benchmark_page_renders_external_guidance_anchors() -> None:
    generator, _ = _load_reference_generator()
    generated = generator.build_ocelot_benchmark_rst()
    # A dedicated, clearly-labelled guidance section (not the gate/tolerance band).
    assert "guidance" in generated.lower()
    # Rendered from the registry's external rows with clickable RST links + labels.
    assert "`best reported (fully-supervised end-to-end) " in generated
    assert "<https://wearewaiv.github.io/histoboard/>`__" in generated
    # Framed as context, never a target soma gates on.
    assert "non-gating" in generated.lower()


def test_eva_benchmark_page_matches_registry() -> None:
    generator, docs_dir = _load_reference_generator()
    generated = generator.build_eva_benchmark_rst().strip()
    checked_in = (
        docs_dir / "eva-patch-classification-benchmark.rst"
    ).read_text(encoding="utf-8").strip()

    assert generated == checked_in
    assert "TBD" not in generated
    assert "soma reproduce eva/bach" in generated


def test_eva_benchmark_page_renders_reproduced_ledger() -> None:
    generator, _ = _load_reference_generator()
    generated = generator.build_eva_benchmark_rst()
    # soma's OWN measured numbers, generated from results/eva.csv, next to the reference band.
    assert "Reproduced numbers" in generated
    assert "soma (mean ± std)" in generated
    # A seeded historical cell renders with its provenance (commit) and delta column.
    assert "0.914 ± 0.007" in generated
    assert "7ef2d7c" in generated
    # Never-run datasets have no measured row (honest: only recorded cells appear).
    reproduced = generated.split("Reproduced numbers", 1)[1].split("Reproduce\n", 1)[0]
    for never_run in ("mhist", "gleason_arvaniti", "patch_camelyon"):
        assert never_run not in reproduced


def test_documented_yaml_examples_load_through_public_config_interface() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example_paths = (
        sorted((repo_root / "examples").glob("*.yaml"))
        + sorted((repo_root / "examples" / "eva").glob("*.yaml"))
        + sorted((repo_root / "examples" / "ocelot").glob("*.yaml"))
    )

    assert example_paths
    for path in example_paths:
        load_config(path)


def test_generated_cli_reference_yaml_matches_bundled_defaults() -> None:
    generator, docs_dir = _load_reference_generator()
    generated = generator.build_cli_rst()
    reference_yaml = _extract_first_yaml_block(generated)

    documented = yaml.safe_load(reference_yaml)
    bundled_defaults = yaml.safe_load(
        (docs_dir.parent / "soma" / "configs" / "default.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert documented == bundled_defaults


def test_reference_example_yaml_matches_bundled_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    documented = yaml.safe_load(
        (repo_root / "examples" / "reference.yaml").read_text(encoding="utf-8")
    )
    bundled_defaults = yaml.safe_load(
        (repo_root / "soma" / "configs" / "default.yaml").read_text(encoding="utf-8")
    )

    assert documented == bundled_defaults


@pytest.mark.parametrize("name", ["getting-started.rst", "pipeline.rst"])
def test_representative_python_docs_examples_construct_public_configs(name: str) -> None:
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    rst = (docs_dir / name).read_text(encoding="utf-8")

    blocks = [
        block
        for block in _extract_code_blocks(rst, "python")
        if "PipelineConfig(" in block
    ]
    assert blocks

    for block in blocks:
        namespace: dict[str, object] = {}
        exec(_without_pipeline_run(block), namespace)
        assert isinstance(namespace["config"], PipelineConfig)


def _without_pipeline_run(code: str) -> str:
    return "\n".join(
        line for line in code.splitlines() if "Pipeline(config).run()" not in line
    )


def _extract_code_blocks(rst: str, language: str) -> list[str]:
    lines = rst.splitlines()
    blocks: list[str] = []
    marker = f".. code-block:: {language}"
    index = 0
    while index < len(lines):
        if lines[index] != marker:
            index += 1
            continue
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        block: list[str] = []
        while index < len(lines):
            line = lines[index]
            if line.startswith("   "):
                block.append(line[3:])
                index += 1
                continue
            if not line.strip():
                block.append("")
                index += 1
                continue
            break
        blocks.append("\n".join(block))
    return blocks


def _extract_first_yaml_block(rst: str) -> str:
    lines = rst.splitlines()
    start = lines.index(".. code-block:: yaml") + 1
    while start < len(lines) and not lines[start].strip():
        start += 1

    block: list[str] = []
    for line in lines[start:]:
        if line.startswith("   "):
            block.append(line[3:])
            continue
        if not line.strip():
            block.append("")
            continue
        break
    return "\n".join(block)


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
