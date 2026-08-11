from __future__ import annotations

import ast
import importlib.util
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tomllib

from jinja2 import Environment
import pytest
import yaml

from soma.config import PipelineConfig, load_config

TUTORIALS = [
    "walkthrough-tile-level.ipynb",
    "walkthrough-slide-mil.ipynb",
    "walkthrough-slide-encoder.ipynb",
    "walkthrough-segmentation.ipynb",
    "walkthrough-detection.ipynb",
    "walkthrough-composite.ipynb",
    "walkthrough-attention-segmentation.ipynb",
]


def test_documentation_always_spells_soma_lowercase() -> None:
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    source_suffixes = {".html", ".md", ".py", ".rst", ".svg"}
    offenders: list[str] = []

    for path in sorted(docs_dir.rglob("*")):
        if (
            not path.is_file()
            or path.suffix not in source_suffixes
            or "_build" in path.parts
        ):
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for match in re.finditer(r"\bsoma\b", line, flags=re.IGNORECASE):
                if match.group() != "soma":
                    offenders.append(
                        f"{path.relative_to(docs_dir)}:{line_number}: {match.group()}"
                    )

    assert offenders == []


def test_sphinx_exposes_the_project_release_without_network_access() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    conf_path = repo_root / "docs" / "conf.py"
    spec = importlib.util.spec_from_file_location("soma_docs_conf", conf_path)
    assert spec is not None and spec.loader is not None

    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)

    expected = tomllib.loads(
        (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    assert (conf.release, conf.html_context) == (
        expected,
        {
            "github_latest_release_tag": expected,
            "github_latest_release_url": (
                f"https://github.com/clemsgrs/soma/releases/tag/{expected}"
            ),
        },
    )


def test_sidebar_renders_repository_and_release_as_accessible_sibling_links() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    template = Environment(autoescape=True).from_string(
        (repo_root / "docs" / "_templates" / "sidebar" / "github.html").read_text(
            encoding="utf-8"
        )
    )
    rendered = template.render(
        github_latest_release_tag="1.8.0",
        github_latest_release_url="https://github.com/clemsgrs/soma/releases/tag/1.8.0",
    )

    class LinkParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.depth = 0
            self.max_depth = 0
            self.links: list[dict[str, str]] = []

        def handle_starttag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            if tag != "a":
                return
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)
            self.links.append({key: value or "" for key, value in attrs})

        def handle_endtag(self, tag: str) -> None:
            if tag == "a":
                self.depth -= 1

    parser = LinkParser()
    parser.feed(rendered)

    assert (parser.max_depth, parser.links) == (
        1,
        [
            {
                "class": "soma-sidebar-github__repo",
                "href": "https://github.com/clemsgrs/soma",
                "aria-label": "View clemsgrs/soma on GitHub",
            },
            {
                "class": "soma-sidebar-github__release",
                "href": "https://github.com/clemsgrs/soma/releases/tag/1.8.0",
                "aria-label": "View release 1.8.0 on GitHub",
            },
        ],
    )


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


@pytest.mark.parametrize(
    ("builder", "filename"),
    [
        ("build_ocelot_benchmark_rst", "ocelot-detection-benchmark.rst"),
        ("build_eva_benchmark_rst", "eva-patch-classification-benchmark.rst"),
        ("build_hest_benchmark_rst", "hest-gene-expression-benchmark.rst"),
    ],
)
def test_benchmark_page_matches_registry(builder: str, filename: str) -> None:
    generator, docs_dir = _load_reference_generator()
    generated = getattr(generator, builder)().strip()
    checked_in = (docs_dir / filename).read_text(encoding="utf-8").strip()

    assert generated == checked_in
    assert "TBD" not in generated


def test_croma_encoder_panel_page_matches_mapping_and_scopes_compatibility() -> None:
    generator, docs_dir = _load_reference_generator()
    generated = generator.build_croma_encoder_panel_rst().strip()
    checked_in = (docs_dir / "croma-encoder-panel.rst").read_text(
        encoding="utf-8"
    ).strip()

    assert generated == checked_in
    assert "does not prove exact numerical identity" in generated
    for unpinned_factor in (
        "weights and checkpoint revision",
        "preprocessing",
        "input geometry",
        "normalization",
        "precision",
        "implementation version",
    ):
        assert unpinned_factor in generated


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


def test_home_page_links_its_primary_routes_to_pages_that_exist() -> None:
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    page = (docs_dir / "index.rst").read_text(encoding="utf-8")

    routes = re.findall(r'<a class="soma-route" href="([^"#]+)(?:#[^"]*)?">', page)

    assert routes
    assert [route for route in routes if not (docs_dir / route).with_suffix(".rst").exists()] == []


def test_sidebar_uses_clickable_parent_pages_with_collapsible_children() -> None:
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    index = (docs_dir / "index.rst").read_text(encoding="utf-8")
    toctrees = index.split(".. toctree::")[1:]

    assert len(toctrees) == 1
    assert [line.strip() for line in toctrees[0].splitlines() if line.startswith("   ")] == [
        ":maxdepth: 2",
        ":hidden:",
        "how-soma-works",
        "getting-started",
        "data",
        "components",
        "tasks",
        "training-evaluation",
        "tutorials/index",
        "benchmarking",
        "reference",
        "system",
    ]

    expected_children = {
        "data.rst": ("dataset", "curation", "preprocessing"),
        "components.rst": ("encoders", "modeling", "aggregators", "decoders"),
        "tasks.rst": (
            "classification",
            "regression",
            "survival",
            "segmentation",
            "detection",
        ),
        "training-evaluation.rst": ("training", "evaluation"),
        "tutorials/index.rst": ("tile-level", "slide-level", "detection", "segmentation"),
        "benchmarking.rst": (
            "eva-patch-classification-benchmark",
            "ocelot-detection-benchmark",
            "hest-gene-expression-benchmark",
            "croma-encoder-panel",
        ),
        "reference.rst": ("api", "cli"),
        "system.rst": ("caching", "outputs", "reporting"),
    }
    for filename, children in expected_children.items():
        page = (docs_dir / filename).read_text(encoding="utf-8")
        tree = page.split(".. toctree::", 1)[1]
        entries = [
            line.strip()
            for line in tree.splitlines()
            if line.startswith("   ") and not line.strip().startswith(":")
        ]
        assert entries == list(children), filename


def test_getting_started_pipeline_example_constructs_public_config() -> None:
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    rst = (docs_dir / "getting-started.rst").read_text(encoding="utf-8")

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


def test_getting_started_python_examples_are_syntactically_valid() -> None:
    page = (
        Path(__file__).resolve().parents[1] / "docs" / "getting-started.rst"
    ).read_text(encoding="utf-8")
    blocks = _extract_code_blocks(page, "python")

    assert blocks
    for block in blocks:
        ast.parse(block)


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
def test_tutorial_notebook_ships_without_outputs(name: str) -> None:
    """Tutorials ship as clean code listings — valid JSON, code cells, no outputs.

    The committed notebooks deliberately carry **no** stored outputs (the docs
    build never executes them — nbsphinx_execute = "never" — and rendering empty
    cells keeps the pages uncluttered). This guards that policy: each notebook is
    valid JSON with code cells, and no code cell carries stored outputs. Smoke-test
    that the code still runs via ``scripts/execute_tutorials.sh``.
    """
    path = Path(__file__).resolve().parents[1] / "docs" / "tutorials" / name
    nb = json.loads(path.read_text(encoding="utf-8"))

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, f"{name} has no code cells"

    for i, cell in enumerate(code_cells):
        assert not cell.get("outputs"), (
            f"{name} code cell {i} carries stored outputs; strip them "
            f"(committed tutorials ship without outputs)"
        )
