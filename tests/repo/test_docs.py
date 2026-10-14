from __future__ import annotations

import ast
import importlib.util
from html.parser import HTMLParser
import os
from pathlib import Path
import re
import tomllib

from jinja2 import Environment
import pytest
import yaml

from soma.config import ExperimentConfig
from soma.plan import check

def test_documentation_always_spells_soma_lowercase() -> None:
    docs_dir = Path(__file__).resolve().parents[2] / "docs"
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


def test_documentation_always_spells_slide2vec_lowercase() -> None:
    docs_dir = Path(__file__).resolve().parents[2] / "docs"
    offenders: list[str] = []

    for path in sorted(docs_dir.rglob("*")):
        if (
            not path.is_file()
            or path.suffix not in {".md", ".py", ".rst"}
            or "_build" in path.parts
        ):
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for match in re.finditer(r"\bslide2vec\b", line, flags=re.IGNORECASE):
                if match.group() != "slide2vec":
                    offenders.append(
                        f"{path.relative_to(docs_dir)}:{line_number}: {match.group()}"
                    )

    assert offenders == []


def test_sphinx_exposes_the_project_release_without_network_access() -> None:
    repo_root = Path(__file__).resolve().parents[2]
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
    repo_root = Path(__file__).resolve().parents[2]
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
    docs_dir = Path(__file__).resolve().parents[2] / "docs"
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
        ("build_croma_benchmark_rst", "croma-robustness-benchmark.rst"),
    ],
)
def test_benchmark_page_matches_registry(builder: str, filename: str) -> None:
    generator, docs_dir = _load_reference_generator()
    generated = getattr(generator, builder)().strip()
    checked_in = (docs_dir / filename).read_text(encoding="utf-8").strip()

    assert generated == checked_in
    assert "TBD" not in generated


def test_croma_benchmark_page_exceptions_table_matches_panel() -> None:
    from soma.benchmark.croma import CROMA_0_3_ENCODER_PANEL

    docs_dir = Path(__file__).resolve().parents[2] / "docs"
    page = (docs_dir / "croma-robustness-benchmark.rst").read_text(encoding="utf-8")

    documented = re.findall(
        r"\* - (\S+)\n     - ``(\S+)``\n     - ``(\S+)``", page
    )
    expected = [
        (published, spec.soma_encoder, spec.output_variant)
        for published, spec in CROMA_0_3_ENCODER_PANEL.items()
        if spec.output_variant != "default"
    ]
    assert sorted(documented) == sorted(expected)
    assert "no\nclaim of numerical identity" in page


def test_documented_yaml_examples_load_through_public_config_interface() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    example_paths = (
        sorted((repo_root / "examples").glob("*.yaml"))
        + sorted((repo_root / "examples" / "eva").glob("*.yaml"))
        + sorted((repo_root / "examples" / "ocelot").glob("*.yaml"))
        + sorted((repo_root / "examples" / "detection_benchmark").glob("*.yaml"))
        + sorted((repo_root / "soma" / "benchmarks" / "configs").rglob("*.yaml"))
    )

    assert example_paths
    for path in example_paths:
        check(ExperimentConfig.from_yaml(path))


def test_generated_cli_reference_yaml_is_the_reference_example() -> None:
    generator, docs_dir = _load_reference_generator()
    reference_yaml = _extract_first_yaml_block(generator.build_cli_rst())
    example = docs_dir.parent / "examples" / "reference.yaml"

    assert yaml.safe_load(reference_yaml) == yaml.safe_load(example.read_text(encoding="utf-8"))


def test_reference_yaml_does_not_depend_on_the_generating_machine(monkeypatch) -> None:
    # slide2vec resolves the GPU and preprocessing-worker counts where a run uses them and
    # leaves the fields unset; the checked-in reference reads the same wherever it is
    # generated.
    import slide2vec.api

    generator, _ = _load_reference_generator()
    for env_name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
        monkeypatch.delenv(env_name, raising=False)
    builds = []
    for cpus, gpus in ((4, 1), (8, 2)):
        monkeypatch.setattr(os, "cpu_count", lambda cpus=cpus: cpus)
        monkeypatch.setattr(slide2vec.api, "_default_num_gpus", lambda gpus=gpus: gpus)
        builds.append((generator.build_reference_yaml(), generator.build_cli_rst()))

    assert builds[0] == builds[1]
    execution = yaml.safe_load(builds[0][0])["features"]["execution"]
    assert (execution["num_gpus"], execution["num_preprocessing_workers"]) == (None, None)


def test_reference_example_yaml_spells_every_default() -> None:
    from soma.config import EvaluationConfig, LoadersConfig, RunConfig, TrainerConfig

    generator, docs_dir = _load_reference_generator()
    example = docs_dir.parent / "examples" / "reference.yaml"
    text = example.read_text(encoding="utf-8")
    assert text == generator.build_reference_yaml()

    config = ExperimentConfig.from_yaml(example)
    check(config)
    # The file reads back as the reference experiment.
    assert config == generator.reference_config()
    # Every field is spelled out, and every section beyond the experiment's own records,
    # encoder and learner is the dataclass default.
    assert config.to_dict() == yaml.safe_load(text)
    assert (config.run, config.loaders, config.trainer, config.evaluation) == (
        RunConfig(),
        LoadersConfig(),
        TrainerConfig(),
        EvaluationConfig(),
    )


def test_home_page_links_its_primary_routes_to_pages_that_exist() -> None:
    docs_dir = Path(__file__).resolve().parents[2] / "docs"
    page = (docs_dir / "index.rst").read_text(encoding="utf-8")

    routes = re.findall(r'<a class="soma-route" href="([^"#]+)(?:#[^"]*)?">', page)

    assert routes
    assert [route for route in routes if not (docs_dir / route).with_suffix(".rst").exists()] == []


def test_sidebar_uses_clickable_parent_pages_with_collapsible_children() -> None:
    docs_dir = Path(__file__).resolve().parents[2] / "docs"
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
        "training",
        "evaluation",
        "extending",
        "benchmarking",
        "reference",
        "system",
    ]

    expected_children = {
        "data.rst": ("dataset", "curation", "preprocessing"),
        "components.rst": ("encoders", "modeling", "aggregators", "decoders"),
        "tasks.rst": (
            "classification",
            "survival",
            "segmentation",
            "detection",
        ),
        "benchmarking.rst": (
            "eva-patch-classification-benchmark",
            "ocelot-detection-benchmark",
            "detection-annotation-efficiency",
            "hest-gene-expression-benchmark",
            "croma-robustness-benchmark",
            "benchmark-in-house-encoder",
        ),
        "reference.rst": ("configuration", "api", "cli"),
        "system.rst": ("caching", "outputs"),
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


def _e2e_scenarios() -> set[str]:
    """The scenario names of ``tests/e2e/README.md``'s table, ``a_{b,c}`` spelled out."""
    readme = (Path(__file__).resolve().parents[1] / "e2e" / "README.md").read_text(encoding="utf-8")
    names: set[str] = set()
    for pattern in re.findall(r"^\| `([^`]+)` \|", readme, flags=re.MULTILINE):
        match = re.fullmatch(r"(.*)\{([^}]*)\}(.*)", pattern)
        alternatives = match.group(2).split(",") if match else [""]
        names |= {f"{match.group(1)}{alt}{match.group(3)}" if match else pattern for alt in alternatives}
    return names


def test_the_use_case_table_names_scenarios_the_e2e_suite_runs() -> None:
    rst = (Path(__file__).resolve().parents[2] / "docs" / "how-soma-works.rst").read_text(encoding="utf-8")
    table = rst.split(".. _use-cases:", 1)[1].split(".. list-table::", 1)[1]
    rows = [row for row in table.split("   * - ")[2:]]
    use_cases = [row.splitlines()[0].strip() for row in rows]
    assert use_cases == [
        "Slide classification", "Survival", "Regression", "Tile-level classification", "Segmentation", "Detection",
    ]
    named = {name for row in rows for name in re.findall(r"``(\w+)``", row.split("     - ")[-1])}
    assert named and named <= _e2e_scenarios()


def test_getting_started_pipeline_example_constructs_public_config() -> None:
    docs_dir = Path(__file__).resolve().parents[2] / "docs"
    rst = (docs_dir / "getting-started.rst").read_text(encoding="utf-8")

    blocks = [
        block
        for block in _extract_code_blocks(rst, "python")
        if "ExperimentConfig(" in block
    ]
    assert blocks

    for block in blocks:
        namespace: dict[str, object] = {}
        exec(_without_pipeline_run(block), namespace)
        assert isinstance(namespace["config"], ExperimentConfig)
        check(namespace["config"])


def test_getting_started_yaml_is_the_pipeline_example(tmp_path: Path) -> None:
    docs_dir = Path(__file__).resolve().parents[2] / "docs"
    rst = (docs_dir / "getting-started.rst").read_text(encoding="utf-8")
    (python,) = [
        block
        for block in _extract_code_blocks(rst, "python")
        if "ExperimentConfig(" in block and "features=FeaturesConfig(" in block
    ]
    (config_yaml,) = [
        block for block in _extract_code_blocks(rst, "yaml") if block.startswith("run:")
    ]
    namespace: dict[str, object] = {}
    exec(_without_pipeline_run(python), namespace)
    path = tmp_path / "config.yaml"
    path.write_text(config_yaml, encoding="utf-8")

    assert ExperimentConfig.from_yaml(path) == namespace["config"]


def test_getting_started_python_examples_are_syntactically_valid() -> None:
    page = (
        Path(__file__).resolve().parents[2] / "docs" / "getting-started.rst"
    ).read_text(encoding="utf-8")
    blocks = _extract_code_blocks(page, "python")

    assert blocks
    for block in blocks:
        ast.parse(block)


def test_getting_started_preset_example_edits_a_checked_config() -> None:
    rst = (Path(__file__).resolve().parents[2] / "docs" / "getting-started.rst").read_text(encoding="utf-8")
    (block,) = [block for block in _extract_code_blocks(rst, "python") if "presets." in block]
    namespace: dict[str, object] = {}
    exec(_without_pipeline_run(block), namespace)

    config = namespace["config"]
    check(config)
    assert (str(config.cohort.records), str(config.run.output_dir), config.trainer.max_epochs) == (
        "dataset.csv",
        "output",
        20,
    )


def _without_pipeline_run(code: str) -> str:
    return "\n".join(
        line
        for line in code.splitlines()
        if not ("Pipeline(config" in line and ".run()" in line)
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

    docs_dir = Path(__file__).resolve().parents[2] / "docs"
    out_dir = tmp_path / "html"
    status = build_main(["-W", "-b", "html", str(docs_dir), str(out_dir)])

    assert status == 0
    index_html = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "Made with" not in index_html
    assert "@pradyunsg" not in index_html
    assert (out_dir / "index.html").exists()
    assert (out_dir / "cli.html").exists()
