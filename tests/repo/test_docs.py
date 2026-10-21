from __future__ import annotations

import ast
import importlib.util
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import subprocess
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


_BENCHMARK_PAGES = {
    "ocelot-detection-benchmark.rst": ("ocelot-results.rst",),
    "midog-detection-benchmark.rst": ("midog-results.rst",),
    "monkey-detection-benchmark.rst": ("monkey-results.rst",),
    "eva-patch-classification-benchmark.rst": (
        "eva-patch-results.rst",
        "eva-segmentation-results.rst",
        "eva-slide-results.rst",
    ),
    "hest-gene-expression-benchmark.rst": ("hest-results.rst",),
    "croma-robustness-benchmark.rst": ("croma-results.rst",),
}


def test_results_fragments_match_the_ledgers() -> None:
    # The benchmark pages are written by hand; only their results tables are generated,
    # from the packaged results and reference ledgers, into docs/_generated/.
    generator, docs_dir = _load_reference_generator()
    for filename in generator.RESULTS_FRAGMENTS:
        generated = generator.build_results_fragment(filename).strip()
        checked_in = (docs_dir / "_generated" / filename).read_text(encoding="utf-8").strip()
        assert generated == checked_in, filename
        assert "TBD" not in generated
    assert set(generator.RESULTS_FRAGMENTS) == {f for fs in _BENCHMARK_PAGES.values() for f in fs}


@pytest.mark.parametrize(
    ("builder", "metric", "label"),
    [
        ("build_midog_results_rst", "f1", "Stitched-ROI F1"),
        ("build_monkey_results_rst", "froc_inflammatory", "Inflammatory FROC"),
    ],
)
def test_detection_results_tables_select_the_native_headline_metric(
    monkeypatch, builder: str, metric: str, label: str
) -> None:
    from soma.benchmark import MeasuredRow

    generator, _ = _load_reference_generator()
    rows = [
        MeasuredRow(key={"encoder": "virchow2"}, metric=metric, measured=0.75, std=0.01, n_seeds=3),
        MeasuredRow(key={"encoder": "virchow2"}, metric="precision", measured=0.95, n_seeds=3),
    ]
    monkeypatch.setattr(generator, "load_results", lambda name: rows)
    monkeypatch.setattr(generator, "expected_rows", lambda *args, **kwargs: [])

    assert getattr(generator, builder)() == (
        ".. list-table::\n"
        "   :header-rows: 1\n\n"
        "   * - Encoder\n"
        f"     - {label} (mean ± seed std)\n"
        "     - Seeds\n"
        "   * - virchow2\n"
        "     - 0.750 ± 0.010\n"
        "     - 3"
    )


def test_missing_monkey_results_request_one_seed_across_five_folds(monkeypatch) -> None:
    generator, _ = _load_reference_generator()
    monkeypatch.setattr(generator, "load_results", lambda name: [])

    assert generator.build_monkey_results_rst() == (
        "No reproductions have been recorded yet. "
        "``soma reproduce detection/monkey --seeds 1 --record`` "
        "appends a measured score to the results ledger this table reads."
    )


@pytest.mark.parametrize(
    ("builder", "dataset", "metric", "label", "spacing"),
    [
        ("build_midog_results_rst", "midog", "f1", "Stitched-ROI F1", "0.25"),
        ("build_monkey_results_rst", "monkey", "froc_inflammatory", "Inflammatory FROC", "0.242"),
    ],
)
def test_detection_table_uses_the_latest_recorder_row_for_each_encoder(
    monkeypatch, builder: str, dataset: str, metric: str, label: str, spacing: str
) -> None:
    from soma.benchmark import MeasuredRow

    generator, _ = _load_reference_generator()
    rows = [
        MeasuredRow(
            key={"dataset": dataset, "encoder": "virchow2", "spacing": spacing},
            metric=metric, measured=0.6, std=0.02, n_seeds=3,
        ),
        MeasuredRow(
            key={"encoder": "virchow2"},
            metric=metric, measured=0.8, std=0.01, n_seeds=3,
        ),
    ]
    monkeypatch.setattr(generator, "load_results", lambda name: rows)
    monkeypatch.setattr(generator, "expected_rows", lambda *args, **kwargs: [])

    assert getattr(generator, builder)() == (
        ".. list-table::\n"
        "   :header-rows: 1\n\n"
        "   * - Encoder\n"
        f"     - {label} (mean ± seed std)\n"
        "     - Seeds\n"
        "   * - virchow2\n"
        "     - 0.800 ± 0.010\n"
        "     - 3"
    )


def test_midog_aggregate_table_does_not_compare_a_seed_mean_with_the_seed_zero_reference(
    monkeypatch,
) -> None:
    from soma.benchmark import MeasuredRow

    generator, _ = _load_reference_generator()
    rows = [
        MeasuredRow(
            key={"dataset": "midog", "encoder": "virchow2", "spacing": "0.25"},
            metric="f1", measured=0.5, std=0.01, n_seeds=3,
        ),
    ]
    monkeypatch.setattr(generator, "load_results", lambda name: rows)

    assert generator.build_midog_results_rst() == (
        ".. list-table::\n"
        "   :header-rows: 1\n\n"
        "   * - Encoder\n"
        "     - Stitched-ROI F1 (mean ± seed std)\n"
        "     - Seeds\n"
        "   * - virchow2\n"
        "     - 0.500 ± 0.010\n"
        "     - 3"
    )


def test_monkey_encoder_only_recorder_row_is_checked_against_the_fixed_recipe(monkeypatch) -> None:
    from soma.benchmark import MeasuredRow

    generator, _ = _load_reference_generator()
    monkeypatch.setattr(generator, "load_results", lambda name: [
        MeasuredRow(
            key={"encoder": "virchow2"}, metric="froc_inflammatory",
            measured=0.8, n_seeds=1,
        ),
    ])

    assert generator.build_monkey_results_rst() == (
        ".. raw:: html\n\n"
        "   <style>.red { color: #d32f2f; font-weight: 600; }</style>\n\n"
        ".. role:: red\n\n"
        ".. list-table::\n"
        "   :header-rows: 1\n\n"
        "   * - Encoder\n"
        "     - Inflammatory FROC (mean ± seed std)\n"
        "     - Seeds\n"
        "   * - virchow2\n"
        "     - :red:`0.800`\n"
        "     - 1"
    )


@pytest.mark.parametrize("filename", sorted(_BENCHMARK_PAGES))
def test_benchmark_pages_share_one_structure_and_include_their_results(filename: str) -> None:
    docs_dir = Path(__file__).resolve().parents[2] / "docs"
    page = (docs_dir / filename).read_text(encoding="utf-8")
    headings = re.findall(r"^(\S.*)\n-+\n", page, flags=re.MULTILINE)
    assert headings[:4] == ["Data and task", "Protocol", "Results", "Reproduce"], filename
    for fragment in _BENCHMARK_PAGES[filename]:
        assert f".. include:: _generated/{fragment}" in page, (filename, fragment)


def test_benchmark_pages_spell_the_protocol_constants_of_the_packages() -> None:
    # The constants the pages quote are the ones the protocols run with.
    from soma.benchmark import eva as eva_bench
    from soma.benchmark import get_benchmark
    from soma.benchmark.detection import dataset_spec
    from soma.benchmark.eva import segmentation as eva_seg_bench
    from soma.benchmark.eva import slide as eva_slide_bench
    from soma.benchmark.hest.probe import DEFAULT_PCA_COMPONENTS, ridge_alpha

    docs_dir = Path(__file__).resolve().parents[2] / "docs"
    eva_page = (docs_dir / "eva-patch-classification-benchmark.rst").read_text(encoding="utf-8")
    for text in (
        f"learning rate ``{eva_bench.LEARNING_RATE:g}``, weight decay ``{eva_bench.WEIGHT_DECAY:g}``, "
        f"batch size ``{eva_bench.HEAD_BATCH_SIZE}``",
        f"``{eva_bench.MAX_STEPS}`` optimizer steps",
        f"learning rate ``{eva_seg_bench.LEARNING_RATE:g}``, weight decay ``{eva_seg_bench.WEIGHT_DECAY:g}``, "
        f"batch size ``{eva_seg_bench.BATCH_SIZE}``",
        f"``{eva_seg_bench.MAX_STEPS}`` optimizer steps",
        f"learning rate ``{eva_slide_bench.LEARNING_RATE:g}``, weight decay ``{eva_slide_bench.WEIGHT_DECAY:g}``, "
        f"batch size ``{eva_slide_bench.BATCH_SIZE}``",
        f"``{eva_slide_bench.EPOCHS}`` epochs, early-stopping patience ``{eva_slide_bench.PATIENCE}``",
        f"``0``–``{get_benchmark('eva/panda_small').canonical_seeds[-1]}``, averaged",
    ):
        assert text in eva_page, text
    for name, spec in eva_slide_bench.DATASETS.items():
        assert f"{spec.max_tiles} tiles at {spec.target_mpp:g} µm/px (``{name}``)" in eva_page, name

    hest_page = (docs_dir / "hest-gene-expression-benchmark.rst").read_text(encoding="utf-8")
    assert f"``PCA`` to {DEFAULT_PCA_COMPONENTS} components" in hest_page
    assert f"``alpha = 100 / ({DEFAULT_PCA_COMPONENTS} · 50)``" in hest_page
    assert abs(ridge_alpha(50) - 100 / (DEFAULT_PCA_COMPONENTS * 50)) < 1e-12

    ocelot = get_benchmark("ocelot")
    ocelot_page = (docs_dir / "ocelot-detection-benchmark.rst").read_text(encoding="utf-8")
    assert f"``{ocelot.facet.fixed['decoder']}``" in ocelot_page
    assert f"``{ocelot.primary_metric}``" in ocelot_page

    for dataset in ("midog", "monkey"):
        benchmark = get_benchmark(f"detection/{dataset}")
        page = (docs_dir / f"{dataset}-detection-benchmark.rst").read_text(encoding="utf-8")
        assert f"``{benchmark.facet.fixed['decoder']}``" in page
        assert f"``{benchmark.primary_metric}``" in page
        assert f"{dataset_spec(dataset).spacing_um:.3g} µm/px" in page
    monkey_page = (docs_dir / "monkey-detection-benchmark.rst").read_text(encoding="utf-8")
    assert "--seeds 1" in monkey_page


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
        + sorted((repo_root / "soma" / "benchmark").glob("*/configs/*.yaml"))
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
    assert [
        route
        for route in routes
        if not any((docs_dir / route).with_suffix(suffix).exists() for suffix in (".rst", ".ipynb"))
    ] == []


def test_sidebar_uses_clickable_parent_pages_with_collapsible_children() -> None:
    docs_dir = Path(__file__).resolve().parents[2] / "docs"
    index = (docs_dir / "index.rst").read_text(encoding="utf-8")
    toctrees = index.split(".. toctree::")[1:]

    assert len(toctrees) == 1
    assert [line.strip() for line in toctrees[0].splitlines() if line.startswith("   ")] == [
        ":maxdepth: 2",
        ":hidden:",
        "how-soma-works",
        "installation",
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
            "midog-detection-benchmark",
            "monkey-detection-benchmark",
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
        "Tile-level classification", "Tile-level regression",
        "Slide classification", "Regression", "Survival",
        "Slide classification with a slide encoder", "Regression with a slide encoder", "Survival with a slide encoder",
        "Segmentation", "Detection",
    ]
    named = {name for row in rows for name in re.findall(r"``(\w+)``", row.split("     - ")[-1])}
    assert named and named <= _e2e_scenarios()


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
    # The getting-started notebook ships without outputs: the build executed it, and the
    # page offers the notebook itself for download.
    notebook_html = (out_dir / "getting-started.html").read_text(encoding="utf-8")
    assert notebook_html.count('class="output_area') > 10
    assert re.search(r'href="_downloads/[0-9a-f]+/getting-started\.ipynb"', notebook_html)


def _committed_notebooks(root: Path) -> list[Path]:
    """Every ``.ipynb`` git tracks (or, outside a git checkout, every one under ``root``)."""
    try:
        listed = subprocess.run(
            ["git", "ls-files", "*.ipynb"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return sorted(p for p in root.rglob("*.ipynb") if not {"_build", ".ipynb_checkpoints"} & set(p.parts))
    return [root / name for name in listed]


def test_no_committed_notebook_carries_outputs_or_execution_counts() -> None:
    """Notebooks are committed clean; the docs build executes them (``nbsphinx_execute``)."""
    root = Path(__file__).resolve().parents[2]
    notebooks = _committed_notebooks(root)
    assert root / "docs" / "getting-started.ipynb" in notebooks
    dirty = []
    for path in notebooks:
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        for index, cell in enumerate(cells):
            if cell["cell_type"] == "code" and (cell.get("outputs") or cell.get("execution_count") is not None):
                dirty.append(f"{path.relative_to(root)} cell {index}")
    assert dirty == [], "strip outputs and execution counts before committing: " + ", ".join(dirty)


# The snippet guard. Prose pages carry concept-sized Python fragments, and every ``soma``
# name they spell must still exist. A block is Python when it is a ``code-block`` (or
# ``code`` / ``sourcecode``) directive whose language is ``python``, ``python3`` or ``py``;
# such a block must compile. A bare ``::`` literal block takes the page's highlight
# language, which is Sphinx's lenient ``default`` (Python, plain text when it does not
# lex) unless a ``.. highlight::`` directive sets one; the guard reads it as Python only
# when it compiles, so shell commands, YAML and file trees written as ``::`` blocks are
# left alone. Write a Python snippet as ``.. code-block:: python`` so a syntax error fails.
_SNIPPET_DIRECTIVE = re.compile(r"^(\s*)\.\. (?:code-block|code|sourcecode)::\s*(\S*)\s*$")
_HIGHLIGHT_DIRECTIVE = re.compile(r"^\s*\.\. highlight::\s*(\S+)\s*$")
_ANY_DIRECTIVE = re.compile(r"^\s*\.\. [\w:-]+::")
_PYTHON_LANGUAGES = {"python", "python3", "py"}
_SOMA_DOTTED_PATH = re.compile(r"soma(?:\.[A-Za-z_]\w*)+")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _block_after(lines: list[str], start: int, parent_indent: int) -> tuple[int, str, int]:
    """The body indented under ``lines[start - 1]``: its first line number, its dedented text, the index after it."""
    end = start
    while end < len(lines) and lines[end].strip().startswith(":") and _indent(lines[end]) > parent_indent:
        end += 1  # directive options
    while end < len(lines) and not lines[end].strip():
        end += 1
    first = end
    while end < len(lines) and (not lines[end].strip() or _indent(lines[end]) > parent_indent):
        end += 1
    body = lines[first:end]
    width = min((_indent(line) for line in body if line.strip()), default=0)
    return first + 1, "\n".join(line[width:] for line in body).rstrip("\n"), end


def _python_snippets(text: str) -> list[tuple[int, str, bool]]:
    """``(line, source, strict)`` for each Python block of an rst page; ``strict`` blocks must compile."""
    lines = text.splitlines()
    snippets: list[tuple[int, str, bool]] = []
    highlight = "default"
    index = 0
    while index < len(lines):
        line = lines[index]
        if match := _HIGHLIGHT_DIRECTIVE.match(line):
            highlight = match.group(1)
        elif match := _SNIPPET_DIRECTIVE.match(line):
            first, source, end = _block_after(lines, index + 1, len(match.group(1)))
            if match.group(2) in _PYTHON_LANGUAGES:
                snippets.append((first, source, True))
            index = end
            continue
        elif line.rstrip().endswith("::") and not _ANY_DIRECTIVE.match(line):
            first, source, end = _block_after(lines, index + 1, _indent(line))
            if highlight in _PYTHON_LANGUAGES:
                snippets.append((first, source, True))
            elif highlight == "default":
                snippets.append((first, source, False))
            index = end
            continue
        index += 1
    return snippets


def _soma_names(tree: ast.AST) -> set[str]:
    """Every dotted ``soma`` name a snippet imports, reads or spells as a dotted-path string."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names if alias.name.split(".")[0] == "soma"}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").split(".")[0] == "soma":
            names |= {f"{node.module}.{alias.name}" for alias in node.names if alias.name != "*"}
        elif isinstance(node, ast.Attribute):
            parts = [node.attr]
            value = node.value
            while isinstance(value, ast.Attribute):
                parts.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name) and value.id == "soma":
                names.add(".".join(["soma", *reversed(parts)]))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and _SOMA_DOTTED_PATH.fullmatch(node.value):
            names.add(node.value)
    return names


def _unresolved(dotted: str) -> str | None:
    """Why ``dotted`` is not a public object of the installed package, or ``None``."""
    parts = dotted.split(".")
    private = [part for part in parts if part.startswith("_") and not (part.startswith("__") and part.endswith("__"))]
    if private:
        return f"private name {private[0]}"
    obj = importlib.import_module(parts[0])
    for depth, part in enumerate(parts[1:], start=2):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            try:  # a submodule that ``import soma`` does not load, such as soma.benchmark
                obj = importlib.import_module(".".join(parts[:depth]))
            except ImportError:
                return f"no {part} in {'.'.join(parts[:depth - 1])}"
    return None


def _snippet_offences(path: Path, text: str) -> list[str]:
    offences: list[str] = []
    for line, source, strict in _python_snippets(text):
        try:
            tree = compile(source, f"{path}:{line}", "exec", ast.PyCF_ONLY_AST, dont_inherit=True)
            compile(tree, f"{path}:{line}", "exec", dont_inherit=True)  # parsing alone accepts a stray ``return``
        except SyntaxError as error:
            if strict:
                offences.append(f"{path}:{line + (error.lineno or 1) - 1}: does not compile ({error.msg})")
            continue
        offences += [
            f"{path}:{line}: {name} ({reason})"
            for name in sorted(_soma_names(tree))
            if (reason := _unresolved(name)) is not None
        ]
    return offences


def test_the_snippet_guard_reports_a_missing_name_a_private_name_and_a_broken_block() -> None:
    page = "\n".join(
        [
            "Text with a literal block::",
            "",
            "    import soma",
            "    soma.learn.NoSuchLearner",
            "",
            "A shell block is not Python::",
            "",
            "    soma run config.yaml",
            "",
            ".. code-block:: python",
            "   :caption: options are skipped",
            "",
            "   from soma.extract import live_encoder, _facade",
            "   name = {'name': 'soma.models.NoSuchAggregator'}",
            "",
            ".. code-block:: yaml",
            "",
            "   learner: {name: soma.models.NoSuchHead}",
            "",
            ".. code-block:: python",
            "",
            "   def broken(:",
            "",
            ".. code-block:: python",
            "",
            "   return 1",
        ]
    )
    assert _snippet_offences(Path("page.rst"), page) == [
        "page.rst:3: soma.learn.NoSuchLearner (no NoSuchLearner in soma.learn)",
        "page.rst:13: soma.extract._facade (private name _facade)",
        "page.rst:13: soma.models.NoSuchAggregator (no NoSuchAggregator in soma.models)",
        "page.rst:22: does not compile (invalid syntax)",
        "page.rst:26: does not compile ('return' outside function)",
    ]


def test_every_python_snippet_in_the_docs_compiles_and_names_public_soma_objects() -> None:
    docs_dir = Path(__file__).resolve().parents[2] / "docs"
    pages = sorted(path for path in docs_dir.rglob("*.rst") if "_build" not in path.parts)
    snippets = 0
    offences: list[str] = []
    for path in pages:
        text = path.read_text(encoding="utf-8")
        snippets += len(_python_snippets(text))
        offences += _snippet_offences(path.relative_to(docs_dir), text)

    assert snippets > 20
    assert offences == []


def test_the_custom_aggregator_example_runs_as_written(monkeypatch) -> None:
    """The worked aggregator of ``extending.rst`` runs as the page prints it: it registers
    with its params, passes ``soma.testing.check_aggregator``, and the preset names it."""
    import sys
    import types

    import soma
    from soma.models.aggregators import aggregator_registry

    # The example registers a name; restore the registry so it does not leak into later tests.
    monkeypatch.setattr(aggregator_registry, "_entries", dict(aggregator_registry._entries))
    page = (Path(__file__).resolve().parents[2] / "docs" / "extending.rst").read_text(encoding="utf-8")
    (source,) = [source for _, source, _ in _python_snippets(page) if "check_aggregator(" in source]
    # Run it as a module of its own, as a user's script would be: ``@dataclass`` looks its
    # class's module up in ``sys.modules``.
    example = types.ModuleType("extending_aggregator_example")
    monkeypatch.setitem(sys.modules, example.__name__, example)
    exec(compile(source, "docs/extending.rst", "exec"), example.__dict__)

    (entry,) = [entry for entry in soma.components("aggregators") if entry["name"] == "projected_mean"]
    assert entry["params"] == {"hidden_dim": 64}
    assert example.config.learner.params["aggregator"].name == "projected_mean"


def test_the_custom_encoder_example_runs_as_written(monkeypatch, tmp_path, request) -> None:
    """The two-file encoder example of ``extending.rst`` runs as the page prints it: the
    run script imports ``my_encoder.py``, slide2vec registers the encoder and fingerprints
    that file, soma lists it, and the config naming it on one GPU passes ``check``."""
    import hashlib
    import sys
    import types

    from slide2vec.encoders import encoder_code, encoder_registry

    import soma

    # The example registers a name; restore the registry so it does not leak into later tests.
    encoder_registry.names()  # discover the built-in encoders before copying the entries
    monkeypatch.setattr(encoder_registry, "_entries", dict(encoder_registry._entries))
    launched = []

    class RecordingPipeline:
        def __init__(self, config):
            self.config = config

        def run(self):
            launched.append(self.config)

    monkeypatch.setattr(soma, "Pipeline", RecordingPipeline)
    page = (Path(__file__).resolve().parents[2] / "docs" / "extending.rst").read_text(encoding="utf-8")
    snippets = [source for _, source, _ in _python_snippets(page)]
    (module_source,) = [source for source in snippets if "register_encoder(" in source]
    (script_source,) = [source for source in snippets if "import my_encoder" in source]
    module_path = tmp_path / "my_encoder.py"
    module_path.write_text(module_source + "\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    request.addfinalizer(lambda: sys.modules.pop("my_encoder", None))
    example = types.ModuleType("extending_encoder_example")
    monkeypatch.setitem(sys.modules, example.__name__, example)
    exec(compile(script_source, "docs/extending.rst", "exec"), example.__dict__)

    assert "my-tile-model" in [entry["name"] for entry in soma.components("encoders")]
    assert encoder_code("my-tile-model")["object"] == "my_encoder.MyTileModel"
    assert encoder_code("my-tile-model")["fingerprint"] == hashlib.sha256(module_path.read_bytes()).hexdigest()
    assert launched == [example.config]
    assert example.config.features.encoder.name == "my-tile-model"
    assert example.config.features.execution.num_gpus == 1
    check(example.config)
