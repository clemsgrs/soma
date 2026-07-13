from __future__ import annotations

import ast
import builtins
import importlib.util
import json
import symtable
import tomllib
from html.parser import HTMLParser
from pathlib import Path

import pytest
import yaml

import soma
from soma.config import PipelineConfig, load_config

TUTORIALS = [
    "walkthrough-slide-level.ipynb",
    "walkthrough-dense.ipynb",
]


@pytest.fixture(scope="module")
def built_docs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("sphinx")
    from sphinx.cmd.build import build_main

    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    out_dir = tmp_path_factory.mktemp("docs") / "html"
    status = build_main(["-W", "-b", "html", str(docs_dir), str(out_dir)])

    assert status == 0
    return out_dir


class _NavigationCaptionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.captions: list[str] = []
        self.level_one_hrefs: list[str] = []
        self._caption_parts: list[str] | None = None
        self._level_one_link_pending = False
        self._sidebar_depth = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        classes = dict(attrs).get("class", "") or ""
        if tag == "div":
            if self._sidebar_depth:
                self._sidebar_depth += 1
            elif "sidebar-tree" in classes.split():
                self._sidebar_depth = 1
        if (
            self._sidebar_depth
            and tag == "span"
            and "caption-text" in classes.split()
        ):
            self._caption_parts = []
        if self._sidebar_depth and tag == "li" and "toctree-l1" in classes.split():
            self._level_one_link_pending = True
        if self._sidebar_depth and tag == "a" and self._level_one_link_pending:
            href = dict(attrs).get("href")
            if href is not None:
                self.level_one_hrefs.append(href)
                self._level_one_link_pending = False

    def handle_data(self, data: str) -> None:
        if self._caption_parts is not None:
            self._caption_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "span" and self._caption_parts is not None:
            self.captions.append("".join(self._caption_parts).strip())
            self._caption_parts = None
        if tag == "div" and self._sidebar_depth:
            self._sidebar_depth -= 1


class _NestedAnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.nested_anchor_hrefs: list[tuple[str | None, str | None]] = []
        self._open_anchor_hrefs: list[str | None] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag != "a":
            return

        href = dict(attrs).get("href")
        if self._open_anchor_hrefs:
            self.nested_anchor_hrefs.append((self._open_anchor_hrefs[-1], href))
        self._open_anchor_hrefs.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._open_anchor_hrefs:
            self._open_anchor_hrefs.pop()


class _RenderedTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


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


def test_configuration_generator_matches_checked_in_file() -> None:
    generator, docs_dir = _load_reference_generator()
    generated = generator.build_configuration_rst().strip()
    checked_in = (docs_dir / "configuration.rst").read_text(encoding="utf-8").strip()

    assert generated == checked_in
    assert generated.startswith("Configuration\n=============")
    assert "Canonical YAML schema" in generated
    assert ":doc:`cli`" in generated


def test_cli_links_to_configuration_instead_of_embedding_full_schema() -> None:
    generator, _ = _load_reference_generator()
    generated = generator.build_cli_rst()

    assert ":doc:`configuration`" in generated
    assert "dense_window_size:" not in generated


def test_ocelot_benchmark_page_matches_registry() -> None:
    generator, docs_dir = _load_reference_generator()
    generated = generator.build_ocelot_benchmark_rst().strip()
    checked_in = (
        docs_dir / "ocelot-detection-benchmark.rst"
    ).read_text(encoding="utf-8").strip()

    assert generated == checked_in
    assert "TBD" not in generated
    assert "soma reproduce ocelot" in generated


def test_ocelot_benchmark_page_keeps_external_guidance_concise() -> None:
    generator, _ = _load_reference_generator()
    generated = generator.build_ocelot_benchmark_rst()

    assert "fully supervised" in generated.lower()
    assert "0.70–0.73" in generated
    assert "<https://wearewaiv.github.io/histoboard/>`__" in generated
    assert "non-gating" in generated.lower()
    assert "Reference environment" not in generated
    assert "NVIDIA GeForce" not in generated


def test_ocelot_benchmark_page_contains_its_own_data_preparation_contract() -> None:
    generator, _ = _load_reference_generator()
    generated = generator.build_ocelot_benchmark_rst()
    normalized = " ".join(generated.split())

    assert "ocelot2023_v1.0.1" in generated
    assert "cell patches" in generated
    assert "preserves OCELOT's train/validation/test split" in normalized
    assert ":doc:`curation`" not in generated


def test_eva_benchmark_page_matches_registry() -> None:
    generator, docs_dir = _load_reference_generator()
    generated = generator.build_eva_benchmark_rst().strip()
    checked_in = (
        docs_dir / "eva-patch-classification-benchmark.rst"
    ).read_text(encoding="utf-8").strip()

    assert generated == checked_in
    assert "TBD" not in generated
    assert "soma reproduce eva/bach" in generated


def test_eva_benchmark_page_renders_a_public_reproduced_reference_comparison() -> None:
    generator, _ = _load_reference_generator()
    generated = generator.build_eva_benchmark_rst()
    results = generated.split("Results\n-------", 1)[1].split(
        "Protocol details", 1
    )[0]

    assert "Reproduced numbers" in generated
    assert "Soma (mean ± std)" in results
    assert "EVA reference" in results
    assert "0.914 ± 0.007" in results
    assert "gleason_arvaniti" in results
    assert "0.778 ± 0.010" in results
    assert "Recorded (date @ commit)" not in results
    assert "Seeds" not in results
    assert "Δ" not in results
    for never_run in ("mhist", "patch_camelyon"):
        assert never_run not in results


def test_eva_benchmark_page_contains_its_own_data_preparation_contract() -> None:
    generator, _ = _load_reference_generator()
    generated = generator.build_eva_benchmark_rst()

    assert "Raw-root contents" in generated
    assert "annotations.csv" in generated
    assert "CRC-VAL-HE-7K" in generated
    assert "tune_is_test" in generated
    assert ":doc:`curation`" not in generated


def test_hest_benchmark_page_matches_registry() -> None:
    generator, docs_dir = _load_reference_generator()
    generated = generator.build_hest_benchmark_rst().strip()
    checked_in = (
        docs_dir / "hest-gene-expression-benchmark.rst"
    ).read_text(encoding="utf-8").strip()

    assert generated == checked_in
    assert "TBD" not in generated
    assert "soma reproduce hest/IDC" in generated


def test_generated_benchmark_pages_exclude_maintainer_instructions() -> None:
    generator, _ = _load_reference_generator()
    pages = {
        "OCELOT": generator.build_ocelot_benchmark_rst(),
        "EVA": generator.build_eva_benchmark_rst(),
        "HEST": generator.build_hest_benchmark_rst(),
    }
    maintainer_instructions = (
        "This page is generated",
        "python docs/_generate_reference.py",
    )

    violations = {
        name: [text for text in maintainer_instructions if text in page]
        for name, page in pages.items()
        if any(text in page for text in maintainer_instructions)
    }

    assert violations == {}


def test_hest_benchmark_page_excludes_contributor_instructions() -> None:
    generator, _ = _load_reference_generator()
    generated = generator.build_hest_benchmark_rst()
    contributor_instructions = (
        "Adding a HEST task",
        "HEST_TASKS",
    )

    assert [text for text in contributor_instructions if text in generated] == []


def test_hest_benchmark_page_documents_scoped_download() -> None:
    generator, _ = _load_reference_generator()
    generated = generator.build_hest_benchmark_rst()
    # A scoped download of one task that excludes the precomputed foundation-model features.
    assert "hf download MahmoodLab/hest-bench" in generated
    assert "--include 'IDC/*'" in generated
    assert "--exclude 'fm_v1/*'" in generated
    assert "precomputed" in generated.lower()


def test_hest_benchmark_page_contains_its_own_data_preparation_contract() -> None:
    generator, _ = _load_reference_generator()
    generated = generator.build_hest_benchmark_rst()

    assert "pip install 'soma-pathology[hest]'" in generated
    assert "hf auth login" in generated
    assert "Omit ``--include`` to download every registered task" in generated
    assert "preserves HEST's supplied fold assignments" in generated
    assert ":doc:`curation`" not in generated


def test_hest_contributor_guide_documents_task_extension_contract() -> None:
    guide = (
        Path(__file__).resolve().parents[1] / "docs" / "development" / "benchmarks.rst"
    ).read_text(encoding="utf-8")
    normalized = " ".join(guide.split())

    assert guide.startswith(":orphan:")
    assert "HEST_TASKS" in guide
    assert "no new curator or probe code" in normalized


def test_public_benchmark_authoring_guide_documents_the_complete_contract() -> None:
    guide = (
        Path(__file__).resolve().parents[1] / "docs" / "add-a-benchmark.rst"
    ).read_text(encoding="utf-8")

    expected_text = (
        "curate",
        "build_config",
        "expected",
        "score",
        "Facet",
        "canonical_seeds",
        "primary_metric",
        "register_benchmark",
        "soma list benchmarks",
        "soma reproduce",
    )

    assert [text for text in expected_text if text not in guide] == []


def test_hest_benchmark_page_keeps_results_focused_on_per_cell_agreement() -> None:
    generator, _ = _load_reference_generator()
    generated = generator.build_hest_benchmark_rst()
    results = generated.split("Results\n-------", 1)[1]

    assert "Soma" in results
    assert "HEST reference" in results
    assert "median relative difference" in results
    assert "Reproduction — is it sound?" not in results
    assert "rank concordance" not in results.lower()
    assert "drift guard" not in results.lower()
    assert "Spearman" not in results


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


def test_generated_configuration_yaml_matches_bundled_defaults() -> None:
    generator, docs_dir = _load_reference_generator()
    generated = generator.build_configuration_rst()
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


def test_api_python_examples_reference_only_defined_module_names() -> None:
    path = Path(__file__).resolve().parents[1] / "docs" / "api.rst"
    source = "\n\n".join(
        _extract_code_blocks(path.read_text(encoding="utf-8"), "python")
    )

    ast.parse(source, filename=str(path))
    module = symtable.symtable(source, str(path), "exec")
    undefined = sorted(
        symbol.get_name()
        for symbol in module.get_symbols()
        if symbol.is_referenced()
        and not symbol.is_imported()
        and not symbol.is_assigned()
        and symbol.get_name() not in vars(builtins)
    )

    assert undefined == []


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


def test_sphinx_docs_build(built_docs: Path) -> None:
    index_html = (built_docs / "index.html").read_text(encoding="utf-8")
    assert "Made with" not in index_html
    assert "@pradyunsg" not in index_html
    assert (built_docs / "index.html").exists()
    assert (built_docs / "cli.html").exists()


def test_root_sidebar_follows_the_two_primary_user_journeys(built_docs: Path) -> None:
    parser = _NavigationCaptionParser()
    parser.feed((built_docs / "index.html").read_text(encoding="utf-8"))

    assert parser.captions == [
        "Start",
        "Build workflows",
        "Benchmark foundation models",
        "Reference",
    ]
    assert parser.level_one_hrefs == [
        "getting-started.html",
        "pipeline.html",
        "tutorials/index.html",
        "benchmarking.html",
        "eva-patch-classification-benchmark.html",
        "ocelot-detection-benchmark.html",
        "hest-gene-expression-benchmark.html",
        "add-a-benchmark.html",
        "reference.html",
    ]


def test_root_page_has_no_nested_anchors(built_docs: Path) -> None:
    parser = _NestedAnchorParser()
    parser.feed((built_docs / "index.html").read_text(encoding="utf-8"))

    assert parser.nested_anchor_hrefs == []


def test_getting_started_renders_a_self_contained_first_run_path(
    built_docs: Path,
) -> None:
    parser = _RenderedTextParser()
    parser.feed((built_docs / "getting-started.html").read_text(encoding="utf-8"))
    rendered_text = "".join(parser.parts)
    required_text = (
        "Python 3.11",
        "sample_id,image_path,label",
        "sample_id,fold,split",
        (
            "data:\n"
            "  dataset_csv: dataset.csv\n"
            "  splits_csv: splits.csv"
        ),
        "soma config.yaml",
        "summary.json",
    )
    forbidden_text = ("YAML mirrors PipelineConfig field for field",)
    findings = {
        "missing": [text for text in required_text if text not in rendered_text],
        "forbidden": [text for text in forbidden_text if text in rendered_text],
    }

    assert findings == {"missing": [], "forbidden": []}


def test_benchmarking_page_distinguishes_reference_semantics(
    built_docs: Path,
) -> None:
    parser = _RenderedTextParser()
    parser.feed((built_docs / "benchmarking.html").read_text(encoding="utf-8"))
    rendered_text = " ".join(" ".join(parser.parts).split())
    expected_text = (
        "Gate references",
        "External references",
        "Only gate references produce PASS/FAIL.",
    )

    assert [text for text in expected_text if text not in rendered_text] == []


def test_benchmarking_page_presents_reproduction_and_authoring_routes() -> None:
    page = (
        Path(__file__).resolve().parents[1] / "docs" / "benchmarking.rst"
    ).read_text(encoding="utf-8")

    assert "Reproduce an included benchmark" in page
    assert "Add your own benchmark" in page
    assert ":doc:`add-a-benchmark`" in page
    assert "curation" not in page.lower()


def test_central_references_make_spatial_expression_discoverable(
    built_docs: Path,
) -> None:
    expected_by_page = {
        "pipeline.html": ("spatial_expression",),
        "dataset.html": ("targets.npy", "genes.json"),
        "regression.html": ("vector targets", "pearson", "ridge_pca_probe"),
        "evaluation.html": ("pearson",),
        "hest-gene-expression-benchmark.html": (
            "soma-pathology[hest]",
            "supplied fold assignments",
        ),
    }
    missing_by_page: dict[str, list[str]] = {}

    for page, expected_terms in expected_by_page.items():
        parser = _RenderedTextParser()
        parser.feed((built_docs / page).read_text(encoding="utf-8"))
        rendered_text = " ".join(" ".join(parser.parts).split())
        missing_terms = [
            term for term in expected_terms if term not in rendered_text
        ]
        if missing_terms:
            missing_by_page[page] = missing_terms

    assert missing_by_page == {}


def test_pipeline_page_explains_the_swappable_end_to_end_workflow() -> None:
    page = (
        Path(__file__).resolve().parents[1] / "docs" / "pipeline.rst"
    ).read_text(encoding="utf-8")

    expected_stages = (
        "Prepare images",
        "Encode",
        "Aggregate or decode",
        "Predict",
        "Evaluate",
    )

    assert [stage for stage in expected_stages if stage not in page] == []
    assert "swap or sweep" in page.lower()
    assert "aggregation:\n     name: mean_pool" in page


def test_reference_navigation_separates_interfaces_components_tasks_and_outputs() -> None:
    page = (
        Path(__file__).resolve().parents[1] / "docs" / "reference.rst"
    ).read_text(encoding="utf-8")

    expected_captions = (
        ":caption: Configure and run",
        ":caption: Data and components",
        ":caption: Prediction tasks",
        ":caption: Results and operations",
    )

    assert [caption for caption in expected_captions if caption not in page] == []


def test_slide_level_guide_has_a_complete_first_run_path() -> None:
    page = (
        Path(__file__).resolve().parents[1] / "docs" / "tutorials" / "slide-level.rst"
    ).read_text(encoding="utf-8")

    assert "sample_id,image_path,label" in page
    assert "sample_id,split" in page
    assert "examples/slide_binary_classification.yaml" in page
    assert "soma slide_binary_classification.yaml" in page


def test_sphinx_release_matches_the_project_version_in_a_source_checkout() -> None:
    conf_path = Path(__file__).resolve().parents[1] / "docs" / "conf.py"
    spec = importlib.util.spec_from_file_location("soma_docs_conf", conf_path)
    assert spec is not None and spec.loader is not None

    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)

    expected = soma.__version__
    if expected == "0.0.0+unknown":
        pyproject = conf_path.parent.parent / "pyproject.toml"
        expected = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
            "version"
        ]

    assert conf.release == expected


def test_sphinx_html_context_has_offline_release_metadata() -> None:
    conf_path = Path(__file__).resolve().parents[1] / "docs" / "conf.py"
    spec = importlib.util.spec_from_file_location("soma_docs_conf", conf_path)
    assert spec is not None and spec.loader is not None

    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)

    expected = soma.__version__
    if expected == "0.0.0+unknown":
        pyproject = conf_path.parent.parent / "pyproject.toml"
        expected = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
            "version"
        ]

    html_context = getattr(conf, "html_context", {})
    assert {
        "github_latest_release_tag": html_context.get("github_latest_release_tag"),
        "github_latest_release_url": html_context.get("github_latest_release_url"),
    } == {
        "github_latest_release_tag": expected,
        "github_latest_release_url": (
            f"https://github.com/clemsgrs/soma/releases/tag/{expected}"
        ),
    }


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
