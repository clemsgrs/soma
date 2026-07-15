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
    "walkthrough-slide-level.ipynb",
    "walkthrough-dense.ipynb",
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
    results = generated.split("Results", 1)[1].split("Protocol details", 1)[0]

    assert "soma (mean ± std)" in generated
    assert "EVA reference" in results
    assert "0.914 ± 0.007" in generated
    assert "gleason_arvaniti" in results
    assert "0.778 ± 0.010" in results
    assert "7ef2d7c" not in results
    for never_run in ("mhist", "patch_camelyon"):
        assert never_run not in results


def test_eva_benchmark_page_matches_the_hest_reader_flow() -> None:
    generator, _ = _load_reference_generator()
    generated = generator.build_eva_benchmark_rst()

    headings = (
        "Prepare the data",
        "Run the benchmark",
        "Results",
        "Protocol details",
    )
    assert [generated.index(heading) for heading in headings] == sorted(
        generated.index(heading) for heading in headings
    )
    assert "EVA provides 6 registered datasets" in generated
    assert "labelled patches → frozen encoder → linear head → balanced accuracy" in generated

    run_section = generated.split("Run the benchmark", 1)[1].split("Results", 1)[0]
    assert run_section.lstrip("\n-").startswith("Pick any tile-level")
    assert run_section.count("--encoder virchow2") == 2
    assert "default" not in run_section.lower()

    results = generated.split("Results", 1)[1].split("Protocol details", 1)[0]
    assert "We benchmarked two encoders" in results
    assert "EVA reference" in results
    assert "median relative difference" in results
    for internal_detail in ("Seeds", "Recorded", "date @ commit", "Δ"):
        assert internal_detail not in results

    for unwanted in (
        "*Maps to task:*",
        "This page is generated",
        "Encoders\n--------",
        "What you can vary",
        "Reproduced numbers",
    ):
        assert unwanted not in generated


def test_eva_benchmark_page_documents_acquisition_then_automatic_curation() -> None:
    generator, _ = _load_reference_generator()
    generated = generator.build_eva_benchmark_rst()
    normalized = " ".join(generated.split())

    for source in (
        "zenodo.org/records/3632035",
        "web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis",
        "zenodo.org/records/1214456",
        "doi:10.7910/DVN/OCYCMP",
        "bmirds.github.io/MHIST/#accessing-dataset",
        "zenodo.org/records/2546921",
    ):
        assert source in generated
    assert "curl -L" in generated
    assert "soma does not download benchmark data" in generated
    assert "soma reproduce`` runs the built-in EVA curator automatically" in normalized
    assert "No separate curation command is required" in normalized
    for expected_raw_input in (
        "BreaKHis_v1/histology_slides/",
        "Gleason_masks_train.tar.gz",
        "camelyonpatch_level_2_split_{train,valid,test}_{x,y}.h5",
    ):
        assert expected_raw_input in generated


def test_hest_benchmark_page_matches_registry() -> None:
    generator, docs_dir = _load_reference_generator()
    generated = generator.build_hest_benchmark_rst().strip()
    checked_in = (
        docs_dir / "hest-gene-expression-benchmark.rst"
    ).read_text(encoding="utf-8").strip()

    assert generated == checked_in
    assert "TBD" not in generated
    assert "soma reproduce hest/IDC" in generated


def test_hest_benchmark_page_documents_reader_facing_data_preparation() -> None:
    generator, _ = _load_reference_generator()
    generated = generator.build_hest_benchmark_rst()
    normalized = " ".join(generated.split())

    assert "pip install 'soma-pathology[hest]'" in generated
    assert "hf auth login" in generated
    assert "hf download MahmoodLab/hest-bench" in generated
    assert "--include 'IDC/*'" in generated
    assert "--exclude 'fm_v1/*'" in generated
    assert "precomputed" in generated.lower()
    assert "Omit ``--include`` to download every registered task" in normalized
    assert "The ``hf`` CLI downloads the data" in normalized
    assert "soma reproduce`` runs the built-in HEST curator automatically" in normalized
    assert "preserves HEST's fold assignments" in normalized
    assert "No separate curation command is required" in normalized
    assert "Adding a HEST task" not in generated
    assert "curate_hest" not in generated
    assert "HEST_TASKS" not in generated


def test_hest_benchmark_page_keeps_results_focused_on_reference_agreement() -> None:
    generator, _ = _load_reference_generator()
    generated = generator.build_hest_benchmark_rst()
    results = generated.split("Results\n-------", 1)[1].split("Protocol details", 1)[0]

    assert "soma" in results
    assert "HEST reference" in results
    assert "median relative difference" in results
    assert "Reproduction — is it sound?" not in generated
    assert "rank concordance" not in generated.lower()
    assert "drift guard" not in generated.lower()
    assert "Spearman" not in generated


def test_hest_benchmark_page_distinguishes_encoder_selection_from_validation() -> None:
    generator, _ = _load_reference_generator()
    generated = generator.build_hest_benchmark_rst()
    results = generated.split("Results\n-------", 1)[1].split("Protocol details", 1)[0]

    assert "Pick any tile-level :doc:`encoder <encoders>` supported by soma" in generated
    assert "We benchmarked three encoders" in results
    assert "Choose one of the encoders supported" not in generated
    assert "HEST backbone" not in generated


def test_hest_benchmark_run_commands_require_an_explicit_encoder() -> None:
    generator, _ = _load_reference_generator()
    generated = generator.build_hest_benchmark_rst()
    run_section = generated.split("Run the benchmark\n-----------------", 1)[1].split(
        "Results", 1
    )[0]

    assert run_section.lstrip().startswith("Pick any tile-level")
    assert run_section.count("--encoder virchow2") == 2
    assert "default" not in run_section.lower()


def test_hest_benchmark_page_presents_cohorts_in_its_compact_overview() -> None:
    generator, _ = _load_reference_generator()
    generated = generator.build_hest_benchmark_rst()

    overview = generated.split("Prepare the data", 1)[0]
    assert "CCRCC, COAD, IDC, LUNG, LYMPH_IDC, PAAD, PRAD, READ, and SKCM" in overview
    assert ".. list-table::" not in overview
    assert "What you can vary" not in generated


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


def test_modeling_hub_routes_readers_through_supported_downstream_paths() -> None:
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    page = (docs_dir / "modeling.rst").read_text(encoding="utf-8")
    index = (docs_dir / "index.rst").read_text(encoding="utf-8")

    expected = (
        "Modeling",
        "frozen foundation-model features",
        "One feature vector per sample",
        "A bag of tile features per slide or patient",
        "A dense feature grid per tile or region",
        "frozen by design",
        ":doc:`aggregators`",
        ":doc:`decoders`",
        ":doc:`tasks`",
        ":doc:`training`",
        ":doc:`slide-level workflow <tutorials/slide-level>`",
        ":doc:`segmentation workflow <tutorials/segmentation>`",
        ":doc:`detection workflow <tutorials/detection>`",
    )

    normalized_page = " ".join(page.split())
    assert [
        text for text in expected if " ".join(text.split()) not in normalized_page
    ] == []
    assert "\n   modeling\n" in index
    assert ".. code-block::" not in page


def test_how_soma_works_stays_conceptual_and_routes_readers_to_details() -> None:
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    page = (docs_dir / "how-soma-works.rst").read_text(encoding="utf-8")

    expected = (
        "How soma works",
        "streamline computational pathology research with foundation models",
        "Define images, labels, and splits",
        "same modular workflow",
        "One workflow, modular blocks",
        "What you choose",
        "Data",
        "Preprocess",
        "Encode",
        "Train",
        "Evaluate",
        ":doc:`dataset`",
        ":doc:`preprocessing`",
        ":doc:`encoders`",
        ":doc:`modeling`",
        ":doc:`evaluation`",
        ":doc:`outputs`",
        "Explore or benchmark",
        "Custom experimentation",
        "optimize a workflow for your data and evaluation objective",
        ":doc:`API <api>`",
        "Benchmarking",
        "Vary one block while holding the source cohort, labels, splits",
        "Reproducible by design",
        "resolved configuration",
        "Where to go next",
        ":doc:`Get started <getting-started>`",
        ":doc:`Explore modeling paths <modeling>`",
        ":doc:`Benchmark a component <benchmarking>`",
        ":doc:`benchmarking`",
    )

    normalized_page = " ".join(page.split())
    assert [
        text for text in expected if " ".join(text.split()) not in normalized_page
    ] == []
    assert ".. code-block::" not in page
    assert "PipelineConfig(" not in page
    assert "dataset_type" not in page
    assert "spatial_expression" not in page
    assert "feature_mode" not in page
    assert ":doc:`curation`" not in page
    assert "Two common ways to use soma" not in page
    assert len(page.split()) <= 400
    assert not (docs_dir / "pipeline.rst").exists()


def test_home_page_exposes_four_primary_routes_in_order() -> None:
    page = (
        Path(__file__).resolve().parents[1] / "docs" / "index.rst"
    ).read_text(encoding="utf-8")

    expected = [
        (
            "how-soma-works.html",
            "How soma works",
            "See how reusable pipeline blocks support custom workflows and "
            "reproducible benchmarks.",
        ),
        (
            "getting-started.html",
            "Get started",
            "Install soma and run one experiment through the modular API, pipeline, "
            "or CLI.",
        ),
        (
            "benchmarking.html",
            "Benchmarking",
            "Reproduce and compare fixed foundation-model evaluation protocols.",
        ),
        (
            "encoders.html#model-zoo",
            "Foundation model zoo",
            "Browse registered tile-, slide-, and patient-level encoders.",
        ),
    ]

    assert '<nav class="soma-route-list" aria-label="Documentation routes">' in page
    assert [
        (href, title, description)
        for href, title, description in expected
        if (
            f'<a class="soma-route" href="{href}">\n'
            f"         <strong>{title}</strong>\n"
            f"         <span>{description}</span>"
        ) in page
    ] == expected
    assert [page.index(f'href="{href}"') for href, _, _ in expected] == sorted(
        page.index(f'href="{href}"') for href, _, _ in expected
    )


def test_sidebar_groups_api_and_cli_under_reference() -> None:
    page = (
        Path(__file__).resolve().parents[1] / "docs" / "index.rst"
    ).read_text(encoding="utf-8")
    toctrees = page.split(".. toctree::")[1:]

    start = toctrees[0]
    reference = next(block for block in toctrees if ":caption: Reference" in block)

    assert [line.strip() for line in start.splitlines() if line.startswith("   ")] == [
        ":maxdepth: 1",
        ":hidden:",
        "how-soma-works",
        "getting-started",
    ]
    assert [line.strip() for line in reference.splitlines() if line.startswith("   ")] == [
        ":maxdepth: 1",
        ":hidden:",
        ":caption: Reference",
        "api",
        "cli",
    ]


def test_getting_started_shows_one_workflow_through_all_public_interfaces() -> None:
    page = (
        Path(__file__).resolve().parents[1] / "docs" / "getting-started.rst"
    ).read_text(encoding="utf-8")

    expected = (
        "slide-level classification walkthrough",
        "pip install soma-pathology",
        "/_static/figures/how-soma-works-workflow.svg",
        "1. Define the data",
        "sample_id,image_path,label",
        "five-fold",
        "Dataset(\"dataset.csv\")",
        "Splits(\"splits.csv\", dataset)",
        "2. Preprocess and encode",
        "FeatureExtractor(",
        ").extract()",
        "must be relative to",
        "requested_spacing_um=0.5",
        "requested_tile_size_px=224",
        "resolved from the encoder's native configuration",
        ":doc:`preprocessing`",
        ":doc:`encoders`",
        "3. Train and evaluate",
        "AggregatorConfig(name=\"abmil\")",
        "train(",
        "run_dir=\"output/abmil\"",
        "TaskConfig(name=\"binary_classification\")",
        "EvalConfig(metrics=[\"auroc\", \"balanced_accuracy\"])",
        ":doc:`aggregators`",
        ":doc:`classification`",
        ":doc:`evaluation`",
        "Pipeline and CLI",
        "same experiment",
        "single call",
        "Pipeline(config).run()",
        "scalable",
        "soma config.yaml",
        ":doc:`CLI reference <cli>`",
        ":doc:`slide-level tutorial <tutorials/slide-level>`",
    )

    normalized_page = " ".join(page.split())
    assert [
        text for text in expected if " ".join(text.split()) not in normalized_page
    ] == []
    assert "aggregator_name" not in page
    assert '.extract("output/features/phikon")' not in page


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

    assert len(blocks) == 3
    for block in blocks:
        ast.parse(block)


def test_getting_started_routes_cli_configuration_to_the_reference() -> None:
    page = (
        Path(__file__).resolve().parents[1] / "docs" / "getting-started.rst"
    ).read_text(encoding="utf-8")

    assert ".. code-block:: yaml" not in page
    assert "soma config.yaml" in page
    assert ":doc:`CLI reference <cli>`" in page


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
