"""Generate the CLI guide page from public soma metadata."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from inspect import signature
from importlib import resources
from pathlib import Path
from textwrap import dedent, indent
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for sibling in (ROOT.parent / "slide2vec", ROOT.parent / "hs2p"):
    if sibling.exists():
        sys.path.insert(0, str(sibling))

from soma.aggregators import aggregator_registry
from soma.benchmarks import (
    expected_rows,
    get_benchmark,
    list_benchmarks,
    load_results,
    reproduction_report,
)
from soma.benchmarks import eva as eva_bench
from soma.benchmarks import hest as hest_bench
from soma.benchmarks import ocelot as ocelot_bench
from soma.config import (
    AggregatorConfig,
    CacheConfig,
    EncoderConfig,
    EvalConfig,
    ExecutionConfig,
    HeatmapConfig,
    PipelineConfig,
    PreprocessingConfig,
    PreviewConfig,
    TaskConfig,
    TrainingConfig,
)
from soma.dataset import Dataset, Splits
from soma.extraction import FeatureExtractor
from soma.features import FeatureStore
from soma.pipeline import Pipeline, train
from soma.tasks import task_registry
from soma.tile_extraction import TileFeatureExtractor
from soma.training.probe import DEFAULT_PCA_COMPONENTS, ridge_alpha


def _field_names(cls: type, *, exclude: set[str] | None = None) -> str:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    excluded = exclude or set()
    return ", ".join(f"``{field.name}``" for field in fields(cls) if field.name not in excluded)


def _constructor_knobs(cls: type) -> str:
    params = [
        f"``{param.name}``"
        for param in signature(cls.__init__).parameters.values()
        if param.name != "self"
    ]
    return ", ".join(params)


def _list_table(rows: list[tuple[str, str, str, str]]) -> str:
    lines = [".. list-table::", "   :header-rows: 1", ""]
    lines.extend(
        [
            "   * - Name",
            "     - Class",
            "     - Constructor knobs",
            "     - Notes",
        ]
    )
    for name, cls_name, knobs, notes in rows:
        lines.extend(
            [
                f"   * - ``{name}``",
                f"     - ``{cls_name}``",
                f"     - {knobs}",
                f"     - {notes}",
            ]
        )
    return "\n".join(lines)


def _api_table(rows: list[tuple[str, str]]) -> str:
    lines = [".. list-table::", "   :header-rows: 1", ""]
    lines.extend(
        [
            "   * - Symbol",
            "     - Description",
        ]
    )
    for symbol, desc in rows:
        lines.extend(
            [
                f"   * - ``{symbol}``",
                f"     - {desc}",
            ]
        )
    return "\n".join(lines)


def _config_table(rows: list[tuple[str, str, str]]) -> str:
    lines = [".. list-table::", "   :header-rows: 1", ""]
    lines.extend(
        [
            "   * - Config",
            "     - Main fields",
            "     - Purpose",
        ]
    )
    for name, fields_text, purpose in rows:
        lines.extend(
            [
                f"   * - ``{name}``",
                f"     - {fields_text}",
                f"     - {purpose}",
            ]
        )
    return "\n".join(lines)


def _default_config_yaml_block() -> str:
    """Return the bundled default YAML as an indented RST code block body."""
    text = (
        resources.files("soma.configs")
        .joinpath("default.yaml")
        .read_text(encoding="utf-8")
        .strip()
    )
    return indent(text, "   ")


def build_cli_rst() -> str:
    return dedent(
        """\
        CLI
        ===

        Use ``soma`` to run YAML experiments, inspect registered components, and
        reproduce benchmarks.

        Basic usage
        -----------

        Run a configuration::

            soma /path/to/config.yaml

        ``python -m soma /path/to/config.yaml`` is equivalent.

        Available commands
        ------------------

        .. list-table::
           :header-rows: 1
           :widths: 42 58

           * - Command
             - Purpose
           * - ``soma CONFIG``
             - Run a pipeline from YAML.
           * - ``soma list encoders [--level LEVEL]``
             - List encoders, optionally filtered to ``tile``, ``slide``, or ``patient``.
           * - ``soma list aggregators``
             - List MIL aggregators.
           * - ``soma list decoders``
             - List dense neural decoders.
           * - ``soma list pixel-classifiers``
             - List decoder-free pixel classifiers.
           * - ``soma list tasks``
             - List task heads.
           * - ``soma list benchmarks``
             - List names accepted by ``reproduce`` and ``leaderboard``.

        Benchmark commands
        ------------------

        Reproduce one registered benchmark::

            soma reproduce NAME --raw-root /path/to/data

        Use ``--curated-dir`` to reuse manifests, ``--from-run-dir`` to score an
        existing run, and ``--seeds 1`` for a smoke run. ``--encoder`` and
        ``--spacing`` select registered benchmark axes. A family name such as
        ``eva`` runs every ``eva/<dataset>`` member. ``--record`` appends the
        measured value and provenance to the packaged results ledger.

        Gate references produce ``PASS`` or ``FAIL``; external references are
        reported as non-gating deltas. See :doc:`benchmarking` for the distinction.

        Build a faceted view over completed run directories::

            soma leaderboard [NAME] --root OUTPUT_ROOT --vary encoder

        ``--fix AXIS=VALUE`` holds an axis constant. ``--like RUN_DIR`` inherits
        fixed axes from an existing run. ``--metric`` and ``--split`` override the
        ranked value.

        What the CLI expects
        --------------------

        YAML is nested by concern and loaded through
        :func:`soma.config.load_config`. Start with :doc:`getting-started`; consult
        :doc:`configuration` for the generated canonical schema and :doc:`pipeline`
        for the corresponding Python API.
        """
    ).rstrip() + "\n"


def write_cli_rst(path: str | Path | None = None) -> Path:
    """Write the generated CLI guide page to disk."""

    target = Path(path) if path is not None else Path(__file__).with_name("cli.rst")
    target.write_text(build_cli_rst(), encoding="utf-8")
    return target


def build_configuration_rst() -> str:
    """Generate the exhaustive YAML configuration reference."""

    config_reference = _default_config_yaml_block()
    introduction = dedent(
        """\
        Configuration
        =============

        Soma configuration is nested by concern. ``run`` names the output,
        ``data`` selects the pipeline mode and manifests, and the remaining sections
        configure preprocessing, components, training, evaluation, and reports.
        Start with the smaller example in :doc:`getting-started`; use this page when
        you need an uncommon option. The :doc:`cli` accepts the same YAML through
        ``soma CONFIG``.

        Canonical YAML schema
        ---------------------

        This block is generated from ``soma/configs/default.yaml``. Copy only the
        sections you need and replace neutral values such as ``encoder: null`` and
        ``aggregation: null`` for your experiment.
        """
    )
    return introduction + "\n.. code-block:: yaml\n\n" + config_reference + "\n"


def write_configuration_rst(path: str | Path | None = None) -> Path:
    """Write the generated configuration reference to disk."""

    target = Path(path) if path is not None else Path(__file__).with_name(
        "configuration.rst"
    )
    target.write_text(build_configuration_rst(), encoding="utf-8")
    return target


# --- benchmark pages (generated from the registered Benchmark definitions) -----------
#
# Each per-benchmark page is generated from its registered ``Benchmark`` object: the
# protocol summary comes from ``facet`` / ``primary_metric`` / ``canonical_seeds`` /
# ``reference_environment``, the reproduce command from ``benchmark.name``, and the
# reference numbers are read from the benchmark's ``expected()`` reference rows (packaged
# ``soma/benchmarks/reference/<name>.csv``) and rendered as a readable ``list-table`` — the
# gate band beside the measured cells, a linked source instead of dumping the CSV's verbose
# ``source`` column (no hand-typed numbers, no drift, no ``TBD``). A generator + a
# checked-in file kept in sync by ``tests/test_docs.py`` mirrors the ``cli.rst`` mechanism.

# Upstream provenance for each benchmark family's published reference band — the leaderboard
# the numbers were captured from, rendered as one clickable link next to the reference
# (replaces dumping the CSV's verbose ``source`` column on the page). A family absent here
# has a self-referential reference (e.g. a soma seed-0 regression gate), so there is no
# external source to link.
_REFERENCE_SOURCE = {
    "eva": (
        "kaiko-ai/eva pathology leaderboard",
        "https://github.com/kaiko-ai/eva/blob/main/tools/data/leaderboards/pathology.csv",
    ),
    "hest": (
        "HEST-Benchmark leaderboard (mahmoodlab/HEST)",
        "https://github.com/mahmoodlab/HEST#hest-benchmark",
    ),
}


def _kv_table(col_a: str, col_b: str, rows: list[tuple[str, str]], *, widths: str = "30 70") -> str:
    """A two-column ``list-table`` (header + ``rows``)."""
    lines = [".. list-table::", "   :header-rows: 1", f"   :widths: {widths}", ""]
    lines.extend([f"   * - {col_a}", f"     - {col_b}"])
    for a, b in rows:
        lines.extend([f"   * - {a}", f"     - {b}"])
    return "\n".join(lines)


def _reference_source_link(family: str) -> str:
    """A single clickable ``label <url>`` for where a benchmark's reference band came from."""
    label, url = _REFERENCE_SOURCE[family]
    return f"`{label} <{url}>`__"


def _reproduced_table(name: str, key_columns: tuple[str, ...]) -> str:
    """A public comparison of reproduced measurements and reference values.

    Only cells that have actually been run appear. Detailed provenance remains in the
    packaged results CSV rather than dominating the public benchmark page.
    """
    rows = load_results(name)
    if not rows:
        return (
            "No reproductions have been recorded yet. Run ``soma reproduce <name> --record`` "
            "to append a measured number + provenance to the results ledger."
        )
    header = [c.capitalize() for c in key_columns] + [
        "Soma (mean ± std)",
        f"{name.upper()} reference",
    ]
    lines = [".. list-table::", "   :header-rows: 1", ""]
    lines.extend([f"   * - {header[0]}"] + [f"     - {col}" for col in header[1:]])
    for row in rows:
        measured = f"{row.measured:.3f}" + (f" ± {row.std:.3f}" if row.std is not None else "")
        gates = [
            g
            for g in expected_rows(name, metric=row.metric, **row.key)
            if not g.is_external
        ]
        reference = f"{gates[0].expected:.3f}" if gates else "—"
        cells = [row.key.get(c, "") for c in key_columns] + [
            measured,
            reference,
        ]
        lines.extend([f"   * - {cells[0]}"] + [f"     - {cell}" for cell in cells[1:]])
    return "\n".join(lines)


def _ocelot_guidance_section(bench) -> str:
    """Summarize non-gating fully supervised context without protocol clutter."""
    external = [r for r in bench.expected() if r.is_external]
    if not external:
        return ""
    low = min(row.expected for row in external)
    high = max(row.expected for row in external)
    url = external[0].url
    return (
        "For context, `fully supervised OCELOT systems on histoboard <"
        + url
        + f">`__ report about {low:.2f}–{high:.2f} mean F1. They use a different,\n"
        "end-to-end protocol, so this range is non-gating context rather than a Soma target."
    )


def build_ocelot_benchmark_rst() -> str:
    """Generate the OCELOT benchmark page from the registered ``ocelot`` benchmark."""
    bench = get_benchmark("ocelot")
    fixed = bench.facet.fixed
    varied = ", ".join(f"``{v}``" for v in bench.facet.varied)
    seeds = ", ".join(str(s) for s in bench.canonical_seeds)
    anchor = f"``{ocelot_bench.ANCHOR_ENCODER}`` @ {ocelot_bench.ANCHOR_SPACING:g} µm/px"

    protocol_rows = [(f"``{key}``", f"``{value}``") for key, value in fixed.items()]
    protocol_rows += [
        ("varied axes", varied),
        ("primary metric", f"``{bench.primary_metric}``"),
        ("canonical seeds", f"``{seeds}``"),
        ("anchor", f"{anchor} (seed 0)"),
    ]
    axes_rows = [
        (f"``{enc}``", f"{spacing:g}")
        for enc, spacing in sorted(ocelot_bench._CONFIG_FILES)
    ]
    gate_rows = [
        (f"``{row.metric}``", f"{row.expected:.4f} ± {row.tolerance:.3f}")
        for row in bench.expected()
        if not row.is_external
    ]

    sections = [
        "OCELOT\n======",
        "What this benchmark measures\n----------------------------\n\n"
        "Evaluate Soma's :doc:`cell-detection path <detection>` on the\n"
        "`OCELOT 2023 <https://ocelot2023.grand-challenge.org/>`_ TCGA patches. A frozen\n"
        "encoder produces a dense token grid; ``lightweight_conv`` predicts class peak\n"
        "heatmaps. OCELOT's greedy matcher reports class-aware **mean F1 @ δ = 3 µm**.\n"
        "Thresholds are selected on ``tune`` and applied once to ``test``.",
        "Prepare the data\n----------------\n\n"
        "Accept the OCELOT terms, download the public release, and unzip\n"
        "``ocelot2023_v1.0.1``. Pass that directory as ``--raw-root``. Soma uses the\n"
        "1024×1024 cell patches and point annotations, and preserves OCELOT's\n"
        "train/validation/test split while preparing its standard manifests.",
        "Run the benchmark\n-----------------\n\n"
        "Prepare the manifests, train the canonical seed, score, and check the gate::\n\n"
        "    soma reproduce ocelot --raw-root /path/to/ocelot\n\n"
        "Use ``--seeds 1`` for a smoke test or ``--from-run-dir <dir>`` to rescore an\n"
        "existing run.",
        "What you can vary\n-----------------\n\n"
        "Use ``--encoder`` and ``--spacing`` to select one of the registered cells:\n\n"
        + _kv_table("Encoder", "Spacing (µm/px)", axes_rows, widths="50 50"),
        "Results\n-------\n\n"
        "This **gate reference** is Soma's Virchow2 @ 0.2 µm/px, seed-0 frozen-probe\n"
        "regression anchor. It is not an external leaderboard result:\n\n"
        + _kv_table("Metric", "Expected ± tolerance", gate_rows, widths="40 60")
        + "\n\n"
        + _ocelot_guidance_section(bench),
        "Protocol details\n----------------\n\n"
        "The fixed recipe varies ``encoder`` × ``spacing``:\n\n"
        + _kv_table("Axis / setting", "Value", protocol_rows),
        "See :doc:`benchmarking` for the shared benchmark workflow and :doc:`detection`\n"
        "for targets, loss, and matching.",
    ]
    return "\n\n".join(sections).rstrip() + "\n"


def build_eva_benchmark_rst() -> str:
    """Generate the EVA benchmark page from the registered ``eva/<dataset>`` family."""
    family = [get_benchmark(n) for n in list_benchmarks() if n.startswith("eva/")]
    head = family[0]  # protocol constants are shared across the family
    seeds = ", ".join(str(s) for s in head.canonical_seeds)

    encoder_rows = [
        (
            f"``{name}``" + (" (default)" if name == eva_bench.DEFAULT_ENCODER else ""),
            f"eva ``{spec.eva_key}``"
            + (f", slide2vec ``{spec.output_variant}`` output" if spec.output_variant else ""),
        )
        for name, spec in eva_bench.ENCODERS.items()
    ]
    protocol_rows = [
        ("head", "linear probe (``aggregation: null`` — each patch is its own bag)"),
        ("optimizer", f"AdamW, lr ``{eva_bench.LEARNING_RATE:g}``, weight_decay ``{eva_bench.WEIGHT_DECAY:g}``"),
        ("batch size", f"``{eva_bench.HEAD_BATCH_SIZE}``"),
        ("budget", f"eva's ``max_steps={eva_bench.MAX_STEPS}`` mapped to soma epochs"),
        ("metric", "``balanced_accuracy``"),
        ("varied axis", "``encoder``"),
        ("primary metric", f"``{head.primary_metric}`` (from ``summary.json``)"),
        ("canonical seeds", f"``{seeds}`` (averaged)"),
    ]

    dataset_rows = []
    for bench in family:
        dataset = bench.facet.fixed["dataset"]
        spec = eva_bench.DATASETS[dataset]
        eval_split = (
            "EVA validation (``tune_is_test: true``)"
            if spec.tune_is_test
            else "EVA test (real val + test)"
        )
        dataset_rows.append((f"``{bench.name}``", f"``{spec.task}``", eval_split))

    raw_layout_rows = [
        ("``bach``", "``ICIAR2018_BACH_Challenge/Photos/<class>/*.tif``"),
        ("``breakhis``", "the original BreaKHis tree; Soma selects 40× patches and EVA classes"),
        ("``crc``", "``NCT-CRC-HE-100K/`` and ``CRC-VAL-HE-7K/``"),
        (
            "``gleason_arvaniti``",
            "``train_validation_patches_750/`` or the original TMA and mask archives",
        ),
        ("``mhist``", "``images/*.png`` and ``annotations.csv``"),
        (
            "``patch_camelyon``",
            "``{train,val,test}/<class>/`` images or the six official HDF5 files",
        ),
    ]

    def _dataset_table() -> str:
        lines = [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 34 40 26",
            "",
            "   * - Benchmark",
            "     - Task head",
            "     - Eval split",
        ]
        for name, task, eval_split in dataset_rows:
            lines.extend([f"   * - {name}", f"     - {task}", f"     - {eval_split}"])
        return "\n".join(lines)

    sections = [
        "EVA\n===",
        "What this benchmark measures\n----------------------------\n\n"
        "Reproduce the `kaiko-ai/eva <https://github.com/kaiko-ai/eva>`_ patch-classification\n"
        "leaderboard with Soma's :doc:`classification` heads. Each ``eva/<dataset>`` entry\n"
        "uses the same frozen-tile linear probe and varies only ``encoder``.\n\n"
        "**Pipeline:** labelled patches → frozen encoder → linear head → balanced accuracy.",
        "Prepare the data\n----------------\n\n"
        "Download a supported EVA dataset and point ``--raw-root`` at the directory with\n"
        "the following contents. Soma converts it to the standard manifests automatically.\n\n"
        + _kv_table("Dataset", "Raw-root contents", raw_layout_rows, widths="30 70")
        + "\n\nFor train/validation-only datasets, EVA validation becomes Soma ``test`` and\n"
        "``tune_is_test`` preserves EVA's train-on-all-train protocol. Datasets with a real\n"
        "test split retain validation as ``tune`` and test as ``test``.",
        "Run the benchmark\n-----------------\n\n"
        "Reproduce one dataset::\n\n"
        "    soma reproduce eva/bach --raw-root /path/to/eva/bach\n\n"
        "Or run the family::\n\n"
        "    soma reproduce eva --raw-root /path/to/eva\n\n"
        "Select an encoder with ``--encoder`` (default ``"
        + eva_bench.DEFAULT_ENCODER
        + "``).",
        "What you can vary\n-----------------\n\n"
        "Compare the supported foundation-model encoders:\n\n"
        + _kv_table("Encoder", "EVA backbone", encoder_rows, widths="30 70")
        + "\n\nRun one dataset or the complete family:\n\n"
        + _dataset_table(),
        "Results\n-------\n\n"
        "Reproduced numbers\n~~~~~~~~~~~~~~~~~~\n\n"
        "Recorded Soma scores appear beside the published EVA balanced accuracies from "
        + _reference_source_link("eva")
        + ". Unrecorded cells are omitted; detailed run provenance remains in the packaged\n"
        "results CSV.\n\n"
        + _reproduced_table("eva", ("dataset", "encoder")),
        "Protocol details\n----------------\n\n"
        + _kv_table("Setting", "Value", protocol_rows),
        "See :doc:`benchmarking` for the shared benchmark workflow and :doc:`classification`\n"
        "for task-head details.",
    ]
    return "\n\n".join(sections).rstrip() + "\n"


def _hest_results_section() -> str:
    """Render the public reproduced-versus-reference HEST comparison."""
    report = reproduction_report("hest")
    if not report.cells:
        return (
            "No reproduced cells have been recorded yet. Run, for example::\n\n"
            "    soma reproduce hest/IDC --encoder uni2 --raw-root /path/to/hest-bench --record\n\n"
            "to record a Soma score next to the published HEST reference."
        )

    lines = [
        "Soma closely reproduces HEST's published Pearson scores using native slide2vec\n"
        "features. The table contains the task–encoder cells currently recorded for this\n"
        "documentation; ``soma reproduce`` prints the matching reference for any registered\n"
        "task. These external values are comparisons, not PASS/FAIL gates.\n",
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 24 28 24 24",
        "",
        "   * - Task",
        "     - Encoder",
        "     - Soma",
        "     - HEST reference",
    ]
    for cell in report.cells:
        lines.extend(
            [
                f"   * - {cell.dataset}",
                f"     - ``{cell.encoder}``",
                f"     - {cell.measured:.4f}",
                f"     - {cell.reference:.4f}",
            ]
        )

    rels = sorted(abs(100 * c.delta / c.reference) for c in report.cells if c.reference)
    if rels:
        mid = len(rels) // 2
        median_rel = rels[mid] if len(rels) % 2 else (rels[mid - 1] + rels[mid]) / 2
        worst = max(report.cells, key=lambda c: abs(c.delta / c.reference) if c.reference else 0)
        summary = (
            f"Across these {len(report.cells)} recorded task–encoder comparisons, the median "
            "relative difference is "
            f"**{median_rel:.2f}%**; the largest is "
            f"**{abs(100 * worst.delta / worst.reference):.2f}%** "
            f"({worst.dataset} with ``{worst.encoder}``)."
        )
    else:
        summary = ""

    return "\n".join(lines) + ("\n\n" + summary if summary else "")


def build_hest_benchmark_rst() -> str:
    """Generate the HEST benchmark page from the registered ``hest/<task>`` family.

    Family-aware (like EVA): renders every registered ``hest/<task>`` sub-benchmark, so a
    registered task appears automatically. The page documents the shared protocol, scoped
    data download, reproduction commands, and recorded results.
    """
    family = [get_benchmark(n) for n in list_benchmarks() if n.startswith("hest/")]
    head = family[0]  # protocol constants are shared across the family
    seeds = ", ".join(str(s) for s in head.canonical_seeds)
    alpha = ridge_alpha(50)  # 50 highly-variable genes per HEST task

    def _section(title: str, body: str) -> str:
        return f"{title}\n{'-' * len(title)}\n\n{body}"

    protocol_rows = [
        ("head", "closed-form Ridge probe — no trained head, no gradient loop"),
        (
            "features",
            "``StandardScaler`` → ``PCA(n_components="
            + f"{DEFAULT_PCA_COMPONENTS})`` fit on the fold's train spots (X only)",
        ),
        (
            "estimator",
            "``Ridge(solver='lsqr', fit_intercept=False)``, penalty "
            + f"``alpha = {alpha:g}`` = 100 / ({DEFAULT_PCA_COMPONENTS}·50)",
        ),
        ("targets", "50-gene ``log1p(counts)`` vector per 112 µm spot (baked by the curator)"),
        (
            "metric",
            "``pearson`` — per gene, pooled over test spots → mean over 50 genes → mean over folds",
        ),
        ("task family", f"``{head.facet.fixed['task']}``"),
        ("varied axis", "``encoder``"),
        ("primary metric", f"``{head.primary_metric}`` (from ``summary.json``)"),
        ("canonical seeds", f"``{seeds}`` (the probe is closed-form — one seed suffices)"),
    ]

    encoder_rows = [
        (
            f"``{hest_bench.DEFAULT_ENCODER}`` (default)",
            "HEST-Benchmark UNI2-h; slide2vec default output (CLS, 1536-d)",
        ),
        (
            "``virchow2``",
            "HEST-Benchmark Virchow2; slide2vec ``"
            + hest_bench.OUTPUT_VARIANTS["virchow2"]
            + "`` output (CLS-only, 1280-d)",
        ),
        (
            "``h-optimus-1``",
            "HEST-Benchmark H-Optimus-1; slide2vec default output (CLS, 1536-d)",
        ),
    ]

    task_rows = [(f"``{b.name}``", f"``{b.facet.fixed['dataset']}``") for b in family]

    download_cmd = (
        "    hf download MahmoodLab/hest-bench --include 'IDC/*' --exclude 'fm_v1/*' \\\n"
        "        --repo-type dataset --local-dir /path/to/hest-bench"
    )

    sections = [
        "HEST\n====",
        _section(
            "What this benchmark measures",
            "Predict a 50-gene expression vector from each 112 µm tile with a frozen encoder,\n"
            "reproducing `HEST-Benchmark <https://github.com/mahmoodlab/HEST>`_ (Jaume et al.,\n"
            "NeurIPS 2024). Each ``hest/<task>`` uses Soma's native slide2vec cache and the\n"
            "same closed-form :doc:`spatial-expression probe <regression>`; only ``encoder``\n"
            "varies. No ``hest`` or TRIDENT runtime is required.\n\n"
            "**Pipeline:** spot tiles → frozen encoder → Ridge+PCA probe → mean Pearson.",
        ),
        _section(
            "Prepare the data",
            "Install Soma with the optional HEST readers (the base package also provides\n"
            "the ``hf`` download command)::\n\n"
            "    pip install 'soma-pathology[hest]'\n\n"
            "Some foundation-model weights require accepting their Hugging Face terms and\n"
            "authenticating once with ``hf auth login``. The HEST data itself is downloaded\n"
            "separately below.\n\n"
            "Download one task while excluding HEST's precomputed ``fm_v1`` features; Soma\n"
            "re-extracts features and prepares the downloaded tree locally::\n\n"
            + download_cmd
            + "\n\nOmit ``--include`` to download every registered task under the same local root.\n"
            "Start with one task and one encoder: feature extraction is the expensive step\n"
            "and a GPU is strongly recommended.\n"
            + "\n\nPass the downloaded task directory as ``--raw-root``. Soma writes its standard\n"
            "manifests automatically and preserves HEST's supplied fold assignments.",
        ),
        _section(
            "Run the benchmark",
            "Reproduce IDC::\n\n"
            "    soma reproduce hest/IDC --raw-root /path/to/hest-bench/IDC\n\n"
            "Or run every downloaded task::\n\n"
            "    soma reproduce hest --raw-root /path/to/hest-bench\n\n"
            "Select an encoder with ``--encoder`` (default ``"
            + hest_bench.DEFAULT_ENCODER
            + "``).",
        ),
        _section(
            "What you can vary",
            "Choose one of the encoders supported by the published HEST campaign:\n\n"
            + _kv_table("Encoder", "HEST backbone", encoder_rows, widths="30 70")
            + "\n\nRun one registered tissue task or the complete downloaded family:\n\n"
            + _kv_table("Benchmark", "HEST task", task_rows, widths="50 50")
            + "\n\nAll nine registered tissue tasks share the same protocol. HCC has no published score and is\n"
            "not registered.",
        ),
        _section(
            "Results",
            _hest_results_section()
            + "\n\nSee the "
            + _reference_source_link("hest")
            + " for the full published leaderboard.",
        ),
        _section("Protocol details", _kv_table("Setting", "Value", protocol_rows)),
        "See :doc:`benchmarking` for the shared benchmark workflow and :doc:`regression`\n"
        "for the probe and metric.",
    ]
    return "\n\n".join(sections).rstrip() + "\n"


def write_benchmark_rst(directory: str | Path | None = None) -> list[Path]:
    """Write the generated per-benchmark pages to disk."""
    base = Path(directory) if directory is not None else Path(__file__).parent
    written = []
    for filename, builder in (
        ("ocelot-detection-benchmark.rst", build_ocelot_benchmark_rst),
        ("eva-patch-classification-benchmark.rst", build_eva_benchmark_rst),
        ("hest-gene-expression-benchmark.rst", build_hest_benchmark_rst),
    ):
        target = base / filename
        target.write_text(builder(), encoding="utf-8")
        written.append(target)
    return written


def main() -> None:
    write_cli_rst()
    write_configuration_rst()
    write_benchmark_rst()


if __name__ == "__main__":
    main()
