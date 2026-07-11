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
    RESOLVABLE_EPS,
    expected_rows,
    get_benchmark,
    list_benchmarks,
    load_reference,
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
    """A ``list-table`` of soma's recorded measurements joined against the reference band.

    Built from the packaged results ledger (``results/<name>.csv``) via ``load_results`` — so
    only cells that have actually been run appear, each next to its reference number, the
    delta, and the provenance (seeds, date, commit) that produced it. Returns a plain
    "nothing recorded yet" note when the ledger is empty (or absent), so a benchmark with no
    reproductions still renders.
    """
    rows = load_results(name)
    if not rows:
        return (
            "No reproductions have been recorded yet. Run ``soma reproduce <name> --record`` "
            "to append a measured number + provenance to the results ledger."
        )
    header = [c.capitalize() for c in key_columns] + [
        "soma (mean ± std)",
        "Seeds",
        "Reference",
        "Δ",
        "Recorded (date @ commit)",
    ]
    lines = [".. list-table::", "   :header-rows: 1", ""]
    lines.extend([f"   * - {header[0]}"] + [f"     - {col}" for col in header[1:]])
    for row in rows:
        measured = f"{row.measured:.3f}" + (f" ± {row.std:.3f}" if row.std is not None else "")
        seeds = "" if row.n_seeds is None else str(row.n_seeds)
        gates = [
            g
            for g in expected_rows(name, metric=row.metric, **row.key)
            if not g.is_external
        ]
        reference = f"{gates[0].expected:.3f}" if gates else "—"
        delta = f"{row.measured - gates[0].expected:+.3f}" if gates else "—"
        commit = f"``{row.soma_commit}``" if row.soma_commit else "—"
        recorded = f"{row.date} @ {commit}" if row.date else commit
        cells = [row.key.get(c, "") for c in key_columns] + [
            measured,
            seeds,
            reference,
            delta,
            recorded,
        ]
        lines.extend([f"   * - {cells[0]}"] + [f"     - {cell}" for cell in cells[1:]])
    return "\n".join(lines)


def _ocelot_guidance_section(bench) -> str:
    """The non-gating external/guidance anchors as a clickable, clearly-labelled section.

    Generated from the registered benchmark's ``external`` reference rows (issue #226): each
    renders as an anonymous RST hyperlink (``label <url>``__) so the docs page links the
    snapshotted source, framed as context — never a target ``soma reproduce`` gates on.
    """
    external = [r for r in bench.expected() if r.is_external]
    bullets = []
    for row in external:
        note = f" — {row.source}" if row.source else ""
        bullets.append(
            f"* `{row.label} <{row.url}>`__ — ``{row.metric}`` ≈ {row.expected:.2f}{note}"
        )
    return (
        "Guidance anchors (non-gating)\n-----------------------------\n\n"
        "These snapshotted `histoboard <https://wearewaiv.github.io/histoboard/>`__\n"
        "values come from fully supervised, end-to-end systems, not Soma's frozen probe.\n"
        "They provide context only; ``soma reproduce`` never gates on them:\n\n"
        + "\n".join(bullets)
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
    env_rows = [(f"``{key}``", f"``{value}``") for key, value in bench.reference_environment.items()]
    gate_rows = [
        (f"``{row.metric}``", f"{row.expected:.4f} ± {row.tolerance:.3f}")
        for row in bench.expected()
        if not row.is_external
    ]

    sections = [
        "OCELOT\n======",
        "Purpose\n-------\n\n"
        "Evaluate Soma's :doc:`cell-detection path <detection>` on the\n"
        "`OCELOT 2023 <https://ocelot2023.grand-challenge.org/>`_ TCGA patches. A frozen\n"
        "encoder produces a dense token grid; ``lightweight_conv`` predicts class peak\n"
        "heatmaps. OCELOT's greedy matcher reports class-aware **mean F1 @ δ = 3 µm**.\n"
        "Thresholds are selected on ``tune`` and applied once to ``test``.",
        "Protocol\n--------\n\n"
        "The fixed recipe varies ``encoder`` × ``spacing``:\n\n"
        + _kv_table("Axis / setting", "Value", protocol_rows)
        + "\n\nAvailable committed configurations:\n\n"
        + _kv_table("Encoder", "Spacing (µm/px)", axes_rows, widths="50 50"),
        "Prepare and run\n---------------\n\n"
        "Curate, train the canonical seed, score, and check the gate::\n\n"
        "    soma reproduce ocelot --raw-root /path/to/ocelot\n\n"
        "Use ``--encoder`` and ``--spacing`` for an ablation, ``--seeds 1`` for a smoke\n"
        "test, or ``--from-run-dir <dir>`` to rescore an existing run.",
        "Results\n-------\n\n"
        "This **gate reference** is Soma's Virchow2 @ 0.2 µm/px, seed-0 frozen-probe\n"
        "regression anchor. It is not an external leaderboard result:\n\n"
        + _kv_table("Metric", "Expected ± tolerance", gate_rows, widths="40 60")
        + "\n\n"
        + _ocelot_guidance_section(bench)
        + "\n\nReference environment\n~~~~~~~~~~~~~~~~~~~~~\n\n"
        + _kv_table("Component", "Version", env_rows, widths="40 60"),
        "See :doc:`benchmarking` for reference semantics, :doc:`curation` for the raw-data\n"
        "layout, and :doc:`detection` for targets, loss, and matching.",
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
        "Purpose\n-------\n\n"
        "Reproduce the `kaiko-ai/eva <https://github.com/kaiko-ai/eva>`_ patch-classification\n"
        "leaderboard with Soma's :doc:`classification` heads. Each ``eva/<dataset>`` entry\n"
        "uses the same frozen-tile linear probe and varies only ``encoder``.",
        "Protocol\n--------\n\n"
        + _kv_table("Setting", "Value", protocol_rows)
        + "\n\nEncoder mappings:\n\n"
        + _kv_table("Encoder", "EVA backbone", encoder_rows, widths="30 70")
        + "\n\nDataset tasks and evaluation splits:\n\n"
        + _dataset_table()
        + "\n\nTrain/validation-only datasets use validation as Soma ``test``; ``patch_camelyon``\n"
        "retains its held-out test split.",
        "Prepare and run\n---------------\n\n"
        "Reproduce one dataset::\n\n"
        "    soma reproduce eva/bach --raw-root /path/to/eva/bach\n\n"
        "Or run the family::\n\n"
        "    soma reproduce eva --raw-root /path/to/eva\n\n"
        "Select an encoder with ``--encoder`` (default ``"
        + eva_bench.DEFAULT_ENCODER
        + "``).",
        "Results\n-------\n\n"
        "Reproduced numbers\n~~~~~~~~~~~~~~~~~~\n\n"
        "Recorded cells from ``soma/benchmarks/results/eva.csv`` appear below with seed,\n"
        "commit, and delta provenance. References are published EVA balanced accuracies\n"
        "keyed by dataset × encoder from "
        + _reference_source_link("eva")
        + ". Unrecorded cells are omitted:\n\n"
        + _reproduced_table("eva", ("dataset", "encoder")),
        "See :doc:`benchmarking` for gate semantics, :doc:`curation` for raw layouts, and\n"
        ":doc:`classification` for task-head details.",
    ]
    return "\n\n".join(sections).rstrip() + "\n"


def _hest_reproduction_section() -> str:
    """Render the A/B/C reproduction proof from ``reproduction_report("hest")``.

    A (per-cell delta), B (pooled pairwise rank concordance — the headline — plus per-task
    Spearman), C (the provenance-pinned append-only ledger as drift guard). Built purely from
    ``results/hest.csv`` ⋈ ``reference/hest.csv``, so it grows as ``--record`` fills the ledger
    and renders an honest "nothing yet" note while empty.
    """
    report = reproduction_report("hest")
    intro = (
        "Soma extracts native slide2vec features rather than HEST's TRIDENT features. Published\n"
        "HEST values are therefore **external, non-gating** references: no cross-stack delta\n"
        "produces PASS/FAIL (ADR 0005). The joined results and reference ledgers show:\n\n"
        "* **A — absolute agreement:** Pearson and signed slide2vec↔TRIDENT delta per cell.\n"
        "* **B — rank agreement:** pooled encoder-pair concordance where HEST's separation is\n"
        f"  greater than {RESOLVABLE_EPS}, plus per-task Spearman ρ.\n"
        "* **C — drift guard:** append-only commit and slide2vec provenance for Soma-to-Soma\n"
        "  comparisons. This is the only comparison suitable for regression gating."
    )

    if not report.cells:
        return (
            intro
            + "\n\nNo cells reproduced yet. Run, e.g.::\n\n"
            "    soma reproduce hest/IDC --encoder uni2 --raw-root /path/to/hest-bench --record\n\n"
            "to append a measured Pearson + provenance to ``soma/benchmarks/results/hest.csv``;\n"
            "this section then renders the A/B/C proof automatically."
        )

    # A — per-cell table. The relative delta is shown next to the absolute one because the same
    # absolute gap means different things at Pearson 0.30 (COAD) and 0.57 (LUNG).
    a_header = ["Task", "Encoder", "soma", "HEST", "Δ", "Δ %", "Recorded"]
    a_lines = [".. list-table::", "   :header-rows: 1", ""]
    a_lines.extend([f"   * - {a_header[0]}"] + [f"     - {c}" for c in a_header[1:]])
    for cell in report.cells:
        commit = f"``{cell.soma_commit}``" if cell.soma_commit else "—"
        recorded = f"{cell.date} @ {commit}" if cell.date else commit
        rel = 100 * cell.delta / cell.reference if cell.reference else 0.0
        vals = [
            cell.dataset,
            f"``{cell.encoder}``",
            f"{cell.measured:.4f}",
            f"{cell.reference:.4f}",
            f"{cell.delta:+.4f}",
            f"{rel:+.2f}%",
            recorded,
        ]
        a_lines.extend([f"   * - {vals[0]}"] + [f"     - {v}" for v in vals[1:]])

    # Spread of the parity gap, stated rather than gated — the reader judges it.
    rels = sorted(abs(100 * c.delta / c.reference) for c in report.cells if c.reference)
    if rels:
        mid = len(rels) // 2
        median_rel = rels[mid] if len(rels) % 2 else (rels[mid - 1] + rels[mid]) / 2
        worst = max(report.cells, key=lambda c: abs(c.delta / c.reference) if c.reference else 0)
        a_spread = (
            f"\n\nAcross {len(report.cells)} cell(s) the parity gap is a median "
            f"**{median_rel:.2f}%** relative, worst **{abs(100 * worst.delta / worst.reference):.2f}%** "
            f"({worst.dataset}/``{worst.encoder}``). Stated, not gated: see ADR 0005."
        )
    else:
        a_spread = ""

    # B — concordance (a bonus) + Spearman.
    def _frac(n: int, d: int) -> str:
        return f"{n}/{d} ({n / d:.0%})" if d else "—"

    ca = report.concordance_all
    n_within_noise = len(report.pairs) - report.n_resolvable
    concordance_line = (
        f"**Pooled pairwise rank concordance: {_frac(report.n_resolvable_concordant, report.n_resolvable)}**"
        f" on resolvable pairs (HEST separates them by more than {RESOLVABLE_EPS})"
        + (f"; {n_within_noise} within-noise pair(s) excluded" if n_within_noise else "")
        + ".\n"
        f"Over *all* pairs (resolvable + within-noise): "
        f"{sum(1 for p in report.pairs if p.concordant)}/{len(report.pairs)}"
        + (f" ({ca:.0%})" if ca is not None else "")
        + "."
    )
    discordant = [p for p in report.pairs if p.resolvable and not p.concordant]
    if discordant:
        disagree = "\n\nResolvable pairs soma orders *differently* from HEST (reported, not gated):\n\n" + "\n".join(
            f"* {p.dataset}: HEST ``{p.encoder_high}`` > ``{p.encoder_low}`` "
            f"(Δref {p.reference_gap:+.4f}) but soma reverses it (Δsoma {p.measured_gap:+.4f})"
            for p in discordant
        )
    else:
        disagree = "\n\nEvery resolvable pair is concordant — soma reproduces HEST's ordering wherever HEST resolves it."
    spearman_rows = [
        (task, "—" if rho is None else f"{rho:+.3f}")
        for task, rho in report.spearman_by_dataset.items()
    ]
    spearman_table = _kv_table("Task", "Spearman ρ (soma vs HEST)", spearman_rows, widths="50 50")

    # C — provenance.
    commits = ", ".join(f"``{c}``" for c in report.soma_commits) or "—"
    versions = ", ".join(report.slide2vec_versions) or "—"
    provenance = (
        f"Recorded at soma commit(s) {commits}, slide2vec {versions}. The ledger "
        "(``soma/benchmarks/results/hest.csv``) is append-only, so re-running a cell at a new "
        "commit adds a row — drift never overwrites history."
    )

    return (
        intro
        + "\n\n**A — per-cell agreement (published, not gated)**\n\n"
        + "\n".join(a_lines)
        + a_spread
        + "\n\n**B — rank concordance (bonus)**\n\n"
        + concordance_line
        + disagree
        + "\n\n"
        + spearman_table
        + "\n\n**C — drift guard**\n\n"
        + provenance
    )


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

    # Keep the inline IDC summary aligned with the three supported campaign encoders. The
    # authoritative leaderboard and packaged CSV linked below retain the exhaustive evidence.
    campaign_encoders = {"uni2", "virchow2", "h-optimus-1"}
    leaderboard_rows = [
        (f"``{row.key['encoder']}``", f"{row.expected:.4f}")
        for row in sorted(
            (
                r
                for r in load_reference("hest")
                if r.is_external
                and r.metric == head.primary_metric
                and r.key.get("dataset") == "IDC"
                and r.key.get("encoder") in campaign_encoders
            ),
            key=lambda r: r.expected,
            reverse=True,
        )
    ]

    sections = [
        "HEST\n====",
        _section(
            "Purpose",
            "Predict a 50-gene expression vector from each 112 µm tile with a frozen encoder,\n"
            "reproducing `HEST-Benchmark <https://github.com/mahmoodlab/HEST>`_ (Jaume et al.,\n"
            "NeurIPS 2024). Each ``hest/<task>`` uses Soma's native slide2vec cache and the\n"
            "same closed-form :doc:`spatial-expression probe <regression>`; only ``encoder``\n"
            "varies. No ``hest`` or TRIDENT runtime is required.",
        ),
        _section(
            "Protocol",
            _kv_table("Setting", "Value", protocol_rows)
            + "\n\nSupported campaign encoders:\n\n"
            + _kv_table("Encoder", "HEST backbone", encoder_rows, widths="30 70")
            + "\n\nRegistered tasks:\n\n"
            + _kv_table("Benchmark", "HEST task", task_rows, widths="50 50")
            + "\n\nAll nine scored tasks share this protocol. HCC has no published score and is not\n"
            "registered.",
        ),
        _section(
            "Prepare and run",
            "Download one task while excluding HEST's precomputed ``fm_v1`` features; Soma\n"
            "re-extracts features and curates the downloaded tree offline (ADR 0004)::\n\n"
            + download_cmd
            + "\n\nReproduce IDC::\n\n"
            "    soma reproduce hest/IDC --raw-root /path/to/hest-bench/IDC\n\n"
            "Or run every downloaded task::\n\n"
            "    soma reproduce hest --raw-root /path/to/hest-bench\n\n"
            "Select an encoder with ``--encoder`` (default ``"
            + hest_bench.DEFAULT_ENCODER
            + "``).",
        ),
        _section(
            "Results",
            "Published IDC references\n~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
            "HEST's Ridge+PCA Pearson values are **external and non-gating**. The inline table\n"
            "shows the supported campaign encoders; see the "
            + _reference_source_link("hest")
            + " and `packaged reference CSV\n"
            "<https://github.com/clemsgrs/soma/blob/main/soma/benchmarks/reference/hest.csv>`__\n"
            "for the complete leaderboard and task × encoder evidence.\n\n"
            + _kv_table("Encoder", "Published ``pearson``", leaderboard_rows, widths="60 40")
            + "\n\nReproduction — is it sound?\n~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
            + _hest_reproduction_section(),
        ),
        "See :doc:`benchmarking` for reference semantics, :doc:`curation` for HEST input\n"
        "contracts, and :doc:`regression` for the probe and metric.",
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
