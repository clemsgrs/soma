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
    config_reference = _default_config_yaml_block()
    body = (
        dedent(
        """\
        CLI
        ===

        ``soma`` exposes a compact command-line interface for running
        experiments from YAML config files and for listing the available model
        presets.

        .. figure:: /_static/figures/run-flow.svg
           :figclass: soma-figure
           :alt: Three input files flow into one soma command that schedules tiling, feature extraction, training, and metrics.

           You provide three files — a dataset, splits, and a config. ``soma`` then
           schedules every step: tiling, feature extraction, training, and metrics.

        Basic usage
        -----------

        The main entrypoint takes a config path directly::

            soma /path/to/config.yaml

        You can also invoke it through Python if you prefer::

            python -m soma /path/to/config.yaml

        Available commands
        ------------------

        ``soma CONFIG``
           Run a full pipeline from the given YAML config file.

        ``soma list encoders [--level {tile,slide,patient}]``
           List all registered encoder presets. ``--level`` narrows results to
           ``tile``, ``slide``, or ``patient`` encoders.

        ``soma list aggregators``
           List all registered MIL aggregator presets.

        ``soma list decoders``
           List all registered dense decoder presets.

        ``soma list pixel-classifiers``
           List all registered per-pixel classifier presets.

        ``soma list tasks``
           List all registered task-head presets.

        ``soma list benchmarks``
           List all registered foundation-model benchmarks — the names
           ``soma reproduce`` and ``soma leaderboard`` accept.

        Benchmarking commands
        ---------------------

        These drive the registered benchmarks (see :doc:`benchmarking` for the
        end-to-end curate → configure → run → leaderboard → reproduce story).

        ``soma reproduce NAME [--raw-root DIR | --curated-dir DIR | --from-run-dir DIR] [--seeds N]``
           Curate → run → score a registered benchmark and tolerance-check its
           primary metric against the packaged reference band. ``NAME`` is a
           registered benchmark (``ocelot``, ``eva/bach``) or a family prefix
           (``eva``) that fans out over every ``eva/<dataset>``. Three manifest
           sources: ``--raw-root`` curates from raw data; ``--curated-dir`` reuses an
           already-curated manifest dir (``dataset.csv`` + ``splits.csv``), skipping
           curation; ``--from-run-dir`` re-scores an existing run without retraining.
           ``--seeds 1`` is the quickest smoke.

        ``soma leaderboard [NAME] --root OUTPUT_ROOT [--vary AXIS] [--fix AXIS=VALUE] [--like DIR]``
           Render a faceted leaderboard over the completed run dirs under an
           output root. A benchmark ``NAME`` supplies the canonical facet and
           reference band; ``--vary`` / ``--fix`` / ``--like`` shape the facet on
           top of it.

        What the CLI expects
        --------------------

        The config file follows the canonical nested schema below. This block
        is generated from ``soma/configs/default.yaml``, the bundled defaults
        merged by :func:`soma.config.load_config`. Copy it when you want the
        baseline public YAML shape, then replace neutral defaults such as
        ``encoder: null`` and ``aggregation: null`` for your run.

        Full config reference
        ---------------------

        .. code-block:: yaml

        """
        )
        + config_reference
        + "\n\n"
        + dedent(
            """\
        See also
        --------

        * :doc:`pipeline` – Python API equivalent of each config section
        * :doc:`getting-started` – end-to-end walkthrough

        """
        )
    )
    return body.rstrip() + "\n"


def write_cli_rst(path: str | Path | None = None) -> Path:
    """Write the generated CLI guide page to disk."""

    target = Path(path) if path is not None else Path(__file__).with_name("cli.rst")
    target.write_text(build_cli_rst(), encoding="utf-8")
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

_GENERATED_PAGE_NOTE = (
    ".. note::\n\n"
    "   This page is generated from the registered benchmark definition — the protocol\n"
    "   summary and reference numbers from the ``Benchmark`` object's ``expected()`` rows\n"
    "   (packaged ``{csv}``), and the command from the benchmark name. Edit the registry\n"
    "   (``{module}``) and the CSV, not this page; ``python docs/_generate_reference.py``\n"
    "   re-emits it and ``tests/test_docs.py`` guards the two from drifting."
)

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
        "External reference points shown for **context only** — the official challenge\n"
        "baseline and best-reported numbers, snapshotted (not live-scraped) from\n"
        "`histoboard <https://wearewaiv.github.io/histoboard/>`__. They measure a\n"
        "*different* protocol than soma's frozen probe (fully-supervised, end-to-end, not\n"
        "tied to any encoder), so ``soma reproduce`` **never gates** on them; they only show\n"
        "how far the frozen-probe result stands from the best reported result:\n\n"
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
        "*Maps to task:* :doc:`detection` — soma's :doc:`detection path <detection>`\n"
        "reproduced on the `OCELOT 2023 <https://ocelot2023.grand-challenge.org/>`_\n"
        "cell-detection challenge.",
        _GENERATED_PAGE_NOTE.format(
            csv="soma/benchmarks/reference/ocelot.csv", module="soma/benchmarks/ocelot.py"
        ),
        "OCELOT 2023 provides paired cell + tissue patches from TCGA. This benchmark is\n"
        "**cell-only**: a **frozen** foundation-model encoder produces a dense token grid,\n"
        "a ``lightweight_conv`` decoder regresses a per-class peak heatmap, and the\n"
        ":class:`~soma.tasks.detection.DetectionHead` scores it with OCELOT's class-aware\n"
        "**mean F1 @ δ = 3 µm**, greedy-matched — the leaderboard-comparable operating\n"
        "point (per-class score thresholds swept on ``tune``, frozen, applied once to\n"
        "``test``). See :doc:`detection` for the canonical matcher and px↔µm definitions.",
        "Protocol\n--------\n\n"
        "The recipe backbone is held fixed; the facet varies ``encoder`` × ``spacing``.\n\n"
        + _kv_table("Axis / setting", "Value", protocol_rows),
        "Axes\n----\n\n"
        "``build_config`` resolves a committed config per ``(encoder, spacing)`` — the\n"
        "2×2 magnification-alignment ablation plus the native anchor:\n\n"
        + _kv_table("Encoder", "Spacing (µm/px)", axes_rows, widths="50 50"),
        "Reference band\n--------------\n\n"
        "The tolerance band ``soma reproduce`` checks against — a **config-agnostic** banner\n"
        "(soma's own frozen-probe Virchow2 @ 0.2 µm/px seed-0 headline, used as a regression\n"
        "anchor, not an external leaderboard number). The non-gating external anchors —\n"
        "fully-supervised end-to-end baselines from a *different* protocol — are surfaced\n"
        "with clickable links under *Guidance anchors* below:\n\n"
        + _kv_table("Metric", "Reference band (expected ± tolerance)", gate_rows, widths="40 60"),
        _ocelot_guidance_section(bench),
        "Reference environment\n---------------------\n\n"
        "The recorded anchor environment the reference number was produced in:\n\n"
        + _kv_table("Component", "Version", env_rows, widths="40 60"),
        "Reproduce\n---------\n\n"
        "One command curates the raw data, trains the anchor for the canonical seed,\n"
        "greedy-scores it, and tolerance-checks ``mean_f1`` against the band above::\n\n"
        "    soma reproduce ocelot --raw-root /path/to/ocelot\n\n"
        "Fast paths: ``--from-run-dir <dir>`` re-scores an existing run with the greedy\n"
        "matcher (no training); ``--seeds 1`` is the quickest smoke. Sweep the ablation\n"
        "with ``--encoder`` / ``--spacing`` (e.g. ``soma reproduce ocelot --encoder uni2\n"
        "--spacing 0.25 --raw-root ...``).",
        ".. seealso::\n\n"
        "   * :doc:`detection` — the detection modeling substrate (head, target encoding,\n"
        "     loss, F1@δ evaluator).\n"
        "   * :doc:`benchmarking` — the shared curate → run → leaderboard → reproduce guide.\n"
        "   * :doc:`curation` — the OCELOT curator and split policy.",
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

    reproduce_lines = "\n".join(
        f"    soma reproduce {bench.name} --raw-root /path/to/eva/{bench.facet.fixed['dataset']}"
        for bench in family
    )

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
        "*Maps to task:* :doc:`classification` — frozen-tile linear-probe runs of the\n"
        "binary / multiclass classification heads reproducing the\n"
        "`kaiko-ai/eva <https://github.com/kaiko-ai/eva>`_ patch-classification leaderboard.",
        _GENERATED_PAGE_NOTE.format(
            csv="soma/benchmarks/reference/eva.csv", module="soma/benchmarks/eva.py"
        ),
        "EVA is registered as **one sub-benchmark per dataset** (``eva/<dataset>``), each\n"
        "sharing the same offline linear-probe recipe and varying only the ``encoder`` axis.\n"
        "``soma reproduce eva`` fans out over the whole family; a single ``eva/<dataset>``\n"
        "reproduces one dataset.",
        "The frozen-tile-probe protocol\n------------------------------\n\n"
        "Stated once, shared by every dataset:\n\n"
        + _kv_table("Setting", "Value", protocol_rows),
        "Encoders\n--------\n\n"
        "The ``encoder`` axis maps a soma encoder onto an EVA leaderboard backbone:\n\n"
        + _kv_table("Encoder", "EVA backbone", encoder_rows, widths="30 70"),
        "Datasets\n--------\n\n"
        "Where EVA ships only train/validation, the EVA validation split becomes soma\n"
        "``test`` and the run sets ``tune_is_test: true`` (train-on-all-train /\n"
        "evaluate-on-validation); ``patch_camelyon`` has a real held-out test split:\n\n"
        + _dataset_table(),
        "Reproduced numbers\n------------------\n\n"
        "What soma has actually measured, recorded by ``soma reproduce --record`` into the\n"
        "packaged results ledger (``soma/benchmarks/results/eva.csv``) alongside the commit\n"
        "and slide2vec version that produced each number. The ``Reference`` column is the\n"
        "published EVA balanced-accuracy band (keyed by ``dataset`` × ``encoder``, from "
        + _reference_source_link("eva")
        + "); only cells that have been run appear, each with its delta to that band:\n\n"
        + _reproduced_table("eva", ("dataset", "encoder")),
        "Reproduce\n---------\n\n"
        "``soma reproduce`` curates the raw layout, trains the linear probe over the\n"
        "canonical seeds, reads ``test/balanced_accuracy`` from ``summary.json``, and\n"
        "tolerance-checks it against the band above. Reproduce one dataset::\n\n"
        + reproduce_lines
        + "\n\n"
        "…or fan out over the whole family in one go (each member owns a per-dataset\n"
        "subdirectory)::\n\n"
        "    soma reproduce eva --raw-root /path/to/eva\n\n"
        "Pick the encoder axis with ``--encoder`` (default ``"
        + eva_bench.DEFAULT_ENCODER
        + "``); ``--seeds 1`` runs a single-seed smoke.",
        ".. seealso::\n\n"
        "   * :doc:`classification` — the task heads the probe trains (binary, multiclass).\n"
        "   * :doc:`benchmarking` — the shared curate → run → leaderboard → reproduce guide.\n"
        "   * :doc:`curation` — the EVA curators and split policy.",
    ]
    return "\n\n".join(sections).rstrip() + "\n"


def _hest_reproduction_section() -> str:
    """Render the A/B/C reproduction proof from ``reproduction_report("hest")``.

    A (per-cell delta), B (pooled pairwise rank concordance — the headline — plus per-task
    Spearman), C (the provenance-pinned append-only ledger as drift guard). Built purely from
    ``results/hest.csv`` ⋈ ``reference/hest.csv``, so it grows as ``--record`` fills the ledger
    and renders an honest "nothing yet" note while empty.
    """
    from soma.benchmarks.reproduction import RESOLVABLE_EPS

    report = reproduction_report("hest")
    intro = (
        "soma reproduces HEST **natively** — its own slide2vec features, not HEST's TRIDENT\n"
        "extraction — so the proof of soundness is **not** that the numbers match to the\n"
        "decimal (the extraction stacks differ), but that soma re-derives HEST's **ranking**\n"
        "of encoders. Three views, computed from the results ledger joined to the published\n"
        "reference:\n\n"
        "* **A — absolute agreement** (per cell): soma's Pearson beside HEST's, and the signed\n"
        "  delta. Shown, never gated — the delta is the accepted slide2vec↔TRIDENT parity gap.\n"
        "* **B — rank agreement** (the headline): **pooled pairwise concordance** — over every\n"
        "  (task, encoder-pair), the fraction soma orders the same way HEST does. A pair is\n"
        f"  *resolvable* when HEST separates it by more than {RESOLVABLE_EPS} on the metric; the\n"
        "  headline is concordance over resolvable pairs, so soma is not graded on within-noise\n"
        "  coin-flips. Per-task Spearman ρ is shown alongside (coarse at few encoders).\n"
        "* **C — drift guard**: the ledger is append-only and provenance-pinned (commit,\n"
        "  slide2vec version), so a re-run at a new commit adds a row and drift is a visible diff."
    )

    if not report.cells:
        return (
            intro
            + "\n\nNo cells reproduced yet. Run, e.g.::\n\n"
            "    soma reproduce hest/IDC --encoder uni2 --raw-root /path/to/hest-bench --record\n\n"
            "to append a measured Pearson + provenance to ``soma/benchmarks/results/hest.csv``;\n"
            "this section then renders the A/B/C proof automatically."
        )

    # A — per-cell table.
    a_header = ["Task", "Encoder", "soma", "HEST", "Δ", "Recorded"]
    a_lines = [".. list-table::", "   :header-rows: 1", ""]
    a_lines.extend([f"   * - {a_header[0]}"] + [f"     - {c}" for c in a_header[1:]])
    for cell in report.cells:
        commit = f"``{cell.soma_commit}``" if cell.soma_commit else "—"
        recorded = f"{cell.date} @ {commit}" if cell.date else commit
        vals = [
            cell.dataset,
            f"``{cell.encoder}``",
            f"{cell.measured:.4f}",
            f"{cell.reference:.4f}",
            f"{cell.delta:+.4f}",
            recorded,
        ]
        a_lines.extend([f"   * - {vals[0]}"] + [f"     - {v}" for v in vals[1:]])

    # B — concordance headline + Spearman.
    def _frac(n: int, d: int) -> str:
        return f"{n}/{d} ({n / d:.0%})" if d else "—"

    cr, ca = report.concordance_resolvable, report.concordance_all
    headline = (
        f"**Pooled pairwise rank concordance: {_frac(report.n_resolvable_concordant, report.n_resolvable)}**"
        " on resolvable pairs"
        + (f" — ρ-headline {cr:.0%}" if cr is not None else "")
        + ".\n"
        f"Over *all* pairs (resolvable + within-noise): "
        f"{sum(1 for p in report.pairs if p.concordant)}/{len(report.pairs)}"
        + (f" ({ca:.0%})" if ca is not None else "")
        + "."
    )
    discordant = [p for p in report.pairs if p.resolvable and not p.concordant]
    if discordant:
        disagree = "\n\nResolvable pairs soma orders *differently* from HEST (honest failures):\n\n" + "\n".join(
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
        + "\n\n**A — per-cell agreement**\n\n"
        + "\n".join(a_lines)
        + "\n\n**B — rank concordance (headline)**\n\n"
        + headline
        + disagree
        + "\n\n"
        + spearman_table
        + "\n\n**C — drift guard**\n\n"
        + provenance
    )


def build_hest_benchmark_rst() -> str:
    """Generate the HEST benchmark page from the registered ``hest/<task>`` family.

    Family-aware (like EVA): renders every registered ``hest/<task>`` sub-benchmark, so a
    fanned-out task appears automatically once ``HestBenchmark(task)`` is registered. The
    page also documents the scoped data download and the "adding a task" fan-out recipe —
    the point being that a new task is data + one ``HEST_TASKS`` entry + reference rows, never
    a change to the curator or the probe.
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

    reproduce_lines = "\n".join(
        f"    soma reproduce {b.name} --raw-root /path/to/hest-bench/{b.facet.fixed['dataset']}"
        for b in family
    )

    download_cmd = (
        "    hf download MahmoodLab/hest-bench --include 'IDC/*' --exclude 'fm_v1/*' \\\n"
        "        --repo-type dataset --local-dir /path/to/hest-bench"
    )

    fanout_body = (
        "All 9 scored tasks are already registered. The one hest-bench task *not* registered\n"
        "is ``HCC`` (liver): the HF hub ships its data tree, but HCC is **unscored** — no\n"
        "published leaderboard number — so it carries no reference row. Adding it (or any future\n"
        "task) is a **fan-out**: **data + one ``HEST_TASKS`` entry + reference rows** — never new\n"
        "machinery. ``curate_hest`` and the closed-form probe are task-agnostic, so a new task\n"
        "**never touches the curator or the probe**:\n\n"
        "**1. Download the task** (scoped; e.g. ``HCC``)::\n\n"
        "    hf download MahmoodLab/hest-bench --include 'HCC/*' --exclude 'fm_v1/*' \\\n"
        "        --repo-type dataset --local-dir /path/to/hest-bench\n\n"
        "**2. Curate** it into a ``spatial_expression`` Manifest with the *same* curator::\n\n"
        "    python -m soma.curation.hest --raw-root /path/to/hest-bench/HCC \\\n"
        "        --output-dir /path/to/curated/HCC --task HCC\n\n"
        "**3. Register** it by adding the task id to ``HEST_TASKS`` in ``soma/benchmarks/hest.py``\n"
        "— the module loop-registers ``HestBenchmark(task)`` for each, no new curator/probe code::\n\n"
        '    HEST_TASKS = (..., "HCC")  # loop-registers hest/HCC\n\n'
        "**4. Add external reference rows** for the task to ``soma/benchmarks/reference/hest.csv``\n"
        "— one ``kind=external`` row per encoder (the published Pearson, a ``label``, a ``url``).\n"
        "Without a published number a task can still run, but it has nothing to reproduce against.\n\n"
        "Then ``python docs/_generate_reference.py`` re-emits this page with the new task,\n"
        "``soma list benchmarks`` shows ``hest/HCC``, and ``soma reproduce hest/HCC`` runs —\n"
        "all from the same curator and the same probe."
    )

    # The published HEST leaderboard for this task, rendered readably (best first) instead of
    # dumping the reference CSV. ``load_reference`` returns every row unfiltered (the
    # benchmark's own ``expected()`` defaults the encoder axis, so it would show only one).
    # The full published IDC leaderboard (all ~18 encoders) is the illustrative one; the other
    # tasks' references (our 3 campaign encoders × 8 tasks) drive the Reproduction section below.
    leaderboard_rows = [
        (f"``{row.key['encoder']}``", f"{row.expected:.4f}")
        for row in sorted(
            (
                r
                for r in load_reference("hest")
                if r.is_external
                and r.metric == head.primary_metric
                and r.key.get("dataset") == "IDC"
            ),
            key=lambda r: r.expected,
            reverse=True,
        )
    ]

    sections = [
        "HEST\n====",
        "*Maps to task:* :doc:`regression` — a **frozen** patch encoder scored on\n"
        "**gene-expression-from-morphology**: predict a 50-gene expression vector from a\n"
        "112 µm tile, reproducing the\n"
        "`HEST-Benchmark <https://github.com/mahmoodlab/HEST>`_ (Jaume et al., NeurIPS 2024).",
        _GENERATED_PAGE_NOTE.format(
            csv="soma/benchmarks/reference/hest.csv", module="soma/benchmarks/hest.py"
        ),
        "HEST is registered as **one sub-benchmark per task** (``hest/<task>``), each sharing\n"
        "the same closed-form spatial-expression probe recipe and varying only the ``encoder``\n"
        "axis. soma reproduces it **natively** — its own slide2vec encoder → its per-spot\n"
        "feature cache → a closed-form Ridge+PCA probe — with **no dependency on the** ``hest``\n"
        "**library or TRIDENT**. All **9 HEST-Benchmark tasks** are registered (see *Tasks*);\n"
        "reproduction soundness is proven by **rank agreement** across them, not by matching the\n"
        "extraction stack (see *Reproduction — is it sound?*).",
        _section(
            "Protocol",
            "Stated once, shared by every task; the ``encoder`` axis is the only variable:\n\n"
            + _kv_table("Setting", "Value", protocol_rows),
        ),
        _section(
            "Encoders",
            "The ``encoder`` axis maps a soma encoder onto a HEST leaderboard backbone. Any\n"
            "slide2vec-registered encoder works (slide2vec validates the name); the variant is\n"
            "pinned only where the leaderboard used a non-default one:\n\n"
            + _kv_table("Encoder", "HEST backbone", encoder_rows, widths="30 70"),
        ),
        _section(
            "Tasks",
            "The registered sub-benchmark family — all 9 HEST-Benchmark tasks, spanning organs\n"
            "(breast, prostate, pancreas, colon, rectum, kidney, lung, skin):\n\n"
            + _kv_table("Benchmark", "HEST task", task_rows, widths="50 50")
            + "\n\nEach shares the *same* curator and closed-form probe; a task is data + a\n"
            "registration line + reference rows (*Adding a HEST task* below). The hest-bench HF\n"
            "dataset also ships an ``HCC`` (liver) tree, but HCC is **not** one of the 9 scored\n"
            "tasks (no published leaderboard number), so it is deliberately not registered.",
        ),
        _section(
            "Published leaderboard (IDC)",
            "HEST's published **external, non-gating** Ridge+PCA Pearson on the IDC task, per\n"
            "encoder (best first). There is **no gate row**: nothing is tolerance-checked.\n"
            "``soma reproduce hest/IDC`` renders soma's Measured row *beside* these, making the\n"
            "slide2vec↔TRIDENT extraction gap an explicit, non-gating delta. The other 8 tasks'\n"
            "references (our reproduction encoders × task) drive the reproduction proof below.\n"
            "Source: "
            + _reference_source_link("hest")
            + ".\n\n"
            + _kv_table("Encoder", "Published ``pearson``", leaderboard_rows, widths="60 40"),
        ),
        _section(
            "Reproduction — is it sound?",
            _hest_reproduction_section(),
        ),
        _section(
            "Download one task",
            "The curator is hermetic and offline (ADR 0004): provision the raw task tree once,\n"
            "out of band. Pull **only the needed task** and **exclude the** ``fm_v1/``\n"
            "**precomputed foundation-model features** (soma re-extracts them natively via\n"
            "slide2vec) — a few-GB task subtree, never the full multi-task / >1 TB HEST corpus::\n\n"
            + download_cmd
            + "\n\nThe scoped ``--include 'IDC/*'`` pulls just that task's ``patches/``, ``adata/``,\n"
            "``splits/`` and ``var_50genes.json``; ``--exclude 'fm_v1/*'`` drops the precomputed\n"
            "features. ``curate_hest`` then runs fully offline over the result.",
        ),
        _section(
            "Reproduce",
            "``soma reproduce`` curates the raw task tree, fits the closed-form probe over the\n"
            "canonical seed, reads ``" + head.primary_metric + "`` from ``summary.json``, and\n"
            "renders it beside the external reference::\n\n"
            + reproduce_lines
            + "\n\nPick the encoder axis with ``--encoder`` (default ``"
            + hest_bench.DEFAULT_ENCODER
            + "``; e.g. ``--encoder virchow2``).",
        ),
        _section("Adding a HEST task", fanout_body),
        ".. seealso::\n\n"
        "   * :doc:`regression` — the task family, the ``pearson`` metric, and the closed-form\n"
        "     Ridge+PCA probe this benchmark drives.\n"
        "   * :doc:`benchmarking` — the shared curate → run → leaderboard → reproduce guide.\n"
        "   * :doc:`curation` — the HEST curator (``curate_hest``) and its split policy.",
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
    write_benchmark_rst()


if __name__ == "__main__":
    main()
