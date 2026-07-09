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
from soma.benchmarks import expected_rows, get_benchmark, list_benchmarks, load_results
from soma.benchmarks import eva as eva_bench
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

        ``soma reproduce NAME [--raw-root DIR | --from-run-dir DIR] [--seeds N]``
           Curate → run → score a registered benchmark and tolerance-check its
           primary metric against the packaged reference band. ``NAME`` is a
           registered benchmark (``ocelot``, ``eva/bach``) or a family prefix
           (``eva``) that fans out over every ``eva/<dataset>``. Full mode needs
           ``--raw-root``; ``--from-run-dir`` re-scores an existing run without
           retraining, and ``--seeds 1`` is the quickest smoke.

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
# reference table is embedded with a ``.. csv-table:: :file:`` directive pointing straight
# at the packaged ``soma/benchmarks/reference/<name>.csv`` — so the numbers Sphinx renders
# are the SAME BYTES as the CSV the registry scores against (no hand-typed numbers, no
# drift, no ``TBD``). A generator + a checked-in file kept in sync by ``tests/test_docs.py``
# mirrors the ``cli.rst`` mechanism.

_GENERATED_PAGE_NOTE = (
    ".. note::\n\n"
    "   This page is generated from the registered benchmark definition — the protocol\n"
    "   summary from the ``Benchmark`` object, the reference table straight from the\n"
    "   packaged ``{csv}`` (same bytes), and the command from the benchmark name. Edit the\n"
    "   registry (``{module}``) and the CSV, not this page; ``python docs/_generate_reference.py``\n"
    "   re-emits it and ``tests/test_docs.py`` guards the two from drifting."
)


def _kv_table(col_a: str, col_b: str, rows: list[tuple[str, str]], *, widths: str = "30 70") -> str:
    """A two-column ``list-table`` (header + ``rows``)."""
    lines = [".. list-table::", "   :header-rows: 1", f"   :widths: {widths}", ""]
    lines.extend([f"   * - {col_a}", f"     - {col_b}"])
    for a, b in rows:
        lines.extend([f"   * - {a}", f"     - {b}"])
    return "\n".join(lines)


def _reference_csv_table(title: str, rel_path: str, widths: str) -> str:
    """A ``csv-table`` reading the packaged reference CSV verbatim at build time."""
    return "\n".join(
        [
            f".. csv-table:: {title}",
            f"   :file: {rel_path}",
            "   :header-rows: 1",
            f"   :widths: {widths}",
        ]
    )


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
        "Reference numbers\n-----------------\n\n"
        "Read verbatim from the packaged reference CSV. The ``kind`` column marks each\n"
        "row's role: a ``gate`` row is the tolerance band ``soma reproduce`` checks against\n"
        "(config-agnostic banner); ``external`` rows are non-gating guidance anchors (also\n"
        "surfaced with clickable links below). The ``source`` cell records provenance and\n"
        "why the tolerance is what it is:\n\n"
        + _reference_csv_table(
            "``soma/benchmarks/reference/ocelot.csv``",
            "../soma/benchmarks/reference/ocelot.csv",
            "5 5 5 8 6 6 5 14 14 32",
        ),
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
        "Reference numbers\n-----------------\n\n"
        "The published EVA balanced-accuracy band, keyed by ``dataset`` × ``encoder`` —\n"
        "read verbatim from the packaged reference CSV (``patch_camelyon`` carries both a\n"
        "``test`` and a ``tune`` row):\n\n"
        + _reference_csv_table(
            "``soma/benchmarks/reference/eva.csv``",
            "../soma/benchmarks/reference/eva.csv",
            "12 10 20 10 10 38",
        ),
        "Reproduced numbers\n------------------\n\n"
        "What soma has actually measured, recorded by ``soma reproduce --record`` into the\n"
        "packaged results ledger (``soma/benchmarks/results/eva.csv``) alongside the commit\n"
        "and slide2vec version that produced each number. Only cells that have been run\n"
        "appear; each is shown next to its reference band above, with the delta:\n\n"
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


def write_benchmark_rst(directory: str | Path | None = None) -> list[Path]:
    """Write the generated per-benchmark pages to disk."""
    base = Path(directory) if directory is not None else Path(__file__).parent
    written = []
    for filename, builder in (
        ("ocelot-detection-benchmark.rst", build_ocelot_benchmark_rst),
        ("eva-patch-classification-benchmark.rst", build_eva_benchmark_rst),
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
