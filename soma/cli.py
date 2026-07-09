"""Command-line interface for soma."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from soma.aggregators import list_aggregators
from soma.config import load_config
from soma.decoders import list_decoders
from soma.encoders import list_models
from soma.pipeline import Pipeline
from soma.pixel_classifiers import list_pixel_classifiers
from soma.tasks import list_task_heads


def _parse_set_overrides(pairs: list[str]) -> dict[str, Any]:
    """Turn ``--set a.b.c=value`` strings into a nested override dict.

    Keys are dotted paths into the config layout (``data.dataset_csv``,
    ``run.output_root``, ``training.epochs`` …). Values are parsed as YAML scalars so
    types come through naturally (``epochs=2`` → int, ``pin_memory=false`` → bool, paths
    stay strings). Lets a committed config be repointed at a new machine without editing
    it on disk.
    """
    overrides: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            print(f"Error: --set expects key=value, got {pair!r}", file=sys.stderr)
            sys.exit(2)
        key, _, raw_value = pair.partition("=")
        key = key.strip()
        if not key:
            print(f"Error: --set has an empty key in {pair!r}", file=sys.stderr)
            sys.exit(2)
        value = yaml.safe_load(raw_value)
        cursor = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            existing = cursor.get(part)
            if not isinstance(existing, dict):
                existing = {}
                cursor[part] = existing
            cursor = existing
        cursor[parts[-1]] = value
    return overrides


def _run_config_path(config_path: Path, overrides: dict[str, Any] | None = None) -> None:
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        config = load_config(config_path, overrides=overrides)
    except Exception as exc:
        print(f"Error: failed to load config from {config_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    Pipeline(config).run()


def _print_table(title: str, values: list[str]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(
        title=title,
        show_header=True,
        header_style="bold",
        min_width=max(len(title) + 4, 12),
    )
    table.add_column("Name")
    if not values:
        table.add_row("(none)")
    else:
        for value in values:
            table.add_row(value)
    console.print(table)


def _cmd_list(args: argparse.Namespace) -> None:
    kind = args.kind
    if kind == "encoders":
        values = list_models(level=args.level)
        title = "Encoders" if args.level is None else f"Encoders ({args.level})"
    elif kind == "aggregators":
        values = list_aggregators()
        title = "Aggregators"
    elif kind == "decoders":
        values = list_decoders()
        title = "Decoders"
    elif kind == "pixel-classifiers":
        values = list_pixel_classifiers()
        title = "Pixel Classifiers"
    elif kind == "benchmarks":
        from soma.benchmarks import list_benchmarks

        values = list_benchmarks()
        title = "Benchmarks"
    else:
        values = list_task_heads()
        title = "Task Heads"
    _print_table(title, values)


def _build_list_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soma list",
        description="List encoders, aggregators, dense registries, or task heads.",
    )
    parser.add_argument(
        "kind",
        choices=(
            "encoders",
            "aggregators",
            "decoders",
            "pixel-classifiers",
            "tasks",
            "benchmarks",
        ),
        help="Component family to list.",
    )
    parser.add_argument(
        "--level",
        choices=("tile", "slide", "patient"),
        default=None,
        help="Restrict encoder listing to one level.",
    )
    parser.set_defaults(func=_cmd_list)
    return parser


def _reproduce_seeds(benchmark, requested: int | None) -> tuple[int, ...]:
    """Seeds to run: the benchmark's canonical set by default, or the first ``requested``.

    ``--seeds N`` is a single-seed-style smoke knob: it runs seeds ``0..N-1`` so the check
    is a fast subset of the canonical set (``--seeds 1`` is the quickest smoke).
    """
    if requested is None:
        return tuple(benchmark.canonical_seeds)
    if requested < 1:
        print("Error: --seeds must be >= 1", file=sys.stderr)
        sys.exit(2)
    return tuple(range(requested))


def _reproduce_reference_row(benchmark, axes: dict[str, Any]):
    """The single **gate** reference row for the benchmark's primary metric at these axes.

    Only ``kind="gate"`` rows are tolerance-checkable. External guidance anchors (issue
    #226) share the metric + a config-agnostic empty key, so they are filtered out here: a
    metric with only external rows errors "no gate reference row" rather than silently
    gating on guidance.
    """
    rows = [
        r
        for r in benchmark.expected(**axes)
        if r.metric == benchmark.primary_metric and not r.is_external
    ]
    if not rows:
        print(
            f"Error: no gate reference row for metric {benchmark.primary_metric!r} "
            f"(axes={axes}) in benchmark {benchmark.name!r}.",
            file=sys.stderr,
        )
        sys.exit(2)
    if len(rows) > 1:
        print(
            f"Error: {len(rows)} gate reference rows match metric {benchmark.primary_metric!r} "
            f"(axes={axes}); refine the axes.",
            file=sys.stderr,
        )
        sys.exit(2)
    return rows[0]


def _report_tolerance(benchmark, measured: float, row) -> bool:
    ok = row.within_tolerance(measured)
    verdict = "PASS" if ok else "FAIL"
    delta = measured - row.expected
    print(
        f"[{verdict}] {benchmark.name} {row.metric} = {measured:.4f}  "
        f"(reference {row.expected:.4f}, Δ {delta:+.4f}, tolerance ±{row.tolerance:.4f})"
    )
    return ok


def _resolve_reproduce_targets(name: str) -> list[Any]:
    """Benchmarks a ``reproduce NAME`` drives: the single benchmark, or a whole family.

    ``NAME`` may be a directly registered benchmark (``ocelot``, ``eva/bach``) or a family
    prefix (``eva``) that fans out over every registered ``NAME/<member>``. Returns an empty
    list when nothing matches (fail-fast handled by the caller).
    """
    from soma.benchmarks import get_benchmark, list_benchmarks

    try:
        return [get_benchmark(name)]
    except KeyError:
        return [get_benchmark(n) for n in list_benchmarks() if n.startswith(f"{name}/")]


def _from_run_dir_axes(benchmark, from_run_dir: str | Path) -> dict[str, Any]:
    """The benchmark's varied axes (encoder/spacing/...) read from a run's OWN recorded spec.

    ``reproduce --from-run-dir`` must tolerance-check against the reference row for the
    encoder/spacing the run actually used — otherwise empty axes fall back to the benchmark
    default and a ``uni`` run is silently compared against the ``uni2`` reference. Reads the
    tolerant ``canonical_spec`` (the same source the leaderboard projects), so it still
    resolves on runs whose full config no longer round-trips through ``load_config``.
    Unresolved axes are skipped; a missing/unrankable run dir yields ``{}``.
    """
    from soma.leaderboard import _MISSING, axis_value, load_run_record

    path = Path(from_run_dir)
    if (path / "summary.json").is_file() or (path / "run.yaml").is_file():
        run_dir = path
    else:  # an output_root above the run(s): mirror score()'s newest-summary resolution.
        summaries = sorted(path.glob("**/summary.json"), key=lambda p: p.stat().st_mtime)
        run_dir = summaries[-1].parent if summaries else path
    record = load_run_record(run_dir)
    if record is None:
        return {}
    resolved: dict[str, Any] = {}
    for axis in benchmark.facet.varied:
        value = axis_value(record.canonical_spec, axis)
        if value is not _MISSING:
            resolved[axis] = value
    return resolved


def _results_table_name(benchmark) -> str:
    """The results/reference table a benchmark records under (its family prefix).

    Family members share one keyed table: ``eva/bach`` records under ``eva`` (keyed by
    ``dataset``), matching ``reference/eva.csv``. A standalone benchmark (``ocelot``) maps
    to itself.
    """
    return benchmark.name.split("/", 1)[0]


def _git_commit() -> str:
    """Short ``HEAD`` SHA of the soma checkout (``-dirty`` if the tree has changes).

    Resolved from the installed package location, not the CWD, so it pins the code that
    actually produced the number. Returns ``"unknown"`` outside a git checkout.
    """
    import subprocess

    import soma

    repo = Path(soma.__file__).resolve().parents[1]
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _provenance() -> tuple[str, str, str]:
    """``(date, soma_commit, slide2vec_version)`` pinning the environment of a recorded row."""
    import datetime
    from importlib import metadata

    date = datetime.date.today().isoformat()
    try:
        slide2vec_version = metadata.version("slide2vec")
    except metadata.PackageNotFoundError:
        slide2vec_version = "unknown"
    return date, _git_commit(), slide2vec_version


def _record_result(benchmark, row, measured: float, std: float | None, n_seeds: int | None) -> None:
    """Append a reproduced-measurement row to the benchmark's results ledger (``--record``).

    Keys and metric are copied from the matched **gate** reference ``row`` so the recorded
    measurement joins its band exactly; provenance is captured at run time.
    """
    from soma.benchmarks import MeasuredRow, append_result

    date, commit, slide2vec_version = _provenance()
    measured_row = MeasuredRow(
        key=dict(row.key),
        metric=row.metric,
        measured=measured,
        std=std,
        n_seeds=n_seeds,
        date=date,
        soma_commit=commit,
        slide2vec_version=slide2vec_version,
        source="soma reproduce --record",
    )
    path = append_result(_results_table_name(benchmark), measured_row, key_order=list(row.key))
    print(f"  recorded → {path}")


def _reproduce_one(benchmark, args: argparse.Namespace, *, family_root: str | None = None) -> int:
    """Curate → run → score one benchmark and tolerance-check its primary metric.

    ``family_root`` is the family name when this benchmark is one member of a fanned-out
    family (e.g. ``eva`` for ``eva/bach``); it nests the member's raw/curated/output paths
    under a per-dataset subdirectory so sibling members do not collide.
    """
    axes: dict[str, Any] = {}
    if args.encoder is not None:
        axes["encoder"] = args.encoder
    if args.spacing is not None:
        axes["spacing"] = args.spacing

    if args.from_run_dir is not None:
        # Constrain the reference lookup to the run's OWN axes; an explicit --encoder/--spacing
        # still wins (setdefault only fills what the CLI left unset).
        for axis, value in _from_run_dir_axes(benchmark, args.from_run_dir).items():
            axes.setdefault(axis, value)

    seeds = _reproduce_seeds(benchmark, args.seeds)
    # ON START: surface the fast paths so a time-conscious user isn't surprised by the
    # full canonical-seed training cost.
    print(
        f"Reproducing benchmark '{benchmark.name}' — canonical seeds "
        f"{list(benchmark.canonical_seeds)}, running {list(seeds)}.\n"
        "  Fast paths: --seeds 1 (single-seed smoke) | "
        "--from-run-dir <dir> (re-score an existing run, no training).\n"
        "  Cache-aware: feature extraction is cached and shared across seeds and repeat "
        "runs, so it runs once per encoder.",
        flush=True,
    )

    row = _reproduce_reference_row(benchmark, axes)

    if args.from_run_dir is not None:
        metrics = benchmark.score(args.from_run_dir)
        measured = float(metrics[benchmark.primary_metric])
        ok = _report_tolerance(benchmark, measured, row)
        if getattr(args, "record", False):
            # A re-scored single run has no seed spread.
            _record_result(benchmark, row, measured, std=None, n_seeds=None)
        return 0 if ok else 1

    if args.raw_root is None:
        print(
            "Error: reproduce needs --raw-root <dir> (full mode) or --from-run-dir <dir> "
            "(re-score an existing run).",
            file=sys.stderr,
        )
        return 2

    import statistics

    # In a fanned-out family each member owns a per-dataset subdirectory so raw roots,
    # curated manifests, and run outputs never collide.
    sub = benchmark.name.split("/", 1)[1] if family_root else None
    raw_root = Path(args.raw_root) / sub if sub else Path(args.raw_root)
    if args.out_dir:
        out_dir = Path(args.out_dir) / sub if sub else Path(args.out_dir)
    else:
        out_dir = raw_root / "curated"
    manifest = benchmark.curate(raw_root, out_dir)
    if args.output_root:
        output_root = Path(args.output_root) / sub if sub else Path(args.output_root)
    else:
        output_root = Path.cwd() / "soma_reproduce" / benchmark.name
    # Feature extraction is seed-independent (the encoder is frozen and the cache key is
    # derived from encoder + tile content, not the seed), so every seed shares one cache
    # root and extraction runs once. --cache-root relocates it (e.g. to fast local storage);
    # by default it sits beside the run outputs, shared across seeds. Without a shared root
    # each seed's per-seed output_root would get its own empty cache and re-extract.
    cache_root = Path(args.cache_root) if args.cache_root else output_root / "feature_cache"
    overrides = {"cache": {"enabled": True, "root_dir": str(cache_root)}}
    print(f"Feature cache (shared across seeds): {cache_root}", flush=True)

    measured_values: list[float] = []
    for seed in seeds:
        seed_root = output_root / f"seed_{seed}"
        config = benchmark.build_config(
            **axes,
            dataset_csv=manifest.dataset_csv,
            splits_csv=manifest.splits_csv,
            output_root=seed_root,
            seed=seed,
            overrides=overrides,
        )
        Pipeline(config).run()
        metrics = benchmark.score(seed_root)
        measured_values.append(float(metrics[benchmark.primary_metric]))

    measured = statistics.fmean(measured_values)
    ok = _report_tolerance(benchmark, measured, row)
    if getattr(args, "record", False):
        std = statistics.stdev(measured_values) if len(measured_values) > 1 else 0.0
        _record_result(benchmark, row, measured, std=std, n_seeds=len(seeds))
    return 0 if ok else 1


def _cmd_reproduce(args: argparse.Namespace) -> int:
    from soma.benchmarks import get_benchmark

    targets = _resolve_reproduce_targets(args.name)
    if not targets:
        try:
            get_benchmark(args.name)  # re-raise for the canonical "Unknown benchmark …" message
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
        return 2

    is_family = any(b.name != args.name for b in targets)
    if is_family and args.from_run_dir is not None:
        print(
            f"Error: --from-run-dir re-scores one run, so it needs a single sub-benchmark "
            f"(e.g. '{targets[0].name}'), not the '{args.name}' family.",
            file=sys.stderr,
        )
        return 2

    codes = [
        _reproduce_one(bench, args, family_root=args.name if is_family else None)
        for bench in targets
    ]
    return max(codes) if codes else 2


def _cmd_leaderboard(args: argparse.Namespace) -> int:
    """Render a faceted leaderboard over the run dirs under ``--root`` (ADR 0003).

    Two entry points share one flat projection: a positional benchmark name supplies the
    canonical facet + primary metric + reference rows, while a bare ``--root`` discovers
    the ``(dataset, splits, task)`` triples under the root and requires disambiguation when
    several exist. ``--vary``/``--fix``/``--like`` shape the facet on top of either.
    """
    from soma.leaderboard import (
        LeaderboardFacet,
        _AXIS_ALIASES,
        _MISSING,
        axis_value,
        discover_triples,
        format_table,
        load_run_record,
        project_leaderboard,
        write_leaderboard,
    )

    benchmark = None
    if args.name is not None:
        from soma.benchmarks import get_benchmark

        try:
            benchmark = get_benchmark(args.name)
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    if args.root is None:
        print("Error: leaderboard needs --root <output_root>.", file=sys.stderr)
        return 2

    triples = discover_triples(args.root)
    if not triples:
        print(f"Error: no completed runs found under {args.root}.", file=sys.stderr)
        return 2

    # Facet: benchmark canonical facet as the base, then CLI overrides.
    vary: tuple[str, ...] = tuple(args.vary) if args.vary else (
        tuple(benchmark.facet.varied) if benchmark is not None else ()
    )
    fixed: dict[str, object] = dict(benchmark.facet.fixed) if benchmark is not None else {}
    for pair in args.fix or []:
        if "=" not in pair:
            print(f"Error: --fix expects axis=value, got {pair!r}", file=sys.stderr)
            return 2
        key, _, value = pair.partition("=")
        fixed[key.strip()] = value.strip()

    like_record = None
    if args.like is not None:
        like_record = load_run_record(args.like)
        if like_record is None:
            print(f"Error: --like run dir is not a completed run: {args.like}", file=sys.stderr)
            return 2
        # Fix every recognised axis except the varied one(s), by the example's value.
        for axis in _AXIS_ALIASES:
            if axis in vary:
                continue
            value = axis_value(like_record.canonical_spec, axis)
            if value is not _MISSING and value is not None:
                fixed.setdefault(axis, value)

    # Resolve the triple to render.
    if len(triples) == 1:
        triple = next(iter(triples))
    elif like_record is not None:
        triple = like_record.triple
    else:
        candidates = triples
        if "task" in fixed:
            candidates = {t: d for t, d in triples.items() if t[2] == str(fixed["task"])}
        if len(candidates) == 1:
            triple = next(iter(candidates))
        else:
            print(
                f"Error: {len(triples)} (dataset, splits, task) triples under {args.root}; "
                "disambiguate with --like <run_dir> or narrower filters:",
                file=sys.stderr,
            )
            for (dataset_ck, splits_ck, task), dirs in sorted(triples.items()):
                print(
                    f"  task={task} dataset={dataset_ck[:8]} splits={splits_ck[:8]} "
                    f"({len(dirs)} runs)",
                    file=sys.stderr,
                )
            return 2

    facet = LeaderboardFacet(vary=vary, fixed=fixed)
    table = project_leaderboard(
        triples[triple], facet, metric=args.metric, benchmark=benchmark, split=args.split
    )
    paths = write_leaderboard(table, args.root, name=args.name)
    print(format_table(table))
    print(f"\nWrote: {paths['csv']}  {paths['json']}  {paths['html']}")
    return 0


def _build_leaderboard_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soma leaderboard",
        description="Render a faceted leaderboard over run dirs under an output root.",
    )
    parser.add_argument("name", nargs="?", default=None, help="Registered benchmark name (canonical facet).")
    parser.add_argument("--root", type=Path, default=None, help="Output root whose run dirs to project.")
    parser.add_argument("--vary", action="append", default=None, help="Axis to surface/rank across (repeatable).")
    parser.add_argument("--fix", action="append", default=None, help="Hold an axis fixed: axis=value (repeatable).")
    parser.add_argument("--like", type=Path, default=None, help="Fix all axes but --vary by this run dir's example.")
    parser.add_argument("--metric", type=str, default=None, help="Override the ranking metric.")
    parser.add_argument("--split", type=str, default=None, help="Override the split ranked on (default: test).")
    parser.set_defaults(func=_cmd_leaderboard)
    return parser


def _build_reproduce_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soma reproduce",
        description="Curate → run → score a registered benchmark and check its tolerance band.",
    )
    parser.add_argument("name", help="Registered benchmark name (see `soma list benchmarks`).")
    parser.add_argument(
        "--from-run-dir",
        type=Path,
        default=None,
        help="Re-score an existing run directory (no curation, no training).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Run seeds 0..N-1 instead of the canonical set (--seeds 1 = fastest smoke).",
    )
    parser.add_argument("--raw-root", type=Path, default=None, help="Raw dataset root (full mode).")
    parser.add_argument("--out-dir", type=Path, default=None, help="Curated manifest dir (default <raw-root>/curated).")
    parser.add_argument("--output-root", type=Path, default=None, help="Where runs are written.")
    parser.add_argument("--cache-root", type=Path, default=None, help="Shared feature-cache root (reused across seeds).")
    parser.add_argument("--encoder", type=str, default=None, help="Encoder axis (benchmark default if omitted).")
    parser.add_argument("--spacing", type=float, default=None, help="Spacing axis in µm/px (benchmark default if omitted).")
    parser.add_argument(
        "--record",
        action="store_true",
        help="Append the measured number + provenance to the results ledger "
        "(soma/benchmarks/results/<name>.csv) so 'reproduced' becomes a committed fact.",
    )
    parser.set_defaults(func=_cmd_reproduce)
    return parser


def _print_top_level_help() -> None:
    print(
        "usage: soma CONFIG\n"
        "       soma list {encoders,aggregators,decoders,pixel-classifiers,tasks,benchmarks} [--level {tile,slide,patient}]\n"
        "       soma reproduce NAME [--from-run-dir DIR] [--seeds N] [--raw-root DIR]\n"
        "       soma leaderboard [NAME] --root OUTPUT_ROOT [--vary AXIS] [--fix AXIS=VALUE] [--like DIR]\n"
        "\n"
        "commands:\n"
        "  CONFIG       run a pipeline from a YAML config file\n"
        "  list         list public model/component/benchmark registries\n"
        "  reproduce    curate → run → score a registered benchmark, check its tolerance band\n"
        "  leaderboard  render a faceted view over the run dirs under an output root\n"
        "\n"
        "examples:\n"
        "  soma /path/to/config.yaml\n"
        "  python -m soma /path/to/config.yaml\n"
        "  soma list benchmarks\n"
        "  soma reproduce ocelot --from-run-dir /runs/ocelot\n"
        "  soma reproduce eva/bach --encoder uni2 --raw-root /data/eva/bach\n"
        "  soma reproduce eva --raw-root /data/eva   # fan out over the eva/<dataset> family\n"
        "  soma leaderboard --root /runs/sweep --vary encoder"
    )


def _print_list_help() -> None:
    print(
        "usage: soma list {encoders,aggregators,decoders,pixel-classifiers,tasks,benchmarks} [--level {tile,slide,patient}]\n"
        "\n"
        "commands:\n"
        "  encoders     list registered encoder presets\n"
        "  aggregators  list registered aggregator presets\n"
        "  decoders     list registered dense decoder presets\n"
        "  pixel-classifiers  list registered per-pixel classifier presets\n"
        "  tasks        list registered task-head presets\n"
        "  benchmarks   list registered benchmarks\n"
        "\n"
        "options:\n"
        "  --level      restrict encoder listing to one level\n"
        "\n"
        "examples:\n"
        "  soma list encoders\n"
        "  soma list encoders --level tile\n"
        "  soma list tasks"
    )


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    if not args:
        _print_top_level_help()
        raise SystemExit(2)

    if args[0] in {"-h", "--help"}:
        _print_top_level_help()
        return

    if args[0] == "list":
        if len(args) == 1 or args[1] in {"-h", "--help"}:
            _print_list_help()
            return
        parser = _build_list_parser()
        parsed = parser.parse_args(args[1:])
        parsed.func(parsed)
        return

    if args[0] == "reproduce":
        parser = _build_reproduce_parser()
        parsed = parser.parse_args(args[1:])
        raise SystemExit(parsed.func(parsed))

    if args[0] == "leaderboard":
        parser = _build_leaderboard_parser()
        parsed = parser.parse_args(args[1:])
        raise SystemExit(parsed.func(parsed))

    if args[0] == "run":
        print(
            "Error: pass the config path directly.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Config-run form: one config path, then any number of `--set key=value` overrides.
    config_path = args[0]
    if config_path.startswith("-"):
        print(
            "Error: expected one config path or the 'list' subcommand.",
            file=sys.stderr,
        )
        sys.exit(2)

    set_pairs: list[str] = []
    rest = args[1:]
    i = 0
    while i < len(rest):
        token = rest[i]
        if token == "--set":
            if i + 1 >= len(rest):
                print("Error: --set requires a key=value argument", file=sys.stderr)
                sys.exit(2)
            set_pairs.append(rest[i + 1])
            i += 2
        elif token.startswith("--set="):
            set_pairs.append(token[len("--set="):])
            i += 1
        else:
            print(
                f"Error: unexpected argument {token!r} "
                "(expected one config path and optional --set key=value)",
                file=sys.stderr,
            )
            sys.exit(2)

    overrides = _parse_set_overrides(set_pairs) if set_pairs else None
    _run_config_path(Path(config_path), overrides)


if __name__ == "__main__":
    main()
