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
    """The single reference row for the benchmark's primary metric at these axes."""
    rows = [r for r in benchmark.expected(**axes) if r.metric == benchmark.primary_metric]
    if not rows:
        print(
            f"Error: no reference row for metric {benchmark.primary_metric!r} "
            f"(axes={axes}) in benchmark {benchmark.name!r}.",
            file=sys.stderr,
        )
        sys.exit(2)
    if len(rows) > 1:
        print(
            f"Error: {len(rows)} reference rows match metric {benchmark.primary_metric!r} "
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


def _cmd_reproduce(args: argparse.Namespace) -> int:
    from soma.benchmarks import get_benchmark

    try:
        benchmark = get_benchmark(args.name)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    axes: dict[str, Any] = {}
    if args.encoder is not None:
        axes["encoder"] = args.encoder
    if args.spacing is not None:
        axes["spacing"] = args.spacing

    seeds = _reproduce_seeds(benchmark, args.seeds)
    # ON START: surface the fast paths so a time-conscious user isn't surprised by the
    # full canonical-seed training cost.
    print(
        f"Reproducing benchmark '{benchmark.name}' — canonical seeds "
        f"{list(benchmark.canonical_seeds)}, running {list(seeds)}.\n"
        "  Fast paths: --seeds 1 (single-seed smoke) | "
        "--from-run-dir <dir> (re-score an existing run, no training).\n"
        "  Cache-aware: a repeat run reuses soma's feature cache (extraction is skipped).",
        flush=True,
    )

    row = _reproduce_reference_row(benchmark, axes)

    if args.from_run_dir is not None:
        metrics = benchmark.score(args.from_run_dir)
        measured = float(metrics[benchmark.primary_metric])
        return 0 if _report_tolerance(benchmark, measured, row) else 1

    if args.raw_root is None:
        print(
            "Error: reproduce needs --raw-root <dir> (full mode) or --from-run-dir <dir> "
            "(re-score an existing run).",
            file=sys.stderr,
        )
        return 2

    import statistics

    out_dir = args.out_dir or (Path(args.raw_root) / "curated")
    manifest = benchmark.curate(args.raw_root, out_dir)
    output_root = Path(args.output_root) if args.output_root else Path.cwd() / "soma_reproduce" / benchmark.name
    overrides = {"cache": {"enabled": True, "root_dir": str(args.cache_root)}} if args.cache_root else None

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
    return 0 if _report_tolerance(benchmark, measured, row) else 1


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
    parser.set_defaults(func=_cmd_reproduce)
    return parser


def _print_top_level_help() -> None:
    print(
        "usage: soma CONFIG\n"
        "       soma list {encoders,aggregators,decoders,pixel-classifiers,tasks,benchmarks} [--level {tile,slide,patient}]\n"
        "       soma reproduce NAME [--from-run-dir DIR] [--seeds N] [--raw-root DIR]\n"
        "\n"
        "commands:\n"
        "  CONFIG     run a pipeline from a YAML config file\n"
        "  list       list public model/component/benchmark registries\n"
        "  reproduce  curate → run → score a registered benchmark, check its tolerance band\n"
        "\n"
        "examples:\n"
        "  soma /path/to/config.yaml\n"
        "  python -m soma /path/to/config.yaml\n"
        "  soma list benchmarks\n"
        "  soma reproduce ocelot --from-run-dir /runs/ocelot"
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
