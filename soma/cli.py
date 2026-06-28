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
        choices=("encoders", "aggregators", "decoders", "pixel-classifiers", "tasks"),
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


def _print_top_level_help() -> None:
    print(
        "usage: soma CONFIG\n"
        "       soma list {encoders,aggregators,decoders,pixel-classifiers,tasks} [--level {tile,slide,patient}]\n"
        "\n"
        "commands:\n"
        "  CONFIG   run a pipeline from a YAML config file\n"
        "  list     list public model/component registries\n"
        "\n"
        "examples:\n"
        "  soma /path/to/config.yaml\n"
        "  python -m soma /path/to/config.yaml\n"
        "  soma list encoders --level tile"
    )


def _print_list_help() -> None:
    print(
        "usage: soma list {encoders,aggregators,decoders,pixel-classifiers,tasks} [--level {tile,slide,patient}]\n"
        "\n"
        "commands:\n"
        "  encoders     list registered encoder presets\n"
        "  aggregators  list registered aggregator presets\n"
        "  decoders     list registered dense decoder presets\n"
        "  pixel-classifiers  list registered per-pixel classifier presets\n"
        "  tasks        list registered task-head presets\n"
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
