"""Command-line interface for soma."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from soma.aggregators import list_aggregators
from soma.config import load_config
from soma.encoders import list_models
from soma.pipeline import Pipeline
from soma.tasks import list_task_heads


def _run_config_path(config_path: Path) -> None:
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"Error: failed to load config from {config_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    Pipeline(config).run()


def _print_table(title: str, values: list[str]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=title, show_header=True, header_style="bold")
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
    else:
        values = list_task_heads()
        title = "Task Heads"
    _print_table(title, values)


def _build_list_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soma list",
        description="List encoders, aggregators, or task heads.",
    )
    parser.add_argument(
        "kind",
        choices=("encoders", "aggregators", "tasks"),
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
        "       soma list {encoders,aggregators,tasks} [--level {tile,slide,patient}]\n"
        "\n"
        "commands:\n"
        "  CONFIG   run a pipeline from a YAML config file\n"
        "  list     list encoders, aggregators, or task heads\n"
        "\n"
        "examples:\n"
        "  soma /path/to/config.yaml\n"
        "  python -m soma /path/to/config.yaml\n"
        "  soma list encoders --level tile"
    )


def _print_list_help() -> None:
    print(
        "usage: soma list {encoders,aggregators,tasks} [--level {tile,slide,patient}]\n"
        "\n"
        "commands:\n"
        "  encoders     list registered encoder presets\n"
        "  aggregators  list registered aggregator presets\n"
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

    if len(args) != 1:
        print(
            "Error: expected one config path or the 'list' subcommand.",
            file=sys.stderr,
        )
        sys.exit(2)

    _run_config_path(Path(args[0]))


if __name__ == "__main__":
    main()
