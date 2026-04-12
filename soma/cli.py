"""Command-line interface for soma."""

import argparse
import sys
from pathlib import Path

from soma.config import load_config
from soma.pipeline import Pipeline


def _cmd_run(args: argparse.Namespace) -> None:
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"Error: failed to load config from {config_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    Pipeline(config).run()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soma",
        description="Modular experimentation framework for computational pathology.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    subparsers.required = True

    run_parser = subparsers.add_parser("run", help="Run a pipeline from a YAML config file.")
    run_parser.add_argument("config", metavar="config.yaml", help="Path to the experiment config file.")
    run_parser.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
