"""Module entrypoint for ``python -m soma`` and console script integration."""

from soma.cli import main


def entrypoint(argv: list[str] | None = None) -> int:
    main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(entrypoint())
