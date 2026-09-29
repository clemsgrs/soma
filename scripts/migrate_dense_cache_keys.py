#!/usr/bin/env python3
"""Re-key legacy dense feature caches after the dense key dropped the registry input size.

Scans ``<cache_root>/dense/`` and ``<cache_root>/dense_image/`` for caches whose recorded
``execution`` block still holds the registry default ``input_size``, and renames each to
the key soma resolves today. It reads recorded metadata only: no encoder, no GPU.

A dry run by default; pass ``--apply`` to rename. A cache whose target folder already
exists is refused and left untouched. Do not run it on a cache that a running job uses.

    python scripts/migrate_dense_cache_keys.py /path/to/feature_cache
    python scripts/migrate_dense_cache_keys.py /path/to/feature_cache --apply
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from soma.cache.migration import migrate_legacy_dense_caches


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cache_root", type=Path, help="feature cache root (holds dense/, dense_image/)")
    parser.add_argument("--apply", action="store_true", help="rename and patch (default: dry run)")
    args = parser.parse_args(argv)

    if not args.cache_root.is_dir():
        parser.error(f"cache root is not a directory: {args.cache_root}")

    results = migrate_legacy_dense_caches(args.cache_root, apply=args.apply)
    if not results:
        print(f"No legacy dense cache under {args.cache_root}.")
        return 0
    for result in results:
        print(f"[{result.status}] {result.source} -> {result.target.name}")
        if result.reason is not None:
            print(f"    {result.reason}")
        if result.unverified_sample_ids:
            print(
                f"    {len(result.unverified_sample_ids)} sample(s) cannot be verified against "
                "manifest.csv; they will be extracted again."
            )
    if not args.apply:
        print("Dry run: nothing was changed. Pass --apply to migrate.")
    return 1 if any(result.status == "refused" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
