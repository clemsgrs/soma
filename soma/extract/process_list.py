"""Process-list normalization and restoration for slide2vec embedding runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence


_PENDING_FEATURE_PATH_SENTINEL = "__soma_pending_feature_path__"


def _rewrite_process_list_rows(
    process_list_path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, str]],
) -> None:
    with process_list_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _normalize_process_list_for_embedding(process_list_path: Path) -> None:
    if not process_list_path.is_file():
        return
    with process_list_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames or not rows:
        return

    # Recover from a previous crash that left sentinel values in the file.
    changed = False
    if "feature_path" in fieldnames:
        for row in rows:
            if row.get("feature_path") == _PENDING_FEATURE_PATH_SENTINEL:
                row["feature_path"] = ""
                changed = True
    if changed:
        _rewrite_process_list_rows(process_list_path, fieldnames=fieldnames, rows=rows)
        changed = False

    for column in ("feature_status", "aggregation_status"):
        if column not in fieldnames:
            continue
        if any((row.get(column) or "").strip() for row in rows):
            continue
        for row in rows:
            row[column] = "tbp"
        changed = True
    if "feature_path" in fieldnames:
        if not any((row.get("feature_path") or "").strip() for row in rows):
            for row in rows:
                row["feature_path"] = _PENDING_FEATURE_PATH_SENTINEL
            changed = True

    if changed:
        _rewrite_process_list_rows(
            process_list_path,
            fieldnames=fieldnames,
            rows=rows,
        )


def _restore_process_list_after_embedding(process_list_path: Path) -> None:
    if not process_list_path.is_file():
        return
    with process_list_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "feature_path" not in fieldnames or not rows:
        return
    changed = False
    for row in rows:
        if row.get("feature_path") == _PENDING_FEATURE_PATH_SENTINEL:
            row["feature_path"] = ""
            changed = True
    if changed:
        _rewrite_process_list_rows(
            process_list_path,
            fieldnames=fieldnames,
            rows=rows,
        )
