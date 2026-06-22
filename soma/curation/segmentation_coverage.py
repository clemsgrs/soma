"""Per-slide, per-class annotation coverage summary — segmentation ingestion block A1.

A thin soma driver over hs2p's ``resolve_annotation_masks`` + ``summarize_annotation_coverage``
(the preprocessing layer owns the scan; see the hs2p > slide2vec > soma hierarchy). Given a
slide manifest (``sample_id, image_path, mask_path``) and a ``pixel_mapping`` / ``min_coverage``
(mirroring hs2p's ``masks`` config 1:1), it emits a **wide** CSV — one row per slide with
``area_mm2_<class>``, ``frac_<class>``, ``est_tiles_<class>`` columns.

This only *informs* the user's split (stratify on area / expected-tile-count, not mere
presence). soma never partitions: split assignment stays a user, slide-level decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hs2p.api import resolve_annotation_masks, summarize_annotation_coverage
from hs2p.wsi.reader import open_slide

REQUIRED_COLUMNS = {"sample_id", "image_path", "mask_path"}
_METRICS = ("area_mm2", "frac", "est_tiles")


def _non_background_classes(pixel_mapping: dict[str, int]) -> list[str]:
    return [name for name in pixel_mapping if name != "background"]


def _coverage_columns(pixel_mapping: dict[str, int]) -> list[str]:
    return ["sample_id"] + [
        f"{metric}_{name}"
        for name in _non_background_classes(pixel_mapping)
        for metric in _METRICS
    ]


def summarize_coverage(
    manifest: pd.DataFrame | str | Path,
    *,
    pixel_mapping: dict[str, int],
    min_coverage: dict[str, float | None] | None,
    tile_size_px: int,
    spacing_um: float,
    seg_downsample: int = 64,
    overlap: float = 0.0,
    backend: str = "auto",
) -> pd.DataFrame:
    """Return a wide coverage DataFrame, one row per manifest slide.

    ``manifest`` is a DataFrame or a path to a CSV with at least
    ``sample_id, image_path, mask_path``.
    """
    if isinstance(manifest, (str, Path)):
        manifest = pd.read_csv(manifest)
    missing = REQUIRED_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(
            f"coverage manifest is missing required column(s): {sorted(missing)}"
        )
    if "background" not in pixel_mapping:
        raise ValueError("pixel_mapping must include a 'background' label")

    classes = _non_background_classes(pixel_mapping)
    rows: list[dict[str, object]] = []
    for record in manifest.itertuples(index=False):
        slide = open_slide(Path(str(record.image_path)), backend=backend)
        try:
            resolved = resolve_annotation_masks(
                slide=slide,
                mask_path=Path(str(record.mask_path)),
                pixel_mapping=pixel_mapping,
                seg_downsample=seg_downsample,
            )
            summary = summarize_annotation_coverage(
                slide=slide,
                resolved_masks=resolved,
                min_coverage=min_coverage,
                requested_tile_size_px=tile_size_px,
                requested_spacing_um=spacing_um,
                overlap=overlap,
            )
        finally:
            close = getattr(slide, "close", None)
            if callable(close):
                close()
        row: dict[str, object] = {"sample_id": record.sample_id}
        for name in classes:
            stats = summary.get(name, {})
            row[f"area_mm2_{name}"] = stats.get("area_mm2")
            row[f"frac_{name}"] = stats.get("frac")
            row[f"est_tiles_{name}"] = stats.get("est_tiles")
        rows.append(row)

    return pd.DataFrame(rows, columns=_coverage_columns(pixel_mapping))


def write_coverage_csv(out_path: str | Path, coverage: pd.DataFrame) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(out_path, index=False)
    return out_path


def _load_masks_config(path: str | Path) -> dict:
    """Load the masks config (``pixel_mapping`` + ``min_coverage``) from JSON or YAML."""
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    masks = cfg.get("masks", cfg)
    return masks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="soma-segmentation-coverage",
        description="Per-slide per-class annotation coverage summary (informs split authoring).",
    )
    parser.add_argument("--manifest", required=True, help="slide manifest CSV (sample_id,image_path,mask_path)")
    parser.add_argument("--masks-config", required=True, help="JSON/YAML with masks.pixel_mapping + masks.min_coverage")
    parser.add_argument("--out", required=True, help="output coverage CSV path")
    parser.add_argument("--tile-size-px", type=int, required=True)
    parser.add_argument("--spacing-um", type=float, required=True)
    parser.add_argument("--seg-downsample", type=int, default=64)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--backend", default="auto")
    args = parser.parse_args(argv)

    masks = _load_masks_config(args.masks_config)
    coverage = summarize_coverage(
        args.manifest,
        pixel_mapping=masks["pixel_mapping"],
        min_coverage=masks.get("min_coverage"),
        tile_size_px=args.tile_size_px,
        spacing_um=args.spacing_um,
        seg_downsample=args.seg_downsample,
        overlap=args.overlap,
        backend=args.backend,
    )
    out = write_coverage_csv(args.out, coverage)
    print(f"Wrote coverage for {len(coverage)} slides to {out}")


if __name__ == "__main__":
    main()
