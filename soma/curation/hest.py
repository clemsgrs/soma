"""Curator for the HEST-Benchmark gene-expression-from-morphology tasks.

Like the EVA and OCELOT curators, this emits soma's unified ``dataset.csv`` +
``splits.csv`` Manifest from a **locally pre-provisioned** raw tree and **never touches
the network** (ADR 0004). The hest-bench data is downloaded once, out of band, with a
documented step (a later issue owns it), e.g.::

    hf download MahmoodLab/hest-bench --include 'IDC/*' --exclude 'fm_v1/*' \\
        --repo-type dataset --local-dir <raw_root>

HEST-Benchmark (Jaume et al., NeurIPS 2024) predicts a 50-dimensional highly-variable-gene
expression vector from a 112x112 µm tile. This curator turns one hest-bench **task** tree
into soma's ``spatial_expression`` Manifest, where **one soma sample = one spot** (a tile +
its gene vector).

Assumed on-disk layout of a hest-bench task subtree (verify against the real download; the
paths are all confined to :func:`_resolve_task_dir` / :func:`_load_task` so a mismatch is a
one-place fix)::

    <raw_root>/                      # the bench root ...
      <task>/                        # ... or pass the task dir directly as raw_root
        patches/<slide>.h5           # HDF5: img [N,H,W,3] uint8, coords [N,2], barcode [N,1]
        adata/<slide>.h5ad           # AnnData: X = raw counts, obs = spot barcodes, var = genes
        splits/train_0.csv test_0.csv train_1.csv test_1.csv ...   # per-fold slide lists
        var_50genes.json             # {"genes": [... task gene symbols ...]}

Each split CSV carries a ``sample_id`` column of **slide** ids (optionally ``patches_path``
/ ``expr_path`` relative to the task dir; absent, they default to ``patches/<slide>.h5`` and
``adata/<slide>.h5ad``). ``n_folds = number of train_i/test_i pairs``.

What the curator produces:

* **Tile pixels** — each slide's ``img[N]`` patch array is exploded into per-spot **lossless
  PNG** files under ``tiles/<slide>/<spot_id>.png``, referenced by ``image_path``. PNG
  preserves HEST's exact pixels; soma runs its own encoder + eval transforms on them.
* **Baked targets** — ``y = log1p(raw counts)`` on float64 for the task's genes (no
  total-count normalization, no spatial smoothing — HEST's default). Written to the
  ``targets.npy`` sidecar with the ordered ``genes.json``; the probe reads them verbatim.
* **Splits** — each HEST fold ``i`` expands slide-level train/test membership to per-spot
  rows ``(spot_id, "train"|"test", fold=i)``. There are **no tune** rows.

Re-curation of the same raw input is byte-identical: slides iterate in sorted order, spots
in patch-array order, PNGs use fixed encoder settings, and the sidecars/summary are written
deterministically.

Curation deps (``anndata`` for ``.h5ad``, ``h5py`` for the patch HDF5) live in the
``soma[hest]`` optional extra and are **lazy-imported inside** :func:`curate_hest`, so
``import soma`` stays light and core CI does not need them.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from soma.curation.manifest import CuratedManifest, write_manifest

_HEST_DATASET_TYPE = "spatial_expression"

# PNG encoder settings pinned for byte-identical, lossless re-curation. PNG is inherently
# lossless; a fixed zlib level keeps the encoded bytes reproducible, and PIL writes no
# timestamp chunk by default.
_PNG_COMPRESS_LEVEL = 6


def curate_hest(
    raw_root: str | Path,
    output_dir: str | Path,
    *,
    task: str = "IDC",
) -> CuratedManifest:
    """Curate one hest-bench task into soma's ``spatial_expression`` Manifest.

    Args:
        raw_root: Either the hest-bench root containing a ``<task>/`` subtree, or that task
            directory itself. The task dir must hold ``patches/``, ``adata/``, ``splits/``
            and ``var_50genes.json`` (see the module docstring for the assumed layout).
        output_dir: Directory where ``dataset.csv``, ``splits.csv``, the ``targets.npy`` /
            ``genes.json`` sidecars, ``summary.json`` and the per-spot ``tiles/`` PNGs are
            written (created if absent).
        task: The hest-bench task name (e.g. ``"IDC"``). Selects the ``<task>/`` subtree
            when ``raw_root`` is the bench root, and labels the summary.

    Returns:
        A :class:`~soma.curation.manifest.CuratedManifest` pointing at the generated
        ``dataset.csv`` / ``splits.csv`` and the two spatial-expression sidecars.
    """
    # Heavy, dataset-specific readers: import lazily so `import soma` stays light and core
    # CI does not require the soma[hest] extra.
    try:
        import anndata  # noqa: F401
        import h5py  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via the extra
        raise ModuleNotFoundError(
            "curate_hest needs the 'soma[hest]' optional dependencies "
            "(anndata for .h5ad, h5py for the patch HDF5). Install with: "
            "pip install 'soma-pathology[hest]'."
        ) from exc

    task_dir = _resolve_task_dir(Path(raw_root), task)
    genes = _load_gene_list(task_dir / "var_50genes.json")
    folds = _load_folds(task_dir / "splits")

    output_dir = Path(output_dir)
    tiles_root = output_dir / "tiles"
    tiles_root.mkdir(parents=True, exist_ok=True)

    # Every slide referenced by any fold (some slides appear only in test folds), sorted
    # for deterministic spot ordering / target-matrix row assignment.
    all_slides = sorted(
        {slide for fold in folds.values() for role_slides in fold.values() for slide in role_slides}
    )

    dataset_rows: list[dict] = []
    target_vectors: list[np.ndarray] = []
    spots_by_slide: dict[str, list[str]] = {}
    seen_spot_ids: set[str] = set()

    for slide in all_slides:
        barcodes, images = _read_patches(task_dir, slide)
        gene_matrix = _read_expression(task_dir, slide, barcodes, genes)  # [n_spots, n_genes] float64
        # Sanitized slide component keeps the tile subdir a safe bare name (no traversal).
        slide_tile_dir = tiles_root / _safe_component(slide)
        slide_tile_dir.mkdir(parents=True, exist_ok=True)
        spot_ids: list[str] = []
        for i, barcode in enumerate(barcodes):
            spot_id = _spot_id(slide, barcode)
            if spot_id in seen_spot_ids:
                raise ValueError(
                    f"Duplicate spot id {spot_id!r} (slide {slide!r}, barcode {barcode!r}); "
                    "spot ids must be unique across the curated Manifest."
                )
            seen_spot_ids.add(spot_id)
            png_path = slide_tile_dir / f"{spot_id}.png"
            _write_png(images[i], png_path)
            target_index = len(target_vectors)
            target_vectors.append(gene_matrix[i])
            dataset_rows.append(
                {
                    "sample_id": spot_id,
                    "image_path": str(png_path.resolve()),
                    "target_index": target_index,
                    "slide_id": slide,
                    "barcode": barcode,
                }
            )
            spot_ids.append(spot_id)
        spots_by_slide[slide] = spot_ids

    if not dataset_rows:
        raise ValueError(f"No spots found for hest-bench task {task!r} under {task_dir}")

    # Per-fold spot-row expansion: a slide's every spot inherits its slide's fold role.
    split_rows: list[dict] = []
    for fold_idx in sorted(folds):
        for role in ("train", "test"):
            for slide in folds[fold_idx][role]:
                for spot_id in spots_by_slide[slide]:
                    split_rows.append({"sample_id": spot_id, "split": role, "fold": fold_idx})

    target_matrix = np.ascontiguousarray(np.stack(target_vectors), dtype=np.float64)

    summary = {
        "dataset": f"HEST-Benchmark ({task})",
        "dataset_type": _HEST_DATASET_TYPE,
        "task": task,
        "num_genes": len(genes),
        "num_slides": len(all_slides),
        "num_spots": len(dataset_rows),
        "num_folds": len(folds),
        "spots_per_slide": {slide: len(spots_by_slide[slide]) for slide in all_slides},
        "folds": {
            str(fold_idx): {
                "train_slides": list(folds[fold_idx]["train"]),
                "test_slides": list(folds[fold_idx]["test"]),
                "train_spots": sum(len(spots_by_slide[s]) for s in folds[fold_idx]["train"]),
                "test_spots": sum(len(spots_by_slide[s]) for s in folds[fold_idx]["test"]),
            }
            for fold_idx in sorted(folds)
        },
    }

    return write_manifest(
        output_dir,
        dataset_type=_HEST_DATASET_TYPE,
        dataset_rows=dataset_rows,
        split_rows=split_rows,
        summary=summary,
        target_matrix=target_matrix,
        genes=genes,
    )


def _resolve_task_dir(raw_root: Path, task: str) -> Path:
    """Return the task subtree, accepting either the bench root or the task dir itself."""
    candidates = [raw_root / task, raw_root]
    for candidate in candidates:
        if (candidate / "patches").is_dir() and (candidate / "splits").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find a hest-bench task tree for task {task!r}. Expected "
        f"'patches/' + 'splits/' under {raw_root / task} or {raw_root}."
    )


def _load_gene_list(genes_json: Path) -> list[str]:
    """Read the task's ordered gene list from ``var_50genes.json`` (``{"genes": [...]}``)."""
    if not genes_json.is_file():
        raise FileNotFoundError(f"Missing HEST gene list: {genes_json}")
    payload = json.loads(genes_json.read_text())
    genes = payload["genes"] if isinstance(payload, dict) else payload
    if not isinstance(genes, list) or not genes or not all(isinstance(g, str) for g in genes):
        raise ValueError(f"{genes_json} must contain a non-empty list of gene symbols.")
    if len(set(genes)) != len(genes):
        raise ValueError(f"{genes_json} contains duplicate gene symbols.")
    return list(genes)


def _load_folds(splits_dir: Path) -> dict[int, dict[str, list[str]]]:
    """Read per-fold slide lists from ``splits/{train,test}_<i>.csv``.

    Returns ``{fold_index: {"train": [slides...], "test": [slides...]}}``. Slide order is
    preserved from each CSV so the summary is stable; spot ordering itself is governed by
    the globally sorted slide list in :func:`curate_hest`.
    """
    if not splits_dir.is_dir():
        raise FileNotFoundError(f"Missing HEST splits directory: {splits_dir}")
    fold_indices = sorted(
        int(m.group(1))
        for p in splits_dir.glob("train_*.csv")
        if (m := re.fullmatch(r"train_(\d+)", p.stem))
    )
    if not fold_indices:
        raise FileNotFoundError(f"No train_<i>.csv fold files found under {splits_dir}")

    folds: dict[int, dict[str, list[str]]] = {}
    for i in fold_indices:
        fold: dict[str, list[str]] = {}
        for role in ("train", "test"):
            csv_path = splits_dir / f"{role}_{i}.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(f"Missing HEST split file: {csv_path}")
            df = pd.read_csv(csv_path)
            if "sample_id" not in df.columns:
                raise ValueError(f"{csv_path} must have a 'sample_id' column (slide ids).")
            fold[role] = [str(s) for s in df["sample_id"]]
        folds[i] = fold
    return folds


def _read_patches(task_dir: Path, slide: str) -> tuple[list[str], np.ndarray]:
    """Read a slide's patch HDF5: return ``(barcodes, img[N,H,W,3] uint8)`` in file order."""
    import h5py

    patch_path = task_dir / "patches" / f"{slide}.h5"
    if not patch_path.is_file():
        raise FileNotFoundError(f"Missing HEST patch HDF5 for slide {slide!r}: {patch_path}")
    with h5py.File(patch_path, "r") as f:
        images = np.asarray(f["img"][:], dtype=np.uint8)
        raw_barcodes = np.asarray(f["barcode"][:]).reshape(-1)
    barcodes = [_decode_barcode(b) for b in raw_barcodes]
    if images.shape[0] != len(barcodes):
        raise ValueError(
            f"{patch_path}: img has {images.shape[0]} tiles but {len(barcodes)} barcodes."
        )
    return barcodes, images


def _read_expression(
    task_dir: Path, slide: str, barcodes: list[str], genes: list[str]
) -> np.ndarray:
    """Return ``log1p(raw counts)`` [n_spots, n_genes] float64, aligned to ``barcodes``.

    Expression is read from the slide's ``.h5ad``, subselected to ``genes`` in list order,
    and reindexed to the patch **barcode** order (obs order in the ``.h5ad`` need not match
    the patch array order). log1p is applied on float64 with no other normalization.
    """
    import anndata

    expr_path = task_dir / "adata" / f"{slide}.h5ad"
    if not expr_path.is_file():
        raise FileNotFoundError(f"Missing HEST expression .h5ad for slide {slide!r}: {expr_path}")
    adata = anndata.read_h5ad(expr_path)

    var_names = [str(v) for v in adata.var_names]
    var_pos = {name: i for i, name in enumerate(var_names)}
    missing_genes = [g for g in genes if g not in var_pos]
    if missing_genes:
        raise ValueError(
            f"{expr_path}: gene(s) {missing_genes} from the task gene list are absent from "
            "the .h5ad var index."
        )
    gene_cols = [var_pos[g] for g in genes]

    counts = _dense(adata.X)
    obs_pos = {str(name): i for i, name in enumerate(adata.obs_names)}
    missing_spots = [bc for bc in barcodes if bc not in obs_pos]
    if missing_spots:
        raise ValueError(
            f"{expr_path}: patch barcode(s) {missing_spots[:5]} (of {len(missing_spots)}) "
            "have no matching spot in the .h5ad obs index."
        )
    rows = [obs_pos[bc] for bc in barcodes]
    raw = counts[np.ix_(rows, gene_cols)].astype(np.float64, copy=False)
    return np.log1p(raw)


def _dense(matrix) -> np.ndarray:
    """Densify an AnnData ``X`` (sparse or dense) to a 2D ``np.ndarray``."""
    if hasattr(matrix, "toarray"):  # scipy sparse
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def _decode_barcode(value) -> str:
    """Decode a HEST barcode cell (bytes or str) to a stripped string."""
    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    return str(value).strip()


def _safe_component(value: str) -> str:
    """Sanitize ``value`` to a bare, filename-safe path component (no separators/``..``)."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _spot_id(slide: str, barcode: str) -> str:
    """A stable, filename-safe per-spot id ``<slide>_<sanitized-barcode>``."""
    return f"{_safe_component(slide)}_{_safe_component(barcode)}"


def _write_png(image: np.ndarray, path: Path) -> None:
    """Write one RGB tile losslessly and deterministically to ``path`` (PNG)."""
    from PIL import Image

    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an [H, W, 3] RGB tile, got shape {arr.shape} for {path}")
    Image.fromarray(arr, mode="RGB").save(
        path, format="PNG", optimize=False, compress_level=_PNG_COMPRESS_LEVEL
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m soma.curation.hest",
        description="Curate a hest-bench task into a soma spatial_expression Manifest.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="hest-bench root (containing <task>/) or the task directory itself",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="curated output dir")
    parser.add_argument("--task", default="IDC", help="hest-bench task name (default: IDC)")
    args = parser.parse_args(argv)

    manifest = curate_hest(args.raw_root, args.output_dir, task=args.task)
    print(f"curated: {manifest.dataset_csv}")
    print(f"         {manifest.splits_csv}")
    print(f"         {manifest.target_matrix_path}")
    print(f"         {manifest.genes_path}")
    if manifest.summary_json is not None:
        print(f"         {manifest.summary_json}")


if __name__ == "__main__":
    main()
