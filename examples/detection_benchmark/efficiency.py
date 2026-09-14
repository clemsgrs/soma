"""Annotation-efficiency ladder for the detection benchmark (issue #237).

Pure, GPU-free half of the per-encoder **sample-efficiency study**: how many annotated
training atoms each frozen encoder needs to approach its full-data detection score. The
protocol is the nested-ladder × K-seed design from #186, generalized to every roster encoder
and to both OCELOT and MIDOG'22:

* the **atom** is the unit a pathologist annotates — an OCELOT FoV patch (``sample_id``)
  or a MIDOG ROI (``source_wsi``; its supervision tiles travel together);
* per seed, the ladder is **nested**: rung ``N`` trains on the first ``N`` atoms of
  ``rng(seed).permutation(sorted_atoms)``, so smaller rungs are prefixes of larger ones
  and composition drift never confounds the curve. The draw is pure random (no
  stratification by object count or domain — stratifying on labels would leak them);
* **tune and test stay full** at every rung, so selection is held constant and ``N`` is the
  only variable (stated limitation: small rungs are optimistic about model selection);
* the training recipe is the committed benchmark recipe except ``epochs`` / ``patience`` are
  multiplied by ``max(1, ceil((full/8) / N))`` so the tiny rungs are not step-starved
  (early stopping and val-best checkpointing stay on);
* the **full rung is never retrained** — its score and seed spread come from the headline
  sweep (#234 / #375).

Deliverables per dataset: the per-encoder learning curve, ``N*@95%`` of the encoder's own
full score (headline) and of the global-best full score (second column), the noise-band
crossing, and the per-rung rank agreement with the full-data ranking. Cross-dataset: the
ordering of encoders by ``N*`` compared between datasets. The GPU orchestration lives in
``examples/detection_benchmark/campaign.py`` (``efficiency`` phase); everything here is
unit-tested on synthetic fixtures.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from soma.benchmarks.detection_benchmark import DATASET_ORDER, dataset_spec

# --- protocol constants ------------------------------------------------------------

#: Per-dataset ladder below the full rung, in atoms (largest first).
LADDERS: dict[str, tuple[int, ...]] = {
    "ocelot": (256, 128, 64, 32, 16, 8),
    "midog": (128, 64, 32, 16, 8, 4),
}

#: The manifest column that identifies one annotation atom.
ATOM_COLUMNS: dict[str, str] = {
    "ocelot": "sample_id",
    "midog": "source_wsi",
    "monkey": "source_wsi",
}

#: The step floor: rungs smaller than ``full / STEP_FLOOR_DIVISOR`` train for more epochs.
STEP_FLOOR_DIVISOR = 8

#: Fraction of the reference score a rung must reach for ``N*``.
NSTAR_FRACTION = 0.95


def ladder_for(dataset: str) -> tuple[int, ...]:
    try:
        return LADDERS[dataset]
    except KeyError:
        known = ", ".join(sorted(LADDERS))
        raise KeyError(f"no efficiency ladder for {dataset!r}; known: {known}.") from None


def atom_column_for(dataset: str) -> str:
    try:
        return ATOM_COLUMNS[dataset]
    except KeyError:
        known = ", ".join(sorted(ATOM_COLUMNS))
        raise KeyError(f"no atom column for {dataset!r}; known: {known}.") from None


# --- split variants (nested per-seed prefixes) -------------------------------------


def train_atoms(dataset_csv: str | Path, splits_csv: str | Path, atom_column: str) -> list[str]:
    """The sorted atom ids of the ``train`` split (fold 0 if the splits carry folds)."""
    import pandas as pd

    dataset = pd.read_csv(dataset_csv)
    splits = pd.read_csv(splits_csv)
    if "fold" in splits.columns:
        splits = splits[splits["fold"] == splits["fold"].min()]
    if atom_column not in dataset.columns:
        raise KeyError(f"atom column {atom_column!r} missing from {dataset_csv}")
    cols = ["sample_id"] if atom_column == "sample_id" else ["sample_id", atom_column]
    merged = dataset[cols].merge(splits, on="sample_id")
    atoms = merged.loc[merged["split"] == "train", atom_column].astype(str).unique()
    return sorted(atoms)


def nested_prefix(atoms: Sequence[str], seed: int, n: int) -> list[str]:
    """The first ``n`` atoms of the seed's permutation — a prefix of every larger rung."""
    ordered = np.asarray(sorted(atoms), dtype=object)
    perm = np.random.default_rng(seed).permutation(len(ordered))
    return [str(a) for a in ordered[perm][: max(0, n)]]


@dataclass(frozen=True)
class SplitVariant:
    """One ``(seed, n)`` split variant: where it lives and what it contains."""

    dataset: str
    seed: int
    n_atoms: int
    path: Path
    dataset_path: Path
    n_train_samples: int
    n_train_objects: int | None
    atoms: tuple[str, ...] = field(default=(), repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "seed": self.seed,
            "n_atoms": self.n_atoms,
            "path": str(self.path),
            "dataset_path": str(self.dataset_path),
            "n_train_samples": self.n_train_samples,
            "n_train_objects": self.n_train_objects,
            "atoms": list(self.atoms),
        }


def _count_points(points_paths: Sequence[str]) -> int | None:
    """Total annotated objects over the per-sample point CSVs (``None`` if any is missing)."""
    import pandas as pd

    total = 0
    for p in points_paths:
        path = Path(p)
        if not path.is_file():
            return None
        total += int(len(pd.read_csv(path)))
    return total


def write_split_variant(
    dataset_csv: str | Path,
    splits_csv: str | Path,
    *,
    dataset: str,
    seed: int,
    n: int,
    out_path: str | Path,
    atom_column: str | None = None,
    count_objects: bool = True,
) -> SplitVariant:
    """Write a ``splits.csv`` whose ``train`` is the seed's first ``n`` atoms, tune/test verbatim.

    Non-train rows are copied unchanged (same ``fold`` column if present), so every rung
    selects and scores on exactly the full tune/test sets. The pipeline requires every
    manifest sample to resolve to a split, so a paired ``dataset.csv`` (same stem,
    ``.dataset.csv``) drops the excluded train rows — the dense cache is keyed per sample, so
    the retained rows still hit the shared grids. Idempotent by construction: the variant is
    a pure function of ``(sorted atoms, seed, n)``.
    """
    import pandas as pd

    atom_column = atom_column or atom_column_for(dataset)
    ds = pd.read_csv(dataset_csv)
    splits = pd.read_csv(splits_csv)
    atoms = train_atoms(dataset_csv, splits_csv, atom_column)
    if n > len(atoms):
        raise ValueError(f"rung n={n} exceeds the {len(atoms)} train atoms of {dataset}")
    keep = set(nested_prefix(atoms, seed, n))
    atom_of = dict(zip(ds["sample_id"].astype(str), ds[atom_column].astype(str)))
    is_train = splits["split"] == "train"
    in_prefix = splits["sample_id"].astype(str).map(atom_of).isin(keep)
    variant = splits[~is_train | in_prefix].reset_index(drop=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    variant.to_csv(out_path, index=False)
    kept = set(variant["sample_id"].astype(str))
    dataset_path = out_path.with_suffix(".dataset.csv")
    ds[ds["sample_id"].astype(str).isin(kept)].reset_index(drop=True).to_csv(dataset_path, index=False)
    train_ids = set(variant.loc[variant["split"] == "train", "sample_id"].astype(str))
    n_objects = None
    if count_objects and "points_path" in ds.columns:
        n_objects = _count_points(
            ds.loc[ds["sample_id"].astype(str).isin(train_ids), "points_path"].tolist()
        )
    return SplitVariant(
        dataset=dataset,
        seed=seed,
        n_atoms=n,
        path=out_path,
        dataset_path=dataset_path,
        n_train_samples=len(train_ids),
        n_train_objects=n_objects,
        atoms=tuple(sorted(keep)),
    )


# --- recipe scaling ------------------------------------------------------------------


def recipe_scale(full_n: int, n: int, *, floor_divisor: int = STEP_FLOOR_DIVISOR) -> int:
    """The integer multiplier for ``epochs``/``patience`` at rung ``n``.

    ``max(1, ceil((full_n / floor_divisor) / n))`` — rungs at or above ``full/divisor`` run
    the committed recipe verbatim; smaller rungs are stretched so they see at least as many
    optimizer steps (and the same cosine horizon in steps) as the ``full/divisor`` rung.
    """
    if full_n <= 0 or n <= 0:
        raise ValueError("full_n and n must be positive")
    floor_atoms = full_n / floor_divisor
    return max(1, math.ceil(floor_atoms / n))


def scaled_recipe(
    full_n: int, n: int, *, epochs: int, patience: int | None,
    floor_divisor: int = STEP_FLOOR_DIVISOR,
) -> dict[str, int]:
    """``{"epochs", "patience", "scale"}`` for rung ``n`` (patience omitted when off)."""
    scale = recipe_scale(full_n, n, floor_divisor=floor_divisor)
    out = {"epochs": int(epochs) * scale, "scale": scale}
    if patience is not None:
        out["patience"] = int(patience) * scale
    return out


# --- learning-curve math -------------------------------------------------------------


@dataclass(frozen=True)
class RungResult:
    """One ``(encoder, n)`` point of a learning curve, aggregated over seeds."""

    encoder: str
    n: int
    per_replicate: tuple[float, ...]
    n_train_objects: tuple[int | None, ...] = ()
    realized_epochs: tuple[int | None, ...] = ()

    @property
    def mean(self) -> float:
        return float(np.mean(self.per_replicate)) if self.per_replicate else float("nan")

    @property
    def std(self) -> float:
        return float(np.std(self.per_replicate)) if self.per_replicate else float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "per_replicate": [round(v, 6) for v in self.per_replicate],
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "n_replicates": len(self.per_replicate),
            "n_train_objects": list(self.n_train_objects),
            "realized_epochs": list(self.realized_epochs),
        }


@dataclass(frozen=True)
class FullReference:
    """The full-data reference for one encoder (from the headline sweep, never retrained)."""

    encoder: str
    n: int
    mean: float
    std: float
    per_replicate: tuple[float, ...] = ()
    source: str = "sweep"

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "per_replicate": [round(v, 6) for v in self.per_replicate],
            "source": self.source,
        }


def curve_points(rungs: Sequence[RungResult], full: FullReference | None) -> list[tuple[int, float]]:
    """``(n, mean)`` sorted by ``n``, the full rung appended when given."""
    pts = [(r.n, r.mean) for r in rungs if r.per_replicate]
    if full is not None:
        pts.append((full.n, full.mean))
    return sorted(pts)


def n_star(points: Sequence[tuple[int, float]], target: float) -> int | None:
    """The smallest ``n`` whose mean score reaches ``target`` (``None`` if never)."""
    for n, mean in sorted(points):
        if mean >= target:
            return n
    return None


def noise_band_crossing(points: Sequence[tuple[int, float]], full: FullReference) -> int | None:
    """The smallest ``n`` whose mean lands inside the full rung's ``mean - std`` band."""
    return n_star(points, full.mean - full.std)


def rank_of(scores: Mapping[str, float]) -> dict[str, int]:
    """1-based dense ranks by descending score (ties broken by name for determinism)."""
    ordered = sorted(scores, key=lambda e: (-scores[e], e))
    return {e: i + 1 for i, e in enumerate(ordered)}


def rank_agreement(a: Mapping[str, int], b: Mapping[str, int]) -> dict[str, Any]:
    """Spearman/Kendall between two rank maps over their shared encoders."""
    from scipy.stats import kendalltau, spearmanr

    shared = sorted(set(a) & set(b))
    if len(shared) < 3:
        return {"n_encoders": len(shared), "spearman": None, "kendall": None}
    ra = [a[e] for e in shared]
    rb = [b[e] for e in shared]
    return {
        "n_encoders": len(shared),
        "spearman": round(float(spearmanr(ra, rb).statistic), 4),
        "kendall": round(float(kendalltau(ra, rb).statistic), 4),
    }


def pairwise_order_matches(a: Mapping[str, int], b: Mapping[str, int]) -> bool:
    """True when every encoder pair is ordered identically in both rank maps."""
    shared = sorted(set(a) & set(b))
    for i, x in enumerate(shared):
        for y in shared[i + 1 :]:
            if (a[x] < a[y]) != (b[x] < b[y]):
                return False
    return True


def rank_crossing(
    rungs_by_encoder: Mapping[str, Sequence[RungResult]],
    full_by_encoder: Mapping[str, FullReference],
) -> dict[str, Any]:
    """Per-rung agreement of the encoder ranking with the full-data ranking.

    Only rungs where every encoder has a score enter, so the per-rung correlation is over a
    fixed encoder set. ``first_n_full_order`` is the smallest rung from which the full-data
    pairwise order holds at that rung and every larger one.
    """
    full_rank = rank_of({e: f.mean for e, f in full_by_encoder.items()})
    encoders = sorted(full_by_encoder)
    ns = sorted({r.n for e in encoders for r in rungs_by_encoder.get(e, ()) if r.per_replicate})
    per_rung: list[dict[str, Any]] = []
    for n in ns:
        scores = {}
        for e in encoders:
            hit = [r for r in rungs_by_encoder.get(e, ()) if r.n == n and r.per_replicate]
            if hit:
                scores[e] = hit[0].mean
        if len(scores) != len(encoders):
            continue
        rank = rank_of(scores)
        per_rung.append(
            {
                "n": n,
                "ranking": [e for e in sorted(rank, key=rank.get)],
                **rank_agreement(rank, full_rank),
                "full_order_holds": pairwise_order_matches(rank, full_rank),
            }
        )
    first_n = None
    for entry in reversed(per_rung):
        if entry["full_order_holds"]:
            first_n = entry["n"]
        else:
            break
    return {
        "full_ranking": [e for e in sorted(full_rank, key=full_rank.get)],
        "per_rung": per_rung,
        "first_n_full_order": first_n,
    }


def build_dataset_efficiency(
    dataset: str,
    rungs_by_encoder: Mapping[str, Sequence[RungResult]],
    full_by_encoder: Mapping[str, FullReference],
    *,
    fraction: float = NSTAR_FRACTION,
) -> dict[str, Any]:
    """Assemble one dataset's efficiency block (curves, N*, noise-band crossing, rank crossing)."""
    spec = dataset_spec(dataset)
    if not full_by_encoder:
        raise ValueError(f"{dataset}: no full-data references — nothing to measure N* against")
    global_best = max(full_by_encoder.values(), key=lambda f: f.mean)
    encoders_out: dict[str, Any] = {}
    for encoder in sorted(set(rungs_by_encoder) | set(full_by_encoder)):
        rungs = sorted(rungs_by_encoder.get(encoder, ()), key=lambda r: r.n)
        full = full_by_encoder.get(encoder)
        points = curve_points(rungs, full)
        block: dict[str, Any] = {
            "rungs": [r.as_dict() for r in rungs],
            "full": full.as_dict() if full else None,
        }
        if full is not None:
            block["n_star_own"] = n_star(points, fraction * full.mean)
            block["n_star_global"] = n_star(points, fraction * global_best.mean)
            block["noise_band_crossing"] = noise_band_crossing(points, full)
        encoders_out[encoder] = block
    return {
        "dataset": dataset,
        "metric_name": spec.metric_name,
        "atom": atom_column_for(dataset),
        "ladder": list(ladder_for(dataset)),
        "n_star_fraction": fraction,
        "global_best": {"encoder": global_best.encoder, "mean": round(global_best.mean, 4)},
        "encoders": encoders_out,
        "rank_crossing": rank_crossing(rungs_by_encoder, full_by_encoder),
    }


def cross_dataset_nstar(per_dataset: Mapping[str, Mapping[str, Any]], key: str = "n_star_own") -> dict[str, Any]:
    """Does the ordering of encoders by ``N*`` agree across datasets? (lower N* = more efficient)."""
    datasets = [d for d in DATASET_ORDER if d in per_dataset]
    table: dict[str, dict[str, int | None]] = {}
    ranks: dict[str, dict[str, int]] = {}
    for d in datasets:
        encs = per_dataset[d]["encoders"]
        table[d] = {e: encs[e].get(key) for e in encs}
        scored = {e: v for e, v in table[d].items() if v is not None}
        # Rank by N* ascending, ties broken by full-data score descending, then name.
        ordered = sorted(
            scored,
            key=lambda e: (scored[e], -float(encs[e]["full"]["mean"]) if encs[e].get("full") else 0.0, e),
        )
        ranks[d] = {e: i + 1 for i, e in enumerate(ordered)}
    pairs: dict[str, Any] = {}
    for i, a in enumerate(datasets):
        for b in datasets[i + 1 :]:
            pairs[f"{a}|{b}"] = rank_agreement(ranks[a], ranks[b])
    return {"key": key, "n_star": table, "ranks": ranks, "pairs": pairs}


def build_efficiency_report(
    per_dataset: Mapping[str, Mapping[str, Any]],
    *,
    git_sha: str | None = None,
    seeds: Sequence[int] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "config": {
            "protocol": "nested-prefix ladder × seeds; tune/test full; full rung from headline sweep",
            "step_floor_divisor": STEP_FLOOR_DIVISOR,
            "n_star_fraction": NSTAR_FRACTION,
            "seeds": list(seeds),
            "git_sha": git_sha,
            **(dict(extra) if extra else {}),
        },
        "datasets": {d: dict(per_dataset[d]) for d in DATASET_ORDER if d in per_dataset},
        "cross_dataset": cross_dataset_nstar(per_dataset) if len(per_dataset) >= 2 else None,
    }


# --- plot ----------------------------------------------------------------------------


def plot_learning_curves(report: Mapping[str, Any], out_path: str | Path) -> Path:
    """Test metric vs log N per encoder, one panel per dataset, full-rung mean±std bands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = list(report["datasets"])
    fig, axes = plt.subplots(1, max(1, len(datasets)), figsize=(6.5 * max(1, len(datasets)), 4.8), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        block = report["datasets"][dataset]
        for encoder, eb in sorted(block["encoders"].items()):
            xs = [r["n"] for r in eb["rungs"] if r["n_replicates"]]
            ys = [r["mean"] for r in eb["rungs"] if r["n_replicates"]]
            es = [r["std"] for r in eb["rungs"] if r["n_replicates"]]
            full = eb.get("full")
            if full:
                xs.append(full["n"]); ys.append(full["mean"]); es.append(full["std"])
            if not xs:
                continue
            line = ax.errorbar(xs, ys, yerr=es, marker="o", capsize=2, label=encoder)
            if full:
                ax.axhspan(full["mean"] - full["std"], full["mean"] + full["std"],
                           color=line[0].get_color(), alpha=0.08)
        ax.set_xscale("log", base=2)
        ax.set_xlabel(f"annotated {block['atom']} (train)")
        ax.set_ylabel(f"test {block['metric_name']}")
        ax.set_title(dataset)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_json(path: str | Path, data: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
