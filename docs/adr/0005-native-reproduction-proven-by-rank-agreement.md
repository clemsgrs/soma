# A native reproduction is proven by rank agreement, not by gating the absolute delta

When soma reproduces a Benchmark **natively** — with its own slide2vec extraction stack rather than the benchmark's original tooling (HEST uses TRIDENT) — the reproduced number is *expected* to differ from the published one, because the features come from a different extractor. So HEST reference rows are `kind=external` (rendered beside the Measured value, never tolerance-checked), and soundness is claimed along three axes: **A** absolute agreement (per-cell delta, shown not gated), **B** rank agreement (the headline — soma re-derives the benchmark's *ordering* of encoders, measured by pooled pairwise concordance over resolvable pairs), and **C** a drift guard (the append-only, provenance-pinned results ledger). The headline is B: a foundation-model benchmark exists to rank encoders, and a ranking survives an extraction-stack change even when absolute values shift.

Chosen over gating the absolute delta because gating a native reproduction on `|Measured − Reference| < ε` would contradict why the rows are external in the first place — the slide2vec↔TRIDENT gap is real and legitimate, so a per-cell tolerance would either be so loose it proves nothing or so tight it fails on sound pipelines. Rank agreement is both the stronger scientific claim (an independent stack recovering the benchmark's *conclusions*) and robust to the extraction gap.

## Considered and rejected

- **Absolute-delta gate rows** (give each HEST cell a `kind=gate` row with a tolerance). Rejected: it re-hides the delta the external rows exist to surface, and any threshold is arbitrary given the extraction-stack difference. It would also mask the HEST comparison, since `soma reproduce` renders external references only when *not* gating.
- **Self-consistency gate** (gate a fresh run against soma's own recorded golden number). Rejected as a *runtime gate*: it bootstraps from a first golden run and adds machinery. Kept instead as **C**, served for free by the append-only, provenance-pinned ledger — a re-run at a new commit appends a row, so drift is a visible diff, not a silent overwrite.
- **Per-dataset Spearman as the headline.** Rejected: degenerate with few encoders (three near-tied backbones per task). Reported alongside, but the headline is pooled pairwise concordance, which aggregates cleanly across tasks; pairs the reference cannot resolve (gap < 0.005) are excluded so soma is not graded on within-noise coin-flips.

## Consequences

- `soma reproduce --record` had to learn to record for an **external-only** benchmark: with no gate row it keys the ledger entry off the single matching external row (previously a silent no-op). `results/hest.csv` is now populated the same way `results/eva.csv` is.
- `reproduction_report(name)` joins `results/<name>.csv` to `reference/<name>.csv` and computes A/B/C; the generated HEST doc page renders it, so the published proof cannot drift from the ledger.
