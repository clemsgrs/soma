# A native reproduction publishes the per-cell delta rather than gating it

When soma reproduces a Benchmark **natively** — with its own slide2vec extraction stack rather than the benchmark's original tooling (HEST uses TRIDENT) — the published reference is another lab's number, measured with another lab's extractor. soma therefore renders its Measured value **beside** that Reference together with the signed delta (`kind=external` rows, blank tolerance, never tolerance-checked), and lets the reader compare. A gate exists to flag *real* regressions; a cross-stack delta is not one, so it is published, not gated.

Drift is still caught, but against the right baseline. **Gate against yourself, publish against them:** the append-only, provenance-pinned results ledger (`results/<name>.csv`, stamped with `soma_commit` and `slide2vec_version`) turns a soma-vs-soma change across commits into an explicit diff, which is the comparison a regression gate should actually make.

Three views are rendered from the ledger joined to the reference:

- **A — absolute agreement** (what is published): per-cell Measured, Reference, and signed delta.
- **B — rank agreement** (a bonus, not the claim): pooled pairwise concordance over *resolvable* pairs — does soma re-derive the benchmark's *ordering* of encoders? Corroborating, since a foundation-model benchmark exists to rank encoders.
- **C — drift guard**: the append-only, provenance-pinned ledger.

## Why the delta is worth publishing

The first 9 cells (PAAD, COAD, LUNG × `uni2`, `virchow2`, `h-optimus-1`) put the slide2vec↔TRIDENT gap at a **median 0.21 % relative, max 2.99 %**; six of the nine agree to within 0.25 %. An earlier draft of this ADR assumed the opposite — that a native stack could not be expected to match in absolute terms, so only the ranking could carry the claim. The measurements refuted that. Because absolute agreement turns out to be tight, the per-cell delta is informative on its own, and rank agreement settles into what it is: a useful secondary check.

## Considered and rejected

- **Gate the delta against the external reference** (e.g. a ±2 % relative band, mirroring the one `eva/<dataset>` uses on its gate rows). Rejected: it would gate soma against another lab's extraction stack, firing on legitimate stack differences rather than on real issues, and any threshold is arbitrary. On the first 9 cells a ±2 % band fails COAD/`uni2` (+2.99 %) and LUNG/`virchow2` (−2.90 %), neither of which is a regression in soma. EVA's gate is not this: its rows gate a protocol soma reproduces through the same extraction path.
- **Rank agreement as the primary claim.** Rejected as over-strong once absolute agreement proved tight: it discards the delta the external rows exist to surface, and pooled concordance is coarse (9 pairs across 3 tasks). Kept as **B**, a bonus.
- **Self-consistency gate** (gate a fresh run against soma's own recorded golden number). Rejected as a *runtime gate*: it bootstraps from a first golden run and adds machinery. Kept instead as **C**, served for free by the append-only ledger — a re-run at a new commit appends a row, so drift is a visible diff, not a silent overwrite.
- **Per-dataset Spearman as the B statistic.** Rejected: degenerate with few encoders (three near-tied backbones per task). Reported alongside pooled pairwise concordance, which aggregates cleanly across tasks; pairs the reference cannot resolve (gap < `RESOLVABLE_EPS` = 0.005) are excluded, so soma is not graded on within-noise coin-flips.

## Consequences

- `soma reproduce --record` had to learn to record for an **external-only** benchmark: with no gate row it keys the ledger entry off the single matching external row (previously a silent no-op). `results/hest.csv` is now populated the same way `results/eva.csv` is.
- `reproduction_report(name)` joins `results/<name>.csv` to `reference/<name>.csv` and computes A/B/C; the generated HEST doc page renders it, so the published numbers cannot drift from the ledger.
- `soma reproduce hest/<task>` never returns a PASS/FAIL verdict against HEST. A cell that moves is caught by C, at review time, as a new ledger row carrying a different `soma_commit`.
- Because the delta is published rather than gated, an out-of-family cell (LUNG/`virchow2`, −2.90 %) stays visible as a finding to investigate instead of becoming a red check to silence.
