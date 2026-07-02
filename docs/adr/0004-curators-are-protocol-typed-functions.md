# Curators are Protocol-typed functions, not a class hierarchy

Curation (raw public dataset → **Manifest**) is expressed as **deterministic functions typed by a structural `Curator` Protocol** — `(raw_root, out_dir, **params) -> CuratedManifest` — with one shared `write_manifest` writer and a load-time schema validator. There is deliberately **no `Curator` base class**.

Chosen against symmetry with the encoder / aggregator / decoder ABCs: those earn a base class through *swappability* (you swap UNI2 for Virchow2 on the same data). Curators are the opposite — dataset-specific adapters (BACH's photo dir, OCELOT's nested zip, BEETLE's `data_overview.csv`) that are **never interchangeable** — so polymorphism buys nothing and a class-per-dataset hierarchy is pure ceremony. What curators share is their *output* (the Manifest) and a few *sub-steps* (writing CSVs, stratifying splits, emitting `summary.json`) — that is shared **machinery** (functions), not a shared **interface** (classes).

## Consequences

- Fixes the schema drift: one `dataset.csv` (retire BEETLE's `manifest.csv`), `splits.csv` always with `fold`, always a `summary.json`, one writer instead of three.
- Determinism is a stated contract (all curators already satisfy it): re-curating the same raw data yields byte-identical files → stable dataset identity → a Leaderboard that does not fragment on re-curate.
