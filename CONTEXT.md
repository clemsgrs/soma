# soma

`soma` is a scientific experimentation engine for computational pathology: it takes a dataset of pathology images plus labels and splits, runs them through a frozen foundation-model encoder and a trained task head, and returns comparable metrics. This glossary fixes the vocabulary used when soma is put to its **foundation-model benchmarking** use.

## Language

### Experiment structure

**Modeling**:
The trainable downstream portion of a soma workflow after frozen foundation-model encoding, including aggregation or decoding, task-specific prediction, and optimization.
_Avoid_: including foundation-model encoding itself under Modeling.

### Benchmarking

**Custom experimentation**:
Exploring combinations of preprocessing, encoding, downstream modeling, and training choices to find or understand a strong workflow for a research task. Unlike Benchmarking, several choices may change together.

**Benchmarking**:
The controlled comparison of one pipeline component on a single dataset: vary that component while holding the rest of the protocol fixed, then measure its effect on downstream performance. Foundation-model encoders are the most common comparison axis, but preprocessing, aggregation, decoding, or task-head choices can be studied the same way; a published Benchmark is a special case.

**Benchmark**:
A *published* evaluation dataset paired with a canonical protocol and recorded expected metrics, reproducible within a stated tolerance. A Benchmark is the pinned, annotated special case of Benchmarking — the same flow plus a fixed recipe and a checkable expected result.
_Avoid_: calling any single run or any user dataset a "benchmark"; reserve the noun for the published, expected-numbers-annotated instance.

**Protocol**:
The recipe a Benchmark encodes *as code* in its `soma/benchmarks/<name>.py` module: curation entry, config construction (load a committed YAML when static, or compute it — e.g. EVA's dataset×encoder grid), the metric/scorer (soma's `summary.json` headline by default, or a benchmark-specific one like OCELOT's greedy matcher), the expected reference table, and the tolerance. Low-code where the recipe is static, more code where it computes or scores specially.

**Reproduction**:
Re-running a Benchmark's canonical protocol and checking the resulting metric lands within the recorded tolerance of its expected number. The pass/fail validation that soma's engine produces competitive, stable results.

**Leaderboard**:
A persistent, per-(dataset + splits + task) ranked view over the run registry: every completed run contributes a row, ranked by the task's primary metric, with seeds collapsed to mean ± std. It is a *projection over runs*, keyed by data identity — not a one-off table owned by a single sweep. It holds two kinds of rows (below).
_Avoid_: "results table", "comparison" (a comparison is the on-demand N-run `compare_runs` report, a different thing).

**Measured row**:
A Leaderboard row backed by an actual Run — its metric comes from that run's `summary.json`.

**Reference row**:
A Leaderboard row backed by no run — a published number a Benchmark carries (e.g. the kaiko-ai/eva leaderboard values in `soma/benchmarks/reference/eva.csv`). A Reproduction is a Measured row landing next to its Reference row, with the tolerance check reading both. Two shapes: a **broad** reference (one config-agnostic scalar, e.g. OCELOT's official-challenge band) renders as a threshold banner the whole Leaderboard is read against; a **keyed** reference (indexed by config axes, e.g. EVA's per-encoder numbers) renders as aligned rows, one per matching config. Comes in two **kinds** (below).

**Gate row**:
A Reference row `soma reproduce` **tolerance-checks** — the checkable target, printed `PASS`/`FAIL` with the delta. The default kind.

**External row**:
A non-gating Reference row: a published third-party number (e.g. HEST's leaderboard Pearson) rendered *beside* the Measured value with the signed delta, but **never** tolerance-checked. Used when soma measures the same quantity through a *different* stack than the source, so the delta is expected guidance, not a target. Carries a human `label` and a linkable `url`.
_Avoid_: turning an External row into a Gate row by giving it a loose tolerance — that hides the delta the External row exists to show.

**Facet**:
The query that defines a Leaderboard: a set of config axes to *hold fixed* plus one axis to *vary and rank by*, over a fixed (dataset + splits + task). A "cross-encoder" Leaderboard holds the downstream recipe fixed and varies `encoder`; an "encoder-specific" one holds `encoder` fixed and varies the rest. One flat registry, many Facets — a Benchmark declares its canonical Facet (what the published comparison varied). `--like <run_dir>` pins a Facet by example (hold everything fixed except the vary-axis).
_Avoid_: treating a Leaderboard as a single fixed keying; "grouping" (a Facet also fixes a fair-comparison control set, not just a group-by).

**Sweep**:
A launcher that fires many configs at once (e.g. an encoder grid) over one dataset. Its runs land in that dataset's Leaderboard; a Sweep does not own its own ranking.
_Avoid_: "campaign" (the OCELOT-specific name for the first hand-built sweep), "grid".

### Reproduction soundness

**Native reproduction**:
Reproducing a Benchmark with soma's *own* extraction stack — its slide2vec encoder → its feature cache → its head/probe — rather than the benchmark's original tooling (HEST uses TRIDENT). The extraction-stack difference is an accepted, non-gating delta: soma publishes Measured beside Reference and lets the reader compare, rather than issuing a PASS/FAIL against another lab's extractor.
_Avoid_: "exact reproduction" for a native one; and do not gate a native reproduction on the published number — **gate against yourself, publish against them**.

**Reproduction soundness**:
The evidence that a native reproduction is trustworthy, along three axes: **A — absolute agreement** (per-cell Measured vs Reference delta — what is *published*, never gated); **B — rank agreement** (a *bonus* — soma re-derives the benchmark's *ordering* of encoders); **C — drift guard** (the append-only, provenance-pinned results ledger makes any change across code/extractor versions an explicit diff). C is the only axis that ever gates, and it compares soma to soma.

**Pairwise concordance**:
The B statistic: over every `(dataset, encoder-pair)`, the fraction soma orders the same way the Reference does, pooled across datasets. Interpretable where per-dataset rank correlation is degenerate (few encoders). Computed over **resolvable** pairs only; per-dataset Spearman ρ is reported alongside. Corroborates A; it does not replace it.

**Resolvable pair**:
An encoder pair the Reference separates by more than a small ε (`RESOLVABLE_EPS` = 0.005 on the metric). Below ε the benchmark itself cannot call the ordering, so a soma flip is a within-noise coin-flip, not a defect — excluded from the concordance.

### Data preparation

**Supervision**:
The target information paired with pathology images for learning or evaluation, including scalar labels, dense masks, and point annotations.
_Avoid_: using "labels" as the umbrella term when masks or point annotations are also in scope.

**Manifest**:
The canonical on-disk schema soma consumes, identical across all task types: a `dataset.csv` (`sample_id`, `image_path`, and exactly one supervision column — `label` / `mask_path` / `points_path`, chosen by `dataset_type` — plus optional `patient_id`, `group_id`, `spacing_at_level_0`), a `splits.csv` (`sample_id`, `split`, `fold`), and a `summary.json`. `group_id` names a non-independent sample group for representation evaluation and remains optional for ordinary task runs. `spacing_at_level_0` is a finite positive µm/px declaration for the source image's level-0 pixels, not a requested extraction spacing. The Manifest is validated against `dataset_type` at load and written by one shared `write_manifest`.
_Avoid_: `manifest.csv` (the retired BEETLE name — the file is always `dataset.csv`); "dataset" for the files (a `Dataset` is the loaded object, a Manifest is the on-disk schema).

**Curator / Curation**:
Curation turns a *raw* public dataset into a Manifest. A Curator is a **deterministic function** `(raw_root, out_dir, **params) -> CuratedManifest`, typed by a structural `Protocol` — not a class hierarchy, because curators are dataset-specific adapters, not interchangeable components. Deterministic so re-curating the same raw data yields byte-identical files and a stable dataset identity.
_Avoid_: a `Curator` base class; "swappable curator" (curators are not interchangeable).

### Extraction geometry

**Declared geometry**:
The regime in which soma *states* the encoder input it wants — pooled `requested_tile_size_px`, dense `spacing_um` + `target_size` — as a claim about physical extent ("224 px covering 112 µm of tissue"). slide2vec must honor it exactly or raise; it never substitutes a different geometry.
_Avoid_: "the pooled path" as a synonym — hierarchical and dense declarations are Declared geometry too.

**Given geometry**:
The regime in which the pixels are supplied without any soma request — pre-cropped tile datasets (`dataset_type="tile"`), whose sizes belong to the upstream dataset, vary per sample, and are often not square. The encoder's shipped transform is the contract, and its resizing is the published protocol rather than a substitution.
_Avoid_: "the tile path" as a synonym; "raw tiles" (they are curated, just not requested).

**Effective encoder input**:
The geometry of the tensor immediately before `encode_tiles` / `encode_tiles_dense` — `requested_tile_size_px` for a pooled declaration, the padded `encoded_size` for whole-tile dense, `window_size` for sliding dense, and whatever the shipped transform produced under Given geometry. The single quantity the encoder-input contract is stated over, so one capability check serves pooled and dense alike.
_Avoid_: "tile size" (ambiguous between requested, read, and encoder-facing sizes).

**Encoder-input contract**:
slide2vec's explicit, non-defaultable statement of which regime a run is in, resolved at model load. soma names the regime and writes no geometry policy itself.
_Avoid_: treating an absent contract as Given geometry — absence is an error, not a default.

**Composed tiling config**:
The hs2p `TilingConfig` a `PreprocessingConfig` resolves to (`tiling_config()`), built at **resolve** time rather than config-parse time because hs2p requires a spacing and a tile size that soma leaves unset until the encoder supplies them. The pooled adapter, the slide-manifest ROI sampler and the feature-cache key all read this one object, so the geometry a run used and the geometry its key describes cannot diverge.
_Avoid_: "mirroring hs2p" (the fields are forwarded, not re-declared); treating it as a config-surface type — it exists only after resolution.

### Run identity (existing soma concepts, load-bearing for the Leaderboard)

**Run**:
A single execution of a config → one `run_dir` with one `summary.json`. The unit that produces measured metrics.

**Experiment**:
The config-hash identity a Run belongs to: a sha256 over the resolved config *including the `dataset.csv`/`splits.csv` checksums but excluding the seed*, so different seeds are Runs of the same Experiment. A Leaderboard row is one Experiment, its seeds collapsed to mean ± std.
_Avoid_: using "experiment" loosely for a single run or a whole study.
