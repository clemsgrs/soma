# soma

`soma` is a scientific experimentation engine for computational pathology: it takes a dataset of pathology images plus labels and splits, runs them through a frozen foundation-model encoder and a trained task head, and returns comparable metrics. This glossary fixes the vocabulary used when soma is put to its **foundation-model benchmarking** use.

## Language

### Benchmarking

**Benchmarking**:
The core activity — running one or more frozen foundation-model encoders through soma's pipeline on a single dataset to obtain comparable metrics, so encoder (and aggregation/decoder/head) choices can be ranked. Works on *any* dataset the user brings; a published benchmark is a special case.

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
A Leaderboard row backed by no run — a published number a Benchmark carries (e.g. the kaiko-ai/eva leaderboard values in `soma/benchmarks/reference/eva.csv`). A Reproduction is a Measured row landing next to its Reference row, with the tolerance check reading both. Two shapes: a **broad** reference (one config-agnostic scalar, e.g. OCELOT's official-challenge band) renders as a threshold banner the whole Leaderboard is read against; a **keyed** reference (indexed by config axes, e.g. EVA's per-encoder numbers) renders as aligned rows, one per matching config.

**Facet**:
The query that defines a Leaderboard: a set of config axes to *hold fixed* plus one axis to *vary and rank by*, over a fixed (dataset + splits + task). A "cross-encoder" Leaderboard holds the downstream recipe fixed and varies `encoder`; an "encoder-specific" one holds `encoder` fixed and varies the rest. One flat registry, many Facets — a Benchmark declares its canonical Facet (what the published comparison varied). `--like <run_dir>` pins a Facet by example (hold everything fixed except the vary-axis).
_Avoid_: treating a Leaderboard as a single fixed keying; "grouping" (a Facet also fixes a fair-comparison control set, not just a group-by).

**Sweep**:
A launcher that fires many configs at once (e.g. an encoder grid) over one dataset. Its runs land in that dataset's Leaderboard; a Sweep does not own its own ranking.
_Avoid_: "campaign" (the OCELOT-specific name for the first hand-built sweep), "grid".

### Data preparation

**Manifest**:
The canonical on-disk schema soma consumes, identical across all task types: a `dataset.csv` (`sample_id`, `image_path`, and exactly one supervision column — `label` / `mask_path` / `points_path`, chosen by `dataset_type` — plus optional `patient_id`, `level0_spacing`), a `splits.csv` (`sample_id`, `split`, `fold`), and a `summary.json`. Validated against `dataset_type` at load; written by one shared `write_manifest`.
_Avoid_: `manifest.csv` (the retired BEETLE name — the file is always `dataset.csv`); "dataset" for the files (a `Dataset` is the loaded object, a Manifest is the on-disk schema).

**Curator / Curation**:
Curation turns a *raw* public dataset into a Manifest. A Curator is a **deterministic function** `(raw_root, out_dir, **params) -> CuratedManifest`, typed by a structural `Protocol` — not a class hierarchy, because curators are dataset-specific adapters, not interchangeable components. Deterministic so re-curating the same raw data yields byte-identical files and a stable dataset identity.
_Avoid_: a `Curator` base class; "swappable curator" (curators are not interchangeable).

### Run identity (existing soma concepts, load-bearing for the Leaderboard)

**Run**:
A single execution of a config → one `run_dir` with one `summary.json`. The unit that produces measured metrics.

**Experiment**:
The config-hash identity a Run belongs to: a sha256 over the resolved config *including the `dataset.csv`/`splits.csv` checksums but excluding the seed*, so different seeds are Runs of the same Experiment. A Leaderboard row is one Experiment, its seeds collapsed to mean ± std.
_Avoid_: using "experiment" loosely for a single run or a whole study.
