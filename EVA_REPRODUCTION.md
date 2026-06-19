# Reproducing the EVA patch-level benchmark in soma

Goal: validate soma's patch-level (`dataset_type="tile"`) path by reproducing
[kaiko-ai/eva](https://github.com/kaiko-ai/eva) leaderboard numbers (`eva.csv`)
for a few foundation models, and flag any cell that lands suspiciously far from
the expected value.

**Scope of this run:** `{uni2, virchow2}` × `{bach, breakhis, crc, patch_camelyon}`,
5 seeds per cell. Success bar: soma mean within ~0.03–0.05 balanced accuracy of
`eva.csv`, ranking preserved; cells outside that band are flagged `SUSPICIOUS`.

## How to run

### Single cell (the user-facing path)

`scripts/reproduce_eva.py` curates one raw EVA dataset and reproduces one
benchmark cell, printing soma's balanced accuracy next to the published value:

```bash
HF_TOKEN=... python scripts/reproduce_eva.py \
    --dataset bach --encoder uni2 --raw-root /path/to/raw/eva/bach
# bach / uni2  (5 seed(s))
#   val : soma 0.9141 ± 0.0072   eva 0.915   Δ -0.001
```

Datasets: `bach, breakhis, crc, mhist, gleason_arvaniti, patch_camelyon`.
Encoders: `uni2, virchow2` (gated on HuggingFace — set `HF_TOKEN`). The EVA
protocol itself lives in `soma/benchmarks/eva.py` (config builder + expected
values); the script is a thin wrapper. Feature extraction runs once per
(encoder, dataset) and is cached across seeds.

### Full sweep (maintainer)

`scripts/eva_sweep.py` runs the whole grid and builds a comparison table against
the bundled leaderboard snapshot (`soma/benchmarks/reference/eva.csv`, the single
source of expected values for both scripts), with optional CIFS→local staging:

```bash
# small datasets (cheap extraction), all encoders, 5 seeds
EVA_DATASETS="bach,breakhis" python scripts/eva_sweep.py run

# heavy datasets: stage tiles off the high-latency CIFS mount first
EVA_DATASETS="crc,patch_camelyon" EVA_STAGE_ROOT=/tmp/eva_stage python scripts/eva_sweep.py stage
EVA_DATASETS="crc,patch_camelyon" EVA_DATA_ROOT=/tmp/eva_stage \
  EVA_OUT=/tmp/eva_repro_local python scripts/eva_sweep.py run

# comparison table vs eva.csv (reads results under EVA_OUT)
python scripts/eva_sweep.py compare
```

`data/eva/<dataset>/` manifests are produced by `soma.curation.eva` and are
gitignored; the sweep reads them via `EVA_DATA_ROOT`. On this HPC the EVA tiles
sit on CIFS (~94 ms/tile), making extraction I/O-bound (~9 tiles/s); `stage`
bulk-copies a dataset's tiles to local storage once (encoders share them),
turning extraction GPU-bound (~140 tiles/s). The node is shared, so other users'
jobs can starve or OOM-kill runs — the sweep is resumable (per-cell result JSON +
on-disk feature and packed caches), so a killed run re-launches and skips
finished work.

## Protocol reconciliation (soma ↔ eva)

Verified against the eva source at `../eva` (offline classification configs and
dataset/metric/backbone classes). The points that matter for matching `eva.csv`:

| Aspect | eva | soma (this harness) |
|---|---|---|
| Head | `nn.Linear(d, C)` + CE (multiclass) / `Linear(d,1)`+BCE (pcam) | `nn.Linear(d, C)` + CE (binary uses C=2; equivalent for balanced acc) |
| Optimizer | AdamW, lr 3e-4, **wd 0.01** (torch default; only lr is set) | AdamW, lr 3e-4, **wd 0.01** |
| Schedule | none | none |
| Length | `max_steps=12500`, validate per epoch | epochs = ⌈12500/⌈N_train/256⌉⌉ (bach 6250, breakhis 2500, crc 32, pcam 13) |
| Early stop | per-dataset patience (bach 1250, breakhis 500, crc 7, pcam 3) | same patience (epochs == eva's validation checks) |
| Batch | 256 | 256 |
| Metric | `MulticlassAccuracy(average="macro")` / `BinaryBalancedAccuracy` | sklearn `balanced_accuracy_score` (== macro recall) |
| Monitor | same balanced-accuracy metric on val | `monitor=balanced_accuracy`, mode max |
| Runs | `n_runs=5`, mean ± std | 5 seeds, mean ± std |
| Input | `ResizeAndCrop(224)` + model norm | encoder's timm transform (Resize+CenterCrop 224 + model norm) |

### Split mapping

- **bach / breakhis / crc**: eva has no test split → reports on val. soma curates
  these as `train` + `test` (test = eva val) and runs `tune_is_test=true`, so
  checkpoint selection and reporting both use the eva-val set, exactly as eva does.
- **patch_camelyon**: real `train` + `val` + `test`. `tune_is_test=false`; monitor
  on val, report **val → `patch_camelyon`** and **test → `patch_camelyon/test`**.

### Curation fidelity

BACH index ranges match eva exactly; curated split counts match eva for all four
(bach 268/132, crc 100000/7180, breakhis 1132/339, pcam 262144/32768/32768).

### The one fix that actually moves numbers

eva's `paige_virchow2` uses `ExtractCLSFeatures(include_patch_tokens=False)` →
**CLS only, 1280-d**. slide2vec's virchow2 default is the 2560-d CLS+mean concat,
which would *not* reproduce `eva.csv`. The harness pins `output_variant="cls"`.
uni2 is CLS/pooled 1536-d and matches the slide2vec default.

## Results

Balanced accuracy, soma mean ± std over 5 seeds vs `eva.csv`. Bar: |Δ| within
~0.03–0.05; cells outside are flagged `SUSPICIOUS`.

| model | dataset | soma (mean ± std) | eva.csv | Δ | flag |
|---|---|---|---|---|---|
| mahmood_uni2_h | bach | 0.914 ± 0.007 | 0.915 | −0.001 | |
| mahmood_uni2_h | breakhis | 0.855 ± 0.006 | 0.859 | −0.004 | |
| mahmood_uni2_h | crc | 0.966 ± 0.001 | 0.965 | +0.001 | |
| paige_virchow2 | bach | 0.870 ± 0.010 | 0.883 | −0.014 | |
| paige_virchow2 | breakhis | 0.812 ± 0.008 | 0.821 | −0.010 | |
| paige_virchow2 | crc | 0.966 ± 0.001 | 0.967 | −0.001 | |
| _both_ | patch_camelyon | _blocked_ (see below) | 0.944 / 0.933 | | |

All completed cells land within **0.014** of eva.csv, ranking preserved
(uni2 ≥ virchow2). This validates the protocol reconciliation and the virchow2
`cls`-variant fix specifically (the 2560-d default would not match). The soma
patch-level path is sound: across 3 datasets × 2 models the reproduction is
within run-to-run noise of the published EVA numbers.

### patch_camelyon: blocked by node instability (not a soma issue)

pcam (327,680 tiles/model) needs a ~40-min uninterrupted extraction window. On
this shared HPC node an external memory-pressure killer (SIGKILL/137, no Python
traceback, fires during memory growth despite 200+ GB free — systemd-oomd-style,
PSI-based) repeatedly reaped the extraction mid-run (at 12k / 55k / 70k tiles
across attempts), independent of soma. Two compounding factors:

- Each extraction has a transient ~6 GB RSS spike during setup/first batch
  (not driven by dataloader worker count — capping workers via
  `EVA_ENC_WORKERS=4` lowered steady-state RSS 3.7→2.2 GB but not the spike).
- slide2vec's per-sample embedding bookkeeping was written for *slide* counts
  (dozens–hundreds of WSIs), but in the tile path **every tile is its own
  "slide" sample**, so N jumps to 327k. The dominant cost was
  `update_process_list_after_embedding` being called once per completed sample —
  each call re-read and rewrote the *entire* process_list CSV, i.e. **O(N²) I/O
  with a full-table DataFrame allocated and freed per tile**. Memory and time
  both grew with progress, which is what compounded the OOM-kills (and made any
  resume re-do that quadratic work from scratch).

  **Fixed in slide2vec** (`runtime/persist_callbacks.py`): the incremental
  persist callback now buffers completed tile samples and rewrites the CSV once
  per `TILE_EMBEDDING_FLUSH_INTERVAL` (1000) instead of once per sample, with a
  single authoritative full-CSV reconciliation at end of run. This makes the
  tile path O(N); on a crash it re-embeds at most one buffered batch of cheap
  tile samples. Slide-/hierarchical-level runs (sample == slide: few, expensive)
  keep a flush interval of 1, so every slide is still checkpointed individually.

To finish pcam: run on a quiet/dedicated node (or under SLURM with a real cgroup
reservation) — the harness + staging are ready (`EVA_DATASETS=patch_camelyon`,
tiles already staged at `/tmp/eva_stage`). The transient setup spike remains the
only node-instability exposure now that the O(N²) bookkeeping is gone.

<!-- RESULTS -->

## A soma fix this exercise surfaced

soma's tile path re-read every per-sample feature file from disk on **every
epoch** (`FeatureStore.load` → `load_array`, ~46 ms/file). With EVA's protocol
mapping to thousands of epochs on the small datasets, a single bach cell took
~31 min and was disk-bound (no GPU would help). Fixed by adding an in-memory
**packed feature cache** to `FeatureStore` for 1-D (single-vector) features,
persisted as `packed_features.pt` so multi-seed sweeps read the matrix once:
bach cell 31 min → ~50 s (~37×), identical numbers. Benefits the slide-encoder
and patient single-vector paths automatically; MIL bags are intentionally left
on the per-file path (ragged shape + RAM). The pack build fills a preallocated
`[N, D]` matrix in bounded chunks (never accumulating all N tensors) — an early
version that did `list(pool.map(...))` over 100k+ files spiked memory enough to
get the process OOM-killed on the large datasets. See `tests/test_features.py`.

## Caveats / known minor divergences (within tolerance)

- Binary head: soma uses `Linear(d,2)`+CE vs eva `Linear(d,1)`+BCE — same decision
  boundary capacity; balanced accuracy is unaffected in practice.
- Transform: timm `create_transform` uses `crop_pct` resize-then-centercrop vs
  eva's `ResizeAndCrop`; interpolation/crop details differ slightly.
- Step→epoch mapping rounds up the last partial epoch and may run a few extra
  optimizer steps vs eva's hard `max_steps` cap; negligible with early stopping.
