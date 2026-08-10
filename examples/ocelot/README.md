# OCELOT 2023 — cell detection (detection-v1)

End-to-end recipe for the [OCELOT 2023](https://ocelot2023.grand-challenge.org/)
cell-detection benchmark on Soma's `dataset_type: detection` path: frozen encoder →
dense token grid → `lightweight_conv` decoder → per-class peak heatmap, scored with
class-aware **F1 @ δ = 3 µm (15 px @ 0.2 µm/px)**, OCELOT's official tolerance.

The encoder is swappable (the configs cover an encoder × spacing ablation). `ocelot.yaml`
here is a standalone CONCH example. The encoder × spacing configs
(`ocelot_{virchow2,uni2}_*.yaml`) are **canonical under
[`soma/benchmarks/configs/ocelot/`](../../soma/benchmarks/configs/ocelot)** — they are the
committed YAML the registered `ocelot` benchmark loads (`build_config`), and `campaign.py`
resolves them from there by `(encoder, spacing)` (no local copies). The **published
anchor** is `ocelot_virchow2_0.20.yaml` (frozen Virchow2 @ 0.2 µm/px, greedy test mean_F1
0.6995) — see [`RESULTS.md`](RESULTS.md).

The dataset itself lives outside the repo (`data/` is git-ignored). Only the curation
code (`soma/curation/ocelot.py`), the packaged benchmark configs, this `ocelot.yaml`, and
`eval_greedy.py` are tracked. OCELOT is also a first-class registered benchmark —
`soma reproduce ocelot` (see §4) supersedes the old `reproduce.py`.

## 1. Download (one-time, gated)

OCELOT is distributed via Zenodo record [8417503](https://zenodo.org/records/8417503)
(`ocelot2023_v1.0.1.zip`, 303 MB, MD5 `215230295b0440b4dc356519ef9ff644`). The record
is open but you must accept its Terms & Conditions on the page once. Then:

```bash
mkdir -p <data_root>/ocelot && cd <data_root>/ocelot
curl -L -o ocelot2023_v1.0.1.zip \
  "https://zenodo.org/api/records/8417503/files/ocelot2023_v1.0.1.zip/content"
echo "215230295b0440b4dc356519ef9ff644  ocelot2023_v1.0.1.zip" | md5sum -c -   # expect OK
unzip -q ocelot2023_v1.0.1.zip
# the zip nests one folder; flatten so images/ + annotations/ sit at the root:
shopt -s dotglob && mv ocelot2023_v1.0.1/* . && rmdir ocelot2023_v1.0.1
```

Expected layout (cell patches are what detection-v1 uses):

```
images/{train,val,test}/cell/{NNN}.jpg      # 1024×1024 @ 0.2 µm/px
annotations/{train,val,test}/cell/{NNN}.csv # headerless x,y,label ; label 1=BC, 2=TC
metadata.json
```

Split sizes (v1.0.1 drops 4 under-annotated test cases): **400 / 137 / 126**.

## 2. Curate → Soma manifests

```bash
python -m soma.curation.ocelot \
  --raw-root   <data_root>/ocelot \
  --output-dir <data_root>/ocelot/curated
```

Writes under `curated/`: `dataset.csv` + `splits.csv` (663 samples; OCELOT
train→`train`, val→`tune`, test→`test`, single fold), one `points/<sample_id>.csv`
(`x,y,class`, label remapped 1→0 BC / 2→1 TC) per sample, and `summary.json`.

> For a cheap end-to-end smoke test, carve a small local subset (e.g. 40/15/15)
> into `dataset_debug.csv` / `splits_debug.csv` and point a local debug config at
> it. This is intentionally kept local (under the git-ignored `data/`), not
> shipped here.

### Magnification variants (encoder × spacing ablation)

Every magnification protocol consumes the same native Manifest. Its images remain
1024×1024 at `spacing_at_level_0: 0.2`, and its annotations remain in that level-0 pixel
frame. A config selects physical scale only through
`preprocessing.requested_spacing_um`; Slide2Vec and hs2p area-downsample the native JPEG
on read and record the resolved source/effective spacing beside each dense grid.

```bash
# All three configs use curated/dataset.csv + curated/splits.csv.
python -m soma soma/benchmarks/configs/ocelot/ocelot_virchow2_0.25.yaml \
  --set data.dataset_csv=<data_root>/ocelot/curated/dataset.csv \
  --set data.splits_csv=<data_root>/ocelot/curated/splits.csv
```

The committed 0.2/0.25/0.5 protocols declare target sizes 1024/819/410 pixels,
respectively. Detection targets are transformed from level 0 into that effective frame;
exported predictions are transformed back to level-0 coordinates.

## 3. Run

The encoder is gated on Hugging Face — authenticate first (`huggingface-cli login` or
`export HF_TOKEN=...`). Repoint the committed config at your paths with `--set`
(`key=value` into the config layout) rather than editing it on disk:

```bash
# full baseline (50 epochs, 663 samples)
python -m soma examples/ocelot/ocelot.yaml \
  --set data.dataset_csv=<data_root>/ocelot/curated/dataset.csv \
  --set data.splits_csv=<data_root>/ocelot/curated/splits.csv

# OCELOT-official greedy re-score of the trained fold (Hungarian is the run headline)
python examples/ocelot/eval_greedy.py \
  --run-dir <output_root>/ocelot_conch_lightconv \
  --config  examples/ocelot/ocelot.yaml --matching greedy
```

## 4. Reproduce the published anchor

OCELOT is now a **first-class registered benchmark** (`soma list benchmarks` shows
`ocelot`), so the whole curate → train → greedy-score → tolerance-check loop is one
command — no bespoke script. The expected band + per-row tolerance ship as package data
(`soma/benchmarks/reference/ocelot.csv`); the human-readable record stays in
[`RESULTS.md`](RESULTS.md). The **Virchow2 @ 0.2 µm/px** anchor is frozen-probe greedy test
mean_F1 **0.6995 ± 0.02**.

```bash
# full: curate → run canonical seed(s) → greedy-score → check   (~3 h on one GPU)
soma reproduce ocelot --raw-root <data_root>/ocelot --output-root <output_root>

# fast: re-score an already-trained run dir and check (seconds, no training)
soma reproduce ocelot --from-run-dir <output_root>/ocelot_virchow2_0p20_lightconv

# smoke: a single seed instead of the canonical set
soma reproduce ocelot --raw-root <data_root>/ocelot --seeds 1
```

The former `reproduce.py` / `expected_metrics.json` are absorbed into this benchmark;
`eval_greedy.py` remains only as the thin CLI that `campaign.py` drives per (cell, seed)
and now re-uses the benchmark's greedy scorer.

The command prints `REFERENCE OK` within the ±0.02 mean_F1 band and `POTENTIAL DRIFT`
outside it. Reference comparisons are diagnostic and do not change the command's exit
status; release gates should inspect the reported value explicitly.
