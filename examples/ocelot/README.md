# OCELOT 2023 — cell detection (detection-v1)

End-to-end recipe for the [OCELOT 2023](https://ocelot2023.grand-challenge.org/)
cell-detection benchmark on Soma's `dataset_type: detection` path: frozen encoder →
dense token grid → `lightweight_conv` decoder → per-class peak heatmap, scored with
class-aware **F1 @ δ = 3 µm (15 px @ 0.2 µm/px)**, OCELOT's official tolerance.

The encoder is swappable (the configs cover an encoder × spacing ablation): `ocelot.yaml`
is a CONCH example, and the `ocelot_{virchow2,uni2}_*.yaml` configs vary encoder and
spacing. The **published anchor** is `ocelot_virchow2_0.20.yaml` (frozen Virchow2 @ 0.2
µm/px, greedy test mean_F1 0.6995) — see [`RESULTS.md`](RESULTS.md).

The dataset itself lives outside the repo (`data/` is git-ignored). Only the curation
code (`soma/curation/ocelot.py`), these configs, and `eval_greedy.py` are tracked.

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

The dense reader reads flat JPEGs with PIL and ignores any `requested_spacing_um`, so
a coarser magnification must be **materialized at curation time**. Pass
`--render-spacing-um` to downsample every cell patch to a target µm/px (≥ the native
0.2) and rescale its points by the same factor; the emitted manifest stamps a
`level0_spacing` column equal to the rendered spacing and the patches land under
`curated/images/`. Each variant is its own output dir (own manifest digest in the
dense cache):

```bash
python -m soma.curation.ocelot \
  --raw-root        <data_root>/ocelot \
  --output-dir      <data_root>/ocelot/curated_0p4 \
  --render-spacing-um 0.4   # half-resolution (512×512) patches
```

Omitting the flag reproduces the native-resolution manifest above unchanged.

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

The **Virchow2 @ 0.2 µm/px** anchor (frozen-probe greedy test mean_F1 **0.6995**) is
recorded in [`RESULTS.md`](RESULTS.md) / [`expected_metrics.json`](expected_metrics.json).
`reproduce.py` runs the whole curate → train → greedy-score → check loop and asserts the
result is within tolerance of that reference:

```bash
# full: curate (if needed) → train → score → check   (~3 h on one GPU)
python examples/ocelot/reproduce.py --data-root <data_root>/ocelot

# fast: re-score an already-trained run-dir and check (seconds, no training)
python examples/ocelot/reproduce.py \
  --from-run-dir <output_root>/ocelot_virchow2_0p20_lightconv
```

Exit code 0 = within the ±0.02 mean_F1 band; 1 = a real environment/plumbing difference.
