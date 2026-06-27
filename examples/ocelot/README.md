# OCELOT 2023 — cell detection (detection-v1)

End-to-end recipe for the [OCELOT 2023](https://ocelot2023.grand-challenge.org/)
cell-detection benchmark on Soma's `dataset_type: detection` path: frozen CONCH →
dense token grid → `lightweight_conv` decoder → per-class peak heatmap, scored with
class-aware **F1 @ δ = 3 µm (15 px @ 0.2 µm/px)**, OCELOT's official tolerance.

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

## 3. Run

CONCH is gated on Hugging Face — authenticate first (`huggingface-cli login` or
`export HF_TOKEN=...`). The config uses batch 1 over a 1024² heatmap, so set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Edit the `dataset_csv` /
`splits_csv` / `output_root` paths in the config to match your environment.

```bash
# full baseline (50 epochs, 663 samples)
python -m soma examples/ocelot/ocelot.yaml

# OCELOT-official greedy re-score of the trained fold (Hungarian is the run headline)
python examples/ocelot/eval_greedy.py \
  --run-dir <output_root>/ocelot_conch_lightconv \
  --config  examples/ocelot/ocelot.yaml --matching greedy
```
