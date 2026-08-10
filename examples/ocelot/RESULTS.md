# OCELOT 2023 — reference results

This is the recorded reference for the OCELOT cell-detection anchor: the exact recipe,
environment, and metrics that `soma reproduce ocelot` checks an independent run against.
Machine-readable band + per-row tolerance: `soma/benchmarks/reference/ocelot.csv`.

## Recipe (the anchor)

Frozen **Virchow2 @ 0.2 µm/px native** → dense token grid → `lightweight_conv` decoder →
per-class peak heatmap, scored with class-aware **F1 @ δ = 3 µm** (OCELOT's official
tolerance). Per-class score thresholds are swept on **tune**, frozen, and applied once to
**test** (leakage-free). Config:
[`ocelot_virchow2_0.20.yaml`](../../soma/benchmarks/configs/ocelot/ocelot_virchow2_0.20.yaml),
**canonical seed 0**.

## Headline

| metric | tune | **test (greedy, official)** | test (hungarian) |
|---|---|---|---|
| mean_F1 | 0.7146 | **0.6995** | 0.6996 |
| F1 — BC (class 0) | — | 0.6739 | 0.6741 |
| F1 — TC (class 1) | — | 0.7250 | 0.7250 |
| mean_F1 / image | — | 0.6006 | 0.6007 |

- **Greedy is the leaderboard-comparable number** (OCELOT scores with greedy matching);
  hungarian is the run's training headline. They agree to ±0.0001 here, so the ~0.70
  result is matcher-independent.
- Frozen per-class thresholds (swept on tune): **BC 0.5317, TC 0.4913**.
- tune→test mean_F1 drop is only −0.015 → the operating point transfers; not overfit to tune.

A frozen probe at **0.70 mean_F1** lands in the OCELOT 2023 competitive band (top
fully-trained methods ~0.70–0.73) — a strong frozen-probe baseline, not merely "plumbing
works." The health-gate context is GitHub issue #151.

> **This file is the single-seed (seed 0) anchor reference** that `soma reproduce ocelot`
> checks against. The 3-seed **encoder × spacing campaign** (Virchow2/UNI2 × 0.25/0.5 + anchor,
> tune-select → test-confirm; GitHub issue #152) lives in
> `docs/ocelot-detection-benchmark.rst`. Its finding: Virchow2 @ 0.2 wins on tune, but on
> test it ties UNI2 @ 0.25 within seed noise (both ~0.70 mF1); 0.5 µm/px is not competitive.
> This seed-0 test greedy mF1 (0.6995) equals the campaign's Virchow2 @ 0.2 seed-0 value.

## Environment

| component | version |
|---|---|
| soma-pathology | 1.5.1 |
| slide2vec | 5.1.1 (`>=5.1.0`) |
| torch | 2.7.1+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 2080 Ti |

Run: dense extraction batch 8, decoder training batch 1 × grad-accum 8, 39 epochs run
(best at epoch 29, early-stopped at patience 10).

## Soma 1.9 release revalidation

The native anchor was reproduced from a fresh cache at soma `ba8e529` with released
slide2vec 5.7.0 on an RTX 3080 Ti. Seed 0 reached greedy test mean_F1 **0.7030**
(Δ **+0.0035** from the 0.6995 anchor), tune mean_F1 0.7149, and Hungarian test
mean_F1 0.7031. Training again selected epoch 29 and stopped at epoch 39. The result is
recorded in `soma/benchmarks/results/ocelot.csv`.

## Spacing-aware migration validation

Issue #320 replaced the separately rendered 0.25/0.5 datasets with on-read resampling
from the one native 0.2 Manifest. The frozen rendered artifacts were compared before
retiring their workflow.

Across all 663 images, native 0.2 reads preserved decoded RGB bytes exactly. Coarser
dimensions also matched every rendered image; pixel values differ because the historical
path used Pillow BOX followed by a second lossy JPEG encode, while Slide2Vec 5.7 uses
OpenCV area resampling directly on the original decoded JPEG.

| spacing (µm/px) | shape | pixel MAE | RMSE | PSNR | max abs | exact channels |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 819×819 | 3.1753 | 4.4478 | 35.17 dB | 66 | 15.81% |
| 0.50 | 410×410 | 4.7285 | 6.4826 | 31.90 dB | 94 | 11.40% |

Dense grids were regenerated through both source paths with the same Slide2Vec 5.7
runtime, six samples spanning train/tune/test, identical sample order, batch size 8,
fp16 compute/fp32 storage, and the committed 448-window/25%-overlap geometry. Every grid
had the same shape and finite values.

| encoder | spacing | grid | flattened cosine, mean (range) | relative L2, mean |
|---|---:|---:|---:|---:|
| Virchow2 | 0.25 | 1280×59×59 | 0.9654 (0.9123–0.9896) | 0.2531 |
| Virchow2 | 0.50 | 1280×30×30 | 0.9453 (0.8452–0.9726) | 0.3257 |
| UNI2 | 0.25 | 1536×59×59 | 0.9720 (0.9214–0.9880) | 0.2237 |
| UNI2 | 0.50 | 1536×30×30 | 0.9531 (0.8773–0.9757) | 0.2928 |

Native-frame annotations transformed to the historical 0.25/0.5 coordinates within
1.14e-13 px and inverse-exported exactly in the committed coordinate tests. The grid
difference is large enough that coarse protocol performance must be validated
empirically; it is not treated as numerical identity.

Fresh-cache seed-0 Virchow2 runs then exercised the complete on-read extraction,
training, tune-threshold sweep, and official greedy test scorer. Their tune scores match
or improve on the corresponding rendered-input seed-0 results and remain consistent with
the historical three-seed sweeps.

| spacing | rendered tune, seed 0 | rendered tune, 3-seed sweep | on-read tune, seed 0 | on-read test, seed 0 |
|---:|---:|---:|---:|---:|
| 0.25 | 0.7024 | 0.6948 ± 0.0065 | 0.7053 | 0.7148 |
| 0.50 | 0.5924 | 0.5928 ± 0.0047 | 0.5970 | 0.6085 |

The native 0.2 fresh-cache release reproduction above independently remained within its
published tolerance (0.7030 versus the 0.6995 anchor). Together, these runs show no
benchmark regression from collapsing the rendered variants onto spacing-aware reads.
The two coarse on-read results and their code/version provenance are recorded separately
in `soma/benchmarks/results/ocelot-spacing-migration.csv`; they are validation evidence,
not cells in the canonical encoder-only benchmark facet.

## Reproducing

See [`README.md`](README.md) for the full download → curate → run → score workflow, or run
the one-shot checker:

```bash
# fast: re-score an existing run-dir against this reference (seconds, no training)
soma reproduce ocelot --from-run-dir <output_root>/ocelot_virchow2_0p20_lightconv

# full: curate (if needed) → train → score → check, from scratch (~3 h on one GPU)
soma reproduce ocelot --raw-root <data_root>/ocelot --output-root output
```

## Reproducibility caveats

- **Not bit-exact across GPUs/drivers, and dense grids are not bit-exact across batch sizes.**
  Encoder features depend on the batch size `B` of the forward (cuBLAS picks a different
  reduction order as `B` changes), so a cache is reproducible only to a *float tolerance*, never
  bit-for-bit. The effect is tiny — ~5.6e-6 in `1 − cosine`, *smaller* than the fp16-vs-fp32
  difference the pipeline already accepts — so it does not matter for scores, but do not rely on
  byte-identity. Note this holds even at a fixed nominal batch size: a slide whose ROI count is
  not a multiple of the batch size sends its tail through a smaller-`B` forward, so "pin to 8"
  never gave a single consistent `B`. The only stochastic stage is decoder training; cuDNN
  nondeterminism plus a different GPU move mean_F1 by ~0.01–0.02. `soma reproduce ocelot`
  highlights |Δ mean_F1| ≤ 0.02 as `REFERENCE OK`; a larger gap is reported as
  `POTENTIAL DRIFT` and signals a real environment/plumbing difference, not noise.
- The dense cache key omits batch size — which is fine, because features are batch-tolerant, not
  batch-identical. There is no need to pin the extraction batch size for reproducibility.
