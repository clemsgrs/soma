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
