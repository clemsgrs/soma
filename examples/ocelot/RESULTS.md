# OCELOT 2023 — reference results

This is the recorded reference for the OCELOT cell-detection anchor: the exact recipe,
environment, and metrics that `reproduce.py` checks an independent run against.
Machine-readable mirror: [`expected_metrics.json`](expected_metrics.json).

## Recipe (the anchor)

Frozen **Virchow2 @ 0.2 µm/px native** → dense token grid → `lightweight_conv` decoder →
per-class peak heatmap, scored with class-aware **F1 @ δ = 3 µm** (OCELOT's official
tolerance). Per-class score thresholds are swept on **tune**, frozen, and applied once to
**test** (leakage-free). Config: [`ocelot_virchow2_0.20.yaml`](ocelot_virchow2_0.20.yaml),
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

## Reproducing

See [`README.md`](README.md) for the full download → curate → run → score workflow, or run
the one-shot checker:

```bash
# fast: re-score an existing run-dir against this reference (seconds, no training)
python examples/ocelot/reproduce.py --from-run-dir <output_root>/ocelot_virchow2_0p20_lightconv

# full: curate (if needed) → train → score → check, from scratch (~3 h on one GPU)
python examples/ocelot/reproduce.py --data-root <data_root>/ocelot --output-root output
```

## Reproducibility caveats

- **Not bit-exact across GPUs/drivers.** The dense grids are cache-identical *only at
  extraction batch 8* (the dense cache key omits batch size; fp16 features differ at other
  batch sizes). The only stochastic stage is decoder training; cuDNN nondeterminism plus a
  different GPU move mean_F1 by ~0.01–0.02. `reproduce.py` asserts |Δ mean_F1| ≤ 0.02
  against the greedy headline; a larger gap signals a real environment/plumbing difference,
  not noise.
- **Pin the extraction batch size to 8** when reusing or resuming the dense cache.
