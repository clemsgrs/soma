# Composite ensemble members are L2-normalised before concatenation

The decoder-ladder's rung 4 (multi-FM ensemble, #235) concatenates several frozen
encoders' dense token grids channel-wise and hands the `Σdᵢ`-channel grid to the same
decoder the single-encoder cells use. The decoder's 1×1 `Σdᵢ→D` projection is what keeps
downstream capacity `d`-invariant, and that fairness device is all the design records.

Feature-concat alone is not fair across members: foundation models emit patch features at
very different scales (norms differ by an order of magnitude between families), so the
member with the largest norm dominates the projection's initial gradients and the
ensemble starts as a noisy copy of that single encoder. The committed rung-4 recipes
therefore set `member_norm: l2` on every member: each member's feature vector is unit
L2-normalised per token before concatenation. The projection then sees members on an
equal footing, and what it learns is the mixing, not a re-scaling.

## Consequences

- Rung-4 configs under `examples/detection_benchmark/ensemble_*.yaml` carry
  `member_norm: l2` on every member; a composite recipe that omits it is not a
  ladder-fair ensemble cell.
- Per-member L2 is a load-time transform in `CompositeDenseFeatureStore`; member caches
  stay raw, so the same caches serve single-encoder cells and ensemble cells.
- The single-encoder cells are *not* L2-normalised (the base recipes are unchanged).
  A member's normalisation is part of the ensemble treatment, not of the backbone.
- Ensemble cells are an ablation: the sweep driver refuses to launch them into the
  headline sweep's `--out-root`, so the roster-driven aggregation never counts them as
  an encoder.
