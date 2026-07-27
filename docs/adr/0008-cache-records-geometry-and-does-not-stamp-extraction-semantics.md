# The feature cache records geometry facts and accepts that pixel-policy changes are undetectable

soma's feature cache records the geometry triple — requested tile size, per-slide read tile size, and **effective encoder input size** — in `cache_metadata.json`, and validates the encoder input size on reuse. It deliberately carries **no extraction-semantics stamp**, which means an upstream change to *how* pixels are produced at unchanged sizes cannot be detected, and the mitigation is deleting caches by hand on a slide2vec upgrade.

## The gap, stated precisely

slide2vec 5.4.0 (`#226`) made every pooled reader area-resize the raw pyramid read to `requested_tile_size_px` before the model transform. For HEST — 224 px at 0.5 µm on a slide with 0.25 µm level-0 spacing — the read is 448 px in both versions:

- 5.3.0: read 448 → shipped transform's `Resize(224, bicubic)` → encoder sees 224
- 5.4.0: read 448 → **area**-resize to 224 → shipped transform (now a near no-op) → encoder sees 224

The recorded triple is `(224, 448, 224)` in both. Different pixels, identical sizes. The interpolation kernel changed and the resize moved stage; neither is a size, so no geometry record can see it. Encoder-recipe changes (the GigaPath transform and MUSK `out_norm` corrections) are equally invisible, being photometric rather than geometric.

What the triple *does* catch is the regime shift: a 512 px declared request on a variable-input encoder records `encoder_input=224` under the shipped transform and `encoder_input=512` under normalization-only. That is why the encoder input size is the one member of the triple worth validating — soma derives its expected value from config + registry without loading the model.

## Considered and rejected

- **An `EXTRACTION_CONTRACT_VERSION` published by slide2vec and folded into soma's cache key or validator.** It closes the gap exactly, preserves caches across releases that change nothing, and is one constant. Rejected as machinery not worth its keep for a single-maintainer, single-user package: the same person releases slide2vec and runs soma, and can delete caches deliberately.
- **Per-encoder `recipe_version` in the registry, with a CI fingerprint test.** Rejected as over-engineering, for the same reason.
- **Recording `slide2vec.__version__` and erroring on any minor/major difference.** Rejected: it invalidates a 433 GB dense cache on every minor bump regardless of whether pixels moved.
- **Contract-versioned cache *keys* rather than validation.** Keys would let both generations coexist for side-by-side comparison; validation cannot. Rejected together with the stamp — existing caches are being invalidated this round anyway, and no back-compat is wanted.

## Consequences

- Ordinary incompleteness keeps returning `CacheValidationResult(complete=False, reason=…)` and recomputes; an encoder-input-size mismatch is a **distinct hard error**, because silently recomputing a 400 GB feature set is the surprise worth preventing.
- No stamping command, no escape-hatch config, no migration path. Existing caches are invalidated by the alignment and re-extracted.
- Accepted standing risk: a future slide2vec change to pixel policy at unchanged sizes will silently reuse stale features. Deleting caches on upgrade is the mitigation, and it is manual.
