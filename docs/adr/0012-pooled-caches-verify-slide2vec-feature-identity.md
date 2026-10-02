# Pooled feature caches record slide2vec's feature identity and verify it after an upgrade

A pooled feature cache records, in `cache_metadata.json`, the feature identity slide2vec wrote with its first features and the slide2vec version. When a cache that holds features is resolved under another slide2vec version, soma compares the recorded identity with the current one before it reuses the cache or hands missing samples to slide2vec. A difference is a hard error that names the fields; `cache.on_identity_mismatch: reextract` deletes the cache and extracts it again. This supersedes the "accepted standing risk" of ADR 0008 for the encoder's image transform.

## Why

ADR 0008 accepted that a pixel-policy change at unchanged sizes reuses stale features, with manual deletion as the mitigation. The risk materialized (#512): slide2vec 6.2.0 changed the normalization of `lunit` and the resize and crop of `dinov2-vitb14`, the pooled keys did not move, and soma served the old features without a log line. A partly filled cache was completed with new features and scored like neither clean run.

The stamp ADR 0008 rejected as too much machinery now exists upstream. slide2vec 6.3 records a feature identity under `compatibility` in every sidecar and compares it on resume. soma reaches none of that guard: a complete hit never calls slide2vec, and a partial hit hands it only the missing samples. soma therefore applies the same comparison one level up, on the cache.

## Decisions

- **The identity is slide2vec's.** soma copies the `compatibility` block of a sidecar slide2vec just wrote, and compares with slide2vec's `differing_fields`. soma defines no identity of its own. For slide and patient caches, soma passes the identity of the cached tiles through the scratch sidecars it builds, so slide2vec carries the tile transform into the slide embeddings it aggregates.
- **Validation, not keying.** Folding the transform into the key needs the transform before any encoder is loaded, which slide2vec cannot report yet, and it would move every pooled key once. Existing keys do not change.
- **Hits stay cheap.** The current transform needs an instantiated encoder. The comparison runs only when the recorded slide2vec version differs from the installed one; a verified cache then records the installed version. A transform that changes without a slide2vec upgrade is not detected.
- **Unverifiable caches are not stamped.** A cache with no record is reused with one warning (`cache.on_unrecorded_identity: reextract` treats it as a miss). soma never writes the current identity onto features it did not see extracted.
- **`reextract` deletes the whole cache directory.** Every feature in it shares the stale identity, including those of samples the current dataset does not use.

## Consequences

- `soma/slide2vec_adapter.py` imports `pooled_feature_identity`, `deferred_transform_record` and `differing_fields` from `slide2vec.runtime.feature_identity`, and calls `Model._declare_given_encoder_input` for pre-cropped images. slide2vec 6.3 exports none of these publicly, so this is an exception to ADR 0007, confined to one function (`pooled_identity_differences`). It should move to a public slide2vec entry point when one exists.
- Changes outside the recorded identity stay undetectable: how slide2vec reads and resamples a tile (the 5.4.0 example of ADR 0008), and encoder weights or code. Deleting caches by hand remains the mitigation for those.
- Dense caches are out of scope: they record no identity in `cache_metadata.json` and a complete dense hit is not verified.
