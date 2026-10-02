# Pooled feature caches record slide2vec's feature identity and verify it after an upgrade

A pooled feature cache records, in `cache_metadata.json`, the feature identity slide2vec wrote with its first features and the slide2vec version it was last verified with. When a cache that holds features is resolved under another slide2vec version, soma compares the recorded identity with the current one before it reuses the cache or hands missing samples to slide2vec. A difference is a hard error that names the fields; `cache.on_identity_mismatch: reextract` deletes the cache and extracts it again. This supersedes two points of ADR 0008: the "accepted standing risk", for the encoder's image transform, and "no escape-hatch config".

## Why

ADR 0008 accepted that a pixel-policy change at unchanged sizes reuses stale features, with manual deletion as the mitigation. The risk materialized (#512): slide2vec 6.2.0 changed the normalization of `lunit` and the resize and crop of `dinov2-vitb14`, the pooled keys did not move, and soma served the old features without a log line. A partly filled cache was completed with new features and scored like neither clean run.

The stamp ADR 0008 rejected as too much machinery now exists upstream. slide2vec 6.3 records a feature identity under `compatibility` in every sidecar and compares it on resume. soma reaches none of that guard: a complete hit never calls slide2vec, and a partial hit hands it only the missing samples. soma therefore applies the same comparison one level up, on the cache.

## Decisions

- **The identity is slide2vec's.** soma copies the `compatibility` block of a sidecar slide2vec just wrote, and compares with slide2vec's `differing_fields`. soma defines no identity of its own.
- **Validation, not keying.** Folding the transform into the key needs the transform before any encoder is loaded, which slide2vec cannot report yet, and it would move every pooled key once. Existing keys do not change.
- **Hits stay cheap.** The current transform needs an instantiated encoder. The comparison runs only when the recorded slide2vec version differs from the installed one; a verified cache then records the installed version. A transform that changes without a slide2vec upgrade is not detected.
- **A record has three states.** A new cache starts with a pending record (no identity yet), which its first commit completes. A cache with no record at all was written before this ADR and is unverifiable. The pending state is what tells the two apart after a run that stopped before its first commit. Features such a run left on disk are reused when the installed slide2vec wrote them, and dropped when another version did.
- **Unverifiable caches are not stamped.** A cache with no record is reused with one warning (`cache.on_unrecorded_identity: reextract` treats it as a miss). soma never writes the current identity onto features it did not see extracted. An identity without a transform is not recorded either.
- **A slide or patient cache inherits from its tile cache.** soma hands slide2vec the identity the tile cache records, in the scratch tile sidecars it builds for aggregation, so slide2vec carries the tile transform into the slide embeddings. A patient embedding, which soma writes itself, carries the identity of the slide embeddings it pools. A cache aggregated from an unverifiable tile cache is unverifiable too.
- **`reextract` deletes the whole cache directory.** Every feature in it shares the stale identity, including those of samples the current dataset does not use.

## Consequences

- `soma.slide2vec_adapter.pooled_identity_differences` imports `pooled_feature_identity`, `deferred_transform_record` and `differing_fields` from `slide2vec.runtime.feature_identity`, and calls `Model._declare_given_encoder_input` for pre-cropped images. slide2vec 6.3 exports none of these publicly, so this is an exception to ADR 0007, confined to that function. It should move to a public slide2vec entry point when one exists.
- Caches written by soma 1.17.0 or earlier stay unverifiable, even when slide2vec 6.3 wrote their sidecars with an identity: soma does not adopt per-sample sidecar identities it did not see written.
- Changes outside the recorded identity stay undetectable: how slide2vec reads and resamples a tile (the 5.4.0 example of ADR 0008), and encoder weights or code. Deleting caches by hand remains the mitigation for those.
- Dense caches are out of scope: they record no identity in `cache_metadata.json` and a complete dense hit is not verified.
