# Pooled feature caches record slide2vec's feature identity and verify it after an upgrade

A pooled feature cache records, in `cache_metadata.json`, the feature identity slide2vec wrote with its first features and the slide2vec version it was last verified with. When a cache that holds features is resolved under another slide2vec version, soma compares the recorded identity with the current one before it reuses the cache or hands missing samples to slide2vec. A difference is a hard error that names the fields; `cache.on_identity_mismatch: reextract` deletes the cache and extracts it again. This supersedes two points of ADR 0008: the "accepted standing risk", for the encoder's image transform, and "no escape-hatch config".

## Why

ADR 0008 accepted that a pixel-policy change at unchanged sizes reuses stale features, with manual deletion as the mitigation. The risk materialized (#512): slide2vec 6.2.0 changed the normalization of `lunit` and the resize and crop of `dinov2-vitb14`, the pooled keys did not move, and soma served the old features without a log line. A partly filled cache was completed with new features and scored like neither clean run.

The stamp ADR 0008 rejected as too much machinery now exists upstream. slide2vec 6.3 records a feature identity under `compatibility` in every sidecar and compares it on resume. soma reaches none of that guard: a complete hit never calls slide2vec, and a partial hit hands it only the missing samples. soma therefore applies the same comparison one level up, on the cache.

## Decisions

- **The identity is slide2vec's.** soma copies the `compatibility` block of a sidecar slide2vec just wrote, and compares through slide2vec's public `Model.pooled_identity_differences`. soma defines no identity of its own.
- **Validation, not keying.** Folding the transform into the key needs the transform before any encoder is loaded, which slide2vec cannot report yet, and it would move every pooled key once. Existing keys do not change.
- **Hits stay cheap.** The current transform needs an instantiated encoder. The comparison runs only when the recorded slide2vec version differs from the installed one; a verified cache then records the installed version. A transform that changes without a slide2vec upgrade is not detected.
- **A record is pending or recorded.** A new cache starts with a pending record (no identity yet), which its first commit completes with the `compatibility` block of the first committed sidecar. A run that stopped before its first commit leaves the record pending. The next run under the same slide2vec version keeps the cache and hands its unsigned features to slide2vec, which validates them or encodes them again; the next commit completes the record. Under another slide2vec version the cache is extracted again.
- **An absent record is a miss.** A cache that holds features and records no identity, such as one written by soma 1.17.0 or earlier, cannot be verified. soma logs a warning, deletes it and extracts it again. There is no setting to reuse it, and soma never adopts sidecar identities to approve it. A cache with no features starts a pending record.
- **No identity, no commit.** When slide2vec writes features without a `compatibility` block, the commit raises `MissingFeatureIdentity` and leaves the samples unsigned. soma checks only that the block exists; its fields are slide2vec's schema, compared through slide2vec's public API.
- **A slide or patient cache inherits from its tile cache.** soma hands slide2vec the identity the tile cache records, in the scratch tile sidecars it builds for aggregation, so slide2vec carries the tile transform into the slide embeddings. A patient embedding carries the identity of the slide embeddings it pools. Because a populated tile cache without an identity is extracted again, aggregation never propagates an unverifiable identity.
- **`reextract` deletes the whole cache directory.** Every feature in it shares the stale identity, including those of samples the current dataset does not use.

## Consequences

- soma requires slide2vec 6.3.2, which added `Model.pooled_identity_differences` for this check (clemsgrs/slide2vec#357). The comparison stays within ADR 0007: soma imports nothing from `slide2vec.runtime`. For a slide or patient cache, slide2vec compares the transform of the encoder's registered tile encoder.
- Caches written by soma 1.17.0 or earlier are extracted again on first use, even when slide2vec 6.3 wrote their sidecars with an identity. This replaces the earlier policy, which reused them with a warning (`cache.on_unrecorded_identity`). That setting is removed; configs that still set it fail with the unknown-key error.
- Changes outside the recorded identity stay undetectable: how slide2vec reads and resamples a tile (the 5.4.0 example of ADR 0008), and encoder weights or code. Deleting caches by hand remains the mitigation for those.
- Dense caches are out of scope: they record no identity in `cache_metadata.json` and a complete dense hit is not verified.
