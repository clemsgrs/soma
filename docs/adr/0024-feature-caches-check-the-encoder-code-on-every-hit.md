# Feature caches check the encoder's code on every hit, and dense caches record their feature identity

Every feature cache that holds features compares the encoder's code it records with the code the encoder registry holds now, each time it is resolved. A difference follows `cache.on_identity_mismatch`, as a feature identity mismatch does. Dense caches record slide2vec's feature identity like pooled caches, and get the same two checks: the code on every hit, and the whole identity after a slide2vec upgrade. A record that lacks a field the current identity requires cannot be verified, and follows the same setting. This supersedes three points of ADR 0012: "Hits stay cheap", encoder code in the list of undetectable changes, and "Dense caches are out of scope".

## Why

A cache key does not cover the encoder's code. After an edit to an encoder, a run reused the features of the old code, while its `run.yaml` recorded the new fingerprint (ADR 0012 compared the identity only after a slide2vec upgrade, and the component record of #648 sees the edit). The docs told users to delete the cache by hand. A dense cache recorded no identity at all, so neither an edit nor an upgrade was noticed.

slide2vec 7.1.3 records the encoder's code in every sidecar (`encoder_code`: the class, a fingerprint and the registry source; `tile_encoder_code` for a slide or patient encoder) and returns the current block from the registry with `slide2vec.encoders.encoder_code(name)`, without building the encoder. The fingerprint is the SHA-256 of the file that defines the class, or the version of the installed plugin package. Comparing it costs one file hash, so it can run on every hit. slide2vec 7.1.4 adds `Model.dense_identity_differences`, the dense counterpart of `Model.pooled_identity_differences`.

There was a second problem. With slide2vec 7.1.3, every existing pooled cache failed its upgrade check once: slide2vec reported `encoder_code` as `MISSING_FIELD`, soma counted it as a difference, and the error said slide2vec "now extracts different features". The features had not changed; the record predated the field.

## Decisions

- **Check, do not key.** Cache keys do not change. Each time a cache that holds features is resolved, for a complete or a partial hit, soma compares the recorded `encoder_code` with `encoder_code(name)`, and, for a slide or patient cache, the recorded `tile_encoder_code` with the code of the registered tile encoder. A difference follows `cache.on_identity_mismatch`. `error` (the default) raises `CacheFeatureIdentityMismatch`, naming the encoder, the class, the recorded and current fingerprints, and the remedies; the message does not blame a slide2vec upgrade. `reextract` deletes the cache and extracts it again.
- **Every encoder is checked**, built-in ones too. A built-in encoder's fingerprint is the hash of its file in slide2vec, so an edit in an editable slide2vec checkout is caught.
- **A missing field is not a difference.** A record that lacks a field the current identity requires (`slide2vec.MISSING_FIELD`), such as `encoder_code` in a cache written before 7.1.3, cannot be verified. soma follows `on_identity_mismatch`, with a message that says the cache predates the field. This holds on both checks, and for any field slide2vec adds later.
- **Dense caches get the pooled record and checks.** The `dense` and `dense_image` caches record `feature_identity` in `cache_metadata.json`, with the states of ADR 0012: pending when created, completed by the first commit with the sidecar's `compatibility` block, no commit without a block, and an extraction interrupted under another slide2vec version extracted again. After an upgrade, soma compares the recorded identity through `Model.dense_identity_differences`, with the `DenseOptions` or `DenseImageOptions` and the execution it hands to the embed method; slide2vec compares the request-wide fields only, so the read fields of the one sample the record holds are ignored. `composite` needs nothing: each member has its own dense cache.
- **An existing dense cache is never deleted silently.** A dense cache with grids and no record follows `on_identity_mismatch`: by default an error that names the cache and says it predates the record, and `reextract` to opt in. Dense grids are costly to extract again, so the user decides. An empty dense cache starts a pending record. The pooled rule for an absent record (a miss, with a warning) does not change.
- **No fingerprint, one warning.** An encoder whose class has no file, as in a notebook cell, records `fingerprint: null`. Its cache is reused, with one warning per encoder and process that soma cannot detect changes to its code. The component record of #648 does the same.
- **The docs put the encoder in a file of its own.** The fingerprint covers the file that defines the class, so a class defined in the run script ties the cache to the whole script. The examples split the encoder into `my_encoder.py`, imported by the run script.

## Considered and rejected

- **Fold the code fingerprint into the cache key.** An edit would then start a new cache beside the old one instead of failing: no error, and stale caches pile up unseen. It would also move every existing key once.
- **Delete dense caches without a record, as pooled ones.** A dense cache can hold hundreds of gigabytes. Deleting it on upgrade without being asked is the surprise ADR 0008 avoided.
- **Count a missing field as a difference.** It produced the false "now extracts different features" message on every pooled cache after slide2vec 7.1.3.

## Consequences

- soma requires slide2vec 7.1.4. It imports only slide2vec's public API (ADR 0007): `slide2vec.encoders.encoder_code`, `slide2vec.MISSING_FIELD`, `Model.pooled_identity_differences` and `Model.dense_identity_differences`.
- Every existing pooled cache records no `encoder_code`, and every existing dense cache records no identity. Each one stops the first run that resolves it, with an error that says it predates the field or the record; `cache.on_identity_mismatch: reextract` extracts it again.
- A cache hit hashes the encoder's file, and the tile encoder's file for a slide or patient cache; it still loads no encoder.
- What stays undetected: the encoder's weights, code the encoder class inherits or imports from other files, and a transform that changes without a slide2vec upgrade or an edit to the encoder's file (for example after an upgrade of `timm`). Deleting the cache by hand remains the mitigation for those.
