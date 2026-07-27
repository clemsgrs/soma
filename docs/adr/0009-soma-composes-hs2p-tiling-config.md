# soma composes hs2p's TilingConfig rather than mirroring its fields

soma's preprocessing configuration **holds** an hs2p `TilingConfig` instead of re-declaring its vocabulary field by field, and adds only what is genuinely soma's (cache, masks vocabulary, previews). Geometry belongs to hs2p; mirroring it means soma is one field behind on every hs2p release, and the failure mode is silent — a knob the user cannot reach rather than an error.

## Why now

hs2p 4.3.0 added `mask_backend` and made mask decoding authoritative: the silent fallback to another reader is gone, so a mask its slide's backend cannot decode now fails. soma had no way to express `mask_backend` at all, making such a mask unfixable from a soma config. That is the drift symptom; `backend`, `tolerance`, `overlap`, `min_coverage`, `seg_downsample`, `ref_tile_size_px`, `a_t` and the rest are the same debt not yet due.

Composition also fixes the cache signature at its source: `preprocessing_signature` derives from hs2p's own dataclass fields, so it cannot fall behind the geometry it is meant to key.

## Considered and rejected

- **Mirror explicitly and add `mask_backend` now.** Typed and discoverable, but this conversation recurs on every hs2p release.
- **Mirror plus an untyped `hs2p_options` passthrough.** Rejected: options outside the schema are unvalidated and either miss the cache signature or enter it opaquely, which undermines the declarative-config premise.

## Consequences

- `preprocessing_signature` changes, so every pooled cache key changes. Landed in the same alignment that already invalidates caches (ADR 0008), so the cost is paid once.
- soma's YAML surface becomes hs2p's vocabulary; `examples/*.yaml` and the docs move with it, and hs2p's defaults become soma's user-visible defaults.
- soma's own hs2p `TilingConfig` construction for slide-manifest dense sampling stops being a separate translation.
