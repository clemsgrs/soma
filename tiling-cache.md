# Tiling Cache Plan

## Goal

Add a small shared tiling cache that reuses tiling artifacts across runs when the dataset and the fully resolved preprocessing geometry are identical.

## Flow Summary

```mermaid
flowchart TD
    A[FeatureExtractor.preprocess()] --> B{tiling cache hit?}
    B -- yes --> C[write local tiling/ stub]
    B -- no --> D[run live tiling]
    D --> E[publish canonical tiling payload to shared tiling_cache/]
    E --> C
    C --> F[FeatureExtractor.extract()]
    F --> G{feature cache hit?}
    G -- yes --> H[reuse shared features]
    G -- no --> I[compute embeddings and populate feature_cache/]
```

This cache is intentionally narrow:

- It caches only tiling outputs.
- It does not introduce a generic stage-cache framework.
- It preserves the existing feature cache design.

## Motivation

Today, feature-cache hits still depend on a concrete tiling directory:

- `FeatureExtractor.run()` always calls `preprocess()` first.
- `FeatureExtractor.extract()` loads tilings before feature-cache resolution.
- The shared cache stores only feature payloads, not tiling artifacts.

That means repeated runs can reuse embeddings, but they still need tiling metadata and often still rerun tiling.

The user-facing cache docs now show this as two separate layers:

- run-local `tiling/` is the entrypoint for loading tilings
- shared `tiling_cache/` and `feature_cache/` hold the canonical payloads
- local run directories may become lightweight stubs when a shared hit is available

With the tiling cache enabled, run-local `skip_existing=True` should no longer
mean "a `process_list.csv` exists". It should mean the local tiling directory is
already a validated stub for the exact shared tiling-cache entry required by the
current resolved preprocessing configuration.

## Scope

### In scope

- Shared reuse of successful tiling artifacts across similar runs.
- Strict cache identity based on dataset manifest plus resolved preprocessing values.
- Strict validation on load, including actual resolved backend provenance.
- Minimal integration into `preprocess()` and existing extraction flow.

### Out of scope

- Generic cache abstractions for arbitrary pipeline stages.
- Unifying tiling and feature cache into one API.
- Partial recomputation or repair beyond the current per-sample completeness checks.
- Relaxed cache matching policies.

## Proposed Layout

Use a sibling shared directory next to `feature_cache/`:

```text
tiling_cache/
└── <hash>/
    ├── cache_metadata.json
    ├── manifest.csv
    ├── process_list.csv
    └── artifacts/
        └── ... coordinate/meta files referenced by process_list.csv
```

`process_list.csv` remains the canonical entrypoint because current loading
already depends on it.

The shared cache entry should be self-contained and canonical. Its
`process_list.csv` should resolve only to files inside that cache entry's
`artifacts/` subtree.

## Cache Identity

The tiling cache key should be based on:

- dataset manifest digest
- resolved preprocessing signature
- schema version

The key should not include encoder name directly. Different encoders should share the tiling cache if they resolve to the same tiling geometry and segmentation settings.

### Dataset identity

Use the existing dataset manifest digest inputs:

- `sample_id`
- `image_path`
- `mask_path`

This keeps tiling reuse strict with respect to input slide and mask files.

### Resolved preprocessing identity

The key must use resolved preprocessing values, not raw user config. In particular:

- `backend` as requested
- `requested_tile_size_px`
- `requested_spacing_um`
- `ref_tile_size_px`
- `requested_region_size_px`
- `region_tile_multiple`
- `read_tile_size_px`
- `read_region_size_px`
- `tissue_method`
- `tissue_threshold`
- `overlap`
- `seg_downsample`
- `tolerance`
- `a_t`
- `tissue_mask_tissue_value`
- `hierarchical`
- `npatch`
- `hierarchical_patch_size_px`

This matters because some fields are encoder-derived after resolution, for example:

- `requested_tile_size_px`
- `requested_spacing_um`
- `ref_tile_size_px`
- hierarchical region geometry

Two runs should share a tiling cache only if these resolved values are identical.

## Metadata Requirements

`cache_metadata.json` should include:

- `cache_kind: "tiling"`
- `cache_key`
- `schema_version`
- `sample_ids`
- dataset manifest digest
- resolved preprocessing signature
- requested backend
- actual backend summary
- actual backend by sample id

Also record debugging provenance that is not part of the key:

- the encoder name used when preprocessing was resolved
- requested preprocessing config before resolution

Those fields help explain why two configs did or did not collide without making the cache less reusable.

## Strict Validation

Validation should be strict both when determining cache completeness and when loading cached tilings.

### Per-sample requirements

For every dataset sample expected by the cache:

- a row must exist in cached `process_list.csv`
- `tiling_status` must be `success`
- all referenced artifact files must exist
- the row must deserialize into a valid tiling result
- `validate_tiling_result_provenance(...)` must pass against the current dataset row

### Backend validation

Do not validate only the requested backend. Validate the actual resolved backend too.

The cache metadata should record:

- `requested_backend`
- `backend` when all samples share one actual backend
- `backend_by_sample_id`

Reuse is allowed only when:

- the current run's actual resolved backend mapping matches the cached
  `backend_by_sample_id`
- the actual backend provenance loaded from the cached tiling artifacts matches
  the metadata

If the runtime produces mixed actual backends across samples, that exact per-sample mapping should be preserved and validated.

For `backend="auto"`, this requires a lightweight probe before tiling. Use
`hs2p.wsi.resolve_backend(...)` against each current dataset row to determine the
actual backend that the present runtime would select, then compare that mapping
against the cached metadata before accepting a hit.

For explicit backends such as `"openslide"` or `"cucim"`, the effective backend
mapping is simply that explicit value for each sample.

### Resolved geometry validation

On a cache hit, validate that the cached tiling results match the current resolved preprocessing values, including at least:

- requested tile size
- requested spacing
- region size when hierarchical
- region multiple when hierarchical
- tissue-mask provenance

If any resolved value differs, treat it as a miss, not a soft warning.

## Integration Plan

### 1. Add tiling-cache resolution utilities

Add a narrow set of functions in `soma.cache` or a small `soma.tiling_cache` module:

- `build_tiling_cache_key(...)`
- `resolve_tiling_cache(...)`
- `write_tiling_cache_payload(...)`
- `write_tiling_cache_stub(...)` for the run-local cache-backed `tiling_dir`
- `probe_resolved_backends(...)` for current-runtime backend validation

Keep this separate from feature-cache helpers unless code reuse is obvious and local.

### 2. Resolve preprocessing before tiling-cache lookup

`FeatureExtractor.preprocess()` should:

- resolve preprocessing first
- compute a tiling-cache key from the resolved preprocessing
- probe current resolved backends for the dataset
- check the shared tiling cache before invoking `Pipeline(...).run(..., tiling_only=True)`

This ensures encoder-derived defaults are already reflected in the cache identity.

When tiling caching is enabled, plain local file existence should not be
authoritative. A stale run-local tiling directory must not suppress cache lookup
or regeneration.

### 3. Reuse cached tilings on hit

On a complete hit, prefer one of these approaches:

1. Materialize a run-local `tiling_dir` from the shared cache via hardlinks or copies.
2. Or treat the shared cache dir itself as the tiling dir.
3. Or write a run-local cache-backed stub with paths pointing into the shared cache.

Recommendation: write a run-local cache-backed stub.

Reason:

- It keeps the rest of the code unchanged.
- `extract()` and `load_tilings()` continue reading a normal tiling directory.
- Run-local paths remain predictable for debugging.
- It avoids accidental mutation of shared cache contents by downstream code.
- It avoids copying or linking large coordinate artifacts when simple path
  indirection is enough.
- It mirrors the current feature-cache UX: a local placeholder plus canonical
  shared storage.

The run-local `tiling_dir` should contain:

- `README.txt` explaining that the canonical payload lives in the shared tiling cache
- `process_list.csv` with absolute paths to the shared tiling cache artifacts

No backward-compatibility constraint requires materializing the full tiling
artifacts into the run-local directory.

### 4. Populate cache on miss

On a miss:

- run the existing tiling pipeline exactly as today
- validate the produced tiling artifacts
- atomically write them into the shared tiling cache
- then write the run-local stub that points at the cached payload

The shared tiling cache entry should be canonical and self-contained. Its
`process_list.csv` should reference only files within that cache entry, not
paths inside the temporary live-tiling output directory.

### 5. Keep extraction unchanged where possible

`extract()` should continue to consume `tiling_dir` through `load_tilings()`.

The design goal is to make `preprocess()` produce a normal tiling directory regardless of whether the source was live tiling or cached tiling reuse.

## File Handling Strategy

Prefer publishing canonical files into the shared cache by plain copy. The
run-local stub should then point at those canonical paths directly.

Do not rely on metadata-preserving copy modes for shared HPC filesystems.

If an internal implementation step still needs temporary staging, follow the
same conservative hardlink-or-copy fallback pattern already used by feature
cache materialization.

## Failure Policy

When validation fails:

- log why the cached tilings were rejected
- treat it as a cache miss
- regenerate tilings

Do not attempt in-place repair of a shared tiling cache entry in the first version.

If cache population fails partway through:

- write into a temp directory first
- publish via atomic rename once complete

## Tests

Add targeted tests for:

- identical resolved preprocessing across different encoders reuses the same tiling cache key
- differing resolved tile size or spacing causes a miss
- differing requested backend causes a miss
- differing actual backend provenance causes a miss
- current-runtime backend probing for `backend="auto"` rejects hits when the
  resolved backend mapping differs
- cached tilings can be surfaced through a run-local stub `tiling_dir`
- `preprocess()` skips live tiling on a complete cache hit
- a stale local `process_list.csv` does not suppress cache lookup or regeneration
- multi-GPU extraction does not mutate shared cached tiling metadata or the
  canonical shared `process_list.csv`
- incomplete or corrupted cache entries are ignored and regenerated
- hierarchical resolved geometry participates in the key and validation

## Implementation Order

- [ ] Add tiling-cache metadata and key helpers.
- [ ] Add strict completeness and provenance validation for cached tilings.
- [ ] Add cache population from a successful live tiling run.
- [ ] Add run-local cache-stub writing for cache hits.
- [ ] Wire `FeatureExtractor.preprocess()` to resolve and reuse tiling cache entries.
- [ ] Add regression tests for hit, miss, and corruption paths.
- [ ] Document the new tiling-cache behavior in user-facing docs if the implementation lands.

## Design Constraints

- Prefer strict correctness over opportunistic reuse.
- Prefer resolved values over raw config values in cache identity.
- Prefer preserving the current extraction API over introducing new cache abstractions.
- Prefer a narrow implementation that can be deleted or replaced later without touching the rest of the pipeline.
