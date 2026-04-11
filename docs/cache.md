# Cache

`soma` uses two related caches during a full pipeline run:

- a shared tiling cache for preprocessing output
- a shared feature cache for extracted embeddings

The local run directory still exists, but on cache hits it mostly becomes a
small stub that points back to the canonical shared payload.

## Full Pipeline Flow

```mermaid
flowchart TD
    A[Pipeline.run()] --> B[FeatureExtractor.run()]
    B --> C[preprocess()]
    C --> D{tiling cache hit?}
    D -- yes --> E[write local tiling stub]
    D -- no --> F[run live tiling]
    F --> G[publish tiling artifacts to shared tiling_cache/]
    G --> E
    E --> H[extract()]
    H --> I{feature cache hit?}
    I -- yes --> J[write local feature stub]
    I -- no --> K[run embedding extraction]
    K --> L[publish embeddings to shared feature_cache/]
    L --> J
    J --> M[FeatureStore + training]
```

## Default Location

If `CacheConfig.root_dir` is not set, shared caches live next to the run
directory:

```text
<output_root>/tiling_cache/
<output_root>/feature_cache/
```

For example, if `output_root` is `experiments`, the default shared cache roots
are `experiments/tiling_cache` and `experiments/feature_cache`.

## Explicit Shared Cache

Set `root_dir` when you want every experiment to reuse the same shared cache
location for embeddings. The tiling cache follows the same sibling layout.

```python
from pathlib import Path
from soma import CacheConfig

cache = CacheConfig(root_dir=Path("shared/feature_cache"))
```

## Common Scenarios

### 1. Live tiling, no shared caches

If caching is disabled, `preprocess()` writes a normal local tiling directory
and `extract()` writes feature files directly into the requested feature
directory.

### 2. Tiling cache hit

If the resolved dataset and preprocessing signature match a complete shared
tiling cache entry, `preprocess()` skips live tiling and writes a local stub:

- `process_list.csv` points to the canonical shared tiling artifacts
- `README.txt` explains that the local directory is a placeholder

On a fresh run, the first cache resolution logs `✗ tiling cache miss: ... (initializing)`,
and after the payload is published it logs `✓ tiling cache populated: ...`.

### 3. Feature cache hit

If the feature cache entry is complete, `extract()` skips embedding
computation and returns a `FeatureStore` that reads from the shared cache.
The requested feature directory still gets a small manifest/README so the run
remains navigable.

On a fresh run, the first cache resolution logs `✗ feature cache miss: ... (initializing)`,
and after the payload is written back into the shared cache it logs
`✓ feature cache populated: ...`.

## Cache Layout

The feature cache is keyed by dataset manifest, preprocessing backend and
geometry, encoder, and output variant. `PreprocessingConfig.backend` is the
requested backend; the actual backend chosen by tiling is preserved on the
loaded tiling result and its artifacts.

```text
feature_cache/
├── tile/<hash>/
│   ├── cache_metadata.json
│   ├── manifest.csv
│   └── features/
│       └── <sample_id>.pt
├── slide/<hash>/
│   └── ...
└── hierarchical/<hash>/
    └── ...
```

The tiling cache uses the same general idea, but stores coordinate and
preview artifacts under a shared `tiling_cache/<hash>/` directory.

## What It Stores

- `cache_metadata.json` records the configuration used to build the cache.
- `cache_metadata.json` also records the requested preprocessing backend plus
  the actual backend used per sample.
- `cache_metadata.json` also records `empty_sample_ids` so zero-tile samples do
  not get mistaken for missing features on cache reuse.
- `manifest.csv` snapshots the dataset rows used for the cache key.
- `features/` contains the serialized embeddings for each sample.
- tiling cache entries store `process_list.csv` plus referenced tiling
  artifacts inside their cache directory.

## Why The Cache Exists

- It avoids recomputing tiling and embeddings for repeated runs.
- It lets slide-level runs reuse upstream tile embeddings when the encoder
  chain requires them.
- It keeps reusable artifacts separate from experiment-specific training
  outputs.
