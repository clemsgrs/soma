# Cache

`soma` uses a shared embedding cache so extraction can be reused across runs when the dataset and configuration match.

## Default Location

If `CacheConfig.root_dir` is not set, the cache lives next to the run directory:

```text
<output_root>/feature_cache/
```

For example, if `output_root` is `experiments`, the default cache root is `experiments/feature_cache`.

## Explicit Shared Cache

Set `root_dir` when you want every experiment to reuse the same cache location.

```python
from pathlib import Path
from soma import CacheConfig

cache = CacheConfig(root_dir=Path("shared/feature_cache"))
```

## Cache Layout

The cache is keyed by dataset manifest, preprocessing backend and geometry, encoder, and output variant.
`PreprocessingConfig.backend` is the requested backend; the actual backend chosen by
tiling is preserved on the loaded tiling result and its artifacts.

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

## What It Stores

- `cache_metadata.json` records the configuration used to build the cache.
- `cache_metadata.json` also records the requested preprocessing backend plus the
  actual backend used per sample.
- `manifest.csv` snapshots the dataset rows used for the cache key.
- `features/` contains the serialized embeddings for each sample.

## Why The Cache Exists

- It avoids recomputing embeddings for repeated runs.
- It lets slide-level runs reuse upstream tile embeddings when the encoder chain requires them.
- It keeps reusable artifacts separate from experiment-specific training outputs.
