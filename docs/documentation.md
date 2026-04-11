# soma — Documentation

## Documentation Notes

- The README now points to the repository's actual AGPL-3.0 license.
- The README now highlights the layered API so users can extract features, train from a feature store, or run the full pipeline.
- `debug-pipeline.py` now seeds a temp-local cuFile config before importing `slide2vec`/`hs2p`, so `cufile.log` no longer falls back into the repository cwd during debug runs.
- The supported encoder reference in `docs/reference.md` now uses a registry-style table with tile and slide presets, output dimensions, spacing, and variant notes.
- The cache docs now describe the full tiling-and-feature cache flow, including run-local stubs and shared cache directories.
- Cache resolution now emits compact rich-friendly hit/miss messages for tiling and feature caches, e.g. `✓ tiling cache hit:` and `✓ feature cache hit:` / `✗ feature cache miss:` with the status word colored when the active reporter supports color.
- Run-local extraction now keeps `tiling/` as a sibling of `features/` when using the `FeatureExtractor.run()` convenience path.
- Evaluation metrics now import `sklearn.metrics` directly and no longer carry local fallback implementations for missing `sklearn` installs.
- The output-layout module no longer carries an unused path-normalization helper.
- Core modules now import `slide2vec`, `hs2p`, `yaml`, and `FeatureExtractor` directly at module scope instead of deferring those imports behind compatibility shims or function-local indirection.
- HIPT runtime validation in `soma` now checks the encoder against resolved preprocessing tile geometry before extraction.
- Evaluation config (`EvalConfig`) is now separate from `TaskConfig`; `EvalConfig` holds `metrics` and `subgroups` (a `SubgroupConfig` with `columns: list[str]`).
- Reports include a **Subgroup Analysis** section when `eval.subgroups.columns` is configured. Per-subgroup metrics are computed on all-fold concatenated predictions; cells are highlighted by significance tier (see below).
- Statistical testing uses permutation tests (group-vs-rest for subgroups; paired sign-permutation on per-fold values for cross-run). All p-values are corrected for multiple comparisons using **Benjamini-Hochberg FDR** correction: within each (column, metric) family for subgroups, and globally across all (metric, run) comparisons for cross-run reports.
- Highlight tiers in subgroup tables (p-values are BH-adjusted): `subgroup-sig` — p_adj < 0.05 and Δ ≥ 10%; `subgroup-flag` — Δ ≥ 10% but not significant; `subgroup-sig-small` — p_adj < 0.05 but small effect.
- Cross-run comparison tables mark significantly worse runs (`sig-worse`, red) when p_adj < 0.05; the best run is marked `best-val` (green). Stats are omitted when any run has fewer than 2 folds or fold counts differ across runs.

## Output Root Design

`soma` manages experiment outputs from a single user-provided `output_root`.

The output system is built around two distinct identities:

- `experiment`: the scientific configuration being evaluated
- `run`: one concrete execution of that experiment

This keeps repeated executions grouped together without overwriting prior results.

### Experiment Identity

An experiment is identified by a canonical experiment spec and a deterministic hash.

The experiment hash should include:

- dataset CSV checksum
- splits CSV checksum
- resolved dataset CSV path
- resolved splits CSV path
- preprocessing config
- encoder config
- aggregator config
- task config
- training config, except run-only fields such as random seed

The experiment hash should not include:

- output paths
- timestamps
- W&B identifiers
- git SHA
- runtime hardware details
- worker counts or similar execution-only settings

The experiment directory name should remain human-readable:

```text
<dataset>-<encoder>-<aggregator>-<task_type>_<short_hash>
```

The slug is for browsing. The hash remains the canonical identifier.

### Run Identity

A run is one execution of an experiment at a specific time.

Run ids should use:

```text
YYYY-MM-DD_HH-MM-SS__<wandb_id-or-localid>
```

Each run should record its own metadata, including:

- run id
- experiment id
- status
- start and finish timestamps
- random seed
- W&B id and URL when available
- git SHA
- whether the worktree was dirty
- hostname
- username
- resolved output path
- summary metrics
- failure metadata when applicable

Run directories should be immutable after completion.

### Directory Layout

The managed layout should look like this:

```text
<output_root>/
├── experiments/
│   └── <dataset>-<encoder>-<aggregator>-<task_type>_<short_hash>/
│       ├── experiment.yaml
│       ├── experiment.json
│       ├── runs/
│       │   ├── 2026-04-09_16-22-10__abc123/
│       │   │   ├── config.yaml
│       │   │   ├── run.yaml
│       │   │   ├── summary.json
│       │   │   ├── fold_0/
│       │   │   └── ...
│       │   └── 2026-04-11_09-07-43__local/
│       └── latest
├── indexes/
│   ├── experiments.csv
│   └── runs.csv
└── feature_cache/
```

`feature_cache/` should remain separate from run directories. It stores reusable extraction artifacts and already follows a content-addressed model.

### Source Of Truth

Per-experiment and per-run metadata files are the source of truth.

- `experiment.yaml` and `experiment.json` define the experiment spec
- `run.yaml` defines one concrete execution

The root-level CSV files are convenience indexes only:

- `indexes/experiments.csv`
- `indexes/runs.csv`

If an index becomes stale, it should be rebuildable from experiment and run metadata.

### Index Updates

The pipeline updates indexes eagerly during execution.

This keeps the indexes useful without requiring a separate maintenance step, while still allowing a future rebuild utility if needed.

Failed runs should remain visible in `runs.csv` with an explicit `status` column so they can be filtered out easily.

### Latest Pointer

Each experiment directory should expose a `latest` pointer.

The pointer should resolve to:

- the most recent successful run when one exists
- otherwise the most recent run regardless of status

### API

User-facing pipeline configuration takes an `output_root` and lets `soma` resolve:

- the experiment directory
- the run directory
- the feature cache root when not explicitly overridden

The final resolved run directory becomes an internal detail recorded in run metadata and returned in pipeline results.

### Canonical Hashing Rules

The experiment hash should be derived from a canonical JSON representation with:

- sorted keys
- normalized path strings
- stable serialization
- explicit omission of run-only fields

Checksums should be computed from the contents of `dataset_csv` and `splits_csv`, while resolved paths should also be preserved for traceability.

### Recommended Metadata Files

`experiment.yaml` should contain:

- experiment id
- slug
- hash
- canonical experiment spec
- dataset and splits checksums
- resolved dataset and splits paths
- creation timestamp

`run.yaml` should contain:

- run id
- experiment id
- status
- started at
- finished at
- seed
- W&B metadata
- git metadata
- machine metadata
- resolved run directory
- summary metrics
- error details when failed

### Implementation Priorities

The implementation follows this order:

1. Add dataclasses for experiment spec and run metadata.
2. Add canonical serialization and hashing helpers.
3. Add output-root path resolution and slug generation.
4. Update pipeline orchestration to create run directories and metadata eagerly.
5. Update indexes as runs start and finish.
6. Update docs and examples to use `output_root`.

## Dataset Format

soma uses two CSV files to define experiments:

**dataset.csv** — one row per sample (slide):
| Column | Required | Description |
|--------|----------|-------------|
| `sample_id` | Yes | Unique slide identifier |
| `image_path` | Yes | Path to WSI file |
| `label` | Yes | Prediction target (string or int) |
| `mask_path` | No | Path to a pre-computed tissue mask used during tiling when present |

Extra columns are preserved as metadata on each `SampleRecord`.

**splits.csv** — user-provided train/tune/test assignments:
| Column | Description |
|--------|-------------|
| `fold` | Fold index (0, 1, 2, ...) |
| `sample_id` | Join key to dataset.csv |
| `split` | One of `train`, `tune`, `test` |

Labels are auto-encoded to integers via `Dataset.label_map` (sorted unique labels). `num_classes` is inferred automatically.

---

## Pipeline API (Two Layers)

### Layer 1 — Standalone Step APIs

Modular building blocks for agent sweeps and custom workflows:

```python
from soma import Dataset, Splits, FeatureStore, train, train_one_fold
from soma import EncoderConfig, AggregatorConfig, TaskConfig, TrainingConfig

dataset = Dataset("dataset.csv")
splits = Splits("splits.csv", dataset)

# Extract features once per encoder
store = FeatureStore("features/uni2")

# Train all folds at once
result = train(
    feature_store=store,
    dataset=dataset,
    splits=splits,
    aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 256}),
    task=TaskConfig(name="binary_classification"),
    training=TrainingConfig(learning_rate=1e-4, epochs=50),
    output_dir="experiments/run1",
)

# Or train a single fold for fine-grained control
fold_result = train_one_fold(
    feature_store=store,
    dataset=dataset,
    fold_split=splits.folds[0],
    aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 256}),
    task=TaskConfig(name="binary_classification"),
    training=TrainingConfig(learning_rate=1e-4, epochs=50),
    output_dir="experiments/run1/fold_0",
)
```

**Agent sweep** — extract once, sweep aggregators:
```python
for agg_name in ["abmil", "clam_sb", "clam_mb", "dsmil", "transmil", "dtfdmil", "hipt", "mean_pool"]:
    result = train(store, dataset, splits,
                   aggregator=AggregatorConfig(name=agg_name),
                   task=TaskConfig(), training=TrainingConfig(),
                   output_dir=f"experiments/{agg_name}")
```

### Layer 2 — Pipeline Orchestrator

Convenience wrapper that chains everything:

```python
from soma import Pipeline, PipelineConfig, EncoderConfig, AggregatorConfig, TrainingConfig

config = PipelineConfig(
    dataset_csv="dataset.csv",
    splits_csv="splits.csv",
    output_dir="experiments/run1",
    encoder=EncoderConfig(name="uni2", batch_size=64),
    aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 256}),
    training=TrainingConfig(learning_rate=1e-4, epochs=50),
)
pipeline = Pipeline(config, feature_dir="features/uni2")
results = pipeline.run()
```

### Output Directory Structure

```
run_dir/
    config.yaml                  # full PipelineConfig snapshot
    run.yaml                     # run metadata (status, timestamps, git SHA, seed)
    summary.json                 # aggregated metrics (mean ± std across folds)
    report.html                  # self-contained HTML experiment report
    fold_0/
        best_model.pt            # checkpoint
        metrics.json             # tune + test metrics
        predictions.csv          # per-sample predictions
        training_history.json    # epoch-by-epoch loss, metrics, and LR
    fold_1/
        ...
```

### Shared Feature Cache

When `PipelineConfig.cache.enabled=True`, `soma` uses a shared canonical feature cache instead of treating each run-local `features/` directory as the only persisted source.

Default behavior:
- cache root defaults to a stable sibling of the run directory, typically `<output_root>/feature_cache/`
- tile features are cached under `tile/<cache_key>/features/`
- hierarchical features are cached under `hierarchical/<cache_key>/features/`
- slide features are cached under `slide/<cache_key>/features/`
- run-local extraction without cache writes canonical `slide2vec` artifact roots (`tile_embeddings/` and optionally `slide_embeddings/`)
- cache-payload writes, reads, and tile-artifact reconstruction for cache reuse are handled by functions in `soma.cache`
- cache resolution emits a single hit/miss line for tiling and feature caches through the active rich progress reporter, or stdout when no reporter is active
- `FeatureStore` can read soma cache payload dirs or native `slide2vec` artifact roots, including `hierarchical_embeddings/`

This enables reuse such as:
- first run `encoder="prism"` populates both a shared Virchow tile cache and a PRISM slide cache
- later run `encoder="virchow"` with matching preprocessing and precision reuses the cached tile features instead of re-embedding tiles

---

## Preprocessing Pipeline

Generic tiling is now owned by `slide2vec`, which in turn delegates to `hs2p`.

1. `FeatureExtractor.preprocess(...)` resolves encoder-driven spacing and tile-size defaults in soma.
2. soma converts each dataset row into a `slide2vec`/`hs2p` slide spec with `sample_id`, `image_path`, optional `mask_path`, and optional `spacing_at_level_0`.
3. `slide2vec.Pipeline(...).run(..., tiling_only=True)` performs generic tiling and writes canonical `hs2p` artifacts plus `process_list.csv`.
4. soma later reloads those tiling artifacts for extraction, cache population, and provenance validation.

soma keeps a small compatibility shim at this boundary so it can read tiling manifests from slide2vec releases that export either `load_tiling_process_df(...)` or the older `load_process_df(...)` helper name.
The same adapter also tolerates the hs2p provenance-validator rename from `tissue_mask_path` to `mask_path` so tiling reload works across recent sibling package changes.
At extraction runtime, soma passes `output_variant` overrides only for tile encoders. Slide encoders such as `prism` keep their fixed output contract and are instantiated without a runtime override, even though cache metadata may still record the resolved fixed variant.

The active contract is the `slide2vec`/`hs2p` one: `process_list.csv` plus per-slide coordinate artifacts. soma no longer owns a separate generic tiling implementation.

Edge-tile handling is owned by `hs2p`/`slide2vec`. Tile generation keeps contour-local edge coordinates, and downstream slide readers white-pad out-of-bounds crop regions so encoded tiles keep their target size. Tissue fractions for padded edge tiles are normalized over the valid in-slide area rather than the padded border.

Tiling artifacts now separate two concerns more cleanly. `TilingResult` (from `hs2p`) carries explicit geometry and provenance fields such as `sample_id`, `image_path`, `backend`, `requested_backend`, `base_spacing_um`, `slide_dimensions`, `level_downsamples`, `overlap`, `min_tissue_fraction`, `step_px_lv0`, `tissue_method`, `seg_downsample`, `seg_level`, `seg_spacing_um`, `ref_tile_size_px`, and `a_t`. The saved `.meta.json` still mirrors that information into nested `provenance`, `slide`, `tiling`, `segmentation`, and `artifact` sections for readability, but loaders hydrate the explicit `TilingResult` attributes rather than returning a loose metadata payload. The loader is intentionally strict: required fields must be present and unexpected top-level or nested keys are rejected.

In particular, `base_spacing_um` is persisted as the level-0 spacing contract and reused by slide-level coordinate preparation. It is intentionally distinct from `effective_spacing_um`, which only describes the selected read level.

`PreprocessingConfig.target_tile_size_px`, `target_spacing_um`, and `ref_tile_size_px` are optional. When omitted, `FeatureExtractor` resolves them from the selected encoder metadata: tile encoders use their own `input_size` and `supported_spacing_um`, while slide encoders inherit both values from their declared `tile_encoder`. `ref_tile_size_px` defaults to the resolved `target_tile_size_px`. For hierarchical runs, `region_tile_multiple` and `target_region_size_px` are resolved alongside the base tile size, and the effective geometry is kept on the same config object via the `effective_*` fields. Multi-spacing encoders still require an explicit spacing choice, and the cutover path fails fast when that spacing is ambiguous.

---

## Encoder Module

Generic encoding is now delegated to `slide2vec`.

- **`Encoder` ABC** — shared lifecycle/device contract
- **`TileEncoder`** — `get_transform()`, `encode_tiles()`
- **`SlideEncoder`** — `encode_slide(tile_features, coordinates, tile_size_lv0=...)`
- **`TimmTileEncoder`** — Base for timm-backed tile encoders
- **Registry** — `encoder_registry` with metadata including explicit `level`, `input_size`, `tile_encoder`, `tile_encoder_output_variant`, `output_variants`, `supported_spacing_um`, and `precision`
- `slide2vec` owns the generic WSI reader stack, supertiles, adaptive batching, and tile/slide embedding runtime.
- soma keeps the experiment-facing config layer, cache keys, feature-store adapters, and `FeatureExtractor` which calls slide2vec `Model`/`Pipeline` directly.
- Non-cache tile extraction writes canonical `slide2vec` `tile_embeddings/`.
- Non-cache hierarchical extraction writes canonical `slide2vec` `hierarchical_embeddings/`.
- Non-cache slide extraction writes canonical `slide2vec` `slide_embeddings/` and can optionally persist `tile_embeddings/` as well.

For slide2vec-backed embedding, `EncoderConfig.num_workers` is an opt-in override for DataLoader workers. Leave it as `None` to keep slide2vec's automatic worker-resolution path.

Tile-level QC helper functions (`filter_whitespace`, `filter_grayspace`) now accept an optional `valid_mask` so padded border pixels can be excluded from white/gray fraction calculations when edge-tile QC is performed.

Tile-level feature files remain plain tensors, but rows are normalized before save so row `i` matches the same `tile_index[i]` / coordinate row from the tiling artifact.

Tile encoders can expose multiple named output variants. Users select them with `EncoderConfig.output_variant`, and caches are variant-aware. Single-output encoders still use the same metadata path with a single `"default"` variant. Slide encoders keep one fixed output and, when they depend on a tile encoder, they also hardcode the required tile-output variant.

`EncoderConfig.name` is explicit now: callers must choose the encoder themselves rather than inheriting a default model name. `EncoderConfig.precision` is optional. When left unset, soma resolves it from encoder metadata and falls back to `fp32` if the encoder provides no recommendation; explicit user overrides are still allowed and mismatches only warn when a recommendation exists. Validation still requires explicit encoder `level` metadata, and slide encoders must also declare both `tile_encoder` and `tile_encoder_output_variant`.

Slide-level models declare their tile-encoder dependency in registry metadata. For example:
- `prism` → `tile_encoder="virchow"`
- `titan` → `tile_encoder="conchv15"`
- `gigapath-slide` → `tile_encoder="gigapath"`

This means `Pipeline(config).run()` can auto-resolve preprocessing geometry from the tile encoder, run the required tile encoder, pool to a single slide embedding, and then train directly with `aggregator=None`.

Encoder spacing validation is defined against the target spacing, not the selected read level's `effective_spacing_um`. When tiling falls back to a finer natural spacing and reads a larger crop that is resized down, that larger read crop remains valid as long as the target spacing still matches the model contract. Slide-level coordinate preparation follows the same rule and normalizes against `target_spacing_um`.

For slide-level coordinate preparation, `base_spacing_um` always means the level-0 slide spacing. Both direct and cached slide extraction now read it from the explicit `TilingResult.base_spacing_um` field; they do not substitute `effective_spacing_um`.

### Distributed Extraction

The shared distributed path is now `slide2vec`’s execution engine for generic extraction. When `FeatureExtractor.extract(..., num_gpus > 1)` is used on the generic path, soma delegates to `slide2vec.Pipeline.run_with_coordinates(<tiling_dir>, ...)` so existing tiling artifacts are reused through a public shared helper and torchrun orchestration stays in one place. soma then adapts the outputs back into either:
- native `slide2vec` artifact roots for non-cache runs
- flat soma cache payload dirs for reusable cache entries

Hierarchical/HIPT-style extraction uses the same runtime boundary and can run on single-GPU or multi-GPU paths, with the feature store selecting `hierarchical_embeddings/` when present.

### CI Coverage

The pull-request workflow now builds a reusable `Dockerfile.ci` base image and installs the repository package inside the test container at runtime. This keeps source changes from invalidating the expensive image build layers while preserving the same in-container test flow. The PRISM step still runs only for same-repo PRs or non-PR dispatches and still requires `HF_TOKEN`.

### Release Publishing

Published GitHub releases now trigger two separate release workflows:
- `.github/workflows/release.yaml` builds the Python distribution, validates it with `twine check`, and uploads it to PyPI.
- `.github/workflows/docker.yaml` builds the repository Docker image and pushes both the release-tagged image and `latest` to Docker Hub.

The repository now also includes `release.py`, a small helper that:
- bumps the version in `pyproject.toml`
- creates a `release-<version>` branch
- commits the version bump
- pushes the branch and matching Git tag
- optionally opens a PR and the GitHub release draft page

Required GitHub repository secrets for release publishing:
- `PYPI_API_TOKEN`
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

### Supported Models (18)

| Name | level | dim | input/tile | spacing | Base |
|---|---|---|---|---|---|
| uni | tile | 1024 | 224 | 0.5 | TimmTileEncoder |
| uni2 | tile | 1536 | 224 | 0.5 | TimmTileEncoder |
| virchow | tile | 1280 / 2560 | 224 | 0.5 | TimmTileEncoder |
| virchow2 | tile | 1280 / 2560 | 224 | 0.5–2.0 | TimmTileEncoder |
| conch | tile | 512 | 448 | 0.5 | TileEncoder |
| conchv15 | tile | 768 | 448 | 0.5 | TileEncoder |
| gigapath | tile | 1536 | 256 | 0.5 | TimmTileEncoder |
| h-optimus-0 | tile | 1536 | 224 | 0.5 | TimmTileEncoder |
| h-optimus-1 | tile | 1536 | 224 | 0.5 | TimmTileEncoder |
| h0-mini | tile | 768 / 1536 | 224 | 0.5 | TimmTileEncoder |
| phikon | tile | 768 | 224 | 0.5 | TileEncoder |
| phikonv2 | tile | 1024 | 224 | 0.5 | TileEncoder |
| hibou-b | tile | 768 | 224 | 0.5 | TileEncoder |
| hibou-l | tile | 1024 | 224 | 0.5 | TileEncoder |
| midnight | tile | 3072 | 224 | 0.25–2.0 | TileEncoder |
| prism | slide | 1280 | 224 | 0.5 | SlideEncoder |
| titan | slide | 768 | 512 | 0.5 | SlideEncoder |
| gigapath-slide | slide | 768 | 256 | 0.5 | SlideEncoder |

---

## Aggregator Module

MIL bag-level aggregation: tile features `(B, N, D)` → slide-level representation.

- **`Aggregator` ABC** — `forward(X, mask) → AggregatorOutput` with `output_dim`
- **`AggregatorOutput`** — `bag_representation: (B, D_out)` or branch-aware `(B, C, D_out)` + optional `tile_attention` + optional `auxiliary: dict`
- **`MeanPool`** / **`MaxPool`** — Simple pooling, no attention
- **`ABMIL`** — Attention-Based MIL (Ilse et al., 2018), gated attention pooling
- **`CLAM_SB`** — Reference-style single-branch CLAM (Lu et al., 2021), gated attention + weighted task-aware auxiliary loss; supports binary/multiclass classification, ordinal classification, and single-target regression
- **`CLAM_MB`** — Reference-style multi-branch CLAM with class-wise attention branches; multiclass classification only
- **`DSMIL`** — Dual-Stream MIL (Li et al., 2021), critical-instance attention via query-key matching
- **`TransMIL`** — Transformer-based MIL (Shao et al., 2021), Nyströmformer layers + PPEG positional encoding, `output_dim = att_dim`
- **`DTFDMIL`** — Double-Tier Feature Distillation MIL (Zhang et al., 2022), pseudo-bag partitioning + Grad-CAM distillation
- **`HIPT`** — Hierarchical Image Pyramid Transformer (Chen et al., 2022), two-level hierarchy: region ViT (`VisionTransformer4K`) aggregates `P = npatch²` tile features per region, then a global transformer + gated attention pools `M` region embeddings into a slide representation. HIPT accepts either flat `(B, N, D)` or native hierarchical `(B, M, P, D)` inputs. `input_dim` is auto-resolved from `FeatureStore`, while `region_size` and `patch_size` are derived from resolved preprocessing.

Canonical `aggregator.name` values:
- `mean_pool`
- `max_pool`
- `abmil`
- `clam_sb`
- `clam_mb`
- `dsmil`
- `transmil`
- `dtfdmil`
- `hipt`

Tile attention flows through `MILModel` for heatmap generation. Aggregators with auxiliary training losses (CLAM, DSMIL, DTFD-MIL) return extra tensors via `AggregatorOutput.auxiliary`. CLAM uses original-style `bag_weight * task_loss + (1 - bag_weight) * instance_loss`, while other aggregators keep the default `task_loss + aux_loss` combination. `clam_mb` emits branch-aware bag representations and is consumed by `BranchAwareClassificationHead` (task name `branch_aware_classification`). When task name is `multiclass_classification`, the pipeline selects this head automatically.

For `clam_sb`, the auxiliary loss follows the paired task family:
- `binary_classification` / `multiclass_classification`: original CLAM instance clustering
- `ordinal_classification`: top-k scalar instance regression toward the bag label plus low-attention regularization
- `regression`: top-k scalar instance regression toward the bag target plus low-attention regularization

`clam_sb` exposes three CLAM-specific knobs for these task-aware auxiliary paths:
- `instance_loss_mode` to explicitly pin the auxiliary mode when it matches the task
- `low_attention_weight` to weight the low-attention regularization term
- `topk_target_weight` to weight the top-k target-matching term

`use_negative_class_instance_loss` remains classification-only, and `clam_sb` currently supports only single-target regression.

### Hierarchical (HIPT) Preprocessing

When using HIPT, the Pipeline derives preprocessing from a multiple-first contract:
- `tile_multiple` becomes the user-facing hierarchical knob
- `patch_size` is the resolved tile size, `region_size = patch_size × tile_multiple`
- `target_tile_size_px`, `target_region_size_px`, `effective_tile_size_px`, and `effective_region_size_px` stay on the preprocessing config
- soma no longer expands coordinates locally; it consumes native `hierarchical_embeddings/` artifacts from `slide2vec`
- HIPT can consume native `(B, M, P, D)` batches directly, while still accepting flat `(B, N, D)` for compatibility

Minimal YAML config:
```yaml
aggregator:
  name: hipt
  params:
    tile_multiple: 16
encoder:
  name: uni2
```

---

## Task Heads

- **`TaskHead` ABC** — `forward(X) → logits`, `compute_loss(logits, targets) → scalar`
- **`ClassificationHead(input_dim, num_classes)`** — Linear + cross-entropy

---

## Training Module

- **`MILModel`** — Composes aggregator + task head
- **`BagDataset(records, feature_store, label_map)`** — Maps samples to `(features, label, sample_id)`
- **`HierarchicalBagDataset(records, feature_store, label_map)`** — Maps samples to `(features, label, sample_id)` with rank-3 hierarchical tensors
- **`SlideDataset(records, feature_store, label_map)`** — For pre-pooled slide-level features `(D,)`
- **`bag_collate_fn`** — Pads variable-length bags, constructs boolean masks → `BagBatch`
- **`hierarchical_bag_collate_fn`** — Pads the region axis of hierarchical bags, constructs region masks → `HierarchicalBagBatch`
- **`slide_collate_fn`** — Stacks `(D,)` features into `SlideBatch`
- **`SlideModel`** — Applies a task head directly to slide-level features
- **`Trainer`** — Pure PyTorch loop with early stopping, best checkpoint saving, optimizer/scheduler factories
- **`seed_everything(seed)`** — Deterministic seeding

---

## Packaging and Release

PyPI packaging is defined in `pyproject.toml` and built with Hatchling.

Typical release flow:

1. Bump the version in `pyproject.toml`.
2. Build artifacts with `./.venv/bin/python -m build --sdist --wheel --no-isolation`.
3. Verify metadata with `./.venv/bin/python -m twine check dist/*`.
4. Upload the final artifacts with `./.venv/bin/python -m twine upload dist/*`.

The wheel is explicitly limited to the `soma` package so the published distribution stays focused on the runtime library.

---

## Evaluation Module

- **`compute_metrics(task_family, metrics, y_true, y_pred, y_prob)`** → `dict[str, float]` — dispatcher for all task families
- **`resolve_metrics(task_family, metrics)`** — returns effective metric list (defaults when empty, validates names)
- **`VALID_METRICS`** / **`DEFAULT_METRICS`** — catalogues of valid and default metric names per task family
- **`EvaluationReport`** — Split name, metrics dict, per-sample predictions
- **`SamplePrediction`** — sample_id, true/predicted labels, probabilities

---

## Reporting Module

`soma.reporting` generates self-contained HTML experiment reports from completed run directories.

### Training history

Each fold now persists epoch-by-epoch training data to `fold_N/training_history.json`:

```json
[
  {"epoch": 0, "train_loss": 0.69, "tune_loss": 0.71, "tune_metrics": {"auroc": 0.55}, "lr": 1e-4},
  ...
]
```

This is written automatically by `train_one_fold()` alongside `metrics.json` and `predictions.csv`.

### Report generation

Reports are generated automatically at the end of every `Pipeline.run()` and saved to `run_dir/report.html`. They can also be generated on demand from any completed run directory:

```python
from soma.reporting import generate_report, generate_report_from_result

# From a saved run directory
path = generate_report("/path/to/run_dir")

# From an in-memory PipelineResult
path = generate_report_from_result(result, config)
```

### Report contents

The HTML report is fully self-contained (Plotly JS embedded, no network access required) and includes:

- **Run header** — run ID, status badge, seed, timestamps, git SHA
- **Configuration** — encoder, aggregator, task, training hyperparameters
- **Test results table** — per-fold metrics and mean ± std across folds
- **Training curves** — train/tune loss, one tune metric curve per user-requested metric, learning rate schedule; all folds overlaid on the same chart
- **Prediction analysis** — task-aware:
  - *Binary classification*: ROC curve (per-fold + pooled), PR curve (per-fold + pooled), confusion matrix (aggregated), score distributions
  - *Multiclass classification*: macro-averaged ROC curve, confusion matrix, score distributions
  - *Ordinal classification*: confusion matrix, score distributions
  - *Regression*: predicted vs. actual scatter, residuals plot
- **Subgroup analysis** — when `eval.subgroups.columns` is configured, shows per-group metric tables and bar charts with deviation highlighting. An optional statistical testing panel (permutation test) can be enabled with `eval.subgroups.statistical_testing = True`.

### Cross-run comparison

`compare_runs` generates a side-by-side comparison report for any list of completed run directories. It automatically detects which config fields differ across runs and highlights only those, collapsing the shared configuration.

```python
from soma.reporting import compare_runs

path = compare_runs([
    "output/experiments/exp_abc/runs/run1",
    "output/experiments/exp_def/runs/run2",
])
```

Labels are auto-derived from the config diff: if runs differ in exactly one field (e.g., `aggregator.name`), that field's value is used as the label. Otherwise the run ID is used. Custom labels can be passed via the `labels` argument.

The comparison report includes:

- **Configuration differences** — only the fields that vary across runs, as a column-per-run table; shared configuration shown in a collapsible block
- **Metrics comparison** — one row per metric, one column per run (mean ± std), best value per metric highlighted
- **Training curves** — loss and per-metric curves overlaid across runs; multi-fold runs show a ±1 std shaded band around the mean

### Public API

| Symbol | Description |
|---|---|
| `generate_report(run_dir, *, output_path=None)` | Generate single-run report from disk |
| `generate_report_from_result(result, config, *, output_path=None)` | Generate single-run report from in-memory result |
| `compare_runs(run_dirs, *, output_path=None, labels=None)` | Generate cross-run comparison report |
| `load_run_data(run_dir)` | Load all run artifacts into a `RunData` object |
| `load_comparison_data(run_dirs, *, labels=None)` | Load multiple runs into a `ComparisonData` object |
| `run_data_from_result(result, config)` | Convert in-memory result to `RunData` |
| `render_report(run_data)` | Render `RunData` to an HTML string |
| `render_comparison_report(comparison_data)` | Render `ComparisonData` to an HTML string |
| `RunData` | Dataclass holding config, metadata, summary, and per-fold data |
| `FoldData` | Dataclass holding one fold's history, metrics, and predictions |
| `ComparisonData` | Dataclass holding multiple runs with config diffs and labels |

---

## Configuration

All configs are frozen dataclasses with YAML serialization:

| Config | Purpose |
|--------|---------|
| `PreprocessingConfig` | Tissue segmentation + tiling parameters |
| `CacheConfig` | Shared feature-cache policy and root |
| `EncoderConfig` | Required foundation model name, optional precision override, batch_size, optional `num_workers` override |
| `AggregatorConfig` | Aggregator name + params dict |
| `TaskConfig` | Task head name + params dict (model architecture only) |
| `EvalConfig` | Evaluation metrics + subgroup analysis settings |
| `SubgroupConfig` | Categorical columns for subgroup analysis, optional statistical testing flag |
| `TrainingConfig` | Epochs, LR, optimizer, scheduler, patience |
| `PipelineConfig` | Bundles all above + dataset/splits/output paths |

`TaskConfig` and `EvalConfig` are separate: `task` defines what is predicted (model head architecture), `eval` defines how it is measured (metrics, subgroup breakdowns). Metric validation (`resolve_metrics`) runs in `PipelineConfig.__post_init__` where both are in scope.

Example:
```python
from soma.config import TaskConfig, EvalConfig, SubgroupConfig

task = TaskConfig(name="binary_classification")
eval = EvalConfig(
    metrics=["auroc", "f1"],
    subgroups=SubgroupConfig(columns=["sex", "grade"], statistical_testing=False),
)
```

```python
from soma.config import save_config, load_config
save_config(config, "config.yaml")
config = load_config("config.yaml")
```

---

## Module Reference

| Module | Purpose |
|---|---|
| `soma.dataset` | `Dataset`, `SampleRecord`, `Splits`, `FoldSplit` |
| `soma.features` | `FeatureStore` |
| `soma.config` | All config dataclasses, YAML serialization |
| `soma.pipeline` | `Pipeline`, `train`, `train_one_fold`, `FoldResult`, `PipelineResult` |
| `soma.preprocessing.*` | Preview rendering (tissue segmentation, tiling, etc. delegated to `hs2p`) |
| `soma.encoders.*` | Encoder config validation (encoders and registry delegated to `slide2vec`) |
| `soma.aggregators.*` | Aggregator ABC, MeanPool, MaxPool, ABMIL, CLAM_SB, CLAM_MB, DSMIL, TransMIL, DTFDMIL, HIPT |
| `soma.tasks.*` | TaskHead ABC, ClassificationHead |
| `soma.training.*` | MILModel, BagDataset, Trainer, collation, seeding |
| `soma.evaluation.*` | Metrics, EvaluationReport, SamplePrediction |
| `soma.reporting` | `generate_report`, `generate_report_from_result`, `compare_runs`, `RunData`, `FoldData`, `ComparisonData` |

## Active Design Decisions

### Tiling cache

The tiling cache is intentionally narrow and separate from the existing feature
cache.

The shared tiling cache is the canonical storage location for cached tiling
artifacts. Run-local tiling directories are lightweight stubs that contain a
`README.txt` plus a `process_list.csv` pointing at the shared cache paths.

Feature and tiling cache resolution now share the same basic structure: a
small shared resolution base plus explicit validation results so completeness
checks are organized consistently across both cache families.

When `FeatureExtractor` is used directly, passing `output_root` aligns both
`feature_cache/` and `tiling_cache/` under that root when `CacheConfig.root_dir`
is not set.

Lower-level extraction APIs now prefer concrete destination names like
`feature_dir` and `tiling_dir`, while the pipeline keeps `output_root` as the
user-facing managed root.

Standalone training APIs follow the same pattern: `train_one_fold()` writes to
`fold_dir`, `train()` writes to `run_dir`, and `PipelineResult.run_dir`
exposes the resolved managed run path.

For `backend="auto"`, cache reuse validates against the current runtime's
actual resolved backend by probing `hs2p.wsi.resolve_backend(...)` per sample
before accepting a cache hit.
