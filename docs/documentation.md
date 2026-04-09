# Documentation

## Output Root Design

`soma` manages experiment outputs from a single user-provided `output_root`.

The output system is built around two distinct identities:

- `experiment`: the scientific configuration being evaluated
- `run`: one concrete execution of that experiment

This keeps repeated executions grouped together without overwriting prior results.

## Experiment Identity

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

## Run Identity

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

## Directory Layout

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

## Source Of Truth

Per-experiment and per-run metadata files are the source of truth.

- `experiment.yaml` and `experiment.json` define the experiment spec
- `run.yaml` defines one concrete execution

The root-level CSV files are convenience indexes only:

- `indexes/experiments.csv`
- `indexes/runs.csv`

If an index becomes stale, it should be rebuildable from experiment and run metadata.

## Index Updates

The pipeline updates indexes eagerly during execution.

This keeps the indexes useful without requiring a separate maintenance step, while still allowing a future rebuild utility if needed.

Failed runs should remain visible in `runs.csv` with an explicit `status` column so they can be filtered out easily.

## Latest Pointer

Each experiment directory should expose a `latest` pointer.

The pointer should resolve to:

- the most recent successful run when one exists
- otherwise the most recent run regardless of status

## API

User-facing pipeline configuration takes an `output_root` and lets `soma` resolve:

- the experiment directory
- the run directory
- the feature cache root when not explicitly overridden

The final resolved run directory becomes an internal detail recorded in run metadata and returned in pipeline results.

## Canonical Hashing Rules

The experiment hash should be derived from a canonical JSON representation with:

- sorted keys
- normalized path strings
- stable serialization
- explicit omission of run-only fields

Checksums should be computed from the contents of `dataset_csv` and `splits_csv`, while resolved paths should also be preserved for traceability.

## Recommended Metadata Files

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

## Implementation Priorities

The implementation follows this order:

1. Add dataclasses for experiment spec and run metadata.
2. Add canonical serialization and hashing helpers.
3. Add output-root path resolution and slug generation.
4. Update pipeline orchestration to create run directories and metadata eagerly.
5. Update indexes as runs start and finish.
6. Update docs and examples to use `output_root`.
