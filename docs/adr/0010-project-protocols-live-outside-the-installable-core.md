# Project protocols live outside the installable core

The installable `soma` package owns reusable execution mechanisms and stable data
interfaces. Named dataset adapters, benchmark recipes, reference/result tables,
publication policies, submission formats, and campaign orchestration live under
`examples/<project>/` (or in an external provider package) and call those interfaces.

A mechanism belongs in `soma` when its interface and synthetic tests require no dataset
name, fixed cohort, fixed class count, known sample IDs, or publication-specific policy.
Project code may be fully tested in this repository without being distributed in the
wheel.

## Consequences

- soma keeps the Manifest contract and writer, class-aware sampling, metrics,
  task execution, cache validation, artifact publication, and generic benchmark runner.
- Concrete Curators remain deterministic Protocol-typed functions, but live with the
  project whose raw layout they understand. This clarifies ADR 0004: sharing the output
  contract does not imply package ownership.
- The generic Benchmark code-object interface remains, but concrete providers and their
  assets are supplied explicitly by project code. This supersedes ADR 0002's decision to
  bundle named providers, reference data, and results in `soma` and register them through
  import side effects.
- Existing named providers under `soma/` are migration debt. New project work must use
  the external-provider boundary; legacy providers move in staged, compatibility-aware
  changes rather than as part of an unrelated experiment.
- A provider can later become a separately distributed package without changing soma's
  core interfaces.

## Placement rule

Deleting `examples/<project>/` must remove that project's names, cohort constants,
submission rules, and paper decisions. It must not remove soma's ability to curate a
Manifest, validate a cache, train a model, compute sufficient statistics, or publish and
recover run artifacts.
