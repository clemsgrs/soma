# Project protocols live outside the installable core

A Project protocol lives under `examples/<project>/` (or in an external provider
package) while its authoritative scientific contract is provisional or it has not been
explicitly accepted as a Built-in Benchmark. It may contain named dataset adapters,
recipes, results, publication policy, submission formats, and campaign orchestration
without making those surfaces part of soma's installed interface.

A mechanism belongs in `soma` when its interface and synthetic tests require no dataset
name, fixed cohort, fixed class count, known sample IDs, or publication-specific policy.
Project code may be fully tested in this repository without being distributed in the
wheel.

A project becomes eligible for promotion only once an authoritative public release is
stable enough to pin its dataset identity, cohort and splits, preprocessing, model and
selection rules, scorer, reported metric, and reference result. Eligibility is not
promotion: maintainers must separately decide to register, distribute, and support it as
a Built-in Benchmark. On promotion, its provider moves into `soma`; the Curator,
reference data, and other required protocol assets move with it under ADR 0002.

| Scientific contract | soma placement | Canonical description |
| --- | --- | --- |
| Provisional or under review | Outside installed interface | Project protocol; not yet a Benchmark |
| Published and sufficiently fixed, not promoted | Outside installed interface | Benchmark represented by a Project protocol |
| Published and sufficiently fixed, explicitly promoted | Built-in registry and wheel | Built-in Benchmark |

## Consequences

- soma keeps the Manifest contract and writer, class-aware sampling, metrics,
  task execution, cache validation, artifact publication, and generic benchmark runner.
- Concrete Curators remain deterministic Protocol-typed functions. Their structural
  interface does not decide package placement: ownership by a Project protocol or a
  Built-in Benchmark does.
- ADR 0002 remains the package and discovery decision for Built-in Benchmarks. This ADR
  does not require existing named providers to leave the wheel.
- A provider can later become a separately distributed package without changing soma's
  core interfaces.

## Placement rule

Before promotion, deleting `examples/<project>/` must remove that project's names,
cohort constants, submission rules, and paper decisions. It must not remove soma's
ability to curate a Manifest, validate a cache, train a model, compute sufficient
statistics, or publish and recover run artifacts.
