# Benchmarks are first-class, registered, protocol-as-code package citizens

> Superseded in package ownership and discovery by ADR 0010. The generic Benchmark
> interface remains; named providers and their assets no longer belong in the core wheel.

A published **Benchmark** is a named entry in a registry that ships **inside** the `soma` package (`soma/benchmarks/`), not a folder convention under `examples/`. Each Benchmark is a thin *code* object (protocol-as-code) implementing a standard interface — `curate` / `build_config` / `expected` / `score` / `tolerance` — **declarative where the recipe is static** (load a committed YAML) and **code where it computes** (EVA's dataset×encoder grid, its size-dependent epochs, the virchow2 CLS-only-1280-d selection) **or scores specially** (OCELOT's greedy matcher, which is not soma's headline metric).

Chosen over a hand-written per-benchmark folder template because a registry makes uniformity and discoverability *structural*: `soma list benchmarks` and `soma reproduce <name>` cannot drift from a doc someone forgot to write — which is exactly the failure that left OCELOT complete and EVA a `TBD` scaffold. The accepted cost is that benchmark definitions are Python and ship in the PyPI wheel.

## Considered and rejected

- **Protocol-as-data (pure declarative specs).** Rejected: EVA's protocol genuinely computes and OCELOT needs a custom scorer, so a static spec only ends up sprouting a `custom_scorer: path.py` escape hatch — protocol-as-code wearing a data costume.
- **Folder convention under `examples/`.** Rejected: leaves discoverability entirely to docs; nothing ties expected numbers to the leaderboard except convention.

## Consequences

- Supersedes the ad-hoc `examples/ocelot/` harness (`eval_greedy.py` → a `score` override; `expected_metrics.json` → reference data + tolerance; `reproduce.py` → the generic `soma reproduce`) and generalizes PR #87's `soma/benchmarks/eva.py` into the registry.
- Expected numbers ship as package data (`soma/benchmarks/reference/<name>.csv`) — the single source for the tolerance check, `soma leaderboard` reference rows, and the docs table.
