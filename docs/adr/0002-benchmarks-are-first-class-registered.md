# Built-in Benchmarks are first-class, registered, protocol-as-code package citizens

A **Built-in Benchmark** is a named entry in a registry that ships **inside** the `soma` package (`soma/benchmarks/`), not a folder convention under `examples/`. Each Built-in Benchmark is a thin *code* object (protocol-as-code) implementing a standard interface — `curate` / `build_config` / `expected` / `score` / `tolerance` — **declarative where the recipe is static** (load a committed YAML) and **code where it computes** (EVA's dataset×encoder grid, its size-dependent epochs, the virchow2 CLS-only-1280-d selection) **or scores specially** (OCELOT's greedy matcher, which is not soma's headline metric).

ADR 0010 distinguishes a Project protocol from a Built-in Benchmark. A published, sufficiently fixed evaluation contract is eligible for promotion, but package inclusion is a separate maintainer decision.

For Built-in Benchmarks, this is chosen over a hand-written folder template because a registry makes uniformity and discoverability *structural*: `soma list benchmarks` and `soma reproduce <name>` cannot drift from a doc someone forgot to write — which is exactly the failure that left OCELOT complete and EVA a `TBD` scaffold. Project protocols remain outside this registry under ADR 0010. The accepted cost is that Built-in Benchmark definitions are Python and ship in the PyPI wheel.

## Considered and rejected

- **Protocol-as-data (pure declarative specs).** Rejected: EVA's protocol genuinely computes and OCELOT needs a custom scorer, so a static spec only ends up sprouting a `custom_scorer: path.py` escape hatch — protocol-as-code wearing a data costume.
- **Folder convention under `examples/` for Built-in Benchmarks.** Rejected: leaves discoverability entirely to docs; nothing ties expected numbers to the leaderboard except convention. This does not reject `examples/<project>/` for Project protocols.

## Consequences

- Supersedes the ad-hoc `examples/ocelot/` harness (`eval_greedy.py` → a `score` override; `expected_metrics.json` → reference data + tolerance; `reproduce.py` → the generic `soma reproduce`) and generalizes PR #87's `soma/benchmarks/eva.py` into the registry.
- Expected numbers ship as package data (`soma/benchmarks/reference/<name>.csv`) — the single source for the tolerance check, `soma leaderboard` reference rows, and the docs table.
