# Four test kinds

soma keeps four kinds of test: end-to-end scenarios through the CLI and the verbs on synthetic data, the conformance suites of `soma.testing` run on soma's own implementations, numerical oracles, and regression pins for scientifically meaningful bugs written through the public API. Every test goes through the public API. `tests/` mirrors the package's namespaces, and a fifth import-linter contract makes `tests` a root package that may import no private module.

## Why

The suite had grown larger than the package (59,180 lines of tests against 50,587 of code) and flat: 138 test files at the root of `tests/`, many of them unit tests of private functions. Those tests pinned how the code was written, not what it does, so every refactor of the soma 2.0 slices had to port them along with the code, and a private rename broke tests whose behaviour was unchanged. Some re-covered, with stubs, a path that an end-to-end scenario already ran for real. Nothing told a contributor, human or agent, where a new test belongs or what it may import.

## Decisions

- **Four kinds.** (1) End-to-end scenarios in `tests/e2e/`: one per shape, head family and aggregator sweep, plus the parity scenarios (ADR 0017). (2) Conformance suites: `check_set_source`, `check_grid_source`, `check_target_source`, `check_image_source`, `check_encoder_stage`, `check_learner` and `check_task_head` live in `soma.testing`; soma runs them on its own implementations in `tests/testing/`, and users run them on their own classes. (3) Numerical oracles: metrics, losses and cache keys against hand-computed values, cheap EVA rows, cache identity against slide2vec. (4) Regression pins for bugs that change a result, written through the public API.
- **The public API only.** A test reaches soma through a verb, the CLI, a public name of a namespace, or a conformance suite. It imports no `_`-prefixed module or name, from soma or from another test module, and does not call a private function through its module. When a behaviour cannot be reached that way, the gap is fixed in the API (as ADR 0022 does for the benchmark package). A test may still patch a private attribute to isolate a seam; that is left to review.
- **Deletion is mechanical.** A test that pins a private name is deleted, not ported: a test that imports a `_`-prefixed soma name or module (directly, or through a helper or fixture of its file) or calls a private function through its module. Where the public spelling is a one-line change, the test keeps it. A file left with no test goes. Unit tests that re-cover an end-to-end path (the `test_pipeline_*` family) are deleted. Everything else moves.
- **`tests/` mirrors `soma/`.** `tests/{data,extract,models,learn,evaluation,pipeline,presets,testing,benchmark}`, one `tests/benchmark/<protocol>` per protocol package, tests of the root modules at the root of `tests/`, plus `tests/e2e/` and `tests/repo/` (packaging, workflow, docs, layout and release checks). Every test directory is a package. `tests/repo/test_layout.py` checks the mirror.
- **Contract 5.** `[tool.importlinter]` names two root packages, `soma` and `tests`. The fifth contract forbids `tests` the private modules that contract 3 forbids the benchmark package. Like contract 3 it lists them explicitly, because wildcards cannot match `_*`, and checks direct imports only. Import-linter sees modules, not names, so a test in `tests/repo/test_import_contracts.py` refuses a private name imported from a public module.
- **Project protocols test beside themselves.** BEETLE's tests move to `examples/beetle/tests/`, outside `testpaths`; they test a project protocol (ADR 0010) and run with `python -m pytest examples/beetle/tests`.
- **The rule is written down.** `tests/AGENTS.md` states it for agents and contributors: a new test goes through a public verb or a conformance suite, and tests mirror the namespaces.

## Considered and rejected

- **Port the private tests to the public API.** Each port is a new test to design. The behaviours that matter are already covered by the end-to-end scenarios, the conformance suites and the oracles, and a port written under deadline tends to re-pin the old internals through a public door.
- **Delete every file that imports a private name.** Many files imported one private helper in one test. Deleting the whole file would have dropped many public tests for one private import, so the test, not the file, is the unit. A file goes when no test in it is left.
- **Hit a size target.** The tests are to end smaller than the package, but no good test is deleted to reach a number; the PR reports both line counts.
- **Enforce the private-name rule with import-linter alone.** It checks modules, so `from soma.x import _y` would pass. The AST check in the test suite closes that gap.

## Consequences

- A refactor that keeps the public API keeps the suite green without editing tests; a test that breaks has found a behaviour change or a public-API change.
- Some private helpers lose their direct tests. Where no end-to-end scenario, conformance suite or oracle covers what they did, the coverage is lost until a public test is written.
- A shared test helper is public: a helper module under `tests/` or a `conftest.py` fixture.
- `python -m pytest` from the repository root runs `tests/` only. BEETLE's tests run when asked for.
