# Tests

These rules hold for every test under `tests/` (design §10, ADR 0023).

## A new test goes through a public verb or a conformance suite

- Write the test through what a user can call: a verb (`soma.extract`, `soma.fit`,
  `soma.predict`, `soma.evaluate`, `soma.save` / `soma.load`, `soma.Pipeline`,
  `soma.resolve`), the CLI, a public class or function of a namespace, or a conformance
  suite of `soma.testing` (`check_set_source`, `check_grid_source`, `check_target_source`,
  `check_image_source`, `check_encoder_stage`, `check_learner`, `check_task_head`).
- Never import a `_`-prefixed module or name, from soma or from another test module.
  Import-linter contract 5 refuses private modules, and
  `tests/repo/test_import_contracts.py` refuses private names. If a behaviour cannot be
  reached through the public API, that is a gap in the API: fix it there.
- Do not call a `_`-prefixed function through its module (`module._helper()`) either. A
  private attribute that a test only patches, to isolate a seam, is left to review.
- A helper that several test files share is public: put it in a helper module (`tests/*.py`
  beside `conftest.py`, or the namespace directory) or in a `conftest.py` fixture. Never
  put a test helper under `soma/`, except a conformance suite users can run too.

## Pick one of the four kinds

1. **End-to-end scenario** (`tests/e2e/`): the CLI or the verbs on synthetic data, with
   checked artifacts. See `tests/e2e/README.md`.
2. **Conformance suite** (`tests/testing/`): a `soma.testing` check run on soma's own
   implementations, and the check's refusal of a broken one.
3. **Numerical oracle**: a metric, a loss or a cache key against a hand-computed value.
4. **Regression pin**: a scientifically meaningful bug, pinned through the public API.

A unit test that re-covers what an end-to-end scenario already runs is not added.

## Tests mirror `soma/`

- A test lives in the directory of the namespace it exercises: `tests/data`,
  `tests/extract`, `tests/models`, `tests/learn`, `tests/evaluation`, `tests/pipeline`,
  `tests/presets`, `tests/testing`, `tests/benchmark` (the generic mechanism) and
  `tests/benchmark/<protocol>`. A test of a root module (`soma.config`, `soma.plan`,
  `soma.registry`, `soma.discovery`, `soma.cli`) sits at the root of `tests/`.
- `tests/e2e/` holds the scenarios; `tests/repo/` holds packaging, workflow, docs, layout
  and release checks; `tests/fixtures/` holds data files.
- Every test directory is a package (`__init__.py`). `tests/repo/test_layout.py` checks the
  mirror.
- A project protocol's tests live beside it and stay out of the suite: BEETLE's run with
  `python -m pytest examples/beetle/tests`.
