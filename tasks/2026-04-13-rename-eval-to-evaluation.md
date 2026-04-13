# Rename eval to evaluation

- [x] Rename the public pipeline config field from `eval` to `evaluation`.
- [x] Update YAML loading/saving, pipeline call sites, reporting helpers, docs, and examples to use `evaluation`.
- [x] Update tests and fixtures to assert the new field/key name.
- [x] Run the targeted test suite.

Note: the docs build test still depends on `sphinx_copybutton` in this environment, so I verified the code-impacting tests separately.
