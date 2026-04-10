# CI Docker Build Speedup

- [x] Rework `Dockerfile.ci` so it only installs shared system and Python tooling, not the repository package itself.
- [x] Move the `"/opt/app[dev]"` install into the PR test container command in `.github/workflows/pr-test.yaml`.
- [x] Update the workflow regression test to match the new CI contract.
- [x] Refresh the CI documentation note to describe the runtime install pattern.
