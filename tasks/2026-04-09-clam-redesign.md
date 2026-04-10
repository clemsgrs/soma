# CLAM Redesign

- [x] Replace `clam` with explicit `clam_sb` and `clam_mb` aggregators.
- [x] Rework CLAM internals to follow the original `../CLAM` semantics.
- [x] Add branch-aware support for `CLAM_MB` in the MIL/task-head boundary.
- [x] Add CLAM-specific weighted task/instance loss mixing in training.
- [x] Update tests, docs, and examples to use `clam_sb` / `clam_mb`.
- [x] Run targeted CLAM, trainer, and pipeline-adjacent tests.
