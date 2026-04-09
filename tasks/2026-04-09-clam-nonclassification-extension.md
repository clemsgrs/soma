# CLAM Non-Classification Extension

- [x] Add task-family-aware CLAM_SB auxiliary modes for classification, ordinal classification, and regression.
- [x] Keep `clam_mb` classification-only and fail early for incompatible task heads.
- [x] Add explicit task-family metadata to task heads and wire aggregator/task compatibility through `MILModel`.
- [x] Reject classification-only CLAM options for non-classification tasks.
- [x] Reject multi-target regression for `clam_sb` auxiliary supervision.
- [x] Update CLAM tests for ordinal/regression auxiliary behavior and compatibility checks.
- [x] Update docs to describe task-aware `clam_sb` behavior and `clam_mb` limits.
- [x] Run environment-appropriate verification and capture constraints.
