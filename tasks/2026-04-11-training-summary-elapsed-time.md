# Training Summary Elapsed Time

- [x] Add elapsed time to the live training summary panel.
- [x] Keep the formatting compact and readable.
- [x] Add regression coverage for the rendered elapsed-time row.
- [x] Update `docs/documentation.md` with the training UX note.
- [x] Run targeted trainer tests and confirm the output still renders cleanly.

# Training Summary Timing Detail

- [x] Add average epoch time to the live training summary panel.
- [x] Add ETA to the live training summary panel.
- [x] Add regression coverage for the new timing rows.
- [x] Update `docs/documentation.md` with the expanded training UX note.
- [x] Run targeted trainer tests and confirm the output still renders cleanly.

# Saved Timing Artifacts

- [x] Persist timing values into `fold_N/training_history.json`.
- [x] Surface elapsed and average epoch time in the HTML report.
- [x] Keep ETA out of the HTML report.
- [x] Add regression coverage for timing persistence and report rendering.
- [x] Update `docs/documentation.md` with the artifact/report behavior.

# ETA Refinement

- [x] Update ETA only after each completed train+tune epoch.
- [x] Remove ETA from persisted training artifacts.
- [x] Keep elapsed and average epoch time in saved history.
- [x] Update `docs/documentation.md` to match the live-only ETA behavior.
- [x] Run focused tests and confirm the JSON schema no longer includes ETA.

# Trainable Parameters

- [x] Add trainable parameter count to the live training progress panel.
- [x] Add regression coverage for the new parameter row.
- [x] Update `docs/documentation.md` with the parameter count note.
- [x] Run focused trainer tests and confirm the panel renders correctly.

# Timing Layout

- [x] Put ETA inline with elapsed time in the live training panel.
- [x] Move elapsed and average epoch time below status in the live training panel.
- [x] Add regression coverage for the new timing row order.
- [x] Update `docs/documentation.md` with the timing layout note.
- [x] Run focused trainer tests and confirm the panel renders correctly.
