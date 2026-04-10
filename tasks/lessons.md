# Lessons

- Keep experiment identity separate from run identity. Content-addressed experiment grouping and immutable per-run output directories solve different problems and should not be conflated.
- Avoid adding package-level import fallbacks just to satisfy a reduced local environment when CI and supported installs include the real dependencies. Prefer keeping the public import surface strict unless optional behavior is explicitly intended.
- When exposing a Soma-side worker override for slide2vec-backed embedding, name the user-facing knob `num_workers` rather than `num_dataloader_workers`.
- When bootstrapping a PPA in a minimal Ubuntu image, install the GPG tooling first. `add-apt-repository` can fail on missing `gpg-agent` even if `software-properties-common` is already present.
- For hard cutovers in soma, avoid broad legacy-compatibility shims; prefer the simplest direct contract and only the minimal explicit failure needed for the deprecated field.
- When a sibling dependency has clearly cut over and the project is aligned to that new boundary, do not add version-compatibility fallbacks unless explicitly requested.
- In soma, treat `slide2vec` as a required dependency rather than adding local test-only import fallbacks; if it is unavailable in the current environment, report that and skip affected verification.
- Distinguish the user-facing managed `output_root` from internal concrete destination directories like run, fold, feature, or tiling output paths. Replacing the former does not mean lower-level APIs should stop accepting resolved leaf directories.
