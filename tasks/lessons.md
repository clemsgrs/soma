# Lessons

- Keep experiment identity separate from run identity. Content-addressed experiment grouping and immutable per-run output directories solve different problems and should not be conflated.
- Avoid adding package-level import fallbacks just to satisfy a reduced local environment when CI and supported installs include the real dependencies. Prefer keeping the public import surface strict unless optional behavior is explicitly intended.
- Distinguish the user-facing managed `output_root` from internal concrete destination directories like run, fold, feature, or tiling output paths. Replacing the former does not mean lower-level APIs should stop accepting resolved leaf directories.
