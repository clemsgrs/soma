# Lessons

- Keep experiment identity separate from run identity. Content-addressed experiment grouping and immutable per-run output directories solve different problems and should not be conflated.
- Avoid adding package-level import fallbacks just to satisfy a reduced local environment when CI and supported installs include the real dependencies. Prefer keeping the public import surface strict unless optional behavior is explicitly intended.
- When exposing a Soma-side worker override for slide2vec-backed embedding, name the user-facing knob `num_workers` rather than `num_dataloader_workers`.
- When bootstrapping a PPA in a minimal Ubuntu image, install the GPG tooling first. `add-apt-repository` can fail on missing `gpg-agent` even if `software-properties-common` is already present.
- For hard cutovers in soma, avoid broad legacy-compatibility shims; prefer the simplest direct contract and only the minimal explicit failure needed for the deprecated field.
- When a sibling dependency has clearly cut over and the project is aligned to that new boundary, do not add version-compatibility fallbacks unless explicitly requested.
- In soma, treat `slide2vec` as a required dependency rather than adding local test-only import fallbacks; if it is unavailable in the current environment, report that and skip affected verification.
- Distinguish the user-facing managed `output_root` from internal concrete destination directories like run, fold, feature, or tiling output paths. Replacing the former does not mean lower-level APIs should stop accepting resolved leaf directories.
- For compact rich logs, keep the visible status phrase exact and color only the status word or symbol; assert against ANSI-stripped text in tests instead of raw escape sequences.
- When debugging a verified runtime bug, do not keep speculative import-structure refactors that are not required by the final repro/fix path; revert or clearly separate them from the minimal validated fix.
- For cache logging, key the user-facing label off the cache type that callers understand (`tiling cache` vs `feature cache`), not the internal directory segment used to store entries.
- When a field is only used as an internal cache selector, name it after that selector (`cache_kind`) instead of overloading a generic public name like `kind`.
- If a sample legitimately produces zero tiles, persist that fact in cache metadata and skip it during cache validation instead of treating the missing `.pt` as a cache miss.
- When a tolerance gate says the resolved spacing is acceptable, treat the actual read geometry as the source of truth for downstream level-0 footprint fields; update both the producer and the consumer together when that contract changes.
- When validating a cross-repo schema rename, put all affected checkouts on `PYTHONPATH` before running tests so you do not accidentally exercise an older installed copy of a sibling package.
- When a cache-population helper has a distributed refresh step, pass the resolved preprocessing and backend provenance explicitly instead of closing over values from a different branch.
- When removing the implicit MIL default, make `PipelineConfig.aggregator` default to `None` for slide-level runs; do not hide the change behind a default `AggregatorConfig.name` value.
- When GitHub Actions reports that it cannot resolve a major version for an action, verify the actual released tags on the upstream repository instead of assuming a floating `vX` tag exists; pin the exact published tag if that is the only available ref.
