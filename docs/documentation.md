# Documentation Notes

- The landing page now describes `soma` as a modular framework to streamline
  computational pathology research.
- The landing page now keeps the main body uncluttered and moves the GitHub
  repository link plus release status into a compact sidebar widget.
- The README and Sphinx home page now share the same unified API framing and
  emphasize reproducible end-to-end results alongside composable building
  blocks.
- The landing page now briefly mentions the upstream `hs2p` and `slide2vec`
  projects that power preprocessing and encoding.
- The docs use line-block formatting in schema-heavy tables and definition
  lists where explicit line breaks improve readability.
- The landing page now briefly names the tile, slide, and patient workflows
  and points readers to the pipeline page for the concrete execution paths.
- The pipeline page now describes the shared run loop first and then explains
  the tile, slide, and patient paths in a way that matches what each level
  actually executes.
- The pipeline page now includes a minimal slide-level `PipelineConfig`
  example so readers can see the expected Python setup directly on the page.
- The pipeline page now uses a simple bullet list for the `dataset_type`
  overview instead of a two-column table.
- The getting-started guide now uses one unified pipeline intro that combines
  the practical workflow, the under-the-hood execution steps, and the
  returned `PipelineResult`, while keeping the CLI out of the initial
  walkthrough.
- The practical workflow in getting-started now includes the run step and a
  reminder about shared cache reuse for sweeps.
- The getting-started pipeline overview now calls out the shared cache as a
  key feature for efficient experiment sweeps and links to the cache guide.
- The getting-started result summary now links to the dedicated outputs,
  reporting, evaluation, and training pages for deeper artifact details.
- The getting-started pipeline example now shows `HeatmapConfig` alongside the
  other end-to-end knobs.
- The dataset and split manifest schema now lives on a dedicated reference
  page, and the getting-started guide only links to it.
- The docs now include a dedicated API reference landing page that points to
  the compact generated reference and the main guide pages.
- The public pipeline config now uses `evaluation` instead of `eval`, and the
  docs/examples/reporting helpers were updated to match the new field name.
- The encoder config now exposes `allow_non_recommended_settings` so soma can
  sweep non-default spacing and input-size combinations without falling back
  to direct `slide2vec` calls.
- The extractor now threads `allow_non_recommended_settings` as an explicit
  boolean instead of wrapping it in a one-field `model_kwargs` dict.
- The API page now links to the dedicated dataset and split manifest reference
  page from its overview table.
- The heatmap example on the API page now uses the lower-level `train(...)`
  path like the rest of the examples and points to the outputs guide for the
  generated artifacts.
- The API examples now define reusable objects before passing them into helper
  functions, which makes the snippets read more consistently.
- The detailed Pythonic modular API examples now live in the dedicated API
  page, while the getting-started guide keeps only a short pointer to it.
- The getting-started pipeline example now includes `EvalConfig` so the
  end-to-end snippet shows the evaluation contract explicitly.
- The API page now stays focused and links out to the reporting guide for
  report contents, subgroup analysis, and comparison statistics.
- The API page now opens with a compact table of the main building blocks
  before the usage examples.
- Evaluation now has its own guide, separate from training, so metric
  contracts and subgroup analysis are documented alongside the evaluation
  results they produce.
- The top-level guide nav now separates training from evaluation instead of
  bundling them into one page.
- The cache guide now focuses on reusable upstream artifacts, while the
  outputs guide documents run-directory layout and per-run artifacts.
- The Sphinx docs now use a local page template override to remove the Furo
  "Made with Sphinx and @pradyunsg's Furo" footer credit.
- The Sphinx API docs no longer show per-object source links, so the code
  reference pages stay focused on the public API instead of exposing source
  buttons under every documented symbol.
- The encoder zoo now groups tile encoders by output-dimension buckets so
  related variants stay easy to scan without adding a visible family column.
- The aggregators page now uses a short overview table plus one section per
  preset so readers can scan the zoo and then see the class docstrings for
  each MIL aggregator.
- The aggregators page now pairs short prose with `autoclass` blocks so the
  constructor signatures stay in sync with the code.
- The tasks page now hides the internal branch-aware classification head from
  the public task table and API reference so the docs stay focused on
  user-facing heads.
- Heatmap rendering now mirrors the nested split directory structure from
  `fold_*/attention/` into `fold_*/heatmaps/`, so rendered PNGs preserve the
  same split provenance as the saved attention files.
- Heatmap rendering now reads tile geometry from the nested coordinates
  metadata structure used by current `hs2p` artifacts.
- Heatmap rendering now takes an explicit run-local `tiling_dir` so the
  coordinate source is part of the API instead of being inferred from the
  feature store.
- Managed CSV index reads now raise the Python parser field-size limit before
  loading existing rows, so very large metadata fields no longer crash run
  index updates.
- Tile feature extraction now trusts `slide2vec` encoders to be frozen by
  construction and no longer calls `eval()` on the loaded wrapper at runtime.
- Tile feature extraction now keeps encoder inputs in float32 instead of
  pre-casting tile batches to the registry's recommended precision, so the
  encoder wrapper remains responsible for any mixed-precision behavior.
- Tile-only feature extraction now reuses the shared `slide2vec` Rich progress
  reporter so encoder loading and batch-level feature extraction follow the
  same spinner/bar UX as slide pipelines, with a single `Embedding tiles`
  bar updated on the cumulative tile count rather than batch count.
- Tile-only feature extraction now uses the shared SLURM-aware CPU worker
  limit when `EncoderConfig.num_workers` is unset, instead of defaulting tile
  loading to the main process.
- Tile-only feature extraction now logs the resolved DataLoader worker count
  at runtime so HPC runs can confirm the effective worker budget in the Rich
  progress output.
- Tile-only feature extraction now uses non-blocking device transfers and
  persistent DataLoader workers whenever worker processes are enabled.
- `EncoderConfig` now exposes `prefetch_factor` and `persistent_workers`, and
  both slide-level and tile-level extraction thread them through the shared
  `build_execution_options(...)` helper so the tile input pipeline matches the
  upstream `slide2vec` execution contract.
