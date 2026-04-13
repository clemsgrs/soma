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
