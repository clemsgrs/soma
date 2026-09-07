Benchmark an in-house encoder
=============================

Install an in-house encoder as a slide2vec preset, then compare it with public
presets under the same soma benchmark protocol. Keep the plugin in its own
package; soma uses its registered name.

Install and inspect
-------------------

Follow the `slide2vec custom encoder plugin guide
<https://clemsgrs.github.io/slide2vec/models.html#custom-encoder-plugin-package>`_
for implementation, packaging, weights, credentials, and worker availability.
Install the plugin in soma's Python environment and check discovery::

   pip install ./my-slide2vec-encoders
   soma list encoders

The private preset should appear beside the public names. If it does not,
resolve the installation or provider error before running a benchmark.

Run the comparison
------------------

List protocols with ``soma list benchmarks`` and choose one whose input mode
and geometry fit the preset. EVA accepts pooled tile features; a dense
benchmark also requires dense encoder outputs.

Name the private preset and each public comparator explicitly::

   soma reproduce eva/bach --encoders my-private-encoder uni2 --raw-root /data/eva/bach --output-root runs/eva-bach --seeds 1

``--seeds 1`` runs seed 0 for a smoke test; omit it for the canonical seeds.
Each comparator runs locally. A packaged reference provides context for a
matching preset and never replaces its local run. A private preset without a
reference still produces a measured result and reports ``REFERENCE SKIPPED``.

Use a family name to apply the same panel to every member::

   soma reproduce eva --encoders my-private-encoder uni2 --raw-root /data/eva --output-root runs/eva --seeds 1

The panel is validated before any work starts. Each member produces its own
leaderboard. See :doc:`benchmarking` for validation requirements, execution
order, and partial-failure behavior.

Read the results
----------------

The generated leaderboard holds the protocol fixed and varies ``encoder``.
Compare the private preset's Measured row with the locally Measured public
rows. ``n`` and the spread describe the collapsed seed results; Reference and
delta columns provide packaged context when available.

Runs and caches use soma's ordinary preset-name-based identity, so reruns reuse
the normal cache.
