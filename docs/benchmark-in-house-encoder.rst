Benchmark an in-house encoder
=============================

Install an in-house encoder as a slide2vec preset, then compare it with public
presets under the same soma benchmark protocol. Keep the plugin in its own
package; soma uses its registered name.

Install and inspect
-------------------

Follow the `slide2vec custom encoder plugin guide
<https://clemsgrs.github.io/slide2vec/models.html#custom-encoder-plugin-package>`_
for implementation, packaging, weights, and credentials. The plugin's
normalization transform must preserve geometry (no resize or crop).

Install the plugin in soma's Python environment and check discovery::

   pip install ./my-slide2vec-encoders
   soma list encoders

The private preset should appear beside the public names. If it does not,
resolve the installation or provider error before running a benchmark.

Run the comparison
------------------

List protocols with ``soma list benchmarks`` and choose one whose input mode
and geometry fit the preset. EVA accepts pooled tile features; a dense
benchmark also requires dense encoder outputs. Name the private preset and
each public comparator explicitly, on one benchmark or a whole family::

   soma reproduce eva/bach --encoders my-private-encoder uni2 --raw-root /data/eva/bach --output-root runs/eva-bach --seeds 1
   soma reproduce eva --encoders my-private-encoder uni2 --raw-root /data/eva --output-root runs/eva --seeds 1

Every comparator runs locally. A private preset has no packaged reference, so
its row reports ``REFERENCE SKIPPED``. See :doc:`benchmarking` for panel
validation, execution order, and partial-failure behavior.

Read the results
----------------

Each benchmark's leaderboard holds the protocol fixed and varies ``encoder``.
Compare the private preset's measured row with the public rows; ``n`` and the
spread summarize the seeds behind each entry.
