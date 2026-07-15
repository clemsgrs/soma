Benchmarking
============

soma packages foundation model benchmarks as registered, reproducible
protocols. Each benchmark fixes data preparation, splits, downstream training,
metrics, and seeds while exposing the components that are meaningful to compare.

This supports two complementary uses:

* **Reproduce a benchmark.** Run a bundled protocol and, when available,
  compare soma's measurement with the official reference.
* **Measure a component.** Hold the protocol fixed, vary one component (e.g. the encoder)
and measure its effect on downstream performance.

Run two encoders under the same protocol, then compare the completed runs::

   soma reproduce eva/bach --encoder uni2 --raw-root /path/to/eva/bach --output-root runs/eva-bach --seeds 1
   soma reproduce eva/bach --encoder virchow2 --raw-root /path/to/eva/bach --output-root runs/eva-bach --seeds 1
   soma leaderboard eva/bach --root runs/eva-bach/seed_0 --vary encoder

``soma reproduce`` handles curation, execution, and scoring. Completed runs
remain ordinary soma experiments. ``soma leaderboard`` reads their resolved
configurations and metrics without retraining or maintaining a separate results
table. See :doc:`cli` for command options and :doc:`outputs` for run artifacts.

Extend a benchmark
------------------

You can pass any supported :doc:`encoder <encoders>` to ``soma reproduce``, allowing you
to go beyond the curated references. For example, to evaluate a new encoder on the EVA/BACH
benchmark, you can run::

   soma reproduce eva/bach --encoder phikon --raw-root /path/to/eva/bach --output-root runs/eva-bach

When a matching reference exists, soma reports the delta and highlights
potential drift.

Included benchmarks
-------------------

* :doc:`EVA <eva-patch-classification-benchmark>` evaluates frozen encoders on
  patch-classification datasets.
* :doc:`OCELOT <ocelot-detection-benchmark>` evaluates dense encoders and image
  spacing for cell detection.
* :doc:`HEST <hest-gene-expression-benchmark>` evaluates frozen encoders for
  spatial gene-expression prediction.

Each page documents data acquisition, the fixed protocol, reproduction commands,
and the relevant packaged reference.

.. toctree::
   :maxdepth: 1
   :hidden:

   eva-patch-classification-benchmark
   ocelot-detection-benchmark
   hest-gene-expression-benchmark
