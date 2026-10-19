Benchmark an in-house encoder
=============================

Register an in-house encoder with slide2vec, then compare it with public presets
under the same soma benchmark protocol. soma uses the encoder's registered name.
Its normalization transform must preserve geometry (no resize or crop).

From your own script
--------------------

The simplest way needs no package: register the encoder in your script with
``slide2vec.register_encoder`` (see :ref:`custom-encoder`), then run each benchmark
from the same script with ``soma.benchmark.run_benchmark``:

.. code-block:: python

   from soma.benchmark import run_benchmark

   import my_encoder  # registers "my-private-encoder" with slide2vec.register_encoder

   for encoder in ("my-private-encoder", "uni2"):
       run_benchmark(
           "eva/bach",
           encoder=encoder,
           raw_root="/data/eva/bach",
           output_root="runs/eva-bach",
           seeds=1,
       )

The registration exists only in the process that ran your script. slide2vec starts
one worker process per visible GPU, and those processes never import it, so make one
GPU visible, for example with ``CUDA_VISIBLE_DEVICES=0``.

As a plugin package
-------------------

``soma reproduce`` on the command line, and extraction on several GPUs, need an
encoder that every new process finds by name. Install it as a slide2vec plugin
package: follow slide2vec's `installed package guide
<https://clemsgrs.github.io/slide2vec/models.html#from-an-installed-package>`_ for
packaging, weights and credentials. Install the plugin in soma's Python environment
and check discovery::

   pip install ./my-slide2vec-encoders
   soma list encoders

The private preset should appear beside the public names. If it does not,
resolve the installation or provider error before running a benchmark.

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
