Benchmarking
============

Compare encoders with registered benchmarks that fix data preparation, splits,
downstream training, metrics, and seeds. ``soma reproduce`` curates the data,
runs the protocol, and scores the result. ``soma leaderboard`` compares completed
runs without retraining.

Reproduce a benchmark
------------------------

List the available protocols with ``soma list benchmarks``, then follow the
benchmark's data-preparation instructions below. Keep its default encoder or
select a compatible installed preset with ``--encoder``::

   soma reproduce eva/bach --encoder uni2 --raw-root /path/to/eva/bach --output-root runs/eva-bach --seeds 1

``--seeds 1`` runs seed 0 for a quick smoke test; omit it to use the canonical
seed set. Use ``--curated-dir`` to reuse prepared manifests or
``--from-run-dir`` to rescore one existing run.

A family prefix such as ``eva`` runs every registered member. EVA and HEST
expect one raw-data subdirectory per member; CRoMa shares one prepared raw root.
Each member remains a separate dataset-, splits-, and task-specific comparison.

When a packaged reference matches the encoder, soma reports the measured value
next to it. The comparison is informational and never determines command
success; an encoder without a reference still runs.

Compare an encoder panel
------------------------

Use ``--encoders`` to run an ordered panel on one benchmark or a family::

   soma reproduce eva/bach --encoders uni2 virchow2 --raw-root /path/to/eva/bach --output-root runs/eva-bach --seeds 1
   soma reproduce eva --encoders uni2 virchow2 --raw-root /path/to/eva --output-root runs/eva --seeds 1

soma validates every benchmark–encoder pairing against slide2vec's encoder
registry before any run starts, and reports incompatibilities in panel order.
Benchmarks then run in canonical order and encoders in the supplied order, with
raw data curated once per benchmark. ``--encoder`` and ``--encoders`` are
mutually exclusive, and ``--from-run-dir`` accepts a single run.

Each benchmark writes a cross-encoder leaderboard beneath its output root, for
example ``runs/eva/bach/leaderboards/eva/bach.*``. Family members are never
combined into a cross-dataset rank. If an encoder fails, later encoders still
run, the leaderboard covers completed runs, and the command reports ``PARTIAL``
with a nonzero exit status.

To install and compare a private preset, follow
:doc:`benchmark-in-house-encoder`.

Compare other experiment choices
--------------------------------

``soma leaderboard`` ranks completed runs along the axis passed to ``--vary``
and writes CSV, JSON, and HTML tables, including any packaged reference. Runs
sharing a ``(dataset, splits, task)`` triple form one comparison. To compare
aggregators, decoders, spacing, or feature modes, run ordinary configs that
differ only in that axis under one output root, then rank them::

   soma leaderboard --root runs/agg-sweep --vary aggregator

See :doc:`cli` for command options, :doc:`outputs` for run artifacts, and
:ref:`benchmark-api` for the equivalent Python workflow.

Included benchmarks
-------------------

* :doc:`EVA <eva-patch-classification-benchmark>`: patch classification with
  frozen tile encoders.
* :doc:`OCELOT <ocelot-detection-benchmark>`: cell detection with dense encoders.
* :doc:`HEST <hest-gene-expression-benchmark>`: spatial gene-expression prediction
  with frozen encoders.
* :doc:`CRoMa <croma-robustness-benchmark>`: representation robustness across
  medical centers with frozen tile encoders.

.. toctree::
   :maxdepth: 1
   :hidden:

   eva-patch-classification-benchmark
   ocelot-detection-benchmark
   hest-gene-expression-benchmark
   croma-robustness-benchmark
   benchmark-in-house-encoder
