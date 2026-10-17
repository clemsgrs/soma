Benchmarking
============

A benchmark is a fixed protocol: the data, its splits, the downstream model, the
metric and the seeds are all held constant, and only the encoder varies. ``soma
reproduce`` prepares the data, runs the protocol and scores the result; ``soma
leaderboard`` ranks completed runs without retraining. Each benchmark page states its
data and task, its protocol, the results soma has recorded and how to reproduce them.

Benchmarks
----------

.. list-table::
   :header-rows: 1
   :widths: 14 30 36 20

   * - Benchmark
     - Task
     - Data
     - Metric
   * - :doc:`EVA <eva-patch-classification-benchmark>`
     - Patch classification, nucleus segmentation, slide classification
     - Ten public datasets of the kaiko-ai/eva leaderboard
     - Balanced accuracy, foreground mean Dice
   * - :doc:`OCELOT <ocelot-detection-benchmark>`
     - Cell detection
     - OCELOT 2023 cell patches
     - Mean F1 at 3 µm
   * - :doc:`MIDOG <midog-detection-benchmark>`
     - Mitotic-figure detection
     - MIDOG 2022 public training regions, local held-out split
     - Stitched-ROI F1 at 7.5 µm
   * - :doc:`MONKEY <monkey-detection-benchmark>`
     - Inflammatory-cell detection
     - MONKEY public training slides, five patient folds
     - Merged inflammatory-cell FROC at 5 µm
   * - :doc:`HEST <hest-gene-expression-benchmark>`
     - Spatial gene-expression prediction
     - Nine HEST-Benchmark tasks
     - Mean Pearson
   * - :doc:`CRoMa <croma-robustness-benchmark>`
     - Robustness to the medical centre, task-free
     - Three PathoROB tile cohorts
     - CRoMa margin

Reproduce a benchmark
---------------------

#. Prepare the raw data as the benchmark's page describes, in a directory you pass as
   ``--raw-root``. ``soma list benchmarks`` prints every registered protocol.
#. Run the protocol with its default encoder, or any compatible installed one::

      soma reproduce eva/bach --encoder uni2 --raw-root /path/to/eva/bach --output-root runs/eva-bach

#. Read the score. When a packaged reference exists for that encoder, soma prints the
   measured value next to it. The comparison is informational: an encoder without a
   reference still runs, and a gap never fails the command.

``--seeds 1`` runs seed 0 only, for a quick smoke test; omit it for the canonical seed
set. MONKEY uses one seed across five patient folds, so its documented command
sets ``--seeds 1`` explicitly. ``--curated-dir`` reuses prepared manifests,
``--from-run-dir`` rescores one existing run, and ``--record`` appends the score to
the benchmark's results ledger, which is what the results tables in these pages
are built from.

Compare encoders
----------------

``--encoders`` runs an ordered panel on one benchmark or on a family::

   soma reproduce eva/bach --encoders uni2 virchow2 --raw-root /path/to/eva/bach --output-root runs/eva-bach
   soma reproduce eva --encoders uni2 virchow2 --raw-root /path/to/eva --output-root runs/eva

A family prefix such as ``eva`` runs every registered member; EVA and HEST expect one
raw-data subdirectory per member, CRoMa shares one prepared root. soma checks every
benchmark and encoder pairing against slide2vec's encoder registry before any run
starts. Benchmarks then run in canonical order and encoders in the given order, with
the raw data curated once per benchmark. Each benchmark writes a cross-encoder
leaderboard under its output root, for example ``runs/eva/bach/leaderboards/eva/bach.*``;
members of a family are never combined into one rank. If an encoder fails, the later
ones still run, the leaderboard covers the completed runs, and the command reports
``PARTIAL`` with a nonzero exit status.

To add a private encoder to the panel, see :doc:`benchmark-in-house-encoder`.

Compare other choices
---------------------

``soma leaderboard`` ranks completed runs along any axis and writes CSV, JSON and HTML
tables, including any packaged reference. Runs that share a dataset, splits and task
form one comparison. To compare aggregators, decoders, spacings or feature modes, run
ordinary configs that differ only in that axis under one output root, then rank them::

   soma leaderboard --root runs/agg-sweep --vary aggregator

Where a benchmark lives
-----------------------

Each benchmark is one package under ``soma/benchmark/``: its curator, its recipes,
its scorer, the packaged reference rows (``reference.csv``) and the results ledger
``--record`` appends to (``results.csv``). The ``soma`` command imports
:mod:`soma.benchmark` at startup; from Python, ``import soma.benchmark`` registers the
benchmarks (``import soma`` alone does not). See :doc:`cli` for every option,
:doc:`outputs` for the run artifacts and :ref:`benchmark-api` for the Python workflow.

.. toctree::
   :maxdepth: 1
   :hidden:

   eva-patch-classification-benchmark
   ocelot-detection-benchmark
   midog-detection-benchmark
   monkey-detection-benchmark
   hest-gene-expression-benchmark
   croma-robustness-benchmark
   benchmark-in-house-encoder
