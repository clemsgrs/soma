Benchmarking
============

soma packages foundation model benchmarks as registered, reproducible
protocols. Each benchmark fixes data preparation, splits, downstream training,
metrics, and seeds, leaving the encoder as the component you choose.

Two commands drive them:

* ``soma reproduce`` runs a benchmark end to end for the encoder you pick —
  curation, execution, and scoring — and, when the benchmark ships a packaged
  reference, reports the delta against it.
* ``soma leaderboard`` reads completed runs — from reproduce or from your own
  configs — and renders a ranked comparison, without retraining.

See :doc:`cli` for every command option and :doc:`outputs` for the artifacts each
run writes.

.. tip::

   Prefer Python? soma ships this as an API — the same curate, run, and score
   flow, driving the same protocol from code. See :ref:`benchmark-api`.

Reproduce a benchmark
---------------------

``soma reproduce NAME`` curates the data, runs the fixed protocol, and scores the
result for a single encoder — your choice of any supported
:doc:`encoder <encoders>`::

   soma reproduce eva/bach --encoder uni2 --raw-root /path/to/eva/bach --output-root runs/eva-bach --seeds 1

``NAME`` is a registered benchmark (e.g. ``ocelot``, ``eva/bach``) or a family prefix
(``eva``) that fans out over every member. ``--seeds 1`` is the quickest smoke; the
benchmark's canonical seed set runs by default. Completed runs remain ordinary soma
experiments.

When the benchmark ships a packaged reference for the encoder you ran, soma reports
the measured value beside it and highlights potential drift. Because the encoder is a free choice, you
can also benchmark models the reference never covered — passing an encoder with no
matching reference simply skips the comparison::

   soma reproduce eva/bach --encoder phikon --raw-root /path/to/eva/bach --output-root runs/eva-bach

Compare runs on a leaderboard
-----------------------------

``soma leaderboard`` projects a set of completed runs into a single
ranked table, comparing them along the axis you pass to ``--vary``. It writes the table as CSV, JSON, and
HTML, with any packaged reference shown alongside. Every run sharing one
``(dataset, splits, task)`` triple joins the table.

The encoder is the axis ``soma reproduce`` varies, so comparing encoders is one
reproduce run per encoder under the same output root, then a leaderboard::

   soma reproduce eva/bach --encoder uni2     --raw-root /path/to/eva/bach --output-root runs/eva-bach --seeds 1
   soma reproduce eva/bach --encoder virchow2 --raw-root /path/to/eva/bach --output-root runs/eva-bach --seeds 1
   soma leaderboard eva/bach --root runs/eva-bach/seed_0 --vary encoder

Any other axis — aggregator, decoder, spacing, feature mode — works the same way, but
you produce the runs yourself with ordinary ``soma <config>`` runs. To compare
aggregators for a fixed encoder:

#. Take one cohort — a shared ``dataset.csv`` + ``splits.csv`` + task
#. Write N configs identical except the ``aggregation:`` key
#. Run each ordinary pipeline: ``soma abmil.yaml``, ``soma transmil.yaml``, …
#. ``soma leaderboard --root runs/agg-sweep --vary aggregator`` — every run sharing that
   ``(dataset, splits, task)`` triple joins the table, ranked by the metric inferred
   from the runs

Included benchmarks
-------------------

* :doc:`EVA <eva-patch-classification-benchmark>` evaluates frozen encoders on
  patch-classification datasets.
* :doc:`OCELOT <ocelot-detection-benchmark>` evaluates dense encoders for cell
  detection.
* :doc:`HEST <hest-gene-expression-benchmark>` evaluates frozen encoders for
  spatial gene-expression prediction.

Each page documents data acquisition, the fixed protocol, reproduction commands,
and the relevant packaged reference.

The :doc:`Croma 0.3 encoder panel audit <croma-encoder-panel>` pins the published
model names to explicit slide2vec output contracts for reuse by robustness
benchmarks.

.. toctree::
   :maxdepth: 1
   :hidden:

   eva-patch-classification-benchmark
   ocelot-detection-benchmark
   hest-gene-expression-benchmark
   croma-encoder-panel
