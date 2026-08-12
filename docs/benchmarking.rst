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

To compare an externally installed private preset with public presets, follow
:doc:`benchmark-in-house-encoder`.

.. tip::

   Prefer Python? soma ships this as an API — the same curate, run, and score
   flow, driving the same protocol from code. See :ref:`benchmark-api`.

Reproduce a benchmark
---------------------

``soma reproduce NAME`` curates the data, runs the fixed protocol, and scores the
result. Keep the Benchmark's default encoder, select one with ``--encoder``, or run an
explicitly ordered panel with ``--encoders``::

   soma reproduce eva/bach --encoder uni2 --raw-root /path/to/eva/bach --output-root runs/eva-bach --seeds 1
   soma reproduce eva/bach --encoders private-pathology uni2 virchow2 --raw-root /path/to/eva/bach --output-root runs/eva-bach
   soma reproduce eva --encoders private-pathology uni2 --raw-root /path/to/eva --output-root runs/eva --seeds 1

The plural form resolves a family to all of its concrete Benchmarks, then validates every
Benchmark × Encoder cell before curation, Pipeline construction, extraction, training, or
Run writes. If any cell is invalid, soma names its concrete Benchmark and Encoder, reports
every incompatibility in panel order, and starts no work. A missing capability means you
must select a compatible Benchmark or fix the Encoder plugin implementation. After a
successful preflight, soma executes one ordinary Run at a time in canonical Benchmark order
and supplied Encoder order. Raw data is curated once per concrete Benchmark;
``--curated-dir`` skips curation for the whole panel. ``--encoder`` and ``--encoders`` are
mutually exclusive, and ``--from-run-dir`` remains a single-Run rescoring path.
Installed-preset discovery and capability preflight require slide2vec 5.8.0 or newer.

Each concrete Benchmark writes its own canonical cross-encoder Leaderboard beneath its
member output root—for example ``runs/eva/bach/leaderboards/eva/bach.*``. A family is a
collection of dataset-, splits-, and task-specific comparisons; soma never combines its
members into a family-wide rank.

A preflight rejection starts no Run. Once a valid panel has started, a runtime failure is
different: later encoders still run, and completed Runs remain ordinary valid Runs. If at
least one Run completed during the panel, soma writes the canonical Leaderboard from the
completed Runs and labels the panel ``PARTIAL`` in command output; the Leaderboard remains
the ordinary canonical projection. If no Run completed, soma writes no Leaderboard. In either
case soma prints one failure summary and exits nonzero, so automation cannot mistake partial
output for complete success.

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

The encoder is the axis ``soma reproduce --encoders`` varies, so the plural command is the
short path to a local comparison::

   soma reproduce eva/bach --encoders uni2 virchow2 --raw-root /path/to/eva/bach --output-root runs/eva-bach --seeds 1

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
* :doc:`PathoROB <pathorob-robustness-benchmark>` evaluates frozen tile
  encoders for representation robustness across medical centers.

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
   pathorob-robustness-benchmark
   croma-encoder-panel
   benchmark-in-house-encoder
