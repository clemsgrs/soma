Benchmark foundation models
===========================

Soma packages pathology foundation-model evaluations as named, versioned
protocols. A benchmark fixes the data preparation, prediction task, training
recipe, metric, and seeds while exposing meaningful comparison axes such as the
encoder or image spacing.

There are two ways to use the facility:

* **Reproduce an included benchmark.** Point Soma at the downloaded data and run
  the published protocol with one command.
* **Add your own benchmark.** Package an existing Soma workflow with its data
  preparation, comparison axes, scorer, and reference values.

Reproduce an included benchmark
-------------------------------

List the protocols available in your installed version:

.. code-block:: console

   soma list benchmarks

Then run the complete data preparation → configuration → training → scoring →
comparison workflow:

.. code-block:: console

   soma reproduce ocelot --raw-root /path/to/ocelot

By default, runs are written under
``soma_reproduce/<benchmark-name>/seed_<seed>``. Soma may create a managed run
subdirectory beneath that root; it contains the resolved config, predictions,
metrics, summary, and HTML report described in :doc:`outputs`.

Useful shortcuts:

* ``--seeds N`` runs seeds ``0`` through ``N-1`` instead of the benchmark's
  canonical seed set; ``--seeds 1`` is the quickest smoke test.
* ``--curated-dir <dir>`` reuses existing Soma manifests.
* ``--from-run-dir <dir>`` scores an existing run without training.
* A family name such as ``soma reproduce eva --raw-root /path/to/eva`` runs every
  downloaded member.

Each benchmark page contains its own download, raw-layout, split, command, and
result guidance:

* :doc:`EVA <eva-patch-classification-benchmark>` — compare frozen encoders on
  patch-classification datasets.
* :doc:`OCELOT <ocelot-detection-benchmark>` — compare dense encoders and image
  spacing for cell detection.
* :doc:`HEST <hest-gene-expression-benchmark>` — compare frozen encoders for
  spatial gene-expression prediction.

Compare completed runs
----------------------

``soma leaderboard`` reads resolved configs and metrics from completed run
directories, so comparisons do not depend on manually maintained spreadsheets:

.. code-block:: console

   soma leaderboard ocelot --root soma_reproduce --vary encoder

Use ``--fix AXIS=VALUE`` to hold an axis constant and ``--like RUN_DIR`` to copy
the fixed axes from an existing run. See :doc:`cli` for the full command surface.

Add your own benchmark
----------------------

If you already have a modular Soma workflow, the benchmark layer is deliberately
thin: define how raw data becomes manifests, how the canonical config is built,
which axes vary, how the primary metric is read, and which values provide the
comparison. Follow :doc:`add-a-benchmark` for the complete contract and a minimal
implementation shape.

How reference values are interpreted
------------------------------------

Reference rows have two public roles:

* **Gate references** are Soma regression anchors with tolerance bands.
  ``soma reproduce`` compares the measured primary metric with the band and
  reports PASS or FAIL.
* **External references** are published values from another implementation or
  protocol. Soma shows the measured value and difference for comparison, without
  a verdict.

Only gate references produce PASS/FAIL. A close external comparison demonstrates
agreement with a publication; it is not a software regression gate.
