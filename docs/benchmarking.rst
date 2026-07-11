Benchmarking
============

Soma benchmarks are named, versioned protocols for comparing foundation models.
Each registry entry binds a curator, config builder, scorer, canonical seeds, and
packaged references. Use this page for the shared workflow and the generated
benchmark pages for protocol details.

.. toctree::
   :maxdepth: 1
   :hidden:

   curation
   eva-patch-classification-benchmark
   ocelot-detection-benchmark
   hest-gene-expression-benchmark

Reference semantics
-------------------

Reference rows have two distinct roles:

* **Gate references** are Soma regression anchors with tolerance bands.
  ``soma reproduce`` compares the measured primary metric with the band and
  reports PASS or FAIL.
* **External references** are published values from another implementation or
  protocol. Soma reports the measured value and delta for comparison, without a
  verdict.

Only gate references produce PASS/FAIL. HEST, for example, contains external
references only.

Workflow
--------

**Curate.** Convert the benchmark's raw public layout into the standard
``dataset.csv`` and ``splits.csv`` manifests. Reproduction preserves the source
benchmark's splits; it does not invent new ones.

**Configure.** The registry's ``build_config`` method fixes the task, training
recipe, metric, and model settings. You choose only declared axes such as
``encoder`` or ``spacing``.

**Run and score.** Soma runs every canonical seed and reads the primary metric
from the resulting :doc:`run bundle <outputs>`. Most scorers read
``summary.json``; OCELOT uses its challenge-specific greedy matcher.

**Compare.** ``soma leaderboard`` reads completed run directories and groups
them by the benchmark facet. ``--vary``, ``--fix``, and ``--like`` refine that
view without retraining.

Reproduce
---------

Run the complete curate → configure → run → score workflow with one command::

    soma reproduce ocelot --raw-root /path/to/ocelot

Useful variants:

* ``--curated-dir <dir>`` reuses existing manifests.
* ``--from-run-dir <dir>`` scores an existing run without training.
* ``--seeds 1`` provides a quick smoke test.
* A family name such as ``soma reproduce eva --raw-root /path/to/eva`` runs
  every registered member.

Reference roles and values come directly from packaged
``reference/<name>.csv`` files. A PASS therefore confirms that the measured run
matches a Soma gate, not an external publication.

Benchmarks
----------

List the registry in your installed version::

    soma list benchmarks

* :doc:`OCELOT <ocelot-detection-benchmark>` — cell detection with an encoder ×
  spacing ablation.
* :doc:`EVA <eva-patch-classification-benchmark>` — patch classification across
  the ``eva/<dataset>`` family.
* :doc:`HEST <hest-gene-expression-benchmark>` — spatial gene-expression
  regression across the ``hest/<task>`` family.

See :doc:`cli` for the full command surface and :doc:`curation` for raw-data
layouts.
