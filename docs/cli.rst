CLI
===

Use ``soma`` to run YAML experiments, inspect registered components, and
reproduce benchmarks.

Basic usage
-----------

Run a configuration::

    soma /path/to/config.yaml

``python -m soma /path/to/config.yaml`` is equivalent.

Available commands
------------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Command
     - Purpose
   * - ``soma CONFIG``
     - Run a pipeline from YAML.
   * - ``soma list encoders [--level LEVEL]``
     - List encoders, optionally filtered to ``tile``, ``slide``, or ``patient``.
   * - ``soma list aggregators``
     - List MIL aggregators.
   * - ``soma list decoders``
     - List dense neural decoders.
   * - ``soma list pixel-classifiers``
     - List decoder-free pixel classifiers.
   * - ``soma list tasks``
     - List task heads.
   * - ``soma list benchmarks``
     - List names accepted by ``reproduce`` and ``leaderboard``.

Benchmark commands
------------------

Reproduce one registered benchmark::

    soma reproduce NAME --raw-root /path/to/data

Use ``--curated-dir`` to reuse manifests, ``--from-run-dir`` to score an
existing run, and ``--seeds 1`` for a smoke run. ``--encoder`` and
``--spacing`` select registered benchmark axes. A family name such as
``eva`` runs every ``eva/<dataset>`` member. ``--record`` appends the
measured value and provenance to the packaged results ledger.

Gate references produce ``PASS`` or ``FAIL``; external references are
reported as non-gating deltas. See :doc:`benchmarking` for the distinction.

Build a faceted view over completed run directories::

    soma leaderboard [NAME] --root OUTPUT_ROOT --vary encoder

``--fix AXIS=VALUE`` holds an axis constant. ``--like RUN_DIR`` inherits
fixed axes from an existing run. ``--metric`` and ``--split`` override the
ranked value.

What the CLI expects
--------------------

YAML is nested by concern and loaded through
:func:`soma.config.load_config`. Start with :doc:`getting-started`; consult
:doc:`configuration` for the generated canonical schema and :doc:`pipeline`
for the corresponding Python API.
