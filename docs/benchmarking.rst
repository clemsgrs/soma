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
and its delta. An encoder without a matching reference still runs; only the
reference comparison is skipped. Reference comparisons are informational and
do not determine command success.

Compare an encoder panel
------------------------

Use ``--encoders`` to run an ordered panel on one benchmark or a family::

   soma reproduce eva/bach --encoders uni2 virchow2 --raw-root /path/to/eva/bach --output-root runs/eva-bach --seeds 1
   soma reproduce eva --encoders uni2 virchow2 --raw-root /path/to/eva --output-root runs/eva --seeds 1

soma validates every benchmark–encoder combination before curation or execution.
It reports incompatibilities in panel order and starts no runs unless the whole
panel is valid. Choose a compatible benchmark or correct the encoder plugin's
capabilities if validation fails. Installed-preset discovery and capability
checks use slide2vec's public encoder registry.

After validation, benchmarks run in canonical order and encoders in the supplied
order. Raw data is curated once per benchmark. With ``--curated-dir``, curation
is skipped; a family expects each member's manifests in its own subdirectory.
``--encoder`` and ``--encoders`` are mutually exclusive, and ``--from-run-dir``
accepts only a single run.

Each benchmark writes a cross-encoder leaderboard beneath its output root, for
example ``runs/eva/bach/leaderboards/eva/bach.*``. Family members are never
combined into a cross-dataset rank.

If an encoder fails during execution, later encoders continue and completed
runs remain valid. When any run completes, soma writes the ordinary leaderboard
from completed runs and labels the panel ``PARTIAL`` in command output. If none
completes, it writes no leaderboard. Either failure case ends with a failure
summary and a nonzero exit status.

To install and compare a private preset, follow
:doc:`benchmark-in-house-encoder`.

Compare other experiment choices
--------------------------------

``soma leaderboard`` ranks completed runs along the axis passed to ``--vary``
and writes CSV, JSON, and HTML tables, including any packaged reference. Runs
sharing a ``(dataset, splits, task)`` triple form one comparison.

To compare aggregators, decoders, spacing, or feature modes, create the runs
with ordinary configs. For an aggregator comparison:

#. Use the same ``dataset.csv``, ``splits.csv``, and task in every config.
#. Change only ``aggregation:`` and use a shared output root, such as
   ``runs/agg-sweep``.
#. Run each config: ``soma abmil.yaml``, ``soma transmil.yaml``, and so on.
#. Run ``soma leaderboard --root runs/agg-sweep --vary aggregator``. The ranking
   metric is inferred from the runs.

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

Each page covers data acquisition, the fixed protocol, commands, and packaged
references. Encoder compatibility depends on the protocol's required outputs
and geometry.

.. toctree::
   :maxdepth: 1
   :hidden:

   eva-patch-classification-benchmark
   ocelot-detection-benchmark
   hest-gene-expression-benchmark
   croma-robustness-benchmark
   benchmark-in-house-encoder
