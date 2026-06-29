Getting started
===============

`soma` takes a dataset of slides and labels to a reproducible result report
through a single, coherent API.

.. figure:: /_static/figures/run-flow.svg
   :figclass: soma-figure
   :alt: Three input files flow into one soma command that schedules tiling, feature extraction, training, and metrics.

   You provide three files — a dataset, splits, and a config. ``soma`` then
   schedules every step: tiling, feature extraction, training, and metrics.

Install
-------

.. code-block:: bash

   pip install soma-pathology

Three ways to use soma
----------------------

The same components, cache, and run outputs back all three workflows, so you can
move between them freely. Pick the one that matches how much control you want.

Pipeline API — one config, one call
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The quickest path: describe the whole run in one
:class:`~soma.config.PipelineConfig` and call ``.run()``.

.. code-block:: python

   from soma import (
       AggregatorConfig,
       EncoderConfig,
       EvalConfig,
       Pipeline,
       PipelineConfig,
       TaskConfig,
       TrainingConfig,
   )

   config = PipelineConfig(
       dataset_csv="dataset.csv",
       splits_csv="splits.csv",
       output_root="output",
       dataset_type="slide",
       encoder=EncoderConfig(name="uni2"),
       aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 256}),
       task=TaskConfig(name="binary_classification"),
       evaluation=EvalConfig(metrics=["auroc", "balanced_accuracy"]),
       training=TrainingConfig(epochs=50, learning_rate=1e-4),
   )

   result = Pipeline(config).run()

The returned ``result`` is a :class:`~soma.pipeline.PipelineResult` — a handle on
the run you just completed: ``result.run_dir`` (the run directory on disk),
``result.summary`` (aggregated metrics, mirroring ``summary.json``), and
``result.fold_results`` (one :class:`~soma.pipeline.FoldResult` per fold, each
carrying the training result, the tune report, and per-split test reports). The
same experiment is also persisted on disk; see :doc:`outputs`.

Under the hood, ``soma`` turns the config into one run directory and runs a fixed
sequence: read the manifests, resolve settings, extract or load features, train one
model per fold (tune split for checkpoint selection), evaluate on the tune and test
splits, then write metrics, predictions, checkpoints, and an HTML report. A shared
:doc:`cache <caching>` reuses preprocessing and features across runs whenever
upstream settings match, so sweeps skip work already done.

The full configuration reference, plus the tile- and patient-level variants, is in
:doc:`pipeline`.

Step-by-step API — compose the building blocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For finer control, drive the individual :doc:`building blocks <api>` yourself —
preprocessing, feature extraction, training, evaluation, reporting, or heatmaps as
separate steps instead of one call. The :doc:`slide-level tutorial
<tutorials/walkthrough-slide-level>` builds a run this way, end to end.

CLI — run from the shell
~~~~~~~~~~~~~~~~~~~~~~~~~~

Prefer the terminal? Point ``soma`` at a YAML config:

.. code-block:: bash

   soma config.yaml

The YAML mirrors ``PipelineConfig`` field for field. See :doc:`cli` for the full
command set and the canonical config schema.
