API
===

Use the public ``soma`` package to compose experiments in Python; use the
focused reference pages for component-specific contracts.

Entry points
------------

.. list-table::
   :header-rows: 1

   * - Entry point
     - Purpose
   * - :class:`soma.pipeline.Pipeline`
     - Run a validated :class:`soma.config.PipelineConfig` end to end.
   * - :class:`soma.extraction.FeatureExtractor`
     - Extract and cache reusable features.
   * - :func:`soma.pipeline.train`
     - Train from an existing feature store.
   * - :mod:`soma.reporting`
     - Regenerate a run report or compare completed runs.

The :doc:`pipeline` guide explains when to use each level. Configuration,
dataset, and output contracts live in :doc:`configuration`, :doc:`dataset`,
and :doc:`outputs`.

Composition example
-------------------

This example constructs one tile-classification run entirely through the
public API:

.. code-block:: python

   from soma import (
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
       output_root="runs",
       dataset_type="tile",
       encoder=EncoderConfig(name="dinov2-vitb14"),
       task=TaskConfig(name="binary_classification"),
       evaluation=EvalConfig(metrics=["auroc", "balanced_accuracy"]),
       training=TrainingConfig(epochs=5, batch_size=16),
   )
   result = Pipeline(config).run()

See :doc:`getting-started` for the matching manifests and CLI workflow.

Discovery
---------

Registry helpers expose the available component names without importing
implementation modules:

.. code-block:: python

   from soma import (
       list_aggregators,
       list_decoders,
       list_models,
       list_pixel_classifiers,
       list_task_heads,
   )

   tile_encoders = list_models(level="tile")
   aggregators = list_aggregators()
   decoders = list_decoders()
   pixel_classifiers = list_pixel_classifiers()
   task_heads = list_task_heads()

Component guides
----------------

Continue with :doc:`preprocessing`, :doc:`encoders`, :doc:`aggregators`,
:doc:`decoders`, :doc:`tasks`, :doc:`training`, and :doc:`evaluation`.
