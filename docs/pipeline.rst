Pipeline
=================

A pipeline run starts by reading the dataset and split manifests, then follows
the stages implied by ``dataset_type``:

- ``tile``: read tile images and labels, extract tile features, train a
  lightweight task head, and evaluate on the test split.
- ``slide``: read whole-slide images and labels, tile each slide, extract
  features with a tile-level or slide-level encoder, and train the
  appropriate downstream model. Tile-level encoders require an aggregator
  plus prediction head. Slide-level encoders only require a prediction head.
- ``patient``: aggregate slide-level outputs across multiple slides per
  patient (experimental).

The main configuration object is :class:`soma.config.PipelineConfig`:

.. autoclass:: soma.config.PipelineConfig
   :members:

Examples
--------

Slide-level pipeline
~~~~~~~~~~~~~~~~~~~~

This is the smallest typical slide-level pipeline configuration:

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


Tile-level pipeline
~~~~~~~~~~~~~~~~~~~

Tile-level runs use the same pipeline entry point, but keep ``aggregator``
set to ``None`` because the model operates on per-tile features directly:

.. code-block:: python

   from soma import EncoderConfig, EvalConfig, Pipeline, PipelineConfig, TaskConfig, TrainingConfig

   config = PipelineConfig(
       dataset_csv="dataset.csv",
       splits_csv="splits.csv",
       output_root="output",
       dataset_type="tile",
       encoder=EncoderConfig(name="uni2"),
       aggregator=None,
       task=TaskConfig(name="binary_classification"),
       evaluation=EvalConfig(metrics=["accuracy"]),
       training=TrainingConfig(epochs=50, learning_rate=1e-4),
   )

   result = Pipeline(config).run()

Run outputs
-----------

The run directory layout is described in :doc:`outputs`.
