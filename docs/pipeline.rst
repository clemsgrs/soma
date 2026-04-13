Pipeline
=================

A pipeline run starts by reading the dataset and split manifests, then follows
the stages implied by ``dataset_type``:

- ``tile``: Read tile images and labels, extract tile features, train a
  lightweight task head, and evaluate on the test split.
- ``slide``: Read whole-slide images and labels, tile each slide, extract
  features with a tile-level or slide-level encoder, and train the
  appropriate downstream model. Tile-level encoders require an aggregator
  plus prediction head. Slide-level encoders only require a prediction head.
- ``patient``: Aggregate slide-level outputs across multiple slides per
  patient (experimental).

Patient-level runs require a ``patient_id`` column in the dataset manifest and
produce one prediction per patient.

The main configuration object is :class:`soma.config.PipelineConfig`:

.. autoclass:: soma.config.PipelineConfig
   :members:

Example
-------

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

Run outputs
-----------

The run directory layout is described in :doc:`outputs`.
