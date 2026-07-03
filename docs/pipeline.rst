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
- ``segmentation`` / ``detection``: dense paths. A **frozen** encoder produces a
  dense token grid and a trained decoder maps it to a per-pixel mask
  (segmentation) or a per-class peak heatmap (detection). Detection supervision is
  a per-sample ``points_path`` and the head scores class-aware **F1 at a matching
  distance δ** (``mean_f1``); see :doc:`detection`.

The main configuration object is :class:`soma.config.PipelineConfig`:

.. autoclass:: soma.config.PipelineConfig
   :members:

Examples
--------

The minimal slide-level run lives in :doc:`getting-started`. The variants below
show how the other ``dataset_type`` values differ from it.

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
