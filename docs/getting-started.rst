Getting Started
===============

`soma` is a modular framework to streamline computational pathology research.
It helps you go from a dataset of slides and labels to a reproducible result
report through a single, coherent API.

Practical workflow
------------------

1. Start with a valid dataset and split manifest.
2. Choose an :doc:`encoder <encoders>`.
3. If working with a tile-level encoder, add an :doc:`aggregator <aggregators>`.
4. Pick a :doc:`task <tasks>`, :doc:`evaluation config <evaluation>`, and
   :doc:`training config <training>`.

Dataset format
--------------

Two CSV files describe the data:

``dataset.csv``
  | Required columns: ``sample_id``, ``image_path``, ``label``.
  | Optional columns: ``mask_path`` (pre-computed tissue mask), ``patient_id`` (required for ``dataset_type="patient"``).
  | Any additional columns are carried along as per-sample metadata.

``splits.csv``
  | Required columns: ``fold``, ``sample_id``, ``split``.
  | Valid split names: ``train``, ``tune``, or any name starting with ``test`` (e.g. ``test``, ``test_external``).

Run a full pipeline
-------------------

The simplest end-to-end path is ``Pipeline(config).run()``:

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
       eval=EvalConfig(metrics=["auroc", "balanced_accuracy"]),
       training=TrainingConfig(epochs=50, learning_rate=1e-4),
   )

   result = Pipeline(config).run()

Modular API
-----------

If you want finer grained control, you can use individual :doc:`building blocks <api>`
directly instead of running the full pipeline in one call. This is useful when you want
to manage preprocessing, feature extraction, training, evaluation, reporting,
or heatmap generation as separate steps in a custom experiment workflow.
