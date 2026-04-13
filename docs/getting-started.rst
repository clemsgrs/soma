Getting Started
===============

`soma` lets you move from a dataset of slides and labels to a reproducible
training run with a single configuration object.

Install
-------

Requires Python 3.11 or later.

.. code-block:: bash

   pip install soma

Run a full pipeline
-------------------

The simplest end-to-end path is ``Pipeline(config).run()``:

.. code-block:: python

   from soma import AggregatorConfig, EncoderConfig, Pipeline, PipelineConfig, TaskConfig, TrainingConfig

   config = PipelineConfig(
       dataset_csv="dataset.csv",
       splits_csv="splits.csv",
       output_root="output",
       dataset_type="slide",
       encoder=EncoderConfig(name="uni2"),
       aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 256}),
       task=TaskConfig(name="binary_classification"),
       training=TrainingConfig(epochs=50, learning_rate=1e-4),
   )

   result = Pipeline(config).run()

Use the CLI when you want a YAML entrypoint:

.. code-block:: bash

   soma run examples/reference.yaml

Dataset format
--------------

Two CSV files describe the data:

``dataset.csv``
  Required columns: ``sample_id``, ``image_path``, ``label``.
  Optional columns: ``mask_path`` (pre-computed tissue mask), ``patient_id``
  (required for ``dataset_type="patient"``).
  Any additional columns are carried along as per-sample metadata.

``splits.csv``
  Required columns: ``fold``, ``sample_id``, ``split``.
  Valid split names: ``train``, ``tune``, or any name starting with ``test``
  (e.g. ``test``, ``test_external``).

Practical workflow
------------------

1. Start with a valid dataset and split manifest.
2. Choose an encoder and spacing that match the biological scale of the
   problem.
3. Add a bag aggregator only when the pipeline uses slide-level MIL.
4. Tune the task and training knobs after the data path is stable.
5. Reuse the shared cache whenever the upstream artifacts do not change.
