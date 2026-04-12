Getting Started
===============

`soma` lets you move from a dataset of slides and labels to a reproducible
training run with a single configuration object.

Install
-------

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

Practical workflow
------------------

1. Start with a valid dataset and split manifest.
2. Choose an encoder and spacing that match the biological scale of the
   problem.
3. Add a bag aggregator only when the pipeline uses slide-level MIL.
4. Tune the task and training knobs after the data path is stable.
5. Reuse the shared cache whenever the upstream artifacts do not change.
