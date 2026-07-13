Getting started
===============

This first run takes six labelled image tiles through encoding, prediction, and
evaluation, then writes a reproducible result bundle. It uses public DINOv2
weights so the example needs no model access token; afterward, changing
``encoder.name`` is enough to run the same workflow with a registered pathology
foundation model such as UNI2 or Virchow2.

Prerequisites
-------------

- Python 3.11 or later.
- Internet access on the first run to download the public
  ``dinov2-vitb14`` model weights. A GPU is recommended but not required.

Install ``soma`` in your environment:

.. code-block:: bash

   pip install soma-pathology

1. Create the manifests
-----------------------

``dataset.csv`` identifies each RGB tile and its binary label. Paths are
resolved from the directory where you launch ``soma``.

.. code-block:: text

   sample_id,image_path,label
   train_0,tiles/train_0.png,0
   train_1,tiles/train_1.png,1
   tune_0,tiles/tune_0.png,0
   tune_1,tiles/tune_1.png,1
   test_0,tiles/test_0.png,0
   test_1,tiles/test_1.png,1

``splits.csv`` assigns the same samples to one fold:

.. code-block:: text

   sample_id,fold,split
   train_0,0,train
   train_1,0,train
   tune_0,0,tune
   tune_1,0,tune
   test_0,0,test
   test_1,0,test

See :doc:`dataset` for slide, patient, dense-prediction, and cross-validation
manifest contracts.

2. Write the configuration
--------------------------

Save this as ``config.yaml`` beside the two manifests:

.. code-block:: yaml

   run:
     output_root: runs

   data:
     dataset_csv: dataset.csv
     splits_csv: splits.csv
     dataset_type: tile

   encoder:
     name: dinov2-vitb14

   task:
     name: binary_classification

   evaluation:
     metrics: [auroc, balanced_accuracy]

   training:
     epochs: 5
     batch_size: 16

The YAML groups settings by concern; ``soma`` validates and translates these
sections into a :class:`~soma.config.PipelineConfig`. See :doc:`configuration`
for every option and :doc:`cli` for the command reference.

3. Run
------

.. code-block:: bash

   soma config.yaml

The command prints the run directory. Inside it, expect:

- ``config.yaml`` -- the resolved configuration
- ``best_model.pt`` and ``training_history.json`` -- training state
- ``predictions_tune.csv`` and ``predictions_test.csv`` -- sample predictions
- ``metrics.json`` and ``summary.json`` -- fold and run metrics
- ``report.html`` -- the rendered evaluation report

See :doc:`outputs` for the complete artifact contract.

Python equivalent
-----------------

The same run can be launched through the public API:

.. code-block:: python

   from soma import EncoderConfig, EvalConfig, Pipeline, PipelineConfig, TaskConfig, TrainingConfig

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

Continue with :doc:`pipeline` to choose a pipeline mode or with the runnable
:doc:`workflow tutorials <tutorials/index>` for larger experiments.
