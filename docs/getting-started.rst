Getting started
===============

This is a compact slide-level classification walkthrough. It first builds one
AB-MIL experiment with soma's modular API, then runs the same experiment through
the pipeline and CLI. The goal is to show how the pieces fit together. The
:doc:`slide-level tutorial <tutorials/slide-level>` covers more hands-on and
advanced workflows.

Install
-------

soma requires Python 3.11 or later:

.. code-block:: bash

   pip install soma-pathology

The first run downloads the selected model weights. Feature extraction is
faster on a GPU, but a GPU is not required. ``phikon`` uses publicly available
weights and needs no access token.

Modular API
-----------

Use the modular API to inspect intermediate results, reuse extracted features,
or sweep one component without rerunning the others.

1. Define the data
~~~~~~~~~~~~~~~~~~

``dataset.csv`` assigns each slide a stable ID, image path, and binary label:

.. code-block:: text

   sample_id,image_path,label
   slide_001,/path/to/slides/slide_001.svs,0
   slide_002,/path/to/slides/slide_002.svs,1
   slide_003,/path/to/slides/slide_003.svs,0
   ...

``splits.csv`` assigns every sample to one split in each fold. This excerpt
shows part of a five-fold definition. The complete file repeats every sample
for folds 0 through 4.

.. code-block:: text

   sample_id,split,fold
   slide_001,test,0
   slide_002,tune,0
   slide_003,train,0
   ...
   slide_001,train,4
   slide_002,tune,4
   slide_003,test,4
   ...

Each fold is an independent assignment. Every sample
appears once per fold as train, tune, or test. See :doc:`dataset` for the
complete manifest contract and split rules.

2. Preprocess and encode
~~~~~~~~~~~~~~~~~~~~~~~~

Choose how to turn tissue into tiles, then select a foundation model. In
this example, feature extraction runs once and returns a reusable feature
store.

.. code-block:: python

   from soma import (
       Dataset,
       EncoderConfig,
       FeatureExtractor,
       PreprocessingConfig,
       Splits,
   )

   dataset = Dataset("dataset.csv")
   splits = Splits("splits.csv", dataset)

   preprocessing = PreprocessingConfig(
       tissue_method="hsv", # tissue detection method
       min_coverage={"tissue": 0.2}, # minimum tissue coverage for a tile to be kept
       overlap=0.0, # fraction of tile overlap (0.0 means no overlap, 0.5 means 50% overlap)
       requested_spacing_um=0.5, # microns per pixel
       requested_tile_size_px=224, # tile width and height in pixels
   )
   encoder = EncoderConfig(
       name="phikon",
   )

   features = FeatureExtractor(
       dataset,
       encoder,
       preprocessing=preprocessing,
       output_root="output",
   ).extract()

The immutable result contains ``source`` (the reusable feature reader), ``dataset``
(the exact samples indexed by that source), ``provenance``, and ``artifacts``.
``extract()`` takes no arguments; ``output_root`` and ``CacheConfig`` fully determine
the artifact and cache locations.

These values match ``phikon``'s native configuration. See :doc:`preprocessing`
and :doc:`encoders` for every option.

.. note::

   If tile size or spacing is omitted, it is resolved from the encoder's native
   configuration automatically.

3. Train and evaluate
~~~~~~~~~~~~~~~~~~~~~

For slide classification, the downstream model pairs an AB-MIL aggregator with a
task head. ``EvalConfig`` selects how its predictions are scored.

.. code-block:: python

   from soma import (
       AggregatorConfig,
       EvalConfig,
       TaskConfig,
       TrainingConfig,
       train,
   )

   task = TaskConfig(name="binary_classification")
   aggregator = AggregatorConfig(name="abmil")
   evaluation = EvalConfig(metrics=["auroc", "balanced_accuracy"])
   training = TrainingConfig(epochs=5, learning_rate=1e-4, seed=0)

   result = train(
       feature_store=features.source,
       dataset=features.dataset,
       splits=splits.project(features.dataset),
       dataset_type="slide",
       aggregator=aggregator,
       task=task,
       training=training,
       evaluation=evaluation,
       run_dir="output/abmil",
   )
   print(result.summary)

See :doc:`aggregators`, :doc:`classification`, :doc:`training`, and
:doc:`evaluation` for the available components and settings.

Pipeline and CLI
----------------

The same experiment can be expressed as one
:class:`~soma.config.PipelineConfig`. With a single call to
:meth:`~soma.pipeline.Pipeline.run`, soma performs every modular step above:
preprocessing, feature extraction, training, and evaluation. Use the modular
API to debug or customize individual stages. Use the pipeline (or the CLI) for
scalable runs.

.. code-block:: python

   from soma import (
       AggregatorConfig,
       EncoderConfig,
       EvalConfig,
       Pipeline,
       PipelineConfig,
       PreprocessingConfig,
       TaskConfig,
       TrainingConfig,
   )

   config = PipelineConfig(
       dataset_csv="dataset.csv",
       splits_csv="splits.csv",
       output_root="output",
       dataset_type="slide",
       preprocessing=PreprocessingConfig(
           tissue_method="hsv",
           min_coverage={"tissue": 0.2},
           overlap=0.0,
           requested_spacing_um=0.5,
           requested_tile_size_px=224,
       ),
       encoder=EncoderConfig(name="phikon"),
       aggregator=AggregatorConfig(name="abmil"),
       task=TaskConfig(name="binary_classification"),
       training=TrainingConfig(epochs=5, learning_rate=1e-4, seed=0),
       evaluation=EvalConfig(metrics=["auroc", "balanced_accuracy"]),
   )

   result = Pipeline(config).run()

``result.run_dir`` locates the run bundle, ``result.summary`` holds aggregate
metrics, and ``result.fold_results`` holds per-fold results. See :doc:`outputs`
for the saved configuration, predictions, metrics, and reports.

We also ship a simple **CLI** that runs the same pipeline from its YAML representation:

.. code-block:: bash

   soma config.yaml

soma validates ``config.yaml`` as a ``PipelineConfig`` before running it. See
the :doc:`CLI reference <cli>` for the YAML schema and command surface.

Go further
----------

* Follow the :doc:`slide-level tutorial <tutorials/slide-level>` for a deeper,
  hands-on MIL workflow.
* Use the :doc:`API reference <api>` to build custom orchestration around
  individual components.
* Explore :doc:`modeling` to choose a path for other input and prediction types.
