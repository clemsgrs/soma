Getting started
===============

Run slide-level binary classification with a frozen ``phikon`` encoder and an
AB-MIL aggregator. The modular API exposes each step; the pipeline and CLI
examples express the same experiment in one configuration.

Install
-------

soma requires Python 3.11 or later:

.. code-block:: bash

   pip install soma-pathology

Optional extras add the dependencies of specific paths: ``[pixel]`` for the
``xgboost`` pixel classifier, ``[croma]`` for CRoMa data preparation,
``[hest]`` for HEST curation, for example ``pip install 'soma-pathology[pixel]'``.

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
   slide_004,/path/to/slides/slide_004.svs,1
   slide_005,/path/to/slides/slide_005.svs,0
   slide_006,/path/to/slides/slide_006.svs,1

Replace the paths with your slides. ``splits.csv`` assigns each sample to one
split per fold. This minimal example defines one fold:

.. code-block:: text

   sample_id,split,fold
   slide_001,train,0
   slide_002,train,0
   slide_003,tune,0
   slide_004,tune,0
   slide_005,test,0
   slide_006,test,0

Use independent subjects across splits and enough samples for meaningful
evaluation. Both classes must appear in each scored split for AUROC to be
defined. For cross-validation, repeat every sample's assignment for each fold
using consecutive fold numbers from 0. See :doc:`dataset` for manifest and split
rules.

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
       tissue_method="hsv",
       min_coverage={"tissue": 0.2},
       overlap=0.0,
       requested_spacing_um=0.5,
       requested_tile_size_px=224,
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

Use :class:`~soma.config.PipelineConfig` to run preprocessing, feature extraction,
training, and evaluation with one call:

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

For the CLI, save the equivalent configuration as ``config.yaml``:

.. code-block:: yaml

   run:
     output_root: output
     seed: 0
   data:
     dataset_csv: dataset.csv
     splits_csv: splits.csv
     dataset_type: slide
   preprocessing:
     tissue_method: hsv
     min_coverage: {tissue: 0.2}
     overlap: 0.0
     requested_spacing_um: 0.5
     requested_tile_size_px: 224
   encoder:
     name: phikon
   aggregation:
     name: abmil
   task:
     name: binary_classification
   training:
     epochs: 5
     learning_rate: 1.0e-4
   evaluation:
     metrics: [auroc, balanced_accuracy]

Then run:

.. code-block:: bash

   soma config.yaml

soma merges the file with bundled defaults and validates it before running.
YAML uses ``aggregation`` where the Python constructor uses ``aggregator``.
See :doc:`cli` for defaults and command-line overrides.

Go further
----------

* Follow the :doc:`slide-level tutorial <tutorials/slide-level>` for a deeper,
  hands-on MIL workflow.
* Use the :doc:`API reference <api>` to build custom orchestration around
  individual components.
* Explore :doc:`modeling` to choose a path for other input and prediction types.
