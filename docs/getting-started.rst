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

Optional extras add the dependencies of specific paths: ``[croma]`` for CRoMa
data preparation and ``[hest]`` for HEST curation, for example
``pip install 'soma-pathology[hest]'``.

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

Replace the paths with your slides. ``splits.csv`` assigns each sample to one
split per fold. This minimal example defines one fold:

.. code-block:: text

   sample_id,split,fold
   slide_001,train,0
   slide_002,train,0
   slide_003,tune,0
   slide_004,test,0

Both classes must appear in each scored split for AUROC to be defined. See
:doc:`dataset` for manifest, split, and cross-validation rules.

2. Preprocess and encode
~~~~~~~~~~~~~~~~~~~~~~~~

Choose how to turn tissue into tiles, then select a foundation model. Tiling
and encoding are slide2vec's: soma takes slide2vec's ``PreprocessingConfig``,
which composes hs2p's tiling, segmentation and mask settings. In this example,
feature extraction runs once and returns a reusable feature store.

.. code-block:: python

   from dataclasses import replace

   from hs2p import DEFAULT_MASK_SAMPLING, SegmentationConfig, TilingConfig
   from slide2vec import PreprocessingConfig

   from soma import Cohort, FeatureExtractor, ImageManifest
   from soma.data._legacy import LegacyFolds, records_for_pipeline

   cohort = Cohort.from_csv("dataset.csv", "splits.csv", targets=["label"])
   dataset = records_for_pipeline(cohort, ImageManifest.from_csv("dataset.csv"))
   splits = LegacyFolds.from_cohort(cohort)

   preprocessing = PreprocessingConfig(
       tiling=TilingConfig(
           requested_spacing_um=0.5,
           requested_tile_size_px=224,
           tolerance=0.05,
           overlap=0.0,
       ),
       segmentation=SegmentationConfig(method="hsv"),
       masks=replace(DEFAULT_MASK_SAMPLING, min_coverage={"tissue": 0.2}),
   )

   features = FeatureExtractor(
       dataset,
       "phikon",
       preprocessing=preprocessing,
       output_root="output",
   ).extract()

The result contains ``source`` (the reusable feature reader), ``dataset`` (the
exact samples indexed by that source), ``provenance``, and ``artifacts``. Without
a ``tiling``, the tile size and spacing resolve from the encoder's native
configuration; see :doc:`preprocessing` and :doc:`encoders` for every option.

3. Train and evaluate
~~~~~~~~~~~~~~~~~~~~~

For slide classification, the downstream model pairs an AB-MIL aggregator with a
task head. ``EvalConfig`` selects how its predictions are scored. A
:class:`~soma.Pipeline` given the extracted ``source`` trains on it directly:

.. code-block:: python

   from soma import (
       AggregatorConfig,
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
       aggregator=AggregatorConfig(name="abmil"),
       task=TaskConfig(name="binary_classification"),
       training=TrainingConfig(epochs=5, learning_rate=1e-4, seed=0),
       evaluation=EvalConfig(metrics=["auroc", "balanced_accuracy"]),
   )
   result = Pipeline(config, source=features.source).run()

Each fold saves its model as a bundle that :func:`soma.load` reopens (see
:ref:`model-bundle`).

See :doc:`aggregators`, :doc:`classification`, :doc:`training`, and
:doc:`evaluation` for the available components and settings.

Pipeline and CLI
----------------

:class:`~soma.config.PipelineConfig` runs preprocessing, feature extraction,
training, and evaluation with one call:

.. code-block:: python

   from dataclasses import replace

   from hs2p import DEFAULT_MASK_SAMPLING, SegmentationConfig, TilingConfig
   from slide2vec import PreprocessingConfig

   from soma import (
       AggregatorConfig, EvalConfig, FeaturesConfig, Pipeline, PipelineConfig,
       TaskConfig, TrainingConfig,
   )

   config = PipelineConfig(
       dataset_csv="dataset.csv",
       splits_csv="splits.csv",
       output_root="output",
       dataset_type="slide",
       features=FeaturesConfig(
           encoder="phikon",
           extraction=PreprocessingConfig(
               tiling=TilingConfig(
                   requested_spacing_um=0.5, requested_tile_size_px=224,
                   tolerance=0.05, overlap=0.0,
               ),
               segmentation=SegmentationConfig(method="hsv"),
               masks=replace(DEFAULT_MASK_SAMPLING, min_coverage={"tissue": 0.2}),
           ),
       ),
       aggregator=AggregatorConfig(name="abmil"),
       task=TaskConfig(name="binary_classification"),
       training=TrainingConfig(epochs=5, learning_rate=1e-4, seed=0),
       evaluation=EvalConfig(metrics=["auroc", "balanced_accuracy"]),
   )
   result = Pipeline(config).run()

``result`` exposes ``run_dir``, ``summary``, and ``fold_results`` (see
:doc:`outputs`). The same experiment as ``config.yaml``:

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

.. _log-output:

Log output
----------

soma reports progress through Python's ``logging`` module, under the ``soma``
logger. The ``soma`` command prints soma's INFO messages to stderr; other
libraries print warnings and errors only.

In a Python script, soma checks the logging setup when you create a
``Pipeline`` or a ``FeatureExtractor``:

- If you have not configured logging (no handler on the root logger or on the
  ``soma`` logger), soma adds one stderr handler to the ``soma`` logger and
  sets that logger to INFO. You see messages such as the fold summary without
  any setup. soma adds this handler only once. A level you set yourself is
  kept: on the ``soma`` logger, or on the root logger when it is not Python's
  default WARNING (``logging.getLogger().setLevel(logging.ERROR)`` keeps soma
  quiet).
- If you have configured logging, for example with ``logging.basicConfig``,
  soma changes nothing and your configuration decides what is shown.
- If you configure logging after you create a ``Pipeline``, soma's handler
  stops writing as soon as the root logger or the ``soma`` logger has a handler
  of yours, so each message prints once, through your handlers. The ``soma``
  logger stays at INFO. If you set ``propagate = False`` on the ``soma`` logger,
  its messages never reach your root handlers, so soma's handler keeps printing
  them unless you add a handler to the ``soma`` logger.

To change the output:

.. code-block:: python

   import logging

   soma_logger = logging.getLogger("soma")
   soma_logger.setLevel(logging.WARNING)   # hide soma's INFO messages
   soma_logger.setLevel(logging.DEBUG)     # or show soma's DEBUG messages too
   soma_logger.handlers.clear()            # remove soma's default handler
   soma_logger.propagate = False           # keep soma out of your root handlers

Next, follow the :doc:`slide-level tutorial <tutorials/index>` or build
custom orchestration with the :doc:`API reference <api>`.
