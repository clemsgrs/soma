Getting started
===============

Run slide-level binary classification with a frozen ``phikon`` encoder and an
AB-MIL aggregator. A preset gives a complete experiment from a few arguments;
the modular API exposes each step; the pipeline and CLI examples express the
same experiment in one configuration.

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

.. _presets:

Presets
-------

A preset is a function that returns a complete experiment for a common task
from a few arguments. It needs a ``dataset.csv`` and a ``splits.csv`` (see
:ref:`define-the-data`). soma ships five presets:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Preset
     - Experiment
   * - ``mil_classification(encoder, aggregator="abmil")``
     - Whole-slide classification: tiles embedded by a frozen encoder, pooled
       by a MIL aggregator, a classification head.
   * - ``tile_classification(encoder)``
     - Tile-image classification: a linear head on each image's embedding.
   * - ``survival(encoder, aggregator)``
     - Whole-slide survival: a discrete-time hazard head over a MIL bag,
       trained on the cohort's ``time``, ``event`` and ``bin`` columns.
   * - ``segmentation(encoder, decoder)``
     - Semantic segmentation: a decoder over the encoder's dense token grid.
   * - ``detection(encoder, decoder)``
     - Cell detection: a decoder predicting one peak heatmap per class.

Run a preset on your files from the command line:

.. code-block:: bash

   soma run --preset mil_classification --encoder phikon \
       --cohort dataset.csv --splits splits.csv --out output

``--seed`` sets the seed; the other flags are the preset's own arguments.
``soma list presets`` lists the presets with their arguments and defaults. A
preset has no override flags. To change anything else, print the preset as an
editable YAML file with every default filled in, edit it, and run the file:

.. code-block:: bash

   soma init mil_classification --encoder phikon > experiment.yaml
   # point cohort.records and cohort.splits at your files, edit the rest
   soma run experiment.yaml

In Python, a preset returns an :class:`~soma.ExperimentConfig`. Change it by
assigning its fields:

.. code-block:: python

   from soma import Pipeline, presets

   config = presets.mil_classification("phikon")
   config.cohort.records = "dataset.csv"
   config.cohort.splits = "splits.csv"
   config.run.output_dir = "output"
   config.trainer.max_epochs = 20
   result = Pipeline(config).run()

The segmentation and detection presets cannot know your classes: before a run,
set the head's ``classes`` (or ``num_classes``), and for detection its
``match_distance`` in µm. See :doc:`segmentation` and :doc:`detection`.

Modular API
-----------

Use the modular API to inspect intermediate results, reuse extracted features,
or sweep one component without rerunning the others.

.. _define-the-data:

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

   from hs2p import MaskSamplingConfig, SegmentationConfig, TilingConfig
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
       masks=MaskSamplingConfig(
           pixel_mapping={"background": 0, "tissue": 1},
           min_coverage={"tissue": 0.2},
       ),
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

An :class:`~soma.ExperimentConfig` describes the rest of the experiment in seven
sections, each holding one public callable's keyword arguments: ``run``
(:class:`~soma.Pipeline`), ``cohort``, ``features``, ``loaders``, ``learner``,
``trainer`` and ``evaluation``. For slide classification the learner is ``mil``,
whose model is an AB-MIL aggregator and a task head. Each component is a
``{name, params}`` leaf; a bare name keeps the component's default params. A
:class:`~soma.Pipeline` given the extracted ``source`` trains on it directly:

.. code-block:: python

   from soma import ExperimentConfig, Pipeline
   from soma.config import (
       CohortConfig,
       EvaluationConfig,
       LearnerConfig,
       RunConfig,
       TrainerConfig,
   )

   config = ExperimentConfig(
       run=RunConfig(output_dir="output", seed=0),
       cohort=CohortConfig(records="dataset.csv", splits="splits.csv"),
       learner=LearnerConfig(
           "mil",
           {
               "aggregator": "abmil",
               "head": "binary_classification",
               "optimizer": {"name": "adam", "params": {"lr": 1e-4}},
           },
       ),
       trainer=TrainerConfig(max_epochs=5),
       evaluation=EvaluationConfig(metrics=["auroc", "balanced_accuracy"]),
   )
   result = Pipeline(config, source=features.source).run()

Each fold saves its model as a bundle that :func:`soma.load` reopens (see
:ref:`model-bundle`).

See :doc:`aggregators`, :doc:`classification`, :doc:`training`, and
:doc:`evaluation` for the available components and settings.

Pipeline and CLI
----------------

With a ``features`` section naming the encoder and the extraction, the same
:class:`~soma.Pipeline` runs preprocessing, feature extraction, training, and
evaluation with one call:

.. code-block:: python

   from hs2p import MaskSamplingConfig, SegmentationConfig, TilingConfig
   from slide2vec import PreprocessingConfig

   from soma import ExperimentConfig, Pipeline
   from soma.config import (
       CohortConfig,
       EvaluationConfig,
       FeaturesConfig,
       LearnerConfig,
       RunConfig,
       TrainerConfig,
   )

   config = ExperimentConfig(
       run=RunConfig(output_dir="output", seed=0),
       cohort=CohortConfig(records="dataset.csv", splits="splits.csv"),
       features=FeaturesConfig(
           encoder="phikon",
           extraction=PreprocessingConfig(
               tiling=TilingConfig(
                   requested_spacing_um=0.5, requested_tile_size_px=224,
                   tolerance=0.05, overlap=0.0,
               ),
               segmentation=SegmentationConfig(method="hsv"),
               masks=MaskSamplingConfig(
                   pixel_mapping={"background": 0, "tissue": 1},
                   min_coverage={"tissue": 0.2},
               ),
           ),
       ),
       learner=LearnerConfig(
           "mil",
           {
               "aggregator": "abmil",
               "head": "binary_classification",
               "optimizer": {"name": "adam", "params": {"lr": 1e-4}},
           },
       ),
       trainer=TrainerConfig(max_epochs=5),
       evaluation=EvaluationConfig(metrics=["auroc", "balanced_accuracy"]),
   )
   result = Pipeline(config).run()

``result`` exposes ``run_dir``, ``summary``, and ``fold_results`` (see
:doc:`outputs`). The same experiment as ``config.yaml`` spells every field the
way the Python call does:

.. code-block:: yaml

   run:
     output_dir: output
     seed: 0
   cohort:
     records: dataset.csv
     splits: splits.csv
   features:
     encoder:
       name: phikon
     extraction:
       tiling:
         requested_spacing_um: 0.5
         requested_tile_size_px: 224
         tolerance: 0.05
         overlap: 0.0
       segmentation:
         method: hsv
       masks:
         pixel_mapping: {background: 0, tissue: 1}
         min_coverage: {tissue: 0.2}
   learner:
     name: mil
     params:
       aggregator:
         name: abmil
       head:
         name: binary_classification
       optimizer:
         name: adam
         params: {lr: 1.0e-4}
   trainer:
     max_epochs: 5
   evaluation:
     metrics: [auroc, balanced_accuracy]

Then run:

.. code-block:: bash

   soma run config.yaml

A field left out keeps its default. soma validates the file, then resolves it
against the cohort into a *plan*: the config with every default spelled out and
what the cohort decides (for example the label map) filled in.
:func:`soma.resolve` does the same in Python. The run directory keeps your
``config.yaml`` next to the resolved ``plan.yaml``. See :doc:`configuration` for
the sections and components, and :doc:`cli` for the full command reference.

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
