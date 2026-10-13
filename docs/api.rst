API
===

Use the Python API to extract features once, train downstream models, and
inspect results. Start with :doc:`getting-started` for the complete pipeline;
the recipes here cover custom orchestration.

Main building blocks
--------------------

.. list-table::
   :header-rows: 1

   * - Page
     - Focus
   * - :doc:`Dataset and splits <dataset>`
     - CSV manifest schema and fold assignment rules
   * - :doc:`Preprocessing <preprocessing>`
     - Tissue segmentation and slide tiling at a given spacing
   * - :doc:`Encoders <encoders>`
     - Feature extraction backends
   * - :doc:`Aggregators <aggregators>`
     - MIL pooling and bag-level aggregation
   * - :doc:`Tasks <tasks>`
     - Prediction heads and metric contracts
   * - :doc:`Evaluation <evaluation>`
     - Metric contracts, subgroup analysis, predictions and evaluations
   * - :doc:`Training <training>`
     - Optimization behavior and training defaults

Extract once, cache, and reuse features across experiments
----------------------------------------------------------

Extract a shared feature source, then train each downstream model in its own
run directory. This example uses ``uni2``; see :doc:`encoders` for model access
requirements.

.. code-block:: python

   from dataclasses import replace

   import soma
   from soma import Cohort, ExperimentConfig, ImageManifest, Pipeline
   from soma.config import CacheOptions, CohortConfig, LearnerConfig, RunConfig, TrainerConfig

   # The cohort holds identity, targets and folds; the image manifest holds the paths.
   cohort = Cohort.from_csv("dataset.csv", "splits.csv", targets=["label"])
   images = ImageManifest.from_csv("dataset.csv")
   cache = CacheOptions(enabled=True, root_dir="shared/feature_cache")

   # Features enter through soma.extract: it reads the image manifest and returns the source.
   features = soma.extract(images, encoder="uni2", cache=cache, output_dir="output")

   def experiment(aggregator: dict) -> ExperimentConfig:
       return ExperimentConfig(
           run=RunConfig(output_dir="output"),
           cohort=CohortConfig(records="dataset.csv", splits="splits.csv"),
           learner=LearnerConfig(
               "mil", {"aggregator": aggregator, "head": "binary_classification"}
           ),
           trainer=TrainerConfig(max_epochs=50),
       )

   abmil = experiment({"name": "abmil", "params": {"hidden_dim": 256}})
   clam = experiment({"name": "clam_sb", "params": {"hidden_dim": 256, "attn_dim": 128}})

   abmil_result = Pipeline(abmil, source=features).run()
   clam_result = Pipeline(clam, source=features).run()

Each config trains in its own run directory under ``run.output_dir``; the source
is attached, never re-extracted.

``soma.extract`` is the single way features enter soma; a :class:`~soma.Pipeline`
without a source extracts through it too. It is also the namespace of
:func:`soma.extract.live_encoder`.

.. autofunction:: soma.extract.extract

.. autoclass:: soma.extract.RoiSet
   :members: children, label_maps, pixels, subset

The source can be reused across experiments as long as the upstream dataset,
preprocessing, and encoder settings do not change (see :doc:`getting-started`
for the result fields).

Evaluation settings and attention heatmaps
------------------------------------------

Both live in the ``evaluation`` section. See :doc:`evaluation` for metric
contracts and :doc:`outputs` for the saved attention scores and overlays:

.. code-block:: python

   from soma.config import EvaluationConfig

   evaluation = EvaluationConfig(
       metrics=["auroc", "balanced_accuracy", "f1"],
       subgroups=["center", "grade"],
       attention_heatmaps={"cmap": "coolwarm", "alpha": 0.5},
   )

   result = Pipeline(replace(abmil, evaluation=evaluation), source=features).run()

Dense features over given images
--------------------------------

Dense extraction is an extraction setting, not a manifest kind: ``dense`` asks
for token grids and ``unit="tile"`` says each row is a pre-cropped image. The
grid recipe is slide2vec's ``DenseOptions`` (read spacing, supervision size,
padding, window, feature kind); batch size and precision are slide2vec's
``ExecutionOptions``:

.. code-block:: python

   from slide2vec import DenseOptions, ExecutionOptions

   import soma
   from soma import ImageManifest
   from soma.config import CacheOptions

   grids = soma.extract(
       ImageManifest.from_csv("segmentation.csv"),
       encoder="phikon",
       unit="tile",
       dense=DenseOptions(spacing_um=0.5, target_size=224),
       execution=ExecutionOptions(batch_size=64),
       cache=CacheOptions(enabled=True, root_dir="shared/feature_cache"),
       output_dir="output/dense",
   )

   grid = grids.load("roi_001")  # float32: (feature_dim, grid_h, grid_w)
   geometry = grids.geometry("roi_001")

Annotation-sampled whole slides
-------------------------------

A table of whole-slide image and annotation paths becomes an annotation-sampled
grid extraction when the extraction carries an annotation vocabulary (hs2p's
``MaskSamplingConfig``) and a grid runs over whole slides. The grids are keyed
by ROI id; their ``rois`` name each ROI's slide, from which the slide cohort
derives the ROI cohort (each ROI inherits its slide's splits):

.. code-block:: python

   from hs2p import MaskSamplingConfig, SegmentationConfig, TilingConfig
   from slide2vec import DenseOptions, PreprocessingConfig

   import soma
   from soma import AnnotationManifest, Cohort, ImageManifest
   from soma.config import CacheOptions

   grids = soma.extract(
       ImageManifest.from_csv("slides.csv"),
       annotations=AnnotationManifest.from_csv("slides.csv"),
       encoder="phikon",
       dense=DenseOptions(spacing_um=0.5, target_size=512),
       extraction=PreprocessingConfig(
           tiling=TilingConfig(
               requested_spacing_um=0.5, requested_tile_size_px=512, tolerance=0.05, overlap=0.0
           ),
           segmentation=SegmentationConfig(method="hsv"),
           masks=MaskSamplingConfig(
               pixel_mapping={"background": 0, "tumor": 1},
               min_coverage={"tumor": 0.1},
               strategy="joint",
               output_mode="merged",
           ),
       ),
       cache=CacheOptions(enabled=True, root_dir="shared/feature_cache"),
       output_dir="output/annotation-rois",
   )

   rois = Cohort.from_csv("slides.csv", "splits.csv").derived(grids.rois.children)
   label_maps = grids.rois.label_maps(size=(512, 512))  # the stored mask crops

Slides with no sampled ROI have no child, so they leave the ROI cohort;
``output/annotation-rois/extraction_provenance.json`` lists them.

Score and compare saved predictions
-----------------------------------

Each test split's ``predictions_<split>.parquet`` can be scored again, with
other metrics or subgroups, and compared across runs. Given two single-fold
runs over the same cohort:

.. code-block:: python

   from soma import evaluate
   from soma.evaluation import Predictions, compare

   targets = cohort.targets("test")
   abmil = Predictions.from_parquet(abmil_result.run_dir / "predictions_test.parquet")
   clam = Predictions.from_parquet(clam_result.run_dir / "predictions_test.parquet")

   evaluation = evaluate(abmil, targets, metrics=["auroc", "f1"], subgroups=["center"])
   p_values = compare([abmil, clam], targets, "auroc")

``compare`` gives the best set ``1.0`` and every other set the p-value of a
paired permutation test against it. See :doc:`evaluation` for the scoring
rules and :doc:`benchmarking` for leaderboards across many runs.

``soma.components()`` names the component kinds (aggregators, heads,
decoders, encoders, optimizers, ...); ``soma.components("aggregators")`` lists
one kind's registered components, each with its params and their defaults.
See :ref:`registering-components` for adding your own.

Segmentation and detection building blocks
------------------------------------------

The verbs ``soma.learn.loaders``, ``soma.fit``, ``soma.predict`` and
``soma.evaluate`` also cover feature grids. :ref:`grid-verbs` walks through a
complete run; these are the parts:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Object
     - Role
   * - :class:`soma.data.GridSource`
     - Cached feature grids, one ``(D, h, w)`` grid per sample
   * - :class:`soma.data.TileImageSource`
     - Pixels of pre-cropped tiles for the live path;
       ``TileImageSource.from_manifest(image_manifest, size=...)``
   * - :class:`soma.data.LabelMapSource`, :class:`soma.data.CachedLabelMapSource`
     - Segmentation targets: label masks, or the ROI mask crops stored by
       annotation sampling
   * - :class:`soma.data.PointSource`
     - Detection targets; ``class_remap`` and ``drop`` fold in the class scheme
   * - :func:`soma.extract.live_encoder`, :class:`soma.extract.FrozenEncoder`
     - A parameter-free frozen encoder stage that re-encodes pixels at every
       step
   * - :class:`soma.models.DenseModel`
     - ``encoder`` (optional) → ``adaptor`` (optional) → decoder → head
   * - :class:`soma.learn.DenseLearner`, :class:`soma.learn.GridBatch`
     - The grid learner and its batch (cached ``features`` or live ``pixels``,
       plus raw ``targets``)
   * - :class:`soma.learn.SegmentationAugment`
     - Joint image and label-mask augmentation for the live path
   * - :class:`soma.evaluation.LabelMapPredictions`,
       :class:`soma.evaluation.PointPredictions`
     - What ``soma.predict`` returns for a segmentation or detection head
   * - :func:`soma.evaluation.sweep_threshold`
     - Picks a detection score threshold on tune predictions
   * - :class:`soma.evaluation.SaveOverlays`,
       :class:`soma.evaluation.SaveProbabilities`,
       :class:`soma.evaluation.SaveHeatmaps`
     - ``predict`` callbacks that write overlays, softmax maps and heatmaps
   * - :class:`soma.testing.FakeEncoder`
     - A weight-free encoder stage for tests

Every target source has ``subset(sample_ids)`` to restrict it to one split.

For whole-image segmentation after a run, load each fold's bundle with
:func:`soma.load` and pass the list to
``SlidingWindowSegmentationPredictor.from_bundles(bundles, encoder=...)``, which
slides the fold ensemble over an image. A bundle trained live carries its frozen
encoder stage; for a run trained on cached grids, pass the stage that extracted
them, ``soma.pipeline.build.live_encoder_for(plan)`` with the run's
:class:`~soma.Plan` (``soma.Plan.from_yaml(run_dir / "plan.yaml")``).

Task-free representation evaluation
-----------------------------------

Evaluate frozen tile embeddings with the fixed CRoMa v1 protocol by leaving out
the ``learner`` and setting ``evaluation.representation``. Save this as a YAML
config and run it with ``soma run config.yaml``:

.. code-block:: yaml

   cohort:
     records: dataset.csv
     splits: splits.csv
   features:
     encoder:
       name: phikon
   evaluation:
     representation:
       confounder_column: medical_center
       split: test
       evaluation_design: all
       m: 5
       alpha: 0.10

Selected dataset rows require non-empty ``label``, ``group_id`` (a metadata column
naming the non-independence group), and ``medical_center`` columns. The selected split must occur in exactly one fold;
cross-validation is unsupported. The cohort must provide enough same- and
other-confounder neighbours for ``m=5``; CRoMa raises if any selected sample
lacks them.

Representation runs return metrics in ``result.summary`` with an empty
``fold_results`` list. They fit no task head and write no task report. Provenance
records the installed CRoMa version and encoder configuration, including output
variant. It does not include checkpoint hashes or an upstream fingerprint, so
these records alone cannot establish byte-identical weights or preprocessing.
See :doc:`croma-robustness-benchmark` for the registered cohorts and metrics.

.. _benchmark-api:

Reproduce a packaged benchmark programmatically
-----------------------------------------------

Every registered :doc:`benchmark <benchmarking>` is a Python object, so the
``soma reproduce`` flow is available from code, with results directly
comparable to the CLI's:

.. code-block:: python

   import statistics

   import soma
   from soma.benchmark import get_benchmark
   from soma.pipeline import Pipeline

   [entry["name"] for entry in soma.components("benchmarks")]  # [..., "eva/bach", "hest/IDC", "ocelot"]
   benchmark = get_benchmark("eva/bach")

   manifest = benchmark.curate("/path/to/eva/bach", "runs/eva-bach/curated")

   measured = []
   for seed in benchmark.canonical_seeds:
       seed_root = f"runs/eva-bach/seed_{seed}"
       config = benchmark.build_config(
           encoder="uni2",                 # the axis a benchmark varies
           dataset_csv=manifest.dataset_csv,
           splits_csv=manifest.splits_csv,
           output_root=seed_root,
           seed=seed,
           # share one feature cache across seeds (extraction is seed-independent)
           overrides={"cache": {"enabled": True, "root_dir": "runs/eva-bach/feature_cache"}},
       )
       result = Pipeline(config).run()
       metrics = benchmark.score(result.run_dir)
       measured.append(metrics[benchmark.primary_metric])

   print(statistics.fmean(measured))

``benchmark.expected(encoder="uni2")`` returns the packaged reference rows to
compare against, and ``benchmark.score(run_dir)`` alone re-scores an existing run
without retraining (the ``--from-run-dir`` fast path).

The one-call equivalent is ``soma.benchmark.run_benchmark``, the importable
orchestration behind ``soma reproduce`` itself: the canonical-seed loop, the
reference-row tolerance status, provenance stamping (git commit, slide2vec/croma
versions), and the results-ledger append used by the CLI. Its keywords
mirror the CLI flags, plus ``results_root`` so an external repository can append
``MeasuredRow`` rows to its own committed ledger instead of the in-package one:

.. code-block:: python

   from soma.benchmark import run_benchmark

   run_benchmark(
       "eva/bach",
       encoder="uni2",
       raw_root="/path/to/eva/bach",
       output_root="runs/eva-bach",
       record=True,
       # host the results ledger outside the soma checkout:
       # appends to <results_root>/eva.csv with full provenance
       results_root="/path/to/leaderboard-repo/results",
   )

Run an external benchmark specification
---------------------------------------

A project outside soma can own its benchmark protocol as a typed
``soma.benchmark.BenchmarkSpec`` — a config builder, the canonical seed set,
and a scorer — and hand it to ``soma.benchmark.run_benchmark_spec`` for
execution. soma runs every canonical seed with one shared feature cache and
returns the aggregated ``BenchmarkRunResult`` together with the per-seed
evidence roots, without touching the registry, reference tables, or ledger.

Runnable demonstrations live in ``examples/``
(`examples/README.md <https://github.com/clemsgrs/soma/blob/main/examples/README.md>`_):

* ``examples/external_benchmark_spec.py`` — build and execute a ``BenchmarkSpec``;
* ``examples/fixed_step_training.py`` — fixed optimizer-step training budgets
  (``TrainerConfig(max_steps=N)``);
* ``examples/aggregator_resolution.py`` — resolve one fixed MIL recipe for
  tile- and slide-level encoders with ``soma.models.encoders.resolve_aggregator``;
* ``examples/portable_identity.py`` — manifest identity is portable across
  storage roots.
