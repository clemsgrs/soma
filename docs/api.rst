API
===

`soma` exposes a modular public API that can be used either end to end or one
piece at a time. For a quick tour of the end-to-end orchestration — from
manifests to reports — before diving in, see :doc:`How soma works <how-soma-works>`.

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
     - Metric contracts, subgroup analysis, and evaluation results
   * - :doc:`Training <training>`
     - Optimization behavior and training defaults
   * - :doc:`Reporting <reporting>`
     - Report contents, subgroup analysis, and comparison statistics

Examples
--------

The examples below show the most common entry points.

Extract once, cache, and reuse features across experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the most common modular workflow when you want to compare several task
heads or aggregators against the same encoder output:

.. code-block:: python

   from soma import (
       AggregatorConfig,
       CacheConfig,
       EncoderConfig,
       FeatureExtractor,
       TaskConfig,
       TrainingConfig,
       Dataset,
       Splits,
       train,
   )

   dataset = Dataset("dataset.csv")
   splits = Splits("splits.csv", dataset)
   encoder = EncoderConfig(name="uni2")
   cache = CacheConfig(enabled=True, root_dir="shared/feature_cache")

   extractor = FeatureExtractor(
       dataset=dataset,
       encoder=encoder,
       cache=cache,
       output_root="output",
   )
   features = extractor.extract()
   effective_splits = splits.project(features.dataset)
   task = TaskConfig(name="binary_classification")
   training = TrainingConfig(epochs=50, learning_rate=1e-4)
   abmil_aggregator = AggregatorConfig(name="abmil", params={"hidden_dim": 256})
   clam_aggregator = AggregatorConfig(name="clam_sb", params={"hidden_dim": 256, "attn_dim": 128})

   abmil_result = train(
       feature_store=features.source,
       dataset=features.dataset,
       splits=effective_splits,
       task=task,
       training=training,
       aggregator=abmil_aggregator,
       run_dir="output/abmil/uni2",
   )

   clam_result = train(
       feature_store=features.source,
       dataset=features.dataset,
       splits=effective_splits,
       task=task,
       training=training,
       aggregator=clam_aggregator,
       run_dir="output/clam_sb/uni2",
   )

``extract()`` takes no arguments and returns an immutable result containing the
feature source, the effective dataset indexed by that source, provenance, and
artifact paths. The source can be reused across experiments as long as the
upstream dataset, preprocessing, and encoder settings do not change. Cache and
artifact locations are fixed by constructor configuration.

Dense features over given images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the task-specific manifest type to select dense extraction. Geometry remains
part of ``PreprocessingConfig``:

.. code-block:: python

   from soma import (
       CacheConfig,
       EncoderConfig,
       FeatureExtractor,
       PreprocessingConfig,
       SegmentationManifest,
   )

   dataset = SegmentationManifest("segmentation.csv")
   features = FeatureExtractor(
       dataset,
       EncoderConfig(name="phikon", batch_size=64),
       preprocessing=PreprocessingConfig(
           requested_tile_size_px=224,
           requested_spacing_um=0.5,
       ),
       cache=CacheConfig(enabled=True, root_dir="shared/feature_cache"),
       output_root="output/dense",
   ).extract()

   grid = features.source.load("roi_001")  # float32: (feature_dim, grid_h, grid_w)
   geometry = features.source.geometry("roi_001")

``EncoderConfig.batch_size`` controls encoder inference batches. It is distinct
from ``TrainingConfig.batch_size``, which controls downstream training batches.

Annotation-sampled whole slides
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A segmentation manifest containing whole-slide image and annotation paths becomes
an annotation-sampled extraction when masks and sampling are configured. The result's
dataset is the persisted ROI manifest; parent-slide splits are projected explicitly:

.. code-block:: python

   from soma import (
       CacheConfig,
       EncoderConfig,
       FeatureExtractor,
       MasksConfig,
       PreprocessingConfig,
       SamplingConfig,
       SegmentationManifest,
       Splits,
   )

   slides = SegmentationManifest("slides.csv")
   slide_splits = Splits("splits.csv", slides)
   features = FeatureExtractor(
       slides,
       EncoderConfig(name="phikon"),
       preprocessing=PreprocessingConfig(
           requested_tile_size_px=512,
           requested_spacing_um=0.5,
           masks=MasksConfig(
               pixel_mapping={"background": 0, "tumor": 1},
               min_coverage={"tumor": 0.1},
           ),
           sampling=SamplingConfig(strategy="joint", output_mode="merged"),
       ),
       cache=CacheConfig(enabled=True, root_dir="shared/feature_cache"),
       output_root="output/annotation-rois",
   ).extract()

   roi_splits = slide_splits.project(features.dataset)
   print(features.artifacts.dataset_csv)
   print(features.provenance.zero_roi_sample_ids)

Slides with no sampled ROI are not represented by fake samples or empty tensors.
They are recorded in provenance, while the sampling cache preserves the zero outcome
for identical reruns.

Breaking-change migration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Before
     - Now
   * - ``TileFeatureExtractor(...).run(feature_dir)``
     - ``FeatureExtractor(TileDataset(...), ..., output_root=...).extract()``
   * - ``DenseTileFeatureExtractor(...).run(feature_dir)``
     - ``FeatureExtractor(SegmentationManifest(...) or DetectionManifest(...), ...).extract()``
   * - ``SlideManifestDenseExtractor`` or private pipeline ROI orchestration
     - ``FeatureExtractor(SegmentationManifest(...), preprocessing=PreprocessingConfig(masks=..., sampling=...), ...).extract()``
   * - ``FeatureExtractor.preprocess()`` then ``FeatureExtractor.run(...)``
     - Configure the constructor fully, then call argument-free ``extract()``
   * - Extractor returns a feature store
     - Use ``result.source``; ``result.dataset`` is the exact indexed dataset
   * - Persist a derived ROI split CSV
     - Use ``original_splits.project(result.dataset)`` in memory

Train with explicit evaluation settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want a more explicit :doc:`evaluation contract <evaluation>`, define the evaluation
config up front and pass it through the pipeline or the lower-level training API.
Subgroup columns are included in the run outputs and summarized in the report:

.. code-block:: python

   from soma import EvalConfig, SubgroupConfig

   evaluation = EvalConfig(
       metrics=["auroc", "balanced_accuracy", "f1"],
       subgroups=SubgroupConfig(columns=["center", "grade"]),
   )

   result = train(
       feature_store=features.source,
       dataset=features.dataset,
       splits=effective_splits,
       task=task,
       training=training,
       aggregator=aggregator,
       evaluation=evaluation,
       run_dir="output/abmil/uni2",
   )

Generate a report for one run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``generate_report`` to generate a :doc:`report <reporting>` from saved artifacts,
rendering key results (e.g., loss curves and evaluation metrics) in an HTML view:

.. code-block:: python

   from soma.reporting import generate_report, generate_report_from_result

   report_dir = "output/abmil/uni2"
   report_path = generate_report(report_dir)

Compare multiple runs
~~~~~~~~~~~~~~~~~~~~~

Use ``compare_runs`` to generate a cross-run comparison report:

.. code-block:: python

   from soma.reporting import compare_runs

   abmil_run_dir = "output/abmil/uni2"
   transmil_run_dir = "output/transmil/uni2"

   comparison_path = compare_runs(
       [abmil_run_dir, transmil_run_dir],
       labels=["ABMIL", "TransMIL"],
   )

The report is written to ``<shared output_root>/comparisons/<comparison-id>/index.html``
unless you pass ``output_dir`` explicitly.

Discover available presets programmatically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the public discovery helpers to list currently registered presets:

.. code-block:: python

   from soma import (
       list_aggregators,
       list_decoders,
       list_models,
       list_pixel_classifiers,
       list_task_heads,
   )

   tile_encoders = list_models(level="tile")
   aggregators = list_aggregators()
   decoders = list_decoders()
   pixel_classifiers = list_pixel_classifiers()
   task_heads = list_task_heads()

For more detail on what the generated HTML report contains, how subgroup
analysis is summarized, and how comparison statistics are computed, see the
:doc:`reporting guide <reporting>`.

Enable heatmaps when you want attention overlays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Attention heatmaps are controlled through ``HeatmapConfig`` and passed through
``train(...)``. This is most useful for attention-based aggregators that
expose per-tile scores. The saved overlays and raw attention scores are
documented in :doc:`outputs`:

.. code-block:: python

   from soma import HeatmapConfig

   heatmaps = HeatmapConfig(enabled=True, cmap="coolwarm", alpha=0.5)

   result = train(
       feature_store=features.source,
       dataset=features.dataset,
       splits=effective_splits,
       task=task,
       training=training,
       aggregator=aggregator,
       evaluation=evaluation,
       heatmaps=heatmaps,
       run_dir="output/abmil/uni2",
   )

   # attention scores land in fold_N/attention/
   # rendered attention overlays in fold_N/heatmaps/

.. _benchmark-api:

Reproduce a packaged benchmark programmatically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every registered :doc:`benchmark <benchmarking>` is a Python object, so the
``soma reproduce`` flow is available from code: discover benchmarks, curate the
data, build the fixed config per seed, run the pipeline, and score. This is the
same protocol the CLI drives, so results are directly comparable:

.. code-block:: python

   import statistics

   from soma.benchmarks import get_benchmark, list_benchmarks
   from soma.pipeline import Pipeline

   list_benchmarks()                       # ["ocelot", "eva/bach", "hest/IDC", ...]
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
       Pipeline(config).run()
       metrics = benchmark.score(seed_root)
       measured.append(metrics[benchmark.primary_metric])

   print(statistics.fmean(measured))

``benchmark.expected(encoder="uni2")`` returns the packaged reference rows to
compare against, and ``benchmark.score(run_dir)`` alone re-scores an existing run
without retraining (the ``--from-run-dir`` fast path). See :doc:`benchmarking`
for the CLI equivalents and :doc:`outputs` for the artifacts each run writes.

The one-call equivalent is ``soma.benchmarks.run_benchmark``, the importable
orchestration behind ``soma reproduce`` itself: the canonical-seed loop, the
reference-row tolerance status, provenance stamping (git commit, slide2vec/croma
versions), and the results-ledger append, byte-identical to the CLI. Its keywords
mirror the CLI flags, plus ``results_root`` so an external repository can append
``MeasuredRow`` rows to its own committed ledger instead of the in-package one:

.. code-block:: python

   from soma.benchmarks import run_benchmark

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A project outside soma can own its benchmark protocol as a typed
``soma.benchmarks.BenchmarkSpec`` — a config builder, the canonical seed set,
and a scorer — and hand it to ``soma.benchmarks.run_benchmark_spec`` for
execution. soma runs every canonical seed with one shared feature cache and
returns the aggregated ``BenchmarkRunResult`` together with the per-seed
evidence roots, without touching the registry, reference tables, or ledger.

Runnable demonstrations live in ``examples/``
(`examples/README.md <https://github.com/clemsgrs/soma/blob/main/examples/README.md>`_):

* ``examples/external_benchmark_spec.py`` — build and execute a ``BenchmarkSpec``;
* ``examples/fixed_step_training.py`` — fixed optimizer-step training budgets
  (``TrainingConfig(max_steps=N, epochs=None)``);
* ``examples/aggregator_resolution.py`` — resolve one fixed MIL recipe for
  tile- and slide-level encoders with ``soma.encoders.resolve_aggregator``;
* ``examples/portable_identity.py`` — manifest identity is portable across
  storage roots.
