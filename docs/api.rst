API Reference
=============

`soma` exposes a modular public API that can be used either end to end or one
piece at a time.

Main building blocks
--------------------

.. list-table::
   :header-rows: 1

   * - Page
     - Focus
   * - :doc:`Pipeline <pipeline>`
     - End-to-end orchestration from manifests to reports
   * - :doc:`Preprocessing <preprocessing>`
     - Tiling, spacing, and geometry
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

   from soma import Dataset, Splits
   from soma import FeatureExtractor, train
   from soma import CacheConfig, EncoderConfig, AggregatorConfig, EvalConfig, TaskConfig, TrainingConfig

   dataset = Dataset("dataset.csv")
   splits = Splits("splits.csv", dataset)

   extractor = FeatureExtractor(
       dataset=dataset,
       encoder=EncoderConfig(name="uni2"),
       cache=CacheConfig(enabled=True, root_dir="shared/feature_cache"),
       output_root="output",
   )
   store = extractor.extract(feature_dir="output/features/uni2")

   abmil_result = train(
       feature_store=store,
       dataset=dataset,
       splits=splits,
       task=TaskConfig(name="binary_classification"),
       training=TrainingConfig(epochs=50, learning_rate=1e-4),
       aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 256}),
       run_dir="output/abmil/uni2",
   )

   clam_result = train(
       feature_store=store,
       dataset=dataset,
       splits=splits,
       task=TaskConfig(name="binary_classification"),
       training=TrainingConfig(epochs=50, learning_rate=1e-4),
       aggregator=AggregatorConfig(name="clam_sb", params={"hidden_dim": 256, "attn_dim": 128}),
       run_dir="output/clam_sb/uni2",
   )

The returned ``FeatureStore`` can be reused across experiments as long as the
upstream preprocessing and encoder settings do not change.

Train with explicit evaluation settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want a more explicit evaluation contract, define the evaluation config
up front and pass it through the pipeline or the lower-level training API.
Subgroup columns are included in the run outputs and summarized in the report:

.. code-block:: python

   from soma import EvalConfig, SubgroupConfig

   eval_config = EvalConfig(
       metrics=["auroc", "balanced_accuracy", "f1"],
       subgroups=SubgroupConfig(columns=["center", "grade"]),
   )

   result = train(
       feature_store=store,
       dataset=dataset,
       splits=splits,
       task=TaskConfig(name="binary_classification"),
       training=TrainingConfig(epochs=50, learning_rate=1e-4),
       aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 256}),
       eval=eval_config,
       run_dir="output/abmil/uni2",
   )

Generate a report for one run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``generate_report`` when you have a saved run directory on disk, or
``generate_report_from_result`` when you still have the in-memory result and
config object:

.. code-block:: python

   from soma.reporting import generate_report, generate_report_from_result

   report_path = generate_report("output/abmil/uni2")
   report_path = generate_report_from_result(result, config)

Compare multiple runs
~~~~~~~~~~~~~~~~~~~~~

Use ``compare_runs`` to generate a cross-run comparison report:

.. code-block:: python

   from soma.reporting import compare_runs

   comparison_path = compare_runs(
       ["output/abmil/uni2", "output/clam_sb/uni2"],
       labels=["ABMIL", "CLAM-SB"],
   )

For more detail on what the generated HTML report contains, how subgroup
analysis is summarized, and how comparison statistics are computed, see the
:doc:`reporting guide <reporting>`.

Enable heatmaps when you want attention overlays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Attention heatmaps are controlled through ``PipelineConfig.heatmaps``. This is
most useful for attention-based aggregators that expose per-tile scores:

.. code-block:: python

   from soma import AggregatorConfig, EvalConfig, HeatmapConfig, Pipeline, PipelineConfig, TaskConfig, TrainingConfig, EncoderConfig

   config = PipelineConfig(
       dataset_csv="dataset.csv",
       splits_csv="splits.csv",
       output_root="output",
       dataset_type="slide",
       encoder=EncoderConfig(name="uni2"),
       aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 256}),
       task=TaskConfig(name="binary_classification"),
       training=TrainingConfig(epochs=50, learning_rate=1e-4),
       eval=EvalConfig(metrics=["auroc", "balanced_accuracy"]),
       heatmaps=HeatmapConfig(enabled=True, cmap="coolwarm", alpha=0.5),
   )

   result = Pipeline(config).run()
