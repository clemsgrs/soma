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
   * - :doc:`Dataset and splits <dataset>`
     - CSV manifest schema and fold assignment rules
   * - :doc:`Pipeline <pipeline>`
     - End-to-end orchestration from manifests to reports
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
   store = extractor.extract(feature_dir="output/features/uni2")
   task = TaskConfig(name="binary_classification")
   training = TrainingConfig(epochs=50, learning_rate=1e-4)
   abmil_aggregator = AggregatorConfig(name="abmil", params={"hidden_dim": 256})
   clam_aggregator = AggregatorConfig(name="clam_sb", params={"hidden_dim": 256, "attn_dim": 128})

   abmil_result = train(
       feature_store=store,
       dataset=dataset,
       splits=splits,
       task=task,
       training=training,
       aggregator=abmil_aggregator,
       run_dir="output/abmil/uni2",
   )

   clam_result = train(
       feature_store=store,
       dataset=dataset,
       splits=splits,
       task=task,
       training=training,
       aggregator=clam_aggregator,
       run_dir="output/clam_sb/uni2",
   )

The returned ``FeatureStore`` can be reused across experiments as long as the
upstream preprocessing and encoder settings do not change.

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
       feature_store=store,
       dataset=dataset,
       splits=splits,
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
       feature_store=store,
       dataset=dataset,
       splits=splits,
       task=task,
       training=training,
       aggregator=aggregator,
       evaluation=evaluation,
       heatmaps=heatmaps,
       run_dir="output/abmil/uni2",
   )

   # attention scores land in fold_N/attention/
   # rendered attention overlays in fold_N/heatmaps/
