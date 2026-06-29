Getting started
===============

`soma` is a modular framework to streamline computational pathology research.
It helps you go from a dataset of slides and labels to a reproducible result
report through a single, coherent API.

Running a pipeline
------------------

The quickest way to get started is to think of a pipeline run as one
beginner-friendly sequence: you provide the dataset and split manifests, pick
the model pieces, and let ``soma`` handle the rest.

Practical workflow
~~~~~~~~~~~~~~~~~~

1. Start with a valid :doc:`dataset and split <dataset>` manifest.
2. Choose an :doc:`encoder <encoders>`.
3. If working with a tile-level encoder, add an :doc:`aggregator <aggregators>`.
4. Pick a :doc:`task <tasks>`, :doc:`evaluation config <evaluation>`, and
   :doc:`training config <training>`.
5. Run ``Pipeline(config).run()``.
6. Inspect the returned ``result`` and the saved run outputs.

What happens under the hood
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you call :class:`soma.pipeline.Pipeline`, the framework turns your
configuration into one reproducible run directory and then executes the
experiment in a fixed order:

1. Read ``dataset.csv`` and ``splits.csv``.
2. Resolve preprocessing, encoder, and aggregator settings.
3. Extract or load features for each sample.
4. Train one model per fold, using the tune split for checkpoint selection.
5. Evaluate the best checkpoint on the tune and test splits.
6. Write metrics, predictions, checkpoints, and a final HTML report to disk.
   If heatmaps are enabled, those are written too.

**Importantly**, ``soma`` builds a shared :doc:`cache <caching>` to make experiment sweeps
more efficient by reusing preprocessing and feature extraction whenever upstream settings
are identical. This allows repeated runs to skip previously completed upstream work.

Run a full pipeline
-------------------

The simplest end-to-end path is ``Pipeline(config).run()``:

.. code-block:: python

   from soma import (
       AggregatorConfig,
       EncoderConfig,
       EvalConfig,
       HeatmapConfig,
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
       encoder=EncoderConfig(name="uni2"),
       aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 256}),
       task=TaskConfig(name="binary_classification"),
       evaluation=EvalConfig(metrics=["auroc", "balanced_accuracy"]),
       heatmaps=HeatmapConfig(enabled=True, cmap="coolwarm", alpha=0.5),
       training=TrainingConfig(epochs=50, learning_rate=1e-4),
   )

   result = Pipeline(config).run()

The returned ``result`` is a :class:`soma.pipeline.PipelineResult`. It gives
you a Python handle on the run you just completed:

- ``result.run_dir`` is the run directory on disk.
- ``result.summary`` is the aggregated metric dictionary saved to
  ``summary.json`` and mirrored in the run report.
- ``result.fold_results`` contains one entry per fold.

Each fold entry is a :class:`soma.pipeline.FoldResult` with:

- ``train_result``: best epoch, tune loss, training history, and the saved
  checkpoint path.
- ``tune_report``: evaluation metrics and per-sample predictions for the tune
  split.
- ``test_reports``: a split-name keyed mapping of evaluation reports for every
  test split.

In other words, the returned object gives you the key artifacts in memory,
while the run directory on disk contains the same experiment in a persistent,
reproducible form. For the files written to disk, see :doc:`outputs`.

Modular API
-----------

If you want finer grained control, you can use individual :doc:`building blocks <api>`
directly instead of running the full pipeline in one call. This is useful when you want
to manage preprocessing, feature extraction, training, evaluation, reporting,
or heatmap generation as separate steps in a custom experiment workflow.
