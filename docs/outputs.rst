Run outputs
===========

Each run writes an immutable, self-contained result bundle beneath
``output_root``. Reusable tiling and feature artifacts live in the separate
:doc:`cache <caching>`.

Core files
----------

A completed run contains:

- ``config.yaml`` and run metadata for reproducibility;
- ``summary.json`` with run-level metrics;
- ``report.html`` with the rendered analysis;
- per-fold checkpoints, metrics, histories, and predictions when applicable.

A single fold writes fold artifacts directly in the run directory. A
cross-validation run writes the same artifacts under ``fold_0/``, ``fold_1/``,
and so on. Typical fold files are ``best_model.pt``, ``metrics.json``,
``training_history.json``, and ``predictions_<split>.csv``. Closed-form methods
may omit checkpoint or training-history files.

Multiple test splits retain the split name in each artifact, for example
``predictions_test.csv`` and ``predictions_test_external.csv``. Summary keys
follow the same convention: ``test/auroc`` for one fold and
``test/auroc_mean`` / ``test/auroc_std`` across folds.

Experiment identity
-------------------

Managed output paths group runs by an identity derived from the manifests and
settings that affect predictions, metrics, artifacts, or experiment indexes.
This includes preprocessing, encoder, aggregation or dense component, task,
training, evaluation, and enabled heatmap settings.

The training seed remains a run-level value, so repeated seeds share an
experiment namespace but produce distinct runs. Inactive rendering options and
``evaluation.overwrite_test`` do not change identity.

Task-specific artifacts
-----------------------

- Attention-capable MIL aggregators can write raw scores under ``attention/``
  and rendered overlays under ``heatmaps/`` when
  :class:`soma.config.HeatmapConfig` is enabled.
- Segmentation writes prediction rasters and optional overlays or probability
  tensors; see :doc:`segmentation` and :doc:`evaluation`.
- Detection writes level-0 point CSVs, selected thresholds, and optional
  visualizations; see :doc:`detection` and :doc:`evaluation`.

Use :doc:`reporting` to regenerate ``report.html`` or compare result bundles.
