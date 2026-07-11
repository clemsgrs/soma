Regression
==========

The ``regression`` task head maps the aggregated bag representation to a
continuous target, trains with MSE, and reports ``mae`` and ``r2``. The
aggregator is the method home for the bag → slide-level step that feeds the
head — see :doc:`aggregators`.

For vector targets, ``dataset_type="spatial_expression"`` requires
``training.method="ridge_pca_probe"``. The closed-form probe fits a feature
standardizer, PCA, and multi-output Ridge model to cached per-spot features;
it reports per-gene ``pearson`` and their ``mean_pearson``. It does not use an
aggregator or a trainable task head.

.. autoclass:: soma.tasks.regression.RegressionHead
   :members:
