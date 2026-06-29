Regression
==========

The ``regression`` task head maps the aggregated bag representation to a
continuous target, trains with MSE, and reports ``mae`` and ``r2``. The
aggregator is the method home for the bag → slide-level step that feeds the
head — see :doc:`aggregators`.

.. autoclass:: soma.tasks.regression.RegressionHead
   :members:
