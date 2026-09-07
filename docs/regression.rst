Regression
==========

The ``regression`` head maps a feature vector to continuous predictions, trains
with MSE, and reports ``mae`` and ``r2`` by default. The vector can come directly
from a frozen encoder or from a slide-level :doc:`aggregator <aggregators>`.
For gene-expression vectors and the closed-form Ridge+PCA probe, see the
:doc:`hest-gene-expression-benchmark`.

.. autoclass:: soma.tasks.regression.RegressionHead
   :members:
