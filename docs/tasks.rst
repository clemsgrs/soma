Tasks
=====

Select a task with ``task.name``. Its head maps a representation to predictions
and defines the loss and compatible metrics. The representation comes directly
from a frozen encoder or from a trained :doc:`aggregator <aggregators>` or
:doc:`decoder <decoders>`; see :doc:`modeling` for the paths that produce it.

Task Zoo
--------

.. list-table::
   :header-rows: 1

   * - Config name
     - Loss
     - Default metrics
     - When to use
   * - ``binary_classification``
     - Cross-entropy
     - ``auroc``, ``balanced_accuracy``, ``auprc``, ``f1``
     - Two-class labels; see :doc:`classification`
   * - ``multiclass_classification``
     - Cross-entropy
     - ``auroc_macro``, ``balanced_accuracy``, ``f1_macro``
     - | Two or more classes; see :doc:`classification`
       | Use ``binary_classification`` when the problem is strictly binary
   * - ``ordinal_classification``
     - MSE
     - ``qwk``, ``balanced_accuracy``
     - Ordered integer grades; see :doc:`classification`
   * - ``regression``
     - MSE
     - ``mae``, ``r2``
     - Continuous targets; see :ref:`regression-task`
   * - ``survival``
     - NLL or CoxPH
     - ``c_index``
     - Time-to-event with right censoring; see :doc:`survival`
   * - ``segmentation``
     - CE + soft-Dice
     - ``mean_dice``, ``mean_iou``
     - Dense per-pixel classification (``dataset_type: segmentation``); see :doc:`segmentation`
   * - ``detection``
     - Foreground-weighted MSE
     - ``mean_f1``
     - Cell / nucleus point detection (``dataset_type: detection``); see :doc:`detection`

Head dropout
------------

The classification, ordinal-classification, regression and survival heads take an
optional ``dropout`` probability, applied to the head's **input** — the aggregated
bag representation, or the frozen embedding itself when ``aggregation: null`` leaves
the head as the only trainable component:

.. code-block:: yaml

   task:
     name: binary_classification
     params:
       dropout: 0.2

The default is ``0.0``. Aggregator dropout, set under ``aggregation.params``, is
independent.

.. _regression-task:

Regression
----------

The ``regression`` head maps a feature vector to continuous predictions, trains
with MSE, and reports ``mae`` and ``r2`` by default. For gene-expression vectors
and the closed-form Ridge+PCA probe, see the :doc:`hest-gene-expression-benchmark`.

.. autoclass:: soma.tasks.regression.RegressionHead
   :members:

Task-head interface
-------------------

.. autoclass:: soma.tasks.base.TaskHead
   :members:

``soma.list_task_heads()`` returns the registered task-head names.

.. toctree::
   :maxdepth: 1
   :hidden:

   classification
   survival
   segmentation
   detection
