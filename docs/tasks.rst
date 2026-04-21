Tasks
=====

Task heads map the aggregated representation to predictions and define the
loss and metric contract.

The base abstraction is :class:`soma.tasks.base.TaskHead`.

.. autoclass:: soma.tasks.base.TaskHead
   :members:

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
     - Two-class labels
   * - ``multiclass_classification``
     - Cross-entropy
     - ``auroc_macro``, ``balanced_accuracy``, ``f1_macro``
     - | Two or more classes
       | Use ``binary_classification`` when the problem is strictly binary
   * - ``ordinal_classification``
     - MSE
     - ``qwk``, ``balanced_accuracy``
     - Ordered integer grades
   * - ``regression``
     - MSE
     - ``mae``, ``r2``
     - Continuous targets

Task details
------------

.. autoclass:: soma.tasks.classification.BinaryClassificationHead
   :members:

.. autoclass:: soma.tasks.classification.MulticlassClassificationHead
   :members:

.. autoclass:: soma.tasks.ordinal_classification.OrdinalClassificationHead
   :members:

.. autoclass:: soma.tasks.regression.RegressionHead
   :members:

Metric compatibility
--------------------

``multiclass_classification`` accepts ``qwk`` as an opt-in metric when the
class labels have an ordinal interpretation. The task still uses
cross-entropy loss; the metric only changes how results are summarized.

Discovery helper
----------------

Use ``soma.list_task_heads()`` to inspect the registered task heads from code
when you need to populate a selector or validate a config name.
