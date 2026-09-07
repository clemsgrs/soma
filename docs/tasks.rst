Tasks
=====

Select a task with ``task.name``. Its head maps a representation to predictions
and defines the loss and compatible metrics. See :doc:`modeling` for the paths
that produce those representations.

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
   * - ``survival``
     - NLL or CoxPH
     - ``c_index``
     - Time-to-event with right censoring
   * - ``segmentation``
     - CE + soft-Dice
     - ``mean_dice``, ``mean_iou``
     - Dense per-pixel classification (``dataset_type: segmentation``); see :doc:`segmentation`
   * - ``detection``
     - Foreground-weighted MSE
     - ``mean_f1``
     - Cell / nucleus point detection (``dataset_type: detection``); see :doc:`detection`

Task details
------------

* :doc:`classification` — binary, multiclass, and ordinal heads.
* :doc:`regression` — continuous targets.
* :doc:`survival` — time-to-event with ``nll`` or ``cox`` loss.
* :doc:`segmentation` — per-pixel labels with a decoder or pixel classifier.
* :doc:`detection` — object centroids scored at a physical matching distance.

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

The default is ``0.0`` (disabled). Dropout has no parameters, so changing this
setting does not change checkpoint compatibility. Aggregator dropout, set under
``aggregation.params``, is independent.

Task-head interface
-------------------

.. autoclass:: soma.tasks.base.TaskHead
   :members:

Discovery helper
----------------

Use ``soma.list_task_heads()`` to inspect the registered task heads from code
when you need to populate a selector or validate a config name.

.. toctree::
   :maxdepth: 1
   :hidden:

   classification
   regression
   survival
   segmentation
   detection
