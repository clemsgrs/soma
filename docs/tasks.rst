Tasks
=====

Task heads turn model representations into predictions and own their loss and
default metrics. Swap the task head without changing compatible upstream
features.

Task substrates
---------------

- Slide-level heads consume a vector produced by an :doc:`aggregator
  <aggregators>`: :doc:`classification`, :doc:`regression`, and :doc:`survival`.
- Dense heads consume a decoder map: :doc:`segmentation` and :doc:`detection`.

.. autoclass:: soma.tasks.base.TaskHead
   :members:

Task catalog
------------

.. list-table::
   :header-rows: 1

   * - Config name
     - Loss
     - Default metrics
   * - ``binary_classification``
     - Cross-entropy
     - ``auroc``, ``balanced_accuracy``, ``auprc``, ``f1``
   * - ``multiclass_classification``
     - Cross-entropy
     - ``auroc_macro``, ``balanced_accuracy``, ``f1_macro``
   * - ``ordinal_classification``
     - MSE
     - ``qwk``, ``balanced_accuracy``
   * - ``regression``
     - MSE
     - ``mae``, ``r2``
   * - ``survival``
     - Discrete-time NLL
     - ``c_index``
   * - ``segmentation``
     - Cross-entropy + soft Dice
     - ``mean_dice``, ``mean_iou``
   * - ``detection``
     - Foreground-weighted MSE
     - ``mean_f1``

Use ``binary_classification`` for exactly two classes and
``multiclass_classification`` for two or more classes under a multiclass
contract. Regression also supports ``pearson``. Spatial-expression probes
report per-gene coefficients and ``mean_pearson``.

Configure an explicit metric subset under ``evaluation.metrics``; see
:doc:`evaluation` for model selection and subgroup analysis. Use
``soma.list_task_heads()`` to discover registered config names.

The :doc:`slide-level walkthrough <tutorials/walkthrough-slide-level>` applies
classification, regression, and survival heads to the same feature store.
