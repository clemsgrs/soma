Tasks
=====

Task heads map a representation to predictions and define the loss and metric
contract. This page is the task-layer index: it lists the task zoo, documents
the :class:`soma.tasks.base.TaskHead` base abstraction, and links to the
per-task pages.

.. seealso::

   The :doc:`slide-level walkthrough <tutorials/walkthrough-slide-level>` runs
   classification, regression, and survival end to end on the **same** extracted
   features — the clearest way to see how swapping a task head works in practice.

Substrate cleavage
------------------

Tasks split by the representation substrate they consume:

* **Slide-level** tasks consume a bag of tile features, which an
  :doc:`aggregator <aggregators>` pools into a single slide- or patient-level
  vector before the head: :doc:`classification`, :doc:`regression`, and
  :doc:`survival`.
* **Dense** tasks consume a token grid, which a decoder turns into a per-pixel
  map before the head: segmentation and
  :doc:`detection`.

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
   * - ``survival``
     - Discrete-time NLL
     - ``c_index``
     - Time-to-event with right censoring
   * - ``segmentation``
     - CE + soft-Dice
     - ``mean_dice``, ``mean_iou``
     - Dense per-pixel classification (``dataset_type: segmentation``)
   * - ``detection``
     - Foreground-weighted MSE
     - ``mean_f1``
     - Cell / nucleus point detection (``dataset_type: detection``); see :doc:`detection`

Task pages
----------

Slide-level tasks (bag → :doc:`aggregator <aggregators>` → head):

* :doc:`classification` — binary, multiclass, and ordinal heads.
* :doc:`regression` — continuous targets.
* :doc:`survival` — time-to-event with the ``nll`` and ``cox`` losses.

Dense tasks (token grid → decoder → head):

* segmentation
* :doc:`detection`

The autoclass directives for the dense heads live here until the dense task
pages are added:

.. autoclass:: soma.tasks.segmentation.SegmentationHead
   :members:

.. autoclass:: soma.tasks.detection.DetectionHead
   :members:

Discovery helper
----------------

Use ``soma.list_task_heads()`` to inspect the registered task heads from code
when you need to populate a selector or validate a config name.
