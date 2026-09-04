Tasks
=====

Task heads map a representation to predictions and define the loss and metric
contract. This page is the task-layer index: it lists the task zoo, documents
the :class:`soma.tasks.base.TaskHead` base abstraction, and links to the
per-task pages.

.. seealso::

   The :doc:`slide-level MIL walkthrough <tutorials/walkthrough-slide-mil>` trains a
   task head on frozen features end to end — swapping classification, regression, or
   survival is just a different ``TaskConfig`` on the **same** extracted features.

Modeling paths
--------------

Task heads receive representations through three modeling paths:

* A **single feature vector** can feed :doc:`classification` or :doc:`regression`
  directly for a tile, region, slide, or patient. :doc:`survival` supports
  slide- and patient-level vectors.
* A **bag of features** can first pass through an :doc:`aggregator <aggregators>`,
  then feed the same classification, regression, or survival heads.
* A **dense feature grid** passes through a :doc:`decoder <decoders>` before a
  :doc:`segmentation` or :doc:`detection` head.

The base abstraction is :class:`soma.tasks.base.TaskHead`.

.. autoclass:: soma.tasks.base.TaskHead
   :members:

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

It defaults to ``0.0``, and at that default no dropout module is built at all: the
head has exactly the modules, checkpoint state and random-number consumption it had
before the knob existed. Dropout carries no parameters, so a checkpoint trained with
it loads into a head built without it and vice versa.

Aggregators carry their own ``dropout`` under ``aggregator.params`` — see
:doc:`aggregators`. The two are independent; with ``aggregation: null`` only the head's
applies.

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

Task pages
----------

Scalar and vector tasks (single vector, or bag → :doc:`aggregator <aggregators>` → head):

* :doc:`classification` — binary, multiclass, and ordinal heads.
* :doc:`regression` — continuous targets.
* :doc:`survival` — time-to-event with the ``nll`` and ``cox`` losses.

Dense tasks (token grid → decoder → head):

* :doc:`segmentation` — the dense contract + neural-decoder default path.
* :doc:`detection` — point detection with the F1@δ metric.

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
