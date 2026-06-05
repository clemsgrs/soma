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
   * - ``survival``
     - Discrete-time NLL
     - ``c_index``
     - Time-to-event with right censoring

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

.. autoclass:: soma.tasks.survival.SurvivalHead
   :members:

.. autoclass:: soma.tasks.survival.CoxSurvivalHead
   :members:

Survival
--------

The ``survival`` task offers two losses, selected via ``task.params.loss``:

* ``nll`` (default) — **discrete-time** survival modeling. The continuous time
  axis is split into ``num_bins`` bins; the head emits one hazard logit per bin
  and trains with the sigmoid-hazard negative log-likelihood (the Gensheimer /
  HIPT ``NLLSurvLoss`` formulation).
* ``cox`` — **continuous-time CoxPH**. The head emits a single risk scalar and
  trains with the Breslow partial-likelihood loss. The risk set must contain
  several samples, and the loader uses an event-balanced sampler so every
  batch/window contains at least one event (``gradient_accumulation`` must be
  ``1``). Two modes, switched by ``task.params.cox_window``:

  * **Padded mode** (``cox_window`` unset / ``1``): the risk set is the batch
    (``batch_size >= 2``). Works on single-embedding slide/patient features (no
    aggregator) *or* on MIL bags (any of ``abmil``/``transmil``/``mean_pool``),
    where variable-length bags are padded and masking keeps the result exact.
  * **Accumulation mode** (``cox_window >= 2``): for large variable-size MIL
    bags. ``batch_size`` is pinned to ``1`` and an aggregator is required; the
    trainer forwards ``cox_window`` bags un-padded, keeps their risk scalars
    graph-connected, and computes one Cox loss over the window (one optimiser
    step per window). This avoids the padding-memory cost of padded mode. Note
    it does ~``cox_window``× fewer optimiser steps per epoch, so budget a higher
    learning rate and/or more epochs.

Both losses rank with Harrell's C-index via scikit-survival.

Survival datasets reuse the ``label`` column for the **time-to-event /
time-to-last-follow-up** and add two columns:

.. list-table::
   :header-rows: 1

   * - Column
     - Meaning
   * - ``label``
     - Continuous time-to-event (uncensored) or time-to-last-follow-up
       (censored).
   * - ``event``
     - ``1`` if the event was observed, ``0`` if right-censored.
   * - ``bin``
     - Index of the discrete time bin **containing** ``label`` — for *every*
       sample, including censored ones (a censored sample's bin is the last bin
       it was known event-free). Compute the bins yourself (e.g. ``qcut`` on the
       uncensored times); ``num_bins`` is inferred as ``max(bin) + 1``.
       **Required for ``loss: nll`` only** — the Cox path ignores ``bin``.

Supported ``dataset_type`` values are ``slide`` and ``patient`` (``tile`` is
rejected). For ``patient`` pipelines, all slides of a patient must agree on the
survival target. The CLAM and DTFD-MIL aggregators are rejected for survival
because their label-aware auxiliary losses assume classification; the discrete
NLL path uses ``abmil``, ``transmil``, or ``mean_pool``, while the Cox path uses
no aggregator at all (single-embedding slide/patient features).

Metric compatibility
--------------------

``multiclass_classification`` accepts ``qwk`` as an opt-in metric when the
class labels have an ordinal interpretation. The task still uses
cross-entropy loss; the metric only changes how results are summarized.

Discovery helper
----------------

Use ``soma.list_task_heads()`` to inspect the registered task heads from code
when you need to populate a selector or validate a config name.
