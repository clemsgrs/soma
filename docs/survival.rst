Survival
========

The ``survival`` task models time-to-event with right censoring. Select the loss
with ``task.params.loss``:

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
    bags. ``batch_size`` is pinned to ``1`` and an aggregator is required. The
    trainer forwards ``cox_window`` bags un-padded and computes one Cox loss and
    one optimizer step per window.

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
     - Index of the discrete time bin containing ``label``, for every sample
       including censored ones. Required for ``loss: nll`` only.

Compute the bins yourself (e.g. ``qcut`` on the uncensored times). Set
``task.params.num_bins`` to fix the head width: any subset of indices in
``[0, num_bins)`` is then valid, including empty intervals, and indices are
never renumbered. Without an explicit width, observed bins must be contiguous
from zero and ``num_bins`` is inferred as ``max(bin) + 1``.

Supported ``dataset_type`` values are ``slide`` and ``patient`` (``tile`` is
rejected). For ``patient`` pipelines, all slides of a patient must agree on the
survival target. Survival MIL supports ``abmil``, ``transmil``, ``mean_pool``,
and hierarchical ``hipt`` features; other aggregators are rejected.

Task heads
----------

.. autoclass:: soma.tasks.survival.SurvivalHead
   :members:

.. autoclass:: soma.tasks.survival.CoxSurvivalHead
   :members:
