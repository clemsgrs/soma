Survival
========

Two heads model time-to-event with right censoring; the head selects the loss:

* ``discrete_survival`` — **discrete-time** survival modeling. The continuous time
  axis is split into ``num_bins`` bins; the head emits one hazard logit per bin
  and trains with the sigmoid-hazard negative log-likelihood (the Gensheimer /
  HIPT ``NLLSurvLoss`` formulation).
* ``cox_survival`` — **continuous-time CoxPH**. The head emits a single risk scalar and
  trains with the Breslow partial-likelihood loss. The risk set must contain
  several samples, and the loader uses an event-balanced sampler so every
  batch/window contains at least one event (``trainer.accumulate_grad_batches``
  must be ``1``). Two modes, switched by the head's ``cox_window`` param:

  * **Padded mode** (``cox_window`` unset / ``1``): the risk set is the batch
    (``batch_size >= 2``). Works on single-embedding slide/patient features (no
    aggregator) *or* on MIL bags (any of ``abmil``/``transmil``/``mean_pool``),
    where variable-length bags are padded and masking keeps the result exact.
  * **Accumulation mode** (``cox_window >= 2``): for large variable-size MIL
    bags. ``batch_size`` is pinned to ``1`` and an aggregator is required. The
    trainer forwards ``cox_window`` bags un-padded and computes one Cox loss and
    one optimizer step per window.

Both losses rank with Harrell's C-index via scikit-survival. The ``survival``
preset takes the loss as its ``loss`` argument: ``"nll"`` builds the
``discrete_survival`` head and ``"cox"`` the ``cox_survival`` head in
accumulation mode, with ``cox_window: 32``.

Survival datasets carry these target columns; the head's ``targets`` param maps
a target to a differently named column, for example ``targets: {time: label}``:

.. list-table::
   :header-rows: 1

   * - Column
     - Meaning
   * - ``time``
     - Continuous time-to-event (uncensored) or time-to-last-follow-up
       (censored).
   * - ``event``
     - ``1`` if the event was observed, ``0`` if right-censored.
   * - ``bin``
     - Index of the discrete time bin containing ``time``, for every sample
       including censored ones. Required for the ``discrete_survival`` head only.

Compute the bins yourself (e.g. ``qcut`` on the uncensored times). Set the
head's ``num_bins`` param to fix the head width: any subset of indices in
``[0, num_bins)`` is then valid, including empty intervals, and indices are
never renumbered. Without an explicit width, observed bins must be contiguous
from zero and ``num_bins`` is inferred as ``max(bin) + 1``.

Survival samples are slides or patients (``cohort.collapse: patient_id``); a
tile-level run is rejected. For patient runs, all slides of a patient must agree
on the survival target. Survival MIL supports ``abmil``, ``transmil``, ``mean_pool``
and ``hipt``; other aggregators are rejected. A hierarchical store works with all
four: ``hipt`` reads it grouped, and the others read it as a flat bag of its tiles
(see :doc:`aggregators`).

Task heads
----------

.. autoclass:: soma.models.heads.survival.DiscreteSurvivalHead
   :members:

.. autoclass:: soma.models.heads.survival.CoxSurvivalHead
   :members:
