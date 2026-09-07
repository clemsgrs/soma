Survival
========

The ``survival`` task models time-to-event with right censoring from a frozen
slide or patient embedding, or a slide representation produced by an
:doc:`aggregator <aggregators>`.

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
    step per window). This avoids padding, but retains each bag's computation
    graph until the window loss is computed. Changing the window size also
    changes the number of optimizer steps per epoch.

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
because their label-aware auxiliary losses assume classification. DSMIL is also
incompatible because it requires binary classification. Survival MIL can use
``abmil``, ``transmil``, ``mean_pool``, or hierarchical ``hipt`` features.

Task heads
----------

.. autoclass:: soma.tasks.survival.SurvivalHead
   :members:

.. autoclass:: soma.tasks.survival.CoxSurvivalHead
   :members:
