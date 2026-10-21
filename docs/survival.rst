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

Both losses rank with Harrell's C-index via scikit-survival. The ``mil_survival``
preset takes the loss as its ``loss`` argument: ``"nll"`` builds the
``discrete_survival`` head with ``binning: train_quantiles`` and the preset's
``num_bins`` (required), and ``"cox"`` the ``cox_survival`` head in
accumulation mode, with ``cox_window: 32`` (it takes no ``num_bins``).

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
       including censored ones. Read by the ``discrete_survival`` head with
       ``binning: column`` only.

Time bins
---------

The ``discrete_survival`` head's ``binning`` param says where each sample's bin
comes from.

``binning: train_quantiles`` (recommended) derives the bins in the run, from each
fold's training split, so the cohort needs only ``time`` and ``event``.
``num_bins`` is required: it is the number of bins and the head width. Binning on
the whole cohort would let the tune and test times shape the cut points; here
only the training times do.

.. code-block:: yaml

   head:
     name: discrete_survival
     params: {binning: train_quantiles, num_bins: 4}

For each fold, with ``K = num_bins``:

* The cut points are the ``K - 1`` inner quantiles (``1/K, ..., (K-1)/K``) of the
  **uncensored** training times, with linear interpolation (``np.quantile``'s
  ``method="linear"``).
* The training times are those of the samples the fold trains on: its training
  slides with features (a slide without tissue never trains), or for a patient
  run (``cohort.collapse: patient_id``) one time per training patient, however
  many slides it has. ``cohort.tune_is_test`` and ``cohort.allow_missing_tune``
  change tune only and do not move the cut points.
* Bins are left-closed, ``[c_j, c_{j+1})``: a time equal to a cut point goes to
  the upper bin. The first and last bins are open, so a tune or test time
  outside the training range falls in bin ``0`` or bin ``K - 1``.
* A censored sample gets the bin its time falls in, like an uncensored one.

A fold whose training split has fewer than ``K`` uncensored times, or whose cut
points tie (several training times equal to one cut point), cannot form ``K``
bins: the run stops with an error naming the fold, ``num_bins``, the number of
uncensored training times and the repeated cut point. Lower ``num_bins``; empty
bins are never created. The check runs when the run starts, on each fold's
splits (before any tiling or extraction), and again when each fold is trained.

The fitted cut points are the head's ``bin_edges``. The fold's log prints them,
and each fold bundle records them in ``model.json``, so :func:`soma.load` rebuilds
a head that carries them. ``plan.yaml`` records ``binning`` and ``num_bins``,
never the cut points, because the plan is the same for every fold. A config
cannot set ``bin_edges``, and with ``binning: train_quantiles`` the cohort may
not have a ``bin`` column: drop it, or set ``binning: column``.

``binning: column`` (the default) reads the bins you computed before the run from
the cohort's ``bin`` column. Set the head's ``num_bins`` param to fix the head
width: any subset of indices in ``[0, num_bins)`` is then valid, including empty
intervals, and indices are never renumbered. Without an explicit width, observed
bins must be contiguous from zero and ``num_bins`` is inferred as
``max(bin) + 1``. Cut points computed on the whole cohort carry information
about the tune and test times into training; ``train_quantiles`` avoids this.

Risk scores, ``time`` and ``event`` are all that prediction and evaluation read;
neither mode changes them.

Samples and aggregators
-----------------------

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
