Training
========

Training fits the task head, aggregator, or dense component while monitoring a
tune loss or metric for checkpoint selection and early stopping.

.. autoclass:: soma.config.TrainingConfig
   :members:

Core settings
-------------

.. list-table::
   :header-rows: 1

   * - Field
     - Default
     - Purpose
   * - ``epochs``
     - ``50``
     - Maximum epochs.
   * - ``learning_rate`` / ``weight_decay``
     - ``1e-4`` / ``1e-5``
     - Optimization and regularization.
   * - ``optimizer`` / ``scheduler``
     - ``adam`` / ``cosine``
     - Optimization schedule.
   * - ``patience``
     - ``10``
     - Early-stopping patience.
   * - ``batch_size``
     - ``1``
     - Samples per step; increase for tile datasets.
   * - ``gradient_accumulation``
     - ``1``
     - Steps contributing to the effective batch.
   * - ``seed``
     - ``0``
     - Run-level reproducibility seed.

Selection protocol
------------------

``monitor`` and ``monitor_mode`` select the tune loss or metric used for early
stopping and the best checkpoint.

Normally, provide distinct train, tune, and test samples. For a published
protocol with only one held-out split, set ``tune_is_test: true`` and declare
either tune or test, not both; soma uses that split for checkpoint selection
and final reporting. This reproduces the protocol but does not provide an
independent test estimate.

Set ``allow_missing_tune: true`` only when intentionally reusing train samples
as tune data. For unbiased candidate selection with a declared test set, use
``evaluation.holdout_test`` as described in :doc:`evaluation`.

Artifacts
---------

Iterative trainers write ``training_history.json`` with per-epoch losses,
metrics, learning rate, and timing. During a run, the live panel also shows the
active fold, patience, trainable parameters, and ETA. See :doc:`outputs` for
single-fold and cross-validation paths.
