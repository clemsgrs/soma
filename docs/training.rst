Training
========

Training takes the selected feature representation and fits the task head.
The main knobs are learning rate, epochs, patience, optimizer, scheduler, and
batch behavior. If a benchmark only exposes train/test splits, set
``tune_is_test=True`` to use the single test split for checkpoint selection and
final reporting. If a dataset has no tune split and you want a train-as-tune
fallback instead, set ``allow_missing_tune=True``.

The main configuration object is :class:`soma.config.TrainingConfig`.

.. autoclass:: soma.config.TrainingConfig
   :members:

Practical defaults
------------------

.. list-table::
   :header-rows: 1

   * - Field
     - Default
     - Notes
   * - ``seed``
     - ``0``
     - Reproducibility
   * - ``epochs``
     - ``50``
     - Maximum training epochs
   * - ``learning_rate``
     - ``1e-4``
     - Primary optimization knob
   * - ``weight_decay``
     - ``1e-5``
     - Regularization
   * - ``optimizer``
     - ``adam``
     - Also supports ``adamw`` and ``sgd``
   * - ``scheduler``
     - ``cosine``
     - Or ``none``
   * - ``patience``
     - ``10``
     - Early stopping on tune loss
   * - ``batch_size``
     - ``1``
     - Good for MIL; raise for tile runs
   * - ``gradient_accumulation``
     - ``1``
     - Effective batch size multiplier
   * - ``tune_is_test``
     - ``False``
     - Use the only test split as tune; intended for reproducing benchmark protocols without an internal validation set
   * - ``allow_missing_tune``
     - ``False``
     - Reuse train as tune when a fold has no tune split; emits a warning

When tuning, keep the task and evaluation contract stable before sweeping
optimizer details.

Live training summary
---------------------

During training, the live summary panel reports the current epoch, loss,
learning rate, tune metrics, patience, status, trainable parameter count, and
epoch timing. For cross-validation runs, it also shows the active fold as
``Fold: x/N``. The estimated time remaining is shown only in the live display.

Saved timing artifacts
----------------------

The training history is saved as ``training_history.json`` (directly in the
run directory for single-fold runs, inside ``fold_N/`` for cross-validation).
It records the elapsed time and average epoch time for each epoch. Those values
also appear in the HTML report so completed runs can be compared without
reopening the live console.
