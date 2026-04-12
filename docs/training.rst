Training and Evaluation
=======================

Training takes the selected feature representation and fits the task head.
The main knobs are learning rate, epochs, patience, optimizer, and batch
behavior.

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

When tuning, keep the task metric contract stable before sweeping optimizer
details.

Live training summary
---------------------

During training, the live summary panel reports the current epoch, loss,
learning rate, tune metrics, patience, status, trainable parameter count, and
epoch timing. The estimated time remaining is shown only in the live display.

Saved timing artifacts
----------------------

The training history saved under ``fold_N/training_history.json`` includes the
elapsed time and average epoch time for each fold. Those values also appear in
the HTML report so completed runs can be compared without reopening the live
console.
