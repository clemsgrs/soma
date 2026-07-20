Training
========

Training takes the selected feature representation and fits the task head.
The main knobs are learning rate, epochs, patience, optimizer, scheduler, and
batch behavior. If a benchmark only exposes train/test splits, set
``tune_is_test=True`` to use the single test split for checkpoint selection and
final reporting. If a dataset has no tune split and you want a train-as-tune
fallback instead, set ``allow_missing_tune=True``.

``checkpoint_selection`` decides *which* epoch's weights are evaluated. The
default ``best`` selects the checkpoint by the monitored tune metric and stops
early once it stops improving. ``checkpoint_selection='last'`` instead evaluates
the final-epoch weights: model selection comes off the tune metric entirely and
early stopping is disabled, so it requires ``patience=None``. Per-epoch tune
metrics are still computed and written to ``training_history.json`` — as
diagnostics only. This is the protocol for benchmarks that predeclare a fixed
epoch budget and forbid per-encoder tuning. It is orthogonal to
``allow_missing_tune``: one governs which checkpoint is evaluated, the other
where the diagnostic tune split comes from, and neither implies the other.

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
   * - ``checkpoint_selection``
     - ``best``
     - ``last`` evaluates the final-epoch weights (no early stopping, no metric-based selection); requires ``patience=None``
   * - ``patience``
     - ``10``
     - Early stopping on tune loss; ``None`` disables early stopping
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

Feature normalization
---------------------

Frozen encoders span 768 to 4608 dimensions with very different activation
scales, so a shared aggregator and its single externally-calibrated learning
rate do not see comparable inputs across encoders. The top-level
``normalization`` section closes that gap by inserting a *feature adaptor* — a
front module applied to the frozen features before anything trainable sees
them.

.. code-block:: yaml

   normalization:
     method: zscore   # none | zscore | l2 | layernorm
     eps: 1.0e-6

``zscore`` is **fitted**: per-feature center and scale are estimated from the
Support (train) split's tiles alone, so the transform is leak-free — held-out
rows only ever pass *through* the adaptor and never move its statistics.
``eps`` floors the scale so a constant or near-constant channel cannot blow up.
``l2`` and ``layernorm`` are stateless and need no fitting. The default,
``none``, means no adaptor at all: the model is structurally identical to one
built before this section existed.

The fitted state is carried as **buffers, not parameters**, so the optimizer
never sees it while it still rides in the checkpoint — the final-checkpoint test
pass therefore re-applies the exact transform that was fit. Each fold writes a
``feature_adapter.json`` QC sidecar next to its checkpoint recording the method,
the ``eps`` floor, and how many channels that floor actually caught.

Turning normalization on does **not** invalidate an extracted feature cache: the
section is not part of the feature-extraction cache key. It *is* part of the
experiment identity, but only when non-default, so every pre-existing
``experiment_id`` is preserved. The saved run config always serializes the block
regardless, as ``{method: none}`` when off.

This is orthogonal to the composite per-member ``member_norm``, which normalizes
each member's block before concatenation and is unaffected.

Today the adaptor is fit on the tile-encoder MIL path; requesting a transform on
a path that does not yet support one is refused rather than silently ignored.

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
