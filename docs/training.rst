Training
========

Training fits an aggregator, task head, or dense decoder to the selected
features. Configure the training budget, checkpoint selection, and feature
transforms below.

Budget and checkpoint selection
-------------------------------

Choose one budget: ``epochs`` counts passes through the training loader;
``max_steps`` counts optimizer updates after gradient accumulation. For a step
budget, set ``epochs: null``. Cosine scheduling advances per epoch or per update
to match the selected budget.

``checkpoint_selection: best`` selects weights using ``monitor`` and
``monitor_mode`` (default: minimize tune loss). ``patience`` controls early
stopping; ``null`` disables it. ``checkpoint_selection: last`` evaluates weights
at the end of the budget and requires ``patience: null``. Tune metrics are still
recorded for diagnostics.

By default, training requires a tune split. Two explicit alternatives
are available:

- ``tune_is_test: true`` uses one held-out split for both checkpoint selection
  and test reporting. Provide either tune or test, not both. Use this only when
  reproducing a benchmark protocol that selects on its reported cohort.
- ``allow_missing_tune: true`` reuses train as tune when tune is absent and emits
  a warning. This controls the diagnostic split independently of checkpoint
  selection.

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
     - Training budget when ``max_steps`` is unset
   * - ``max_steps``
     - ``None``
     - Optimizer-update budget; set ``epochs: null`` when using it
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
     - ``last`` evaluates weights at the end of the budget; requires ``patience=None``
   * - ``patience``
     - ``10``
     - Early stopping on the monitored tune value; ``None`` disables it
   * - ``batch_size``
     - ``1``
     - Good for MIL; raise for tile runs
   * - ``gradient_accumulation``
     - ``1``
     - Effective batch size multiplier
   * - ``tune_is_test``
     - ``False``
     - Use one held-out split as both tune and test
   * - ``allow_missing_tune``
     - ``False``
     - Reuse train as tune when a fold has no tune split; emits a warning

Feature normalization
---------------------

The top-level ``normalization`` section applies a feature adaptor before the
trainable model, allowing encoders with different activation scales to share
training settings.

.. code-block:: yaml

   normalization:
     method: zscore   # none | zscore | l2 | layernorm
     eps: 1.0e-6

``zscore`` estimates per-feature center and scale from the train split alone.
Held-out samples use the fitted transform without updating it. ``eps`` floors
the scale of constant or near-constant channels and must be finite and positive.
``l2`` and ``layernorm`` are stateless. ``none`` adds no adaptor and requires
``eps`` to retain its default.

The adaptor supports cached tile-encoder MIL, slide-encoder embeddings, and
single-encoder dense paths. Unsupported paths reject an active transform.
For slide embeddings, fitting uses one row per training slide; MIL and dense
paths supply feature rows from their training samples. Composite
``member_norm`` separately normalizes each encoder's block before concatenation.

Feature projection
------------------

The top-level ``projection`` section maps frozen features to a common width
without labels. Matching the downstream input dimension helps compare encoders
without changing the aggregator's capacity simply because their embeddings have
different widths.

.. code-block:: yaml

   projection:
     method: pca      # none | pca | random
     target_dim: 512  # required when method != none
     seed: 0           # configurable only for random

Projection follows normalization. Either stage can be enabled independently.

``pca`` fits centered principal components on the train split alone, without
whitening. It fixes each component's sign by making its largest-magnitude entry
positive. ``target_dim`` must not exceed either the encoder dimension or the
number of training feature rows; preflight checks both limits. With 12 training
slide embeddings, for example, at most 12 components can be fitted.

``random`` uses a fixed Gaussian matrix scaled by ``1/sqrt(target_dim)`` and can
reduce or expand the input while approximately preserving inner products and
distances. A private generator derives its seed from the
configured seed, encoder identity, and input/output dimensions. It does not
alter the global random state.

``target_dim`` and ``seed`` must be exact integers, excluding booleans, strings,
and fractional values. Inactive fields retain their defaults; PCA requires the
default seed. The downstream aggregator and head use ``target_dim`` when
projection is active.

Adaptor state and provenance
----------------------------

Both stages are frozen checkpoint buffers, so evaluation restores the fitted
transform. Neither changes the extracted feature cache. Both configuration
blocks are recorded in the saved config and :doc:`experiment identity <outputs>`,
including their default values.

Each fold with an active transform writes ``feature_adapter.json`` beside its
checkpoint. It records normalization and projection methods, the ``eps`` floor,
the number of floored channels, dimensions, projection seed, and PCA explained
variance where applicable. ``n_support_samples`` counts training samples;
each stage's ``n_fit_samples`` counts feature rows used to fit it. Active
stateless stages report ``0`` and inactive stages report ``null``.

Live training summary
---------------------

During training, the live summary panel reports the current epoch, loss,
learning rate, tune metrics, patience, status, trainable parameter count, and
epoch timing. For cross-validation runs, it also shows the active fold as
``Fold: x/N``. The estimated time remaining is shown only in the live display.

The same epoch timing and loss history are saved in
``training_history.json`` and shown in the HTML report. See :doc:`outputs` for
single-fold and cross-validation layouts.
