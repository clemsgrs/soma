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

.. autoclass:: soma.config.TrainingConfig
   :members:

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

Fitting uses feature rows from the training samples only. Cached tile-encoder
MIL, slide-encoder embeddings, and single-encoder dense paths are supported;
other paths reject an active transform. Composite ``member_norm`` normalizes
each encoder's block separately before concatenation.

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
number of training feature rows; preflight checks both limits.

``random`` uses a fixed Gaussian matrix scaled by ``1/sqrt(target_dim)`` and can
reduce or expand the input while approximately preserving inner products and
distances. Its seed is derived from the configured seed, encoder identity, and
dimensions, without touching the global random state. PCA requires the default
seed. The downstream aggregator and head use ``target_dim`` when projection is
active.

Both stages are frozen checkpoint buffers, so evaluation restores the fitted
transform. Neither changes the extracted feature cache, and both blocks are part
of the :doc:`experiment identity <outputs>`. Each fold with an active transform
writes ``feature_adapter.json``; see :doc:`outputs`.

During training, the live summary panel reports epoch, loss, learning rate,
tune metrics, patience, trainable parameter count, timing, and the active fold
for cross-validation. The same history is saved in ``training_history.json``
and shown in the HTML report.
