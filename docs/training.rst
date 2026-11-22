Training
========

Training fits the downstream model (an aggregator or a decoder, and a task head) to
the features of a source. A config sets it in three sections, each holding the
keywords of one callable: ``loaders`` those of :func:`soma.learn.loaders`,
``learner`` the name and params of the :class:`soma.learn.Learner`, and ``trainer``
those of the :class:`soma.learn.Trainer`. From Python, :func:`soma.fit` takes the
same values.

Budget, optimizer and schedulers
--------------------------------

.. code-block:: yaml

   loaders:
     batch_size: 1
   learner:
     name: mil
     params:
       aggregator: {name: abmil}
       head: {name: binary_classification}
       optimizer:
         name: adamw
         params: {lr: 1.0e-4, weight_decay: 1.0e-5}
       scheduler:
         - {name: constant, params: {factor: 0.333, total_iters: 5}}
         - cosine
   trainer:
     max_epochs: 50
     callbacks:
       - {name: checkpoint, params: {monitor: tune/auroc, mode: max}}
       - {name: early_stopping, params: {monitor: tune/auroc, mode: max, patience: 10}}

``loaders`` holds ``batch_size``, ``num_workers``, ``pin_memory``,
``persistent_workers`` and the optional ``sampler`` and ``augment`` leaves.
``learner.params`` holds the model's ``aggregator`` or ``decoder`` leaf and its
``head`` leaf, the ``optimizer`` and ``scheduler`` leaves, the feature
``normalization`` and ``projection``, and the keywords of a custom learner. The
``mil`` learner also takes ``checkpoint_cox_window``, which saves GPU memory in a
Cox window (see :doc:`survival`).
``learner.name`` is ``mil`` for the set shape and ``dense`` for the grid shape; a
registered or dotted :class:`soma.learn.Learner` subclass owns custom step logic
(see :ref:`custom-learner`).

Choose one budget: ``trainer.max_epochs`` counts passes through the training
loader; ``trainer.max_steps`` counts optimizer updates after
``accumulate_grad_batches``, and replaces the epoch budget when set. A cosine
scheduler anneals over the budget, per epoch or per update.

Every component is a ``{name, params}`` leaf whose name is a registry key or a
dotted import path; a bare name keeps the component's default params. The
optimizer defaults to Adam (``lr`` 1e-4, ``weight_decay`` 1e-5) when the learner
names none. ``scheduler`` may be one leaf or a list, each stepped at its own
``interval`` (``epoch`` unless set); a warm-up is a ``constant`` leaf. With no
scheduler the learning rate stays constant.

A model with no trainable parameter, such as one with the ``attention_map``
detection decoder, is not fitted: the trainer returns it as built, with an
empty training history. It still predicts on the trainer's device.

Checkpoint selection and early stopping
---------------------------------------

The ``checkpoint`` callback selects the weights: ``selection: best`` (the
default) keeps the epoch that is best on ``monitor`` in direction ``mode``
(default: minimize ``tune/loss``; ``tune/<metric>`` selects that tune metric), and
``selection: last`` keeps the final weights; it refuses an ``early_stopping``
callback and needs ``trainer.val_every_n_epochs: 1``. ``early_stopping`` stops after ``patience`` tune evaluations without
improvement; leave it out to train the full budget.

``trainer.val_every_n_epochs: N`` evaluates the tune split after every N-th
epoch and after the final epoch, for runs where tune evaluation dominates the
epoch cost; selection and patience then count tune evaluations, not epochs.
Tune metrics are always recorded for diagnostics.

.. _monitors:

Monitors
~~~~~~~~

A monitor names a value of the tune evaluation by its full name, with the
``tune/`` prefix (see :ref:`logged-values`). These are valid monitors:

- ``tune/loss``, the default, and ``tune/<metric>`` for a metric the head
  reports: values over the whole tune split.
- ``tune/mean_over_<column>/loss`` and ``tune/mean_over_<column>/<metric>``:
  the mean over the groups of a column that ``evaluation.subgroups`` names (see
  :ref:`subgroup-metrics`).

The plan refuses any other name before the run starts:

- A bare name, such as ``loss`` or ``auroc``. The error names the ``tune/`` form.
- A ``train/`` value of the train pass (see :ref:`train-pass`). These values
  show the gap between train and tune; selecting on them would select on the
  training data.
- The value of one group, such as ``tune/center=A/auroc``. A fold's tune split
  can lack the group, and the monitor then does not exist. The error names the
  mean over the column's groups instead.
- A mean over a column that ``evaluation.subgroups`` does not name, or a metric
  the head does not report.

A monitor that is NaN at an evaluated epoch stops the run with an error, in
``early_stopping`` and in a ``checkpoint`` with ``selection: best``, because it
can never improve. Under ``selection: last``, the monitor is only logged. AUROC,
AUPRC and the C-index are NaN on a tune split with one class or no comparable
pairs, a Cox loss is NaN on a tune split without an event, and a mean over groups
is NaN when no group has a value.

**Monitor names from soma 2.4.0 and earlier.** Monitor names now carry the
``tune/`` prefix. Replace ``monitor: loss`` with ``monitor: tune/loss``, and
``monitor: auroc`` with ``monitor: tune/auroc``. A bare name is refused, not
mapped. A run made before the rename cannot be resumed.

.. _logged-values:

Logged values
-------------

While a fold trains, a summary panel shows the epoch, the training loss, the
learning rate, the pooled tune values, the monitors, patience, the trainable
parameter count, timing and, for cross-validation, the active fold. Each fold
writes every logged value to ``training_history.json``, one row per epoch:
``epoch``, ``train_loss``, ``lr``, ``elapsed_seconds``, then one column per value
of the evaluation, ``null`` on an epoch without an evaluation.

Each name starts with the split. A value of one group, or the mean over a
column's groups, names it after the split:

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Name
     - Value
   * - ``tune/loss``, ``tune/<metric>``
     - The loss and each metric of the head, pooled over the whole tune split.
   * - ``tune/<column>=<group>/loss``, ``tune/<column>=<group>/<metric>``
     - The same over the samples of one group of an ``evaluation.subgroups``
       column, for example ``tune/center=A/auroc``.
   * - ``tune/mean_over_<column>/loss``, ``tune/mean_over_<column>/<metric>``
     - The unweighted mean of the column's group values.
   * - ``tune/mean_over_<column>/<name>_groups``
     - The number of groups in that mean.
   * - ``train/loss``, ``train/<metric>``, ``train/<column>=<group>/...``,
       ``train/mean_over_<column>/...``
     - The same values over the train split, with ``trainer.eval_train: true``
       (see :ref:`train-pass`).

:ref:`subgroup-metrics` explains the group values and their mean. The panel
shows the pooled values and the monitors, and ``train/loss`` when the train pass
is on. The values of each group go to ``training_history.json`` only.

With the ``mil`` learner, ``tune/loss`` is the head's loss computed once over the
whole tune split, not a mean of per-batch losses. Each sample weighs the same,
whatever the batch size. With the ``dense`` learner, it is the mean of the batch
losses.

``train_loss`` is not an evaluation of the train split: see :ref:`train-pass`
for how it differs from ``train/loss``.

.. _train-pass:

The train pass
--------------

``trainer.eval_train: true`` also scores the train split at each tune
evaluation, to show the gap between train and tune:

.. code-block:: yaml

   trainer:
     eval_train: true

At each epoch that evaluates tune, the trainer runs the model in eval mode over
every train sample, in split order and in batches like tune. It logs
``train/loss`` and ``train/<metric>`` and, with ``evaluation.subgroups``,
``train/<column>=<group>/...`` and ``train/mean_over_<column>/...``. These are the
model's values at the end of the epoch, computed as the tune values are, so
``train/auroc`` and ``tune/auroc`` compare directly.

``train_loss`` is not comparable. It is the mean of the losses of the epoch's
training batches, computed in train mode (dropout on) while the weights change
through the epoch, and it includes any auxiliary loss of the aggregator. For a
Cox head, it is computed over the event-balanced training batches, while
``train/loss`` is the partial-likelihood loss over the whole train split.

The pass changes nothing in the training: ``train_loss``, every ``tune/`` value
and the selected epoch are the same with it or without it. A ``train/`` value is
never a monitor (see :ref:`monitors`).

The pass costs one more forward pass over the train split at each evaluated
epoch, about the time of a tune evaluation scaled to the size of the train split.
It is for the set shape, which reads cached features, so this is cheap: no
encoder runs. With live encoding, the same pass would encode every train tile
again. The grid shape refuses ``eval_train``. To run the pass less often, raise
``trainer.val_every_n_epochs``.

Without a tune loader, or with an empty one (``soma.fit`` without ``tune``, or a
tune split where no sample has features), nothing is evaluated, so the pass does
not run and the trainer logs a warning. When ``cohort.allow_missing_tune``
reuses train as tune, the pass runs. ``eval_train`` is not part of the
experiment identity, but a resume refuses a change to it, as it does for
``trainer.device``.

The tune split
--------------

By default, training requires a tune split. The ``cohort`` section offers two
explicit alternatives:

- ``cohort.tune_is_test: true`` uses one held-out split for both checkpoint
  selection and test reporting. Provide either tune or test, not both. Use this
  only when reproducing a benchmark protocol that selects on its reported
  cohort.
- ``cohort.allow_missing_tune: true`` reuses train as tune when tune is absent
  and emits a warning. This controls the diagnostic split independently of
  checkpoint selection.

Feature normalization
---------------------

``learner.params.normalization`` applies a feature adaptor before the trainable
model, so encoders with different activation scales can share training settings.

.. code-block:: yaml

   learner:
     params:
       normalization:
         method: zscore   # none | zscore | l2 | layernorm
         eps: 1.0e-6

``zscore`` estimates per-feature center and scale from the feature rows of the
train split alone; held-out samples use the fitted transform without updating it.
``eps`` floors the scale of constant or near-constant channels and must be finite
and positive. ``l2`` and ``layernorm`` are stateless. ``none`` adds no adaptor and
requires ``eps`` to keep its default.

The adaptor runs on cached tile-encoder MIL bags, hierarchical stores (one
``(M, P, D)`` grid of tiles per sample, read grouped by ``hipt`` and flat by any
other aggregator), slide-encoder embeddings and single-encoder dense grids; other
paths refuse an active transform. On a hierarchical store, the adaptor is fitted
on the stored tiles of the train samples, so the fit is the same as on those
tiles stored as a flat bag. The padding added when a batch is collated never
enters the fit. A composite encoder normalizes each member's block before
concatenation instead (``member_norm``, see :doc:`encoders/composite`).

Feature projection
------------------

``learner.params.projection`` maps frozen features to a common width without
labels. With the same downstream input width, encoders can be compared without
the aggregator's capacity changing with their embedding widths.

.. code-block:: yaml

   learner:
     params:
       projection:
         method: pca      # none | pca | random
         target_dim: 512  # required when method != none
         seed: 0          # configurable only for random

Projection follows normalization; either can be enabled alone.

``pca`` fits centered principal components on the train split alone, without
whitening, and fixes each component's sign by making its largest-magnitude entry
positive. ``target_dim`` must not exceed the encoder dimension or the number of
training feature rows; the adaptor refuses either before training starts.

``random`` uses a fixed Gaussian matrix scaled by ``1/sqrt(target_dim)``. It can
reduce or expand the input while approximately preserving inner products and
distances. Its seed is derived from the configured seed, the encoder identity and
the dimensions, without touching the global random state. ``pca`` requires the
default seed. The aggregator and head take ``target_dim`` as their input width
when a projection is active.

Both stages are buffers of the model, so the fold's bundle stores the fitted
transform and evaluation reuses it. Neither changes the feature cache. The plan
spells both out (``method: none`` when off), so they are part of the
:doc:`experiment identity <outputs>`. A fold with an active transform also writes
``feature_adapter.json``; see :doc:`outputs`.

.. _live-mode:

Live re-encoding and augmentation
---------------------------------

Segmentation and detection read cached feature grids by default
(``features.kind: cached``): the frozen encoder runs once and every epoch reuses
its output. With ``features.kind: live``, soma re-encodes each pre-cropped tile
through the frozen encoder at every step instead, so pixel augmentation reaches
the features. This costs one encoder forward pass per tile and step.

.. code-block:: yaml

   features:
     kind: live
     encoder: {name: uni2}
   loaders:
     augment:
       name: segmentation
       params:
         horizontal_flip: 0.5
         vertical_flip: 0.5
         rotation_degrees: 90
         crop_scale: [0.08, 1.0]

``loaders.augment`` is ``null`` by default. It is a ``{name, params}`` leaf: the
name is a registry key (``segmentation`` is
:class:`soma.learn.SegmentationAugment`) or a dotted import path, and ``params``
go to its constructor. It applies to the train split only. Geometric operations
transform the image and its label mask together (the mask with
nearest-neighbour resampling); photometric operations change the image only.
Pixels that a rotation or translation brings into the tile get ``ignore_index``
in the mask, so they carry no supervision. ``crop_scale`` takes a random crop of
that area fraction and resizes it to the tile size before the other operations.
For detection, the points move with the image: a point that leaves the tile is
dropped, and the ignore mask moves like a label mask, so the pixels brought into
the tile are ignored.

Live mode has these limits:

- ``loaders.augment`` requires ``features.kind: live``; cached grids cannot be
  augmented.
- Live mode reads pre-cropped tiles only. It does not support annotation-sampled
  whole slides, composite encoders, an active ``normalization`` or
  ``projection``, or a ``loaders.sampler``.

Training from Python
--------------------

:func:`soma.fit` trains a learner, or a bare model wrapped in its shape's
learner. It installs a ``Checkpoint`` and an ``EarlyStopping`` built from
``monitor``, ``mode`` and ``patience`` (``patience=None`` installs none), passes
its other keywords to the :class:`soma.learn.Trainer`, and returns a
:class:`soma.FitResult`: the learner in its selected state, the history frame and
the selected epoch. :func:`soma.learn.loaders` builds one fold's loaders from a
source and a cohort: ``train`` shuffles, ``tune`` and each of ``tests`` keep split
order.

.. code-block:: python

   fold = soma.learn.loaders(source, cohort, head=model.task_head, fold=0, batch_size=1)
   result = soma.fit(model, fold.train, fold.tune, max_epochs=50, monitor="tune/auroc", mode="max")
   result.selected_epoch, result.history

``tune_groups`` adds the subgroup values to each evaluation. It maps each column
to the group of every tune sample: ``{column: {sample_id: group}}``. With
``eval_train=True``, pass ``train_eval``, a second loader over the train samples
that does not shuffle, and ``train_groups`` for its groups:

.. code-block:: python

   from torch.utils.data import DataLoader


   def groups(split):
       return {"center": cohort.targets(split)["center"].dropna().astype(str).to_dict()}


   train_eval = DataLoader(fold.train.dataset, batch_size=1, collate_fn=fold.train.collate_fn)
   result = soma.fit(
       model, fold.train, fold.tune, max_epochs=50,
       monitor="tune/mean_over_center/auroc", mode="max", tune_groups=groups("tune"),
       eval_train=True, train_eval=train_eval, train_groups=groups("train"),
   )

``fit`` refuses the training loader itself as ``train_eval``: a second pass over
it would change the training's shuffle. The pass keeps the training's random
state, but not the state that a dataset keeps for itself, so a dataset that draws
random numbers needs its own instance for ``train_eval``.

Section 3 of the getting-started notebook, :ref:`the same run from the verbs
</getting-started.ipynb#3.-The-same-run-from-the-verbs>`, runs a whole fold this
way; :doc:`evaluation` covers what follows ``fit``.

.. _grid-verbs:

The grid shape
~~~~~~~~~~~~~~

The same verbs train a decoder over feature grids, with three differences:

- The source is a :class:`~soma.data.GridSource` of cached grids, or a
  :class:`~soma.data.TileImageSource` of pixels that the model re-encodes.
- :func:`soma.learn.loaders` needs ``targets=``: a
  :class:`~soma.data.LabelMapSource` for a segmentation head or a
  :class:`~soma.data.PointSource` for a detection head. The head binds the grid
  geometry when it is built, and the loaders check that every sample matches it.
- The model is a :class:`soma.models.DenseModel`, trained by
  :class:`soma.learn.DenseLearner`. With an ``encoder`` stage it reads pixels;
  without one it reads cached grids.

:func:`soma.extract.live_encoder` wraps a slide2vec encoder as a parameter-free
:class:`soma.extract.FrozenEncoder` stage, so the foundation model never enters a
checkpoint or an optimizer. ``preprocess=stage.preprocessor()`` runs the encoder's
preprocessing in the loader workers, after augmentation; ``augment`` is refused
for cached grids:

.. code-block:: python

   from soma.models.decoders.registry import build_decoder_for_grid

   stage = soma.extract.live_encoder("uni2", target_size=448, spacing_um=0.5, window=224)
   images = soma.data.TileImageSource.from_manifest(soma.ImageManifest.from_csv("dataset.csv"), size=448, spacing_um=0.5)
   labels = soma.data.LabelMapSource.from_manifests(soma.AnnotationManifest.from_csv("dataset.csv"), size=(448, 448))
   head = soma.models.SegmentationHead(num_classes=3, geometry=stage.geometry)
   decoder = build_decoder_for_grid(
       "lightweight_conv", {}, geometry=stage.geometry, input_dim=stage.feature_dim, num_classes=3
   )
   model = soma.models.DenseModel(decoder, head, encoder=stage)
   fold = soma.learn.loaders(
       images, cohort, head=head, targets=labels, batch_size=8,
       preprocess=stage.preprocessor(),
       augment=soma.learn.SegmentationAugment(horizontal_flip=0.5, vertical_flip=0.5),
   )
   result = soma.fit(model, fold.train, fold.tune, max_epochs=50)

:class:`soma.testing.FakeEncoder` is a weight-free stage with the same contract,
for trying the grid flow without downloading model weights.

.. _runtime-observers:

Watching a managed run
----------------------

A fold writes ``training_history.json`` only after it is trained and scored. To
read each epoch while a :class:`~soma.Pipeline` run trains, pass a
``callback_factory``. It returns runtime observers: Trainer callbacks that watch
the fold.

.. code-block:: python

   from soma import Pipeline
   from soma.pipeline import FoldContext


   class EpochPrinter:
       def __init__(self, context: FoldContext):
           self.context = context

       def on_epoch_end(self, trainer, learner):
           row = trainer.history.rows[-1]
           print(self.context.fold, trainer.epoch, row["train_loss"], row.get("tune/loss"))


   result = Pipeline(config, callback_factory=lambda context: [EpochPrinter(context)]).run()

**Runtime observers and config callbacks.** ``trainer.callbacks`` in the config
is the selection policy (``checkpoint``, ``early_stopping``). It is part of
``config.yaml``, ``plan.yaml`` and the experiment identity. A runtime observer is
a Python input, like ``source``. It is not part of ``config.yaml``,
``plan.yaml``, the experiment identity, the component record in ``run.yaml`` or
the resume check. A run started with observers resumes without them, and the
other way round. The CLI has no option for them.

**When the factory is called.** ``run()`` calls ``callback_factory(context)``
once for each fold it trains, before it seeds the fold. Building observers
therefore cannot change the fold's random state. The factory is not called for a
fold that is already complete on resume, or for a fold outside ``run.folds``. A
fold that was not complete restarts from epoch 0 on resume, so its new observers
see the epochs from 0 again. A representation run (``evaluation.representation``)
trains no fold and never calls the factory. A model with no trainable parameter
is not fitted, so its observers see no epoch.

The ``context`` is a frozen :class:`soma.pipeline.FoldContext`:

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - ``run_id``, ``run_dir``
     - The run's id and directory.
   * - ``fold_dir``
     - The fold's directory. With a single fold, it is ``run_dir``.
   * - ``fold``, ``num_folds``
     - The fold index (from 0) and the number of folds in the splits.
   * - ``seed``
     - The base seed, ``run.seed``. The fold trains under
       ``soma.seed_everything(seed, fold=fold)``, so its effective seed is
       ``seed + fold``.

**Order and hooks.** The observers run after the config's callbacks and the
progress panel. ``History`` is always the first callback, so in ``on_epoch_end``
``trainer.history.rows[-1]`` is the current epoch's row: the same row that
``training_history.json`` writes at the end of the fold. An epoch without a tune
evaluation has only ``epoch``, ``train_loss``, ``lr`` and ``elapsed_seconds`` in
its row, where the file writes ``null`` for the evaluation values. Observers use the
fit-callback hooks (see :ref:`custom-callbacks`); there is no other event.

**The observer contract.** An observer may read these Trainer attributes:
``epoch``, ``total_epochs``, ``train_loss``, ``lr``, ``optimizer_steps``,
``summary`` (the epoch's evaluation values under the names of
:ref:`logged-values`, or ``None`` on an epoch without a tune evaluation)
and ``history.rows``. The running training values are ``train_loss`` and ``lr``. An
observer must not set ``trainer.should_stop``, change the learner or the
optimizer, or draw from the global random state (Python ``random``, NumPy or
torch). An observer that follows these rules changes nothing in the run: its
predictions, metrics and ``bundle/weights.safetensors`` are byte-identical to the
same run without it.

**Failures.** soma does not catch an observer's errors. An exception from the
factory or from an observer hook fails the run like any callback error. During
the fit, every callback's ``on_exception`` runs first. ``run.yaml`` then records
``status: failed`` and the error. To make your telemetry best-effort, catch the
errors inside your own observer.

Reference
---------

.. autoclass:: soma.config.LoadersConfig

.. autoclass:: soma.config.LearnerConfig

.. autoclass:: soma.config.TrainerConfig

.. autoclass:: soma.learn.SegmentationAugment

.. autoclass:: soma.learn.GridBatch

.. autoclass:: soma.models.DenseModel

.. autofunction:: soma.extract.live_encoder

.. autoclass:: soma.data.TileImageSource
   :members: from_manifest
