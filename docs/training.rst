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
       - {name: checkpoint, params: {monitor: auroc, mode: max}}
       - {name: early_stopping, params: {monitor: auroc, mode: max, patience: 10}}

``loaders`` holds ``batch_size``, ``num_workers``, ``pin_memory``,
``persistent_workers`` and the optional ``sampler`` and ``augment`` leaves.
``learner.params`` holds the model's ``aggregator`` or ``decoder`` leaf and its
``head`` leaf, the ``optimizer`` and ``scheduler`` leaves, the feature
``normalization`` and ``projection``, and the keywords of a custom learner.
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
(default: minimize the tune ``loss``; a metric name selects that metric), and
``selection: last`` keeps the final weights; it refuses an ``early_stopping``
callback and needs ``trainer.val_every_n_epochs: 1``. ``early_stopping`` stops after ``patience`` tune evaluations without
improvement; leave it out to train the full budget.

``trainer.val_every_n_epochs: N`` evaluates the tune split after every N-th
epoch and after the final epoch, for runs where tune evaluation dominates the
epoch cost; selection and patience then count tune evaluations, not epochs.
Tune metrics are always recorded for diagnostics.

While a fold trains, a summary panel shows the epoch, loss, learning rate, tune
metrics, patience, trainable parameter count, timing and, for cross-validation,
the active fold. Each fold writes the same history to ``training_history.json``,
one row per epoch: ``epoch``, ``train_loss``, ``lr``, ``elapsed_seconds``, then
``loss`` and one column per tune metric, ``null`` on an epoch without a tune
evaluation.

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
``(M, P, D)`` grid of tiles per sample, read by ``hipt``), slide-encoder
embeddings and single-encoder dense grids; other paths refuse an active
transform. On a hierarchical store, the adaptor is fitted on the stored tiles of
the train samples, so the fit is the same as on those tiles stored as a flat
bag. The padding added when a batch is collated never enters the fit. A
composite encoder normalizes each member's block before concatenation instead
(``member_norm``, see :doc:`encoders/composite`).

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
   result = soma.fit(model, fold.train, fold.tune, max_epochs=50, monitor="auroc", mode="max")
   result.selected_epoch, result.history

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

:class:`soma.testing.FakeEncoder` is a weight-free stage with the same contract.
The end-to-end scenarios ``grid_live_segmentation`` and ``grid_live_detection``
(`tests/e2e/test_grid.py
<https://github.com/clemsgrs/soma/blob/main/tests/e2e/test_grid.py>`_) run the whole
grid flow on synthetic tiles with it, from loaders to evaluation.

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
