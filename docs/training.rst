Training
========

Training fits an aggregator, task head, or dense decoder to the selected
features. Configure the training budget, checkpoint selection, and feature
transforms below.

Budget and checkpoint selection
-------------------------------

Training is configured in three sections, one per callable. ``loaders`` holds
:func:`soma.learn.loaders`' keywords (``batch_size``, ``num_workers``,
``pin_memory``, ``persistent_workers`` and the optional ``sampler`` and
``augment`` leaves). ``learner`` names the :class:`soma.learn.Learner` and its
params: the model's ``aggregator`` or ``decoder`` and ``head`` leaves, the
``optimizer`` and ``scheduler`` leaves, and the feature ``normalization`` and
``projection``. ``trainer`` holds the :class:`soma.learn.Trainer`'s keywords.

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

Choose one budget: ``trainer.max_epochs`` counts passes through the training
loader; ``trainer.max_steps`` counts optimizer updates after
``accumulate_grad_batches``, and replaces the epoch budget when set. A cosine
scheduler anneals over the budget, per epoch or per update.

Every component is a ``{name, params}`` leaf whose name is a registry key or a
dotted import path; a bare name keeps the component's default params. The
optimizer defaults to Adam (``lr`` 1e-4, ``weight_decay`` 1e-5) when the learner
names none. ``scheduler`` may be one leaf or a list, each stepped at its own
``interval`` (``epoch`` unless set); a warm-up is a ``constant`` leaf. With no
scheduler the learning rate stays constant. ``learner.name`` is ``mil`` for
the set shape and ``dense`` for the grid shape; a registered or dotted
:class:`soma.learn.Learner` subclass owns custom step logic (see
:doc:`extending`).

The ``checkpoint`` callback selects the weights: ``selection: best`` (the
default) keeps the epoch that is best on ``monitor`` in direction ``mode``
(default: minimize the tune ``loss``; a metric name selects that metric), and
``selection: last`` keeps the final weights and refuses an ``early_stopping``
callback. ``early_stopping`` stops after ``patience`` tune evaluations without
improvement; leave it out to train the full budget.
``trainer.val_every_n_epochs: N`` evaluates the tune split after every N-th
epoch and after the final epoch, for runs where tune evaluation dominates the
epoch cost; selection and patience then count tune evaluations, not epochs.
Tune metrics are always recorded for diagnostics.

By default, training requires a tune split. The ``cohort`` section offers two
explicit alternatives:

- ``cohort.tune_is_test: true`` uses one held-out split for both checkpoint
  selection and test reporting. Provide either tune or test, not both. Use this
  only when reproducing a benchmark protocol that selects on its reported
  cohort.
- ``cohort.allow_missing_tune: true`` reuses train as tune when tune is absent
  and emits a warning. This controls the diagnostic split independently of
  checkpoint selection.

.. autoclass:: soma.config.LoadersConfig

.. autoclass:: soma.config.LearnerConfig

.. autoclass:: soma.config.TrainerConfig

.. _live-mode:

Live re-encoding and augmentation
---------------------------------

Segmentation and detection read cached feature grids by default
(``features.kind: cached``): the frozen encoder runs once and every epoch
reuses its output. With ``features.kind: live``, soma re-encodes each
pre-cropped tile through the frozen encoder at every step instead, so pixel
augmentation reaches the features. This costs one encoder forward pass per
tile and step.

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
:class:`soma.learn.SegmentationAugment`) or a dotted import path, and
``params`` go to its constructor. It applies to the train split only.
Geometric operations transform the image and its label mask together (the mask
with nearest-neighbour resampling); photometric operations change the image
only. Pixels that a rotation or translation brings into the tile get
``ignore_index`` in the mask, so they carry no supervision. ``crop_scale``
takes a random crop of that area fraction and resizes it to the tile size
before the other operations.

Live mode has these limits:

- ``loaders.augment`` requires ``features.kind: live``; cached grids cannot
  be augmented.
- Detection can run with ``features.kind: live`` but cannot be augmented yet:
  joint image and point augmentation waits for torchvision 0.23.
- Live mode reads pre-cropped tiles only. It does not support annotation-sampled
  whole slides, composite encoders, an active ``normalization`` or
  ``projection``, or a ``loaders.sampler``.

.. autoclass:: soma.learn.SegmentationAugment

Feature normalization
---------------------

``learner.params.normalization`` applies a feature adaptor before the
trainable model, allowing encoders with different activation scales to share
training settings.

.. code-block:: yaml

   learner:
     params:
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

``learner.params.projection`` maps frozen features to a common width without
labels. Matching the downstream input dimension helps compare encoders without
changing the aggregator's capacity simply because their embeddings have
different widths.

.. code-block:: yaml

   learner:
     params:
       projection:
         method: pca      # none | pca | random
         target_dim: 512  # required when method != none
         seed: 0          # configurable only for random

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
transform. Neither changes the extracted feature cache, and the plan spells
both out (``method: none`` when off), so they are part of the
:doc:`experiment identity <outputs>`. Each fold with an active transform
writes ``feature_adapter.json``; see :doc:`outputs`.

During training, the live summary panel reports epoch, loss, learning rate,
tune metrics, patience, trainable parameter count, timing, and the active fold
for cross-validation. The same history is saved in ``training_history.json``,
one row per epoch (``epoch``, ``train_loss``, ``lr``, ``elapsed_seconds``, then
``loss`` and one column per tune metric, ``null`` on an epoch without a tune
evaluation).

Predict and evaluate from Python
--------------------------------

:func:`soma.fit` trains a learner (or a bare model). :func:`soma.learn.loaders`
builds one fold's loaders from a set source and a cohort. :func:`soma.predict`
and :func:`soma.evaluate` then score the trained model (see :doc:`evaluation`):

.. code-block:: python

   import soma
   from soma.learn import loaders

   fold = loaders(source, cohort, head=model.task_head, fold=0, batch_size=4)
   result = soma.fit(model, fold.train, fold.tune, max_epochs=50)
   predictions = soma.predict(result.learner, fold.tests["test"])
   evaluation = soma.evaluate(predictions, cohort.targets("test"))
   predictions.to_parquet("predictions_test.parquet")

``predict`` takes callbacks for side artefacts:
``soma.evaluation.heatmaps.SaveAttention(dir)`` writes each bag's normalised tile attention
to ``<dir>/<sample_id>.npz``.

.. _grid-verbs:

Segmentation and detection from Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same verbs train a decoder over feature grids. The source is a
:class:`~soma.data.GridSource` of cached grids, or a
:class:`~soma.data.TileImageSource` of pixels for a model that re-encodes them.
``loaders`` then needs ``targets=``: a :class:`~soma.data.LabelMapSource` for a
segmentation head or a :class:`~soma.data.PointSource` for a detection head.
The model is a :class:`soma.models.DenseModel`; with an ``encoder``
stage it reads pixels, without one it reads cached grids. The example below
re-encodes the tiles with a frozen encoder and augments the train split:

.. code-block:: python

   import soma
   from soma.data import (
       AnnotationManifest, Cohort, ImageManifest, LabelMapSource, TileImageSource,
   )
   from soma.models.decoders.registry import build_decoder_for_grid
   from soma.extract import live_encoder
   from soma.learn import SegmentationAugment, loaders
   from soma.models.heads.segmentation import SegmentationHead
   from soma.models import DenseModel

   cohort = Cohort.from_csv("dataset.csv", "splits.csv", targets=[])
   stage = live_encoder("uni2", target_size=448, spacing_um=0.5, window=224)
   images = TileImageSource.from_manifest(
       ImageManifest.from_csv("dataset.csv"), size=448, spacing_um=0.5
   )
   labels = LabelMapSource.from_manifests(
       AnnotationManifest.from_csv("dataset.csv"), size=(448, 448)
   )
   head = SegmentationHead(num_classes=3, geometry=stage.geometry)
   decoder = build_decoder_for_grid(
       "lightweight_conv", {}, geometry=stage.geometry,
       input_dim=stage.feature_dim, num_classes=3,
   )
   model = DenseModel(decoder, head, encoder=stage)

   fold = loaders(
       images, cohort, head=head, targets=labels, batch_size=8,
       preprocess=stage.preprocessor(),
       augment=SegmentationAugment(horizontal_flip=0.5, vertical_flip=0.5),
   )
   result = soma.fit(model, fold.train, fold.tune, max_epochs=50)
   predictions = soma.predict(result.learner, fold.tests["test"])
   evaluation = soma.evaluate(predictions, labels.subset(cohort.folds[0].tests["test"]))
   predictions.save("predictions_test")

:func:`soma.extract.live_encoder` wraps a slide2vec encoder as a
parameter-free :class:`soma.extract.FrozenEncoder` stage: the foundation model
never enters a checkpoint or an optimizer. :class:`soma.testing.FakeEncoder` is
a weight-free stage with the same contract, for tests.
``preprocess=stage.preprocessor()`` runs the encoder's preprocessing in the
loader workers, after augmentation. ``augment`` is refused for cached grids and
for point targets.

For a grid head, :func:`soma.predict` returns a
:class:`~soma.evaluation.LabelMapPredictions` (segmentation) or a
:class:`~soma.evaluation.PointPredictions` (detection); see
:ref:`grid-predictions`. Detection predictions hold every peak above the head's
``min_score``; pick a threshold on tune and apply it to test::

   from soma.data import PointSource
   from soma.evaluation import sweep_threshold

   points = PointSource.from_manifest(AnnotationManifest.from_csv("dataset.csv"))
   split = cohort.folds[0]
   # ... a DetectionHead, loaders(..., targets=points) and soma.fit as above ...
   tune = soma.predict(result.learner, fold.tune)
   threshold, _ = sweep_threshold(tune, points.subset(split.tune), per_class=True)
   test = soma.predict(result.learner, fold.tests["test"])
   evaluation = soma.evaluate(test.with_threshold(threshold), points.subset(split.tests["test"]))

:class:`soma.learn.DenseLearner` is the learner for both feature paths. Its
``predict_step`` runs the head's ``postprocess``.

Grid ``predict`` callbacks write side artefacts:
:class:`~soma.evaluation.SaveOverlays` (predictions and supervision drawn over
the tile), :class:`~soma.evaluation.SaveProbabilities` (per-sample float16
softmax, segmentation) and :class:`~soma.evaluation.SaveHeatmaps` (per-sample
float16 heatmaps, detection)::

   from soma.evaluation import SaveOverlays, SaveProbabilities

   predictions = soma.predict(
       result.learner,
       fold.tests["test"],
       [SaveOverlays("overlays/test", images=images), SaveProbabilities("probabilities/test")],
   )

.. autoclass:: soma.learn.GridBatch

.. autoclass:: soma.models.DenseModel

.. autofunction:: soma.extract.live_encoder

.. autoclass:: soma.data.TileImageSource
   :members: from_manifest
