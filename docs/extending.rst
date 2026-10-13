Extending soma
==============

soma's trainable components are classes held in registries. To change how one
of them behaves, subclass the built-in class, override the methods that should
differ, and reference the subclass from the config, by its import path or by a
name you register. soma validates a config by the component's *family*, not by
its name. Subclasses inherit family validation and must also support the
representations produced by the other components in the config.

.. _registering-components:

Registering components
----------------------

Every swappable component kind has a registry: aggregators, heads, decoders,
learners, optimizers, schedulers, callbacks, samplers, augmentations and
benchmarks. Encoders are listed too, but they belong to slide2vec and are
registered there. :func:`soma.components` names the kinds, and
``soma.components(kind)`` lists the components of one kind with their params
and defaults; ``soma list [KIND]`` prints the same.

A config can name a component in two ways:

* **By import path**, such as ``mylab.models.FancyMIL``. There is nothing to
  register: soma imports the class and passes the ``params`` to its
  constructor as they are. soma does not validate them, does not list the
  class, and emits a one-line warning the first time it resolves the path.
* **By a registered name.** Register the class once, when your module is
  imported, together with a *params dataclass*: one field for each constructor
  keyword a user sets, with its default. soma then lists the component with its
  defaults, accepts the short name, and checks the params when it builds the
  component::

      from dataclasses import dataclass

      import soma


      @dataclass
      class FancyMILParams:
          hidden_dim: int = 128
          dropout: float = 0.25

          def __post_init__(self):
              if not 0.0 <= self.dropout <= 1.0:
                  raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")


      soma.register("aggregators", "fancy_mil", FancyMIL, params=FancyMILParams)

  Validation goes in ``__post_init__``: an unknown param, or a value it
  rejects, fails the run before training starts. The params of every
  component kind are checked this way, heads included. Values the run supplies
  are not fields: the feature width ``input_dim``, a head's ``metrics``, a
  dense head's ``geometry``. soma reads only the dataclass passed to
  ``register``; a ``Params`` attribute on the class is ignored.

A head can also register an ``infer(cohort, params)`` hook. It returns the
params the cohort decides, such as the class count and label map of a
classification head or the bin count of a survival head; values set in the
config take precedence. A registration does not inherit anything from the
registration of its base class, so a subclass of a built-in head names the
params and the hook it reuses. Both are public::

    from soma.models.heads import ClassificationParams, infer_label_map

    soma.register(
        "heads",
        "label_smoothing_classification",
        LabelSmoothingHead,
        params=ClassificationParams,
        infer=infer_label_map,
    )

A head named by import path gets no ``infer`` hook, so its config sets the
values a hook would infer (``num_classes``, ``label_map``).

Custom task head
----------------

A task head owns the loss, the metrics and the post-processing of one task
family. The family is the ``task_family`` class attribute, which a subclass
inherits. Every head receives its leaf's ``params`` as constructor keyword
arguments.

The example below trains a segmentation decoder with a pure Dice loss instead
of the default cross-entropy plus Dice::

    import soma
    from soma.models.heads import SegmentationParams
    from soma.models.heads.dense_metrics import soft_dice_loss
    from soma.models.heads.segmentation import SegmentationHead


    class DiceOnlySegmentationHead(SegmentationHead):
        def compute_loss(self, predictions, targets):
            return soft_dice_loss(
                predictions,
                targets["mask"],
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
            )


    soma.register(
        "heads", "dice_only_segmentation", DiceOnlySegmentationHead, params=SegmentationParams
    )

Then select it in the config::

    learner:
      name: dense
      params:
        decoder: {name: lightweight_conv}
        head:
          name: dice_only_segmentation
          params:
            num_classes: 3

The methods worth overriding are:

* ``compute_loss(predictions, targets)`` for a custom training objective.
* ``compute_metrics`` for classification, regression and survival metrics.
  Segmentation and detection heads instead return per-image statistics from
  ``dense_stats`` and reduce them with ``finalize_eval_metrics(counts)``; the
  learner uses these for the validation metrics computed during training. Split
  evaluation scores the saved predictions through :func:`soma.evaluate`.
  Metric *names* are validated against the family's list in
  :mod:`soma.evaluation.metrics`, so a new name must be added
  there before a config can request it.
* ``postprocess(raw_output)`` for a custom prediction format. The segmentation
  head returns one label map per sample; the detection head returns every peak
  above ``min_score`` in level-0 pixels. Whole-slide sliding-window
  segmentation inference bypasses it: the predictor blends the folds' softmaxes
  across tiles and takes the argmax itself.
* ``prediction_columns`` and ``to_columns(processed)`` for what
  :func:`soma.predict` writes. The base implementation covers the
  classification, ordinal, regression and survival families; a head of a new
  family overrides both. Segmentation and detection heads also implement
  ``assemble_predictions(sample_ids, columns)``, which builds the
  :class:`~soma.evaluation.LabelMapPredictions` or
  :class:`~soma.evaluation.PointPredictions` that ``predict`` returns.
  :func:`soma.testing.check_task_head` checks a set-shape
  head on one batch: the declared columns match ``to_columns``, every column is
  numeric, the family :class:`~soma.evaluation.Baseline` yields the same
  columns, and :func:`soma.evaluate` on a parquet round-trip agrees with
  ``compute_metrics``.

Every head family can be subclassed this way: classification, regression,
survival, segmentation and detection.

With ``clam_mb``, custom multiclass heads must support per-class branch
representations. Subclass :class:`~soma.models.heads.classification.BranchAwareClassificationHead`
to preserve custom loss and metric behavior. soma automatically adapts the
built-in ordinary multiclass head, but rejects registered ordinary multiclass
subclasses instead of replacing them.

Survival subclasses retain the objective of their base class and inherit its
Cox or discrete-time NLL config constraints: a subclass of
:class:`~soma.models.heads.survival.CoxSurvivalHead` trains the Cox objective, any other
survival head the discrete-time NLL.

Custom decoder
--------------

A decoder maps a dense feature grid ``(B, d, h, w)`` to class logits. Subclass
:class:`soma.models.decoders.base.Decoder`, implement ``forward`` and the
``num_classes`` property, and register it::

    from dataclasses import dataclass

    from torch import nn

    import soma
    from soma.models.decoders.base import Decoder


    @dataclass
    class TwoStageConvParams:
        hidden_dim: int = 64


    class TwoStageConvDecoder(Decoder):
        def __init__(self, *, input_dim, num_classes, hidden_dim=64):
            super().__init__()
            self._num_classes = num_classes
            self.layers = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(hidden_dim, num_classes, kernel_size=3, padding=1),
            )

        def forward(self, X):
            return self.layers(X)

        @property
        def num_classes(self):
            return self._num_classes


    soma.register("decoders", "two_stage_conv", TwoStageConvDecoder, params=TwoStageConvParams)

The constructor always receives ``input_dim`` and ``num_classes``; everything
under ``decoder.params`` is checked against the params dataclass and passed as
keyword arguments. If the constructor accepts ``num_upsample_blocks``, soma
derives it from the grid geometry unless the config pins it. The head resizes the decoder's logits to
the mask size, so a decoder may output any spatial resolution.

Custom learner
--------------

A learner owns the model and the step logic: how a batch is forwarded, how the
loss is composed, how a validation split is reduced, and which optimizer and
schedulers train it. soma ships one per shape: :class:`soma.learn.MILLearner`
for bags and single embeddings, :class:`soma.learn.DenseLearner` for feature
grids. The generic :class:`soma.learn.Trainer` knows nothing about tasks or
heads, so an intermediate loss is one override of ``training_step``::

    from soma.learn import MILLearner


    class AttentionEntropyLearner(MILLearner):
        def __init__(self, model, *, weight=0.01, **kwargs):
            super().__init__(model, **kwargs)
            self.weight = weight

        def training_step(self, batch):
            output = self.forward_batch(batch)
            attention = output.tile_attention.softmax(-1)
            entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1).mean()
            return self.compute_loss(output, batch) + self.weight * entropy

Select it as the ``learner`` by import path (no registration needed) or by a
name registered with ``soma.register("learners", ...)``. Its params hold the
model's leaves, the ``optimizer`` / ``scheduler`` leaves and the learner's own
keywords, which go to the constructor. The run's ``plan.yaml`` keeps the import
path as written and spells out every param, and the bundle's ``provenance.json``
records the installed version of the package the path imports from::

    learner:
      name: my_package.learners.AttentionEntropyLearner
      params:
        aggregator: {name: abmil}
        head: {name: binary_classification}
        weight: 0.01

The methods worth overriding are ``training_step`` (the loss), ``validation_step``
and ``validation_summary`` (how a split is reduced; the summary must contain
``loss``), ``configure_optimizers(budget)`` (parameter groups, unfreezing),
``transfer_batch`` (device placement), ``trainable_state_dict`` (what a
checkpoint restores in memory: the model's state by default, so a learner that owns
trainable modules of its own must widen it to cover them; the fold's bundle holds the
bare model's state either way) and ``predict_step`` (the prediction columns of
a batch). A ``predict_step`` override can add a column; it lands in
``predictions_<split>.parquet`` next to the head's columns::

    from soma.learn import MILLearner, PredictOutput


    class EntropyReportingLearner(MILLearner):
        def predict_step(self, batch):
            step = super().predict_step(batch)
            attention = step.raw.tile_attention.softmax(-1)
            entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1)
            return PredictOutput(columns={**step.columns, "attention_entropy": entropy}, raw=step.raw)

A grid learner is a :class:`soma.learn.DenseLearner` subclass, named the same
way in ``learner.name``. Its batches are :class:`soma.learn.GridBatch`
objects holding cached ``features`` or live ``pixels`` and the raw supervision
in ``targets``.

:func:`soma.testing.check_learner`
checks a learner against the contract on one batch. From Python, :func:`soma.fit`
trains a learner, or a bare model wrapped in its shape's learner, with
``Checkpoint``, ``EarlyStopping`` and ``Progress`` installed.

.. _hand-composition:

A run composed by hand
----------------------

:class:`soma.Pipeline` is the public verbs called in a fixed order, so the same
fold can be written by hand, for example to change one step. Each fold runs:

1. ``soma.seed_everything(seed, fold=fold)``
2. fit the feature adaptor on the train split only
3. build the model (weight initialisation draws random numbers here)
4. build the loaders
5. ``Baseline.fit`` on the train targets, for samples without features
6. ``soma.fit`` with ``Checkpoint`` and ``EarlyStopping``
7. a tune-time step: detection sweeps its score threshold on tune
8. per split: ``soma.predict`` (plus baseline rows), then ``soma.evaluate``
9. ``soma.save`` the fold's bundle

The order matters because random numbers are drawn in it: a hand composition
that follows it, with the config's values, gives byte-identical predictions and
weights. soma's own test suite checks this for a MIL, a segmentation and a
detection run. A set-shape fold over cached features looks like this::

    import soma
    from soma.models.aggregators import ABMIL
    from soma.data import CachedSetSource, Cohort
    from soma.evaluation import Baseline, Predictions
    from soma.learn import Checkpoint, EarlyStopping, MILLearner, loaders
    from soma.models.heads.classification import BinaryClassificationHead
    from soma.learn.feature_adaptor import FeatureAdaptor
    from soma.models import MILModel

    cohort = Cohort.from_csv("dataset.csv", "splits.csv", targets=["label"])
    source = CachedSetSource("features/")
    split = cohort.folds[0]

    soma.seed_everything(0, fold=0)
    adaptor = FeatureAdaptor(method="zscore", num_features=source.feature_dim)
    adaptor.fit([source.load(sid) for sid in split.train])
    aggregator = ABMIL(input_dim=source.feature_dim)
    head = BinaryClassificationHead(
        input_dim=aggregator.output_dim, num_classes=2, label_map={"normal": 0, "tumor": 1}
    )
    learner = MILLearner(
        MILModel(aggregator, head, feature_adaptor=adaptor),
        optimizer={"name": "adam", "params": {"lr": 1e-4, "weight_decay": 1e-5}},
        schedulers=("cosine",),
    )
    fold = loaders(source, cohort, head=head, batch_size=1)
    baseline = Baseline.fit(
        cohort.targets("train"), task_family=head.task_family, classes=head.classes,
        target_keys=head.target_keys,
    )
    result = soma.fit(
        learner, fold.train, fold.tune,
        callbacks=[Checkpoint(monitor="loss", mode="min"), EarlyStopping(patience=10)],
        max_epochs=50,
    )
    predictions = Predictions.concat([soma.predict(result.learner, fold.tests["test"]), baseline.predict([])])
    evaluation = soma.evaluate(predictions, cohort.targets("test"), metrics=["auroc"])
    soma.save(result, "runs/by-hand/bundle")

A ``Checkpoint`` or ``EarlyStopping`` passed in ``callbacks`` replaces the one
``soma.fit`` would build from ``monitor``, ``mode`` and ``patience``. A model
built this way saves, but :func:`soma.load` cannot rebuild it (see
:ref:`model-bundle`).

Custom encoder stage or image source
------------------------------------

The live path (see :ref:`live-mode`) re-encodes pixels through an encoder
stage inside :class:`soma.models.DenseModel`. Any parameter-free
``nn.Module`` with the stage contract can replace
:class:`soma.extract.FrozenEncoder`: a ``geometry``, a ``feature_dim``, an
``input_device``, a ``preprocessor()`` and a deterministic ``forward`` over
preprocessed or raw uint8 pixel batches.
:func:`soma.testing.check_encoder_stage` checks a stage, and
:class:`soma.testing.FakeEncoder` is a weight-free example. An image source
serves ``(3, H, W)`` uint8 pixels by sample id with a ``geometry`` and a
``spacing``, like :class:`soma.data.TileImageSource`;
:func:`soma.testing.check_image_source` checks one.

Registering before the run
--------------------------

Registration is a Python side effect, so the module that registers a component
must be imported before the config is loaded. With the Python API, import it
before building the :class:`~soma.pipeline.Pipeline`. From the command line,
run soma from a small script that imports your module and then calls
``soma.cli.main``::

    import my_components  # registers the head and decoder

    from soma.cli import main

    main(["run", "config.yaml"])

``soma list heads`` and ``soma list decoders`` print every registered
component, built-in or not, with its params.

The same import is needed when reopening the run from Python: the persisted
config names the custom head, and soma resolves its family through the
registry.

Components that ship with soma's benchmarks (for example the EVA segmentation
head ``eva_segmentation``, decoder ``eva_conv_ms`` and aggregator ``eva_abmil``)
follow the same rule: :mod:`soma.benchmark` registers them, and the benchmarks
themselves, when it is imported. ``import soma`` does not import it; the
``soma`` command does at startup. From Python, ``import soma.benchmark``
before loading a saved benchmark config, or name the component by its dotted
path. Until then, ``soma.components("benchmarks")`` and a lookup of one of
those names fail with an error that says so.
