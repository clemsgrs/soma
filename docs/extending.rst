Extending soma
==============

soma's trainable components are registered classes. To change how one of them
behaves, subclass the built-in class, override the methods that should differ,
register the subclass under a new name, and reference that name from the
config. soma validates a config by the component's *family*, not by its
registered name. Subclasses inherit family validation and must also support
the representations produced by the other components in the config.

Custom task head
----------------

A task head owns the loss, the metrics and the post-processing of one task
family. The family is the ``task_family`` class attribute, which a subclass
inherits. Every head receives ``task.params`` as constructor keyword arguments.

The example below trains a segmentation decoder with a pure Dice loss instead
of the default cross-entropy plus Dice::

    from soma.tasks.dense_metrics import soft_dice_loss
    from soma.tasks.registry import task_registry
    from soma.tasks.segmentation import SegmentationHead


    class DiceOnlySegmentationHead(SegmentationHead):
        def compute_loss(self, predictions, targets):
            return soft_dice_loss(
                predictions,
                targets["mask"],
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
            )


    task_registry.register("dice_only_segmentation", DiceOnlySegmentationHead)

Then select it in the config::

    task:
      name: dice_only_segmentation
      params:
        num_classes: 3

The methods worth overriding are:

* ``compute_loss(predictions, targets)`` for a custom training objective.
* ``compute_metrics`` for classification, regression and survival metrics.
  Dense heads use ``finalize_eval_metrics(counts)`` to reduce metrics during
  tuning and split evaluation. ``dense_stats`` supplies the per-image statistics
  during dense tuning and segmentation split evaluation.
  Metric *names* are validated against the family's list in
  :mod:`soma.evaluation.metrics`, so a new name must be added
  there before a config can request it.
* ``postprocess(raw_output)`` for a custom prediction format in classification,
  regression and survival split evaluation. Dense segmentation and detection
  tuning and split evaluation bypass it. Whole-slide sliding-window segmentation
  inference also bypasses it: the predictor blends the folds' softmaxes across
  tiles and takes the argmax itself. Detection point decoding uses
  ``_predict_points(heatmap)``.
* ``prediction_columns`` and ``to_columns(processed)`` for what
  :func:`soma.predict` writes. The base implementation covers the
  classification, ordinal, regression and survival families; a head of a new
  family overrides both. :func:`soma.testing.check_task_head` checks a set-shape
  head on one batch: the declared columns match ``to_columns``, every column is
  numeric, the family :class:`~soma.evaluation.Baseline` yields the same
  columns, and :func:`soma.evaluate` on a parquet round-trip agrees with
  ``compute_metrics``.

Every head family can be subclassed this way: classification, regression,
survival, segmentation and detection.

With ``clam_mb``, custom multiclass heads must support per-class branch
representations. Subclass :class:`~soma.tasks.classification.BranchAwareClassificationHead`
to preserve custom loss and metric behavior. soma automatically adapts the
built-in ordinary multiclass head, but rejects registered ordinary multiclass
subclasses instead of replacing them.

Survival subclasses retain the objective of their base class and inherit its
Cox or discrete-time NLL config constraints. ``task.params.loss`` selects the
objective only for the built-in ``survival`` registration.

Custom decoder
--------------

A decoder maps a dense feature grid ``(B, d, h, w)`` to class logits. Subclass
:class:`soma.decoders.base.Decoder`, implement ``forward`` and the
``num_classes`` property, and register it::

    from torch import nn

    from soma.decoders.base import Decoder
    from soma.decoders.registry import decoder_registry


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


    decoder_registry.register("two_stage_conv", TwoStageConvDecoder)

The constructor always receives ``input_dim`` and ``num_classes``; everything
under ``decoder.params`` is passed through as keyword arguments. If the
constructor accepts ``num_upsample_blocks``, soma derives it from the grid
geometry unless the config pins it. The head resizes the decoder's logits to
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

Select it in the config by dotted import path (no registration needed) or by a
name registered in :data:`soma.learn.learner_registry`; ``params`` go to the
constructor, and the config's ``optimizer`` / ``scheduler`` leaves still apply::

    training:
      learner:
        name: my_package.learners.AttentionEntropyLearner
        params:
          weight: 0.01

The methods worth overriding are ``training_step`` (the loss), ``validation_step``
and ``validation_summary`` (how a split is reduced; the summary must contain
``loss``), ``configure_optimizers(budget)`` (parameter groups, unfreezing),
``transfer_batch`` (device placement), ``trainable_state_dict`` (what a
checkpoint restores in memory: the model's state by default, so a learner that owns
trainable modules of its own must widen it to cover them; ``best_model.pt`` holds
the bare model's state either way) and ``predict_step`` (the prediction columns of
a batch). A ``predict_step`` override can add a column; it lands in
``predictions_<split>.parquet`` next to the head's columns::

    from soma.learn import MILLearner, PredictOutput


    class EntropyReportingLearner(MILLearner):
        def predict_step(self, batch):
            step = super().predict_step(batch)
            attention = step.raw.tile_attention.softmax(-1)
            entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1)
            return PredictOutput(columns={**step.columns, "attention_entropy": entropy}, raw=step.raw)

:func:`soma.testing.check_learner`
checks a learner against the contract on one batch. From Python, :func:`soma.fit`
trains a learner, or a bare model wrapped in its shape's learner, with
``Checkpoint``, ``EarlyStopping`` and ``Progress`` installed.

Registering before the run
--------------------------

Registration is a Python side effect, so the module that registers a component
must be imported before the config is loaded. With the Python API, import it
before building the :class:`~soma.pipeline.Pipeline`. From the command line,
run soma from a small script that imports your module and then calls
``soma.cli.main``::

    import my_components  # registers the head and decoder

    from soma.cli import main

    main(["config.yaml"])

The names ``soma list tasks`` and ``soma list decoders`` print include every
registered component, built-in or not.

The same import is needed when reopening the run from Python: the persisted
config names the custom head, and soma resolves its family through the
registry.

Components that ship with soma's benchmarks (for example the EVA segmentation
head ``eva_segmentation``, decoder ``eva_conv_ms`` and aggregator ``eva_abmil``)
need no such import: the task, decoder and aggregator registries import
:mod:`soma.benchmarks` the first time a lookup misses, so a saved benchmark
config loads in a fresh process as is.
