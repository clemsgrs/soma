Extending soma
==============

Any step of a soma run can run your own code instead: a feature source, a learner,
a registered component, an encoder stage, a callback. Each extension is an ordinary
Python object that you pass to a verb or name in a config, and a check in
:mod:`soma.testing` proves it meets its contract before a run depends on it. The
sections below follow the order in which the pipeline runs the steps.

Where to cut in
---------------

:class:`soma.Pipeline` calls the public verbs in a fixed order: it builds the
features once, then runs the same steps for every fold. Random numbers are drawn in
this order, so a run composed by hand in the same order, with the plan's values,
gives byte-identical predictions and weights.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Step
     - What runs
     - Your code
   * - Features, once per run
     - :func:`soma.extract` returns a source
     - A new foundation model (:ref:`custom-encoder`); a source of your own
       (:ref:`bring-your-own-features`); an encoder stage or an image source on the
       live path (:ref:`custom-encoder-stage`)
   * - 1. Seed
     - ``soma.seed_everything(seed, fold=fold)``
     -
   * - 2. Feature adaptor
     - Normalization and projection, fitted on the train split only
     - Configured, not replaced (see :doc:`training`)
   * - 3. Model
     - The aggregator or decoder and the head; weight initialisation draws here
     - Registered components (:ref:`registering-components`)
   * - 4. Loaders
     - :func:`soma.learn.loaders`
     - An ``augment`` callable on the live grid path; a ``sampler`` leaf for
       cached segmentation
   * - 5. Baseline
     - ``Baseline.fit`` on the train targets, for samples without features
     -
   * - 6. Fit
     - :func:`soma.fit` with ``Checkpoint`` and ``EarlyStopping``
     - A learner (:ref:`custom-learner`); fit callbacks (:ref:`custom-callbacks`)
   * - 7. Tune step
     - Detection sweeps its score threshold on tune
     -
   * - 8. Predict and evaluate
     - :func:`soma.predict` (plus baseline rows), then :func:`soma.evaluate`, per
       split
     - The learner's ``predict_step``; predict callbacks; metric callables
   * - 9. Save
     - :func:`soma.save` writes the fold's bundle
     - ``soma.learn.record_builder``, so that :func:`soma.load` can rebuild a model
       you built by hand

Section 3 of the getting-started notebook, :ref:`the same run from the verbs
</getting-started.ipynb#3.-The-same-run-from-the-verbs>`, runs these steps one cell
each and compares the result with the pipeline's run.

.. _custom-encoder:

A new foundation model
----------------------

Encoders belong to slide2vec, which extracts the features, so a new foundation model
is registered with slide2vec, not with :func:`soma.register`. The simplest way needs
no package. Write the class in a file of its own, ``my_encoder.py``, and register it
with ``slide2vec.register_encoder``:

.. code-block:: python
   :caption: my_encoder.py

   import torch
   from slide2vec import TorchTileEncoder, register_encoder


   @register_encoder(
       "my-tile-model",
       encode_dim=768,
       input_size=224,
       supports_variable_input_size=False,
       supported_spacing_um=0.5,
       precision="fp16",
       source="/models/my-tile-model.ts",
   )
   class MyTileModel(TorchTileEncoder):
       def __init__(self, *, output_variant=None):
           model = torch.jit.load("/models/my-tile-model.ts", map_location="cpu")
           super().__init__(
               model,
               encode_dim=768,
               input_size=224,
               mean=(0.485, 0.456, 0.406),
               std=(0.229, 0.224, 0.225),
               output_variant=output_variant,
           )

Then import the file in the script that runs soma, and name the encoder in
``features.encoder``:

.. code-block:: python
   :caption: run.py

   from dataclasses import replace

   import soma

   import my_encoder  # registers "my-tile-model"

   config = soma.presets.mil_classification("my-tile-model", "abmil")
   config.features.execution = replace(config.features.execution, num_gpus=1)
   result = soma.Pipeline(config).run()

Keep the class out of the run script. The feature cache checks the encoder's code on
every run: it compares the file that defines the class with the file that extracted
the cached features (see :ref:`feature-identity`). If the class is in the run script,
the check covers the whole script, so any edit to it, even to a training setting,
fails the check.

The decorator records the model's input size, spacing and precision. soma reads them
to tile the slides and to check the extraction settings, without building the model,
so load the weights in the constructor, never at import time. ``mean`` and ``std``
are the model's own normalization. slide2vec's `bring your own encoder
<https://clemsgrs.github.io/slide2vec/models.html#bring-your-own-encoder>`_ guide
describes the base classes (``TorchTileEncoder``, ``TimmTileEncoder``) and every
field of the decorator.

The registration exists only in the Python process that ran your script. Two cases
start processes that never import it:

- **The command line.** ``soma run`` and ``soma reproduce`` start a fresh process.
- **Extraction on several GPUs.** slide2vec runs one worker process per GPU, and it
  uses every visible GPU when ``features.execution.num_gpus`` is unset. The example
  sets it to ``1`` for this reason.

For these cases, install the encoder as a slide2vec plugin package (see
:doc:`benchmark-in-house-encoder`); every process then finds it by name.

After you edit ``my_encoder.py``, the feature cache refuses the features of the old
code (see :ref:`feature-identity`). Two changes stay undetected: new weights behind
the same path, and edits to code that the class inherits or imports from other files.
After such a change, delete the encoder's feature cache directory (see
:doc:`caching`), or a run reuses the features of the old model.

.. _bring-your-own-features:

Bring your own features
-----------------------

Features enter a run as a *source*, an object that serves one tensor per sample
id. A set source (:class:`soma.SetSource`) serves ``(D,)``, ``(N, D)`` or
``(M, K, D)`` features of one declared ``rank``, with optional level-0 tile
coordinates. A grid source (:class:`soma.GridSource`) serves ``(D, H, W)`` token
grids with their ``geometry`` and ``spacing``. Two adapters cover the common cases:
:func:`soma.data.from_arrays` wraps a mapping from sample id to tensor or array, and
:func:`soma.data.from_directory` reads ``<id>.pt``, ``.npy``, ``.npz`` or ``.h5``
files, with optional ``<id>.coords.*`` siblings. Ids are never guessed from the
directory structure; ``ids=`` maps sample ids to file stems.

Any object with the protocol's members is a source, with nothing to subclass or
register:

.. code-block:: python

   import h5py
   import torch

   import soma


   class H5Bags:
       """One (N, D) bag per sample, read from one HDF5 file."""

       rank = 2

       def __init__(self, path):
           self.file = h5py.File(path, "r")
           self.sample_ids = list(self.file)
           self.feature_dim = self.file[self.sample_ids[0]].shape[-1]

       def load(self, sample_id):
           return torch.from_numpy(self.file[sample_id][()])

       def coords(self, sample_id):
           return None


   source = H5Bags("features.h5")
   soma.testing.check_set_source(source)

:func:`soma.testing.check_set_source` and :func:`soma.testing.check_grid_source`
load the samples (``limit=`` bounds how many) and raise a
:class:`~soma.testing.ConformanceError` that names the method and the sample id
breaking the contract. A source that passes trains like an extracted one: pass it
to :func:`soma.learn.loaders`, or run ``soma.Pipeline(config, source=source)``,
which then extracts nothing. From the command line, ``soma run config.yaml
--feature-dir DIR`` attaches a directory of cached features (see :doc:`cli`). The
pipeline refuses a supplied source for annotation-sampled whole slides, whose
regions are drawn during extraction.

In the notebook, the Source cell of :ref:`section 3
</getting-started.ipynb#3.-The-same-run-from-the-verbs>` wraps extracted features
with ``from_arrays`` and checks them.

.. _custom-learner:

A custom learner
----------------

A learner owns the model and the step logic. soma ships one per shape,
:class:`soma.learn.MILLearner` for sets and :class:`soma.learn.DenseLearner` for
grids, and the generic :class:`soma.learn.Trainer` knows nothing about tasks or
heads. Subclass the learner of your shape and override the steps that should
differ:

- ``training_step(batch)`` returns the loss, or a dict holding ``"loss"``. An extra
  loss term is one override that calls ``forward_batch`` and ``compute_loss`` and
  adds the term.
- ``validation_step(batch)`` and ``validation_summary(outputs)`` reduce a split. The
  summary must contain ``loss``; its keys are what ``Checkpoint`` and
  ``EarlyStopping`` can monitor.
- ``predict_step(batch)`` returns a :class:`soma.learn.PredictOutput`. A column it
  adds lands in ``predictions_<split>.parquet`` next to the head's columns.
- ``configure_optimizers(budget)`` builds the optimizer and schedulers, for
  parameter groups or unfreezing.
- ``transfer_batch`` places a batch on the device. ``trainable_state_dict`` is what
  a checkpoint keeps in memory: the model's state by default, so a learner that owns
  trainable modules of its own must widen it. The bundle holds the bare model's
  state either way.

.. code-block:: python

   class EntropyMIL(soma.learn.MILLearner):
       """MIL plus a penalty on the entropy of each bag's attention."""

       def __init__(self, model, *, weight=0.01, **kwargs):
           super().__init__(model, **kwargs)
           self.weight = weight

       def training_step(self, batch):
           output = self.forward_batch(batch)
           attention = output.tile_attention.masked_fill(~batch.mask, float("-inf")).softmax(-1)
           entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1).mean()
           return self.compute_loss(output, batch) + self.weight * entropy

       def predict_step(self, batch):  # adds an attention_entropy column
           step = super().predict_step(batch)
           attention = step.raw.tile_attention.softmax(-1)
           entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1)
           return soma.learn.PredictOutput(columns={**step.columns, "attention_entropy": entropy}, raw=step.raw)

:func:`soma.testing.check_learner` runs one batch through every method soma calls
and names the one that breaks the contract: every trainable parameter in an
optimizer group, a finite loss with gradients, a validation summary of finite floats
that includes ``loss``, and a ``trainable_state_dict`` that round-trips. A grid
learner subclasses ``DenseLearner`` the same way; its batches are
:class:`soma.learn.GridBatch` objects holding cached ``features`` or live ``pixels``
and the raw supervision in ``targets``.

A config names the learner by its dotted import path, or by a name registered under
``learners``. Its params hold the model's leaves, the ``optimizer`` and
``scheduler`` leaves, and the learner's own keywords, which go to its constructor:

.. code-block:: yaml

   learner:
     name: my_package.learners.EntropyMIL
     params:
       aggregator: {name: abmil}
       head: {name: binary_classification}
       weight: 0.01

The run's ``plan.yaml`` keeps the import path as written and spells out every
param, and the run records the code the path resolved to (see
:ref:`component-record`). In the notebook, :ref:`section 4
</getting-started.ipynb#4.-Change-a-step>` writes and checks this learner, and
:ref:`section 5 </getting-started.ipynb#5.-Hand-it-back-to-the-pipeline>` runs it
through the pipeline by its dotted path.

.. _registering-components:

A custom component through the registry
---------------------------------------

Every swappable component kind has a registry: ``aggregators``, ``heads``,
``decoders``, ``learners``, ``optimizers``, ``schedulers``, ``callbacks``,
``samplers`` and ``augmentations``, plus ``benchmarks`` once :mod:`soma.benchmark`
is imported. Encoders are listed too, but they belong to slide2vec and are
registered there (see :ref:`custom-encoder`). ``soma.components()``
names the kinds and ``soma.components(kind)`` lists one kind's entries, built-in or
not, with their params and defaults; ``soma list [KIND]`` prints the same.

A config names a component in one of two ways:

- **By dotted import path**, such as ``my_package.models.FancyMIL``. Nothing is
  registered: soma imports the class and passes the ``params`` to its constructor
  as they are. It does not validate or list the class, and warns once when it
  resolves the path.
- **By a registered name.** Register the class when your module is imported,
  together with a *params dataclass*: one field per constructor keyword a user
  sets, with its default. soma then lists the component with its defaults, accepts
  the short name, and checks the params before the run starts.

The first worked example is a task head. A head owns the loss, the metrics and the
post-processing of one task family. The family is the ``task_family`` class
attribute, which a subclass inherits, and soma validates a config by family, not by
name; a subclass must still accept the representations the other components
produce. This head trains a binary classifier with label-smoothed cross-entropy:

.. code-block:: python

   import torch
   import torch.nn.functional as F

   import soma
   from soma.learn.sample_dataset import SampleBatch
   from soma.models.heads import ClassificationParams, infer_label_map


   class LabelSmoothingHead(soma.models.BinaryClassificationHead):
       def compute_loss(self, predictions, targets):
           return F.cross_entropy(predictions, targets["label"], label_smoothing=0.1)


   soma.register(
       "heads", "label_smoothing_classification", LabelSmoothingHead,
       params=ClassificationParams, infer=infer_label_map,
   )

   soma.testing.check_task_head(
       LabelSmoothingHead(input_dim=8, num_classes=2),
       SampleBatch(
           features=torch.randn(8, 8),
           targets={"label": torch.tensor([0, 1] * 4)},
           sample_ids=tuple(f"s{i}" for i in range(8)),
       ),
   )

``params`` is the dataclass soma checks a leaf's params against. Validation goes in
its ``__post_init__``, so an unknown param or a rejected value fails before
training. Values the run supplies are not fields: the feature width ``input_dim``,
a head's ``metrics``, a grid head's ``geometry``. soma reads only the dataclass
passed to ``register``; a ``Params`` attribute on the class is ignored. A head can
also register an ``infer(cohort, params)`` hook that returns the params the cohort
decides, such as a classification head's class count and label map or a survival
head's bin count; a value set in the config wins. A ``fit(records, params)`` hook
returns the params one fold's training samples decide, such as the survival head's
bin edges under ``binning: train_quantiles``: it runs once per fold, and what it
returns goes into that fold's model, never into the plan. A hook may also only check
a fold and return ``{}``, as the survival head's does under ``binning: fixed``: it
logs the training cases in each bin and refuses a fold whose cases all fall in one.
A ``ValueError`` from the hook stops the run when it starts, before anything is
extracted, and names the fold. A registration inherits nothing from the registration
of its base class, so a subclass names the params and the hooks it reuses
(``infer_num_bins`` and ``fit_bin_edges`` for a survival head); all are public. A head named by dotted path gets no hook, so its config sets
``num_classes`` and ``label_map`` itself.

:func:`soma.testing.check_task_head` checks a head on one batch: its declared
``prediction_columns``, the columns of the family's
:class:`~soma.evaluation.Baseline`, and that :func:`soma.evaluate` on a disk round
trip agrees with ``compute_metrics`` (for a grid head, also the logits' size and
the prediction type it assembles).

The head methods worth overriding are:

- ``compute_loss(predictions, targets)``, the training objective.
- ``compute_metrics``, the training-time validation metrics of a set head. A grid
  head returns per-image statistics from ``dense_stats`` and reduces them with
  ``finalize_eval_metrics(counts)``. Split evaluation always scores the saved
  predictions through :func:`soma.evaluate`.
- ``postprocess(raw_output)``, the prediction format. The segmentation head returns
  one label map per sample and the detection head every peak above ``min_score``,
  in level-0 pixels. Whole-slide sliding-window segmentation bypasses it: the
  predictor blends the folds' softmaxes across tiles and takes the argmax itself.
- ``prediction_columns`` and ``to_columns(processed)``, what :func:`soma.predict`
  writes. The base class covers the classification, ordinal, regression and
  survival families; a head of a new family overrides both. A grid head also
  implements ``assemble_predictions(sample_ids, columns)``.

Two families constrain their subclasses. With ``clam_mb``, a custom multiclass head
must handle per-class branch representations: subclass
:class:`~soma.models.heads.classification.BranchAwareClassificationHead`. soma swaps
the built-in multiclass head for it, but refuses a registered ordinary multiclass
subclass. A survival subclass keeps its base class's objective: a subclass of
:class:`~soma.models.heads.survival.CoxSurvivalHead` trains the Cox objective, any
other survival head the discrete-time NLL.

.. _custom-aggregator:

The second worked example is an aggregator, the component a MIL user most often
writes. It subclasses :class:`soma.models.aggregators.base.Aggregator`, and its
constructor receives ``input_dim`` and the leaf's params. ``forward(X, mask)`` maps a
``(B, N, D)`` batch of bags to an
:class:`~soma.models.aggregators.base.AggregatorOutput`. The bags of a batch are
padded to the longest one, and the ``(B, N)`` ``mask`` is ``True`` on the real tiles.
``mask`` can be ``None``, which means that every tile is real. The ``output_dim``
property is the width of the ``bag_representation`` the head receives. This
aggregator projects each tile and averages the real ones:

.. code-block:: python

   from dataclasses import dataclass

   import torch
   from torch import nn

   import soma
   from soma.models.aggregators.base import Aggregator, AggregatorOutput


   class ProjectedMean(Aggregator):
       def __init__(self, input_dim, hidden_dim=64):
           super().__init__()
           self.hidden_dim = hidden_dim
           self.project = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

       def forward(self, X, mask=None):
           if mask is None:  # nothing is padded
               mask = torch.ones(X.shape[:2], dtype=torch.bool, device=X.device)
           H = self.project(X).masked_fill(~mask.unsqueeze(-1), 0.0)  # padded tiles add 0
           z = H.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
           return AggregatorOutput(bag_representation=z)

       @property
       def output_dim(self):
           return self.hidden_dim


   @dataclass
   class ProjectedMeanParams:
       hidden_dim: int = 64


   soma.register("aggregators", "projected_mean", ProjectedMean, params=ProjectedMeanParams)
   soma.testing.check_aggregator(ProjectedMean(input_dim=16), input_dim=16)

   config = soma.presets.mil_classification("uni2", aggregator="projected_mean")

:func:`soma.testing.check_aggregator` runs the aggregator in eval mode on random bags
and raises a :class:`~soma.testing.ConformanceError` when the ``bag_representation``
is not ``(B, output_dim)``, when ``mask=None`` fails or gives another output than a
mask that marks every tile real, or when padded tiles change a bag's output. It fills
the padding with large noise, so an aggregator that pools padded tiles fails. In a
config, the leaf is ``learner.params.aggregator: {name: projected_mean, params:
{hidden_dim: 128}}``.

The other kinds follow the same pattern:

- **Decoders** subclass :class:`soma.models.decoders.base.Decoder`. The constructor
  receives ``input_dim``, ``num_classes`` and the leaf's params, plus
  ``num_upsample_blocks`` derived from the grid geometry when it accepts one and the
  config leaves it unset. ``forward`` maps a ``(B, D, h, w)`` grid to class logits
  at any resolution, since the head resizes them to the mask, and ``num_classes``
  is a property.
- **Metrics** are ``(predictions, targets) -> float`` callables, passed to
  :func:`soma.evaluate` or by dotted path to ``soma evaluate --metrics``; a config's
  ``evaluation.metrics`` takes the built-in names only (see :doc:`evaluation`).
- **Augmentations** are ``(pixels, target) -> (pixels, target)`` callables applied
  to the train split of the live grid path. soma passes ``ignore_index`` and
  ``target_size`` to the constructor when it accepts them and the params leave them
  unset.

.. _custom-encoder-stage:

A custom encoder stage or image source
--------------------------------------

On the live grid path (see :ref:`live-mode`), the frozen encoder runs inside the
model, as the ``encoder`` stage of :class:`soma.models.DenseModel`, and an image
source serves the pixels it re-encodes. Either can be your own:

- An **encoder stage** is a parameter-free ``nn.Module`` like
  :class:`soma.extract.FrozenEncoder`: a ``geometry``, a ``feature_dim``, an
  ``input_device``, a ``preprocessor()`` that maps a ``(3, H, W)`` uint8 tile to the
  encoder's input, and a deterministic ``forward`` from a preprocessed or a raw
  uint8 batch to ``(B, D, h, w)`` grids. Having no parameters keeps it out of every
  checkpoint and optimizer. :func:`soma.testing.check_encoder_stage` checks a stage,
  and :class:`soma.testing.FakeEncoder` is a weight-free example.
- An **image source** serves ``(3, H, W)`` uint8 pixels by sample id, with a
  ``geometry`` and a ``spacing``, like :class:`soma.data.TileImageSource`.
  :func:`soma.testing.check_image_source` checks one.

.. code-block:: python

   soma.testing.check_encoder_stage(stage)
   soma.testing.check_image_source(images)
   model = soma.models.DenseModel(decoder, head, encoder=stage)
   fold = soma.learn.loaders(images, cohort, head=head, targets=labels, preprocess=stage.preprocessor())

A config cannot name a stage: a managed run builds its stage from
``features.encoder``. ``soma.Pipeline(config, source=images)`` accepts your image
source and re-encodes it with that stage. To bring a new foundation model to a
managed run, register it with slide2vec (see :ref:`custom-encoder`).

.. _custom-callbacks:

Callbacks on fit and on predict
-------------------------------

Callbacks carry run-management side effects, such as logging, timing or extra
artefacts, and never change the loss. A **fit callback** implements any of
``on_fit_start(trainer, learner)``, ``on_epoch_start(trainer, learner)``,
``on_train_batch_end(trainer, learner, step_output)``,
``on_validation_end(trainer, learner, summary)``, ``on_epoch_end(trainer, learner)``,
``on_fit_end(trainer, learner)`` and ``on_exception(trainer, learner, exception)``.
It stops the run by setting ``trainer.should_stop = True``.

.. code-block:: python

   class PrintTuneLoss:
       def on_validation_end(self, trainer, learner, summary):
           print(trainer.epoch, summary["loss"])


   result = soma.fit(learner, fold.train, fold.tune, callbacks=[PrintTuneLoss()])

:func:`soma.fit` installs a ``Checkpoint``, an ``EarlyStopping`` and a ``Progress``
itself; one of those passed in ``callbacks`` replaces the one it would build, so a
run can state its whole selection policy with plain constructors. In a config,
``trainer.callbacks`` lists ``{name, params}`` leaves, by registered name or dotted
path.

A **predict callback** implements any of ``on_predict_start(learner)``,
``on_predict_batch(batch, output)`` and ``on_predict_end()``, where ``output`` is the
learner's ``PredictOutput``. Pass it to ``soma.predict(learner, loader,
callbacks=[...])``. soma's own predict callbacks, which write attention, overlays,
probabilities and heatmaps, are listed in :doc:`evaluation`. The pipeline installs
them from the ``evaluation`` save settings, so a predict callback of your own runs
from Python.

Making it a package
-------------------

Registration is a Python side effect. The module that registers your components
must be imported before soma reads a config that names them, and again whenever a
saved run is reopened from Python, because its plan names them and soma resolves
their family through the registry. Two ways keep this reliable:

- **Name the classes by dotted path.** soma imports the module itself when it reads
  the config, wherever the run starts. In exchange, the params are not validated or
  listed, and a head gets no ``infer`` hook.
- **Register at import.** Put the ``soma.register`` calls in your package, import
  it before building the :class:`~soma.Pipeline`, and on the command line start
  soma from a short script that imports it first:

  .. code-block:: python

     import my_package  # registers its heads, decoders and learners

     from soma.cli import main

     main(["run", "config.yaml"])

Either way, editing your component's code does not change the experiment id: a
rerun after an edit lands in the same experiment. The run records the code
instead, so a resume across the edit is refused, and ``soma leaderboard`` refuses
to pool seeds that ran different code until you rerun the stale ones (see
:ref:`component-record`).

A model you build by hand saves like any other, but :func:`soma.load` can only
rebuild it when ``soma.learn.record_builder(model, builder, **params)`` recorded an
importable function that builds it, as the notebook's Model cell does; otherwise,
load ``weights.safetensors`` into the same model yourself (see
:ref:`model-bundle`).

soma's own benchmarks follow the same rule. :mod:`soma.benchmark` registers its
components (for example the EVA segmentation head ``eva_segmentation``, decoder
``eva_conv_ms`` and aggregator ``eva_abmil``) and the benchmarks themselves when it
is imported. ``import soma`` does not import it; the ``soma`` command does, at
startup. From Python, ``import soma.benchmark`` before loading a saved benchmark
config, or name the component by its dotted path. Until then,
``soma.components("benchmarks")`` and a lookup of one of those names fail with an
error that says so.
