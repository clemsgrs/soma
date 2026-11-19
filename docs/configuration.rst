Configuration
=============

An experiment is one :class:`soma.ExperimentConfig`. Its seven sections each
hold the keyword arguments of one public callable, so a YAML file and a Python
config spell every field the same way: a knob that cannot be passed in Python
cannot appear in YAML.

.. list-table::
   :header-rows: 1
   :widths: 16 30 54

   * - Section
     - Callable
     - Holds
   * - ``run``
     - :class:`~soma.Pipeline`
     - ``output_dir``, ``seed``, ``run_id``, ``resume``, ``folds``,
       ``mirror_root``
   * - ``cohort``
     - :class:`~soma.data.Cohort`
     - ``records`` and ``splits`` (the two CSV files), ``collapse``,
       ``tune_is_test``, ``allow_missing_tune``, ``train_subset``
   * - ``features``
     - the feature source
     - the ``encoder`` leaf, slide2vec's ``extraction``, ``execution`` and
       ``dense`` options, the ``cache``, ``kind`` and ``spacing_policy``
       (see :doc:`preprocessing`)
   * - ``loaders``
     - :func:`soma.learn.loaders`
     - ``batch_size``, worker settings, the ``sampler`` and ``augment`` leaves
   * - ``learner``
     - :class:`~soma.learn.Learner`
     - the learner's name and params: the model's ``aggregator`` or ``decoder``
       and ``head`` leaves, ``optimizer``, ``scheduler``, ``normalization``,
       ``projection`` and the learner's own keywords
   * - ``trainer``
     - :class:`~soma.learn.Trainer`
     - ``max_epochs`` or ``max_steps``, ``accumulate_grad_batches``,
       ``val_every_n_epochs``, ``device``, ``precision``, ``callbacks``
   * - ``evaluation``
     - :func:`soma.evaluate`
     - ``metrics``, ``subgroups``, ``holdout_test``, ``overwrite_test``, the
       saved artifacts, ``attention_heatmaps`` and ``representation``

Only ``cohort`` is required. A field left out keeps its default;
``examples/reference.yaml`` spells every section at its defaults (see the
:doc:`CLI reference <cli>`).

Components
----------

Every swappable component is a ``{name, params}`` leaf: the encoder, the
aggregator, decoder and head, the optimizer and each scheduler, the sampler, the
augmentation, each trainer callback and the learner itself. ``name`` is a
registry key or a dotted import path to a class; ``params`` are its constructor
keywords, validated against the params the component declares (ADR 0018). A bare
name stands for ``{name, params: {}}``. ``soma list KIND`` prints every
registered component of a kind with its params and their defaults.

.. code-block:: yaml

   learner:
     name: mil
     params:
       aggregator: abmil
       head:
         name: multiclass_classification
         params: {dropout: 0.1}
       optimizer: {name: adamw, params: {lr: 3.0e-4}}

The learner decides the training path. ``mil`` trains the set shape (a bag or a
vector per sample) and ``dense`` the grid shape (a feature grid per sample,
segmentation and detection). The head decides the task family. A set run's
sample is a patient with ``cohort.collapse: patient_id``, a tile when a tile
encoder runs without an aggregator, and otherwise a slide.

.. _train-subset:

Training on a subset
--------------------

``cohort.train_subset`` names a text file with one sample id per line. Each
line is read literally, like the ids in the CSV files: a space is part of the
id. Empty lines are skipped. Each fold then trains on the listed ids of its
train split only, in the split's order. Tune and test do not change. Use it
to train the same experiment on less data, for example to measure how a model
scales with the number of annotated samples.

.. code-block:: yaml

   cohort:
     records: slides.csv
     splits: splits.csv
     train_subset: subsets/rung_0128.txt

- One file serves every fold of the run. Every listed id must be in the train
  split of every fold. The run refuses an empty file, an id listed twice, an id
  the cohort does not have, and an id that a fold puts in tune or test. The
  error names the fold and the id.
- The ids are the run's samples. When the run samples ROIs from whole slides,
  they are ROI ids, not slide ids (see :ref:`segmentation-train-subset`). A
  patient run (``cohort.collapse: patient_id``) refuses ``cohort.train_subset``.
- What the run fits on the train split sees the subset only: the feature
  normalization, the ROI batch sampler and the survival bins.
- The listed ids are part of the :doc:`experiment identity <outputs>`: two
  subsets are two experiments. The feature caches do not depend on the subset,
  so runs over several subsets share them.

Editing a config
----------------

A config has no override mechanism: you edit it. In Python, assign its fields:

.. code-block:: python

   from soma import presets

   config = presets.mil_classification("uni2", "abmil")
   config.run.seed = 1
   config.trainer.max_steps = 2000
   config.learner.params["head"].params["dropout"] = 0.2

A field write runs no check of its own; :func:`soma.plan.check` builds every
section again, so a value its constructor would refuse is refused before
anything runs. A step budget written over an epoch budget replaces it, as at
construction. On the command line, edit the YAML file: ``soma init PRESET``
prints a preset (see :ref:`presets`) with every default filled in, and
``soma run config.yaml`` runs a file. ``ExperimentConfig.from_yaml(path)`` reads
a YAML file and ``to_yaml(path)`` writes one.

Checking and resolving
----------------------

:func:`soma.plan.check` validates a config across its sections (for example that
a decoder belongs to the ``dense`` learner, or that a survival head is not given
tile samples) without reading any file. :func:`soma.resolve` then expands the
config against its cohort into a :class:`soma.Plan`: every component's params
with their defaults, plus what the cohort decides, such as a classification
head's label map. A value the config states wins; a contradiction raises. A
``null`` value counts as unset, so the cohort decides it.
:class:`~soma.Pipeline` runs both at its start. A plan is frozen: its sections
refuse field writes.

The run directory keeps ``config.yaml`` (the config as given) and ``plan.yaml``
(the plan). ``plan.yaml`` loads as a config, so ``soma run plan.yaml`` replays a
run with every value pinned, while ``soma run config.yaml`` resolves again
against the current code and cohort. See :ref:`plan-yaml` for how a resume compares plans and
what the experiment identity digests.

.. autoclass:: soma.ExperimentConfig
   :members: from_yaml, to_yaml, from_dict, to_dict

.. autofunction:: soma.resolve

.. autoclass:: soma.Plan
   :members: comparable, from_yaml, to_yaml

.. autoclass:: soma.config.Component

.. _configuration-presets:

Presets
-------

A preset returns a complete config for a common task. The :ref:`use-case table
<presets>` shows which preset fits your encoder and task, and
``soma.components("presets")`` lists them with their arguments.

``segmentation`` and ``detection`` decode each tile's token grid, so they take a
tile-level encoder. Each preset checks the level of its encoder when it is
built, and refuses an encoder of another level by naming the preset to use:
``presets.tile_classification("titan")`` asks for ``slide_classification``, and
``presets.slide_classification("uni2")`` for ``mil_classification`` or
``tile_classification``. No preset covers patient-level encoders (``moozy``) yet.

A choice that changes the model has no default: you name the aggregator of a
``mil_*`` preset, the ``loss`` of ``mil_survival`` and ``slide_survival``
(``"nll"`` or ``"cox"``, and ``num_bins`` for ``"nll"``) and the
``num_classes`` of ``segmentation`` and ``detection``.

The ``mil_*`` presets take the tiling as arguments: ``spacing`` (µm per pixel)
and ``tile_size`` (pixels). An argument you leave out takes the encoder's
recommended value, and the preset keeps its tiles with at least 10% tissue:

.. code-block:: python

   config = presets.mil_classification("uni2", "abmil", spacing=0.5, tile_size=224)

On the command line, they are the ``--spacing`` and ``--tile-size`` flags of
``soma init`` and ``soma run --preset``. The ``slide_*`` presets keep the same
tissue, at the slide encoder's own tiling: a slide encoder is tied to the tile
size and spacing of its tile encoder, so they take no tiling arguments.

.. autofunction:: soma.presets.tile_classification

.. autofunction:: soma.presets.tile_regression

.. autofunction:: soma.presets.mil_classification

.. autofunction:: soma.presets.mil_regression

.. autofunction:: soma.presets.mil_survival

.. autofunction:: soma.presets.slide_classification

.. autofunction:: soma.presets.slide_regression

.. autofunction:: soma.presets.slide_survival

.. autofunction:: soma.presets.segmentation

.. autofunction:: soma.presets.detection
