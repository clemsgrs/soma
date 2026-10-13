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
       ``tune_is_test``, ``allow_missing_tune``
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

YAML and overrides
------------------

``ExperimentConfig.from_yaml(path)`` reads a YAML file and ``to_yaml(path)``
writes one; ``from_yaml(path, overrides)`` merges a nested mapping into the file
first. On the command line, ``--set`` takes dotted keys into the sections:

.. code-block:: bash

   soma config.yaml --set run.seed=1 --set learner.params.head.params.dropout=0.2

Values are parsed as YAML, so numbers and booleans keep their types. A mapping is
merged key by key; a list is replaced whole.

Checking and resolving
----------------------

:func:`soma.plan.check` validates a config across its sections (for example that
a decoder belongs to the ``dense`` learner, or that a survival head is not given
tile samples) without reading any file. :func:`soma.resolve` then expands the
config against its cohort into a :class:`soma.Plan`: every component's params
with their defaults, plus what the cohort decides, such as a classification
head's label map. A value the config states wins; a contradiction raises. A
``null`` value counts as unset, so the cohort decides it.
:class:`~soma.Pipeline` runs both at its start.

The run directory keeps ``config.yaml`` (the config as given) and ``plan.yaml``
(the plan). ``plan.yaml`` loads as a config, so ``soma plan.yaml`` replays a run
with every value pinned, while ``soma config.yaml`` resolves again against the
current code and cohort. See :ref:`plan-yaml` for how a resume compares plans and
what the experiment identity digests.

.. autoclass:: soma.ExperimentConfig
   :members: from_yaml, to_yaml, from_dict, to_dict

.. autofunction:: soma.resolve

.. autoclass:: soma.Plan
   :members: comparable, from_yaml, to_yaml

.. autoclass:: soma.config.Component
