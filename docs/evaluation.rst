Evaluation
==========

Scoring is two verbs with one noun between them. :func:`soma.predict` runs a model
over a loader and returns :class:`~soma.evaluation.Predictions`: one row per sample,
with the columns the task head declares. :func:`soma.evaluate` joins those
predictions with the cohort's targets by ``sample_id`` and returns an
:class:`~soma.evaluation.Evaluation`. Targets never travel inside predictions: they
come from :meth:`soma.data.Cohort.targets`, so predictions saved by a run, or made
outside soma, are scored the same way.

:func:`soma.predict` runs the model where its parameters live, or on the ``device`` it
is given (``"auto"``, ``"cpu"``, ``"cuda:1"``). A model with no parameter runs on the
GPU when there is one. A run predicts on its ``trainer.device``.

Score predictions
-----------------

.. code-block:: python

   import soma

   cohort = soma.Cohort.from_csv("dataset.csv", "splits.csv", targets=["label"])
   predictions = soma.Predictions.from_parquet("run/predictions_test.parquet")
   # or, after soma.fit: predictions = soma.predict(result.learner, fold.tests["test"])
   evaluation = soma.evaluate(
       predictions,
       cohort.targets("test"),
       metrics=["auroc", "balanced_accuracy"],
       subgroups=["center"],
   )
   evaluation.to_json("evaluation.json")

``evaluate`` scores exactly the ids it is given. It refuses a prediction
without a target, a target without a prediction, and duplicate ids; the error
names the count. Fill the samples a model could not score with
:class:`~soma.evaluation.Baseline` (class prior, mean value or constant risk,
fitted on the training targets), or filter the targets. When the predictions
carry ``is_placeholder``, the metrics add ``coverage``, ``num_samples``,
``num_real_samples`` and ``num_placeholder_samples``. The pipeline fills
samples without features this way.

``soma evaluate`` does the same from the command line (see :doc:`cli`), without a
run directory.

``predict`` writes side artefacts through callbacks (see :ref:`custom-callbacks`):
``soma.evaluation.heatmaps.SaveAttention(dir)`` writes each bag's normalised tile
attention to ``<dir>/<sample_id>.npz``.

Choose metrics
--------------

``metrics`` is ``"default"`` (the family's defaults, listed in the :doc:`task zoo
<tasks>`), or a list of registry keys, dotted import paths and
``(predictions, targets) -> float`` callables. A callable receives the two frames
aligned by ``sample_id``, with labels already mapped to class indices.
:data:`soma.evaluation.metric_registry` lists each metric's families and direction;
a key outside the predictions' family is refused. In a config,
``evaluation.metrics`` takes the built-in names of the head's family (an empty
list means the defaults); score a metric of your own from Python, or with
``soma evaluate --metrics`` by its dotted path.

:func:`soma.evaluation.compare` runs a paired permutation test of several
prediction sets against the best one.

Subgroup metrics
----------------

``evaluation.subgroups`` names ``dataset.csv`` columns whose distinct values
receive their own metrics:

.. code-block:: yaml

   evaluation:
     subgroups: [center, grade]

For each column, every value with at least two samples gets the run's metrics
and its sample count. A significance test then compares each group with the
rest of the split by permuting group membership 1,000 times; it needs at least
10 samples in the group and two in the remainder. Patient-level runs read each
patient's metadata from its first slide. Results are written to
``subgroup_metrics_<split>.json`` for each test split.

Holding out test data
---------------------

Set ``evaluation.holdout_test: true`` to train and report tune results without
test inference or test artifacts. Checkpoint selection and tune evaluation
continue normally. This supports selecting a candidate on tune results before
evaluating it on the held-out test cohort.

Test identities are recorded in ``test_results.json``. Re-scoring an already
recorded test identity is skipped unless ``evaluation.overwrite_test: true``;
resuming unfinished folds is allowed. See :doc:`outputs` for experiment identity.

.. _grid-predictions:

Segmentation and detection predictions
--------------------------------------

For a segmentation or detection head, :func:`soma.predict` runs the head's
``postprocess`` and returns one of two prediction types instead of a table, and
:func:`soma.evaluate` scores it against a target source rather than a targets
frame:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - :class:`~soma.evaluation.LabelMapPredictions`
     - :class:`~soma.evaluation.PointPredictions`
   * - Holds
     - One ``(H, W)`` uint8 label map per sample
     - A frame ``sample_id, x, y, class, score`` in level-0 pixels, plus every
       sample id (a sample with no detection is still scored)
   * - Saved as
     - ``save(dir)``: one PNG per sample and ``index.parquet``;
       ``load(dir)`` reads it back
     - ``to_parquet(path)`` / ``from_parquet(path)``
   * - Scored against
     - :class:`~soma.data.LabelMapSource` or :class:`~soma.data.CachedLabelMapSource`
     - :class:`~soma.data.PointSource`
   * - Metrics
     - ``mean_dice``, ``mean_iou``, ``dataset_global_mean_dice``,
       ``foreground_mean_dice``, ``dice_per_class``
     - ``mean_f1``, ``f1_per_class``, ``precision``, ``recall``,
       ``mean_f1_per_image``

Label maps are scored from per-sample confusion counts. Points are matched to
the annotations per class within the matching distance δ, with Hungarian
assignment by default. A metric given as a ``{name, params}`` leaf can change
the matching: ``matching: greedy`` matches by descending score, and
``match_distance`` overrides δ (in level-0 pixels).

Point predictions keep every peak above the head's ``min_score``. Choosing an
operating point is a separate step: ``with_threshold`` takes one threshold or a
``{class: threshold}`` mapping, and :func:`~soma.evaluation.sweep_threshold`
picks the threshold that maximises a metric on the given targets (one per
class with ``per_class=True``). Sweep on tune and apply the result to test:

.. code-block:: python

   import soma
   from soma.evaluation import PointPredictions, sweep_threshold

   targets = soma.data.PointSource.from_manifest(soma.AnnotationManifest.from_csv("dataset.csv"))
   tune = PointPredictions.from_parquet("run/predictions_tune.parquet")
   test = PointPredictions.from_parquet("run/predictions_test.parquet")

   threshold, _ = sweep_threshold(tune, targets.subset(tune.sample_ids), per_class=True)
   evaluation = soma.evaluate(
       test.with_threshold(threshold),
       targets.subset(test.sample_ids),
       metrics=[{"name": "mean_f1", "params": {"matching": "greedy"}}],
   )

Grid ``predict`` callbacks write side artefacts:
:class:`~soma.evaluation.SaveOverlays` (predictions and supervision drawn over the
tile), :class:`~soma.evaluation.SaveProbabilities` (per-sample float16 softmax,
segmentation) and :class:`~soma.evaluation.SaveHeatmaps` (per-sample float16
heatmaps, detection), for example
``soma.predict(learner, loader, [SaveOverlays("overlays/test", images=images)])``.

Per-sample confusion matrices are recomputable from saved label maps with
:func:`soma.evaluation.confusion_records`.

``soma evaluate`` also accepts a label-map directory or a points parquet. It
scores label maps against the masks in the cohort CSV's ``label_mask_path``
column and points against the files in its ``points_path`` column; both must
already hold class indices. ``--threshold`` keeps the points scoring at least
that value.

.. _whole-slide-scoring:

Whole-slide label maps
~~~~~~~~~~~~~~~~~~~~~~

:func:`soma.predict_slides` (``soma predict``) writes one label TIFF per slide.
:class:`~soma.evaluation.SlideLabelMaps` is a handle on one bundle's directory of
them, and :func:`soma.evaluate` scores it against a
:class:`~soma.data.LabelMapSource` of the slides' annotation rasters:

.. code-block:: python

   import soma
   from soma.data import AnnotationManifest, ImageManifest, LabelMapSource
   from soma.evaluation import SlideLabelMaps

   maps = SlideLabelMaps("predictions/beetle-test")
   targets = LabelMapSource.from_manifests(
       AnnotationManifest.from_csv("dataset.csv"),
       ImageManifest.from_csv("dataset.csv"),  # aligns each raster to its slide
       size=(0, 0),
       backend=maps.mask_backend,
       image_backend=maps.backend,
   )
   evaluation = soma.evaluate(maps, targets.subset(maps.sample_ids))
   evaluation.per_sample  # one row of counts per slide

``soma evaluate predictions/beetle-test --cohort dataset.csv`` does the same.
The directory is a prediction output with one bundle's maps, or one of its
bundle directories. A slide is predicted when its ``<sample_id>.json`` record
exists.

The slides and the annotation rasters are read with the hs2p readers the
prediction's ``plan.yaml`` records (``maps.backend`` and ``maps.mask_backend``).
``soma predict`` reads the slides with the reader the bundles were trained with,
unless its config sets ``extraction.tiling.backend``, and the masks with
``extraction.tiling.mask_backend``. To read with other readers, pass
``--backend`` for the slides and ``--mask-backend`` for the annotation rasters:

.. code-block:: bash

   soma evaluate predictions/beetle-test --cohort dataset.csv \
       --backend openslide --mask-backend tifffile

In Python, pass the readers to ``LabelMapSource.from_manifests`` as
``image_backend`` and ``backend``. Both options apply to a ``soma predict``
output only; ``soma evaluate`` refuses them for other predictions.

Each slide is read one window at a time on the map's pixel grid, so neither the
map nor the annotation raster is held in memory. hs2p aligns an annotation
raster at another resolution to the slide and reads it at the map's spacing.
The class scheme is the one each map records from its bundle's head: the raw
annotation values of each class, the raw values to ignore, and the vocabulary
the raster is read against, as in training. The class scheme, spacing and
size of the target source are not used.

- A pixel whose annotation value is ignored is not scored.
- An annotated pixel of class ``c`` that no kept tile covered (value 0 in the
  map) is a false negative for ``c`` and a false positive for no class.
- A map pixel that lies partly off the slide is not scored.

Each slide gives one row of counts to the same reducers as
:class:`~soma.evaluation.LabelMapPredictions`. ``Evaluation.per_sample`` holds
the rows: ``annotated_px``, ``not_predicted_px``, ``not_predicted_fraction``
(the annotated pixels not predicted, over all annotated pixels) and
``tp_class_<c>``, ``fp_class_<c>``, ``fn_class_<c>`` per class.

These scores are not comparable to a run's test metrics. A run scores only the
ROIs it samples where an annotation class meets ``min_coverage``; a whole-slide
score covers every annotated pixel of the slide.

.. _whole-slide-points:

Whole-slide detection points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With detection bundles, :func:`soma.predict_slides` (``soma predict``) writes one
points parquet per slide and bundle, ``<output_dir>/<bundle>/<sample_id>.parquet``.
The bundles of one call are all segmentation bundles or all detection bundles.

The parquet has four columns: ``x`` and ``y`` in the slide's level-0 pixels,
``class`` (the class index) and ``score``. It holds every peak above the head's
``min_score``. No threshold is applied, so a threshold can be chosen later.
Its schema metadata (key ``soma``) records:

- the class scheme the bundle was trained with: ``classes`` maps each class
  name to the annotated ids that form it, in class order, and ``drop`` lists
  the dropped ids (see :ref:`detection-classes`);
- the matching and NMS distances, in µm;
- the slide's level-0 spacing and the read spacing;
- the bundle's identity;
- the bundle's tune threshold (``threshold``).

The scheme comes from the bundle's ``preprocessing.json``. A bundle trained
before soma recorded it has no ``class_scheme``, and ``soma predict`` refuses
it before it writes anything. Train the bundle again, or add ``class_scheme``
to its ``preprocessing.json`` from the run's config, for example
``{"classes": {"mnl": [0, 1]}, "drop": [2]}``. The class names must be the
bundle's own.

Each tile's peaks are found in the tile, then placed on the slide from the
tile's level-0 origin and the slide's spacings. Each tile keeps only the
peaks in the region it contributes, by the label maps' pick-one rule:

- With ``overlap: 0``, a tile keeps all its peaks.
- With an overlap, the margin is ``tile * overlap / 2`` pixels. On each axis, a
  tile keeps the peaks from ``margin`` to ``margin + step``, where ``step`` is
  the grid's step in the tile's pixels. The step is usually
  ``tile - 2 * margin``. On a slide resampled to the read spacing, it can differ
  by a fraction of a pixel, because hs2p rounds the step in the pixels it reads.
  The tiles of the slide's first row and first column also keep the peaks in
  their outer margin.

One NMS per class over the whole slide then removes the duplicates that two
tiles found on either side of a seam. It uses the head's NMS distance, which
is the matching distance unless the head sets one. With an overlap, the margin
must be at least the NMS distance, or the config is refused.

With ``ensemble: true``, the bundles' heatmaps are averaged before the peaks are
found. The bundles must then share the class scheme (the classes, their
annotated ids and the dropped ids), both distances and ``min_score``. An
ensemble records no threshold.

A slide with no kept tile, or with no peak, gets a parquet with no rows. The
``index.parquet`` row of each slide gives ``width``, ``height`` and ``spacing_um``
of its level 0, ``num_tiles`` and ``num_points``.

.. _whole-slide-points-scoring:

Scoring whole-slide points
~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~soma.evaluation.SlidePoints` is a handle on one bundle's directory of
points parquets, and :func:`soma.evaluate` scores it against a
:class:`~soma.data.PointSource` of the slides' annotated points:

.. code-block:: python

   import soma
   from soma.data import AnnotationManifest, PointSource
   from soma.evaluation import SlidePoints

   points = SlidePoints("predictions/midog-test")
   targets = PointSource.from_manifest(AnnotationManifest.from_csv("dataset.csv"))
   evaluation = soma.evaluate(points, targets.subset(points.sample_ids))
   evaluation.per_sample  # one row of counts per slide

``soma evaluate predictions/midog-test --cohort dataset.csv`` does the same.
The directory is a prediction output with one bundle's points, or one of its
bundle directories. The cohort CSV gives each slide's ``points_path``, a
``x,y,class`` file in level-0 pixels that holds the dataset's own annotated
ids, and optionally its ``ignore_mask_path``.

The class scheme is the one each parquet records from its bundle: the annotated
ids of each class, and the ids dropped. A points file written before soma
recorded the scheme is refused: predict the slides again. The class scheme of
the target source is not used.

- An annotated point whose id is dropped is removed. A prediction there is a
  false positive.
- An annotated id that the scheme neither gives a class nor drops is refused.
- A prediction scoring below the threshold is removed.
- With an ignore mask, a point (annotated or predicted) on a mask pixel of 255
  is not scored. The mask may have any resolution that covers the slide: hs2p
  aligns it to the slide's level 0 and reads it with the reader the
  prediction's ``plan.yaml`` records (``points.mask_backend``), or
  ``--mask-backend``. Its pixels must be 0 (annotated) or 255 (not annotated);
  another value is refused. Without a mask, the whole slide is annotated.
- Predictions are matched to the annotations of their class within the
  bundle's matching distance, in µm, at each slide's own spacing. Matching is
  Hungarian by default. A metric leaf with ``matching: greedy`` matches by
  descending score, and ``match_distance`` sets another distance, in µm.

The threshold is the bundle's tune threshold, which the parquet records
(``points.threshold``). An ensemble's points record none, and are refused until
one is set. Pick one on a tune split with
:func:`~soma.evaluation.sweep_threshold`, which takes
:class:`~soma.evaluation.SlidePoints` too, and set it with ``with_threshold``.
``soma evaluate --threshold T`` sets one value for every class:

.. code-block:: python

   from soma.evaluation import sweep_threshold

   tune = SlidePoints("predictions/tune")  # an ensemble's points
   threshold, _ = sweep_threshold(tune, targets.subset(tune.sample_ids), per_class=True)
   test = SlidePoints("predictions/test").with_threshold(threshold)
   evaluation = soma.evaluate(test, targets.subset(test.sample_ids))

Each slide gives one row of counts to the same reducers as
:class:`~soma.evaluation.PointPredictions`. ``Evaluation.per_sample`` holds the
rows: ``annotated_area_mm2`` (the annotated mask pixels times their area, or the
whole slide without a mask) and ``tp_class_<c>``, ``fp_class_<c>``,
``fn_class_<c>`` per class.

Matching lists only the pairs of points within the matching distance, so a
slide with many points takes little memory: 100,000 predictions against 20,000
annotations of one class match in about a second.

These scores are not comparable to a run's test metrics. A run scores the ROIs
or tiles of its test split; a whole-slide score covers every annotated point of
the slide, and every prediction outside the ignored region.

Reference
---------

.. autoclass:: soma.config.EvaluationConfig

.. autofunction:: soma.evaluation.predict

.. autofunction:: soma.evaluation.evaluate

.. autoclass:: soma.evaluation.Predictions
   :members:

.. autoclass:: soma.evaluation.Evaluation
   :members:

.. autoclass:: soma.evaluation.Baseline
   :members:

.. autofunction:: soma.evaluation.compare

.. autoclass:: soma.evaluation.LabelMapPredictions
   :members: save, load

.. autoclass:: soma.evaluation.SlideLabelMaps
   :members: sample_ids, classes, backend, mask_backend

.. autoclass:: soma.evaluation.SlidePoints
   :members: sample_ids, classes, class_scheme, match_distance_um, threshold, mask_backend, with_threshold

.. autoclass:: soma.evaluation.PointPredictions
   :members: with_threshold, to_parquet, from_parquet

.. autofunction:: soma.evaluation.sweep_threshold

.. autofunction:: soma.evaluation.confusion_records
