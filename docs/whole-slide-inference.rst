.. _whole-slide-inference:

Whole-slide inference
=====================

A segmentation or detection run trains and scores on the ROIs or tiles it
samples. Whole-slide inference applies the trained model to every tissue tile of
a slide. Two commands do the work:

- ``soma predict CONFIG`` predicts. It writes one file per slide and bundle: a
  label TIFF for a segmentation bundle, a points parquet for a detection
  bundle.
- ``soma evaluate PREDICTIONS --cohort dataset.csv`` scores those files against
  the slides' annotations.

Use it to:

- score a model on a test set of whole slides, over every annotated pixel or
  point instead of the ROIs a run samples;
- predict slides that have no annotations. Prediction needs only the slides;
  only scoring needs annotations.

The model comes from a bundle (see :ref:`model-bundle`). A bundle holds all that
prediction needs: the decoder and head, the class scheme, the tile size, the
read spacing and the recipe of the frozen encoder. Prediction reads the pixels
and runs the encoder again. It reads no feature cache.

.. _whole-slide-predict:

Predict
-------

Write a config with three sections:

.. code-block:: yaml
   :caption: predict.yaml

   predict:
     output_dir: predictions/beetle-test
     bundles: runs/beetle          # a run directory: the bundle of each fold
     ensemble: false
     batch_size: 8
     num_workers: 8
     devices: null                 # or [0, 1]: one process per GPU
   cohort:
     records: dataset.csv
     splits: splits.csv
     subset: test
     fold: 0
   extraction:
     tiling:
       overlap: 0.0
       mask_backend: openslide
     segmentation:
       method: hsv
     masks:
       min_coverage:
         tissue: 0.01

Then run it:

.. code-block:: bash

   soma predict predict.yaml

The same in Python. :func:`soma.predict_slides` takes one keyword per config
key, and leaves the others at their defaults:

.. code-block:: python

   import soma

   index = soma.predict_slides(
       output_dir="predictions/beetle-test",
       bundles="runs/beetle",
       records="dataset.csv",
       splits="splits.csv",
       subset="test",
       num_workers=8,
       extraction={"tiling": {"overlap": 0.0, "mask_backend": "openslide"}},
   )

It returns the index of the files it wrote (see
:ref:`whole-slide-prediction-outputs`). Relative paths resolve against the
working directory. The keys:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Key
     - Meaning
   * - ``predict.output_dir``
     - Where the plan, the predictions and their index go. Required.
   * - ``predict.bundles``
     - The trained bundles: one bundle directory, a list of them, or one run
       directory. A run directory stands for its bundles: ``fold_<k>/bundle``
       for each fold, or ``bundle``. Required.
   * - ``predict.ensemble``
     - ``false`` (default): one prediction per bundle. ``true``: one prediction
       from all the bundles. See :ref:`whole-slide-bundles`.
   * - ``predict.batch_size``
     - Tiles per forward pass (default 8).
   * - ``predict.num_workers``
     - Worker processes that read tiles while the model runs (default 0: the
       main process reads them).
   * - ``predict.devices``
     - ``null`` (default): predict in this process, on the GPU when one is
       visible, else on the CPU. A list such as ``[0, 1]`` or ``["cpu"]``: one
       new process per device, with the slides split across them. Each process
       logs to ``<output_dir>/logs/worker_<k>.log``.
   * - ``cohort.records``
     - The slides: a CSV with ``sample_id`` and ``image_path``, and optionally
       ``mask_path`` and ``spacing_at_level_0`` (see :doc:`dataset`). Required.
   * - ``cohort.splits``
     - A splits CSV. Without it, every row of ``records`` is predicted.
   * - ``cohort.subset``
     - The split to predict (default ``test``). It needs ``cohort.splits``.
   * - ``cohort.fold``
     - The fold of that split (default 0). It needs ``cohort.splits``.
   * - ``extraction``
     - How the slides are tiled and read: the ``features.extraction`` schema
       of :doc:`preprocessing`, with the rules below. Optional.
   * - ``extraction.tiling.requested_tile_size_px``,
       ``extraction.tiling.requested_spacing_um``
     - Taken from the bundles. Leave them out: another value is refused.
   * - ``extraction.tiling.tolerance``
     - How far a pyramid level's spacing may be from the read spacing
       (default: the bundles' own).
   * - ``extraction.tiling.overlap``
     - The share of a tile's side that neighbouring tiles overlap, in
       ``[0, 1)`` (default 0). See :ref:`whole-slide-stitching`.
   * - ``extraction.tiling.grid``
     - Always ``slide``. soma sets it, and refuses another value.
   * - ``extraction.tiling.backend``
     - The hs2p reader of the slides (default: the one the bundles were
       trained with). See :ref:`whole-slide-tiling`.
   * - ``extraction.tiling.mask_backend``
     - The hs2p reader of the tissue masks (default ``auto``). ``soma
       evaluate`` reads the annotation rasters and ignore masks with it too.
   * - ``extraction.segmentation.method``
     - The tissue segmentation of slides without a ``mask_path`` (default
       ``hsv``). The other ``segmentation`` keys are those of
       :doc:`preprocessing`.
   * - ``extraction.masks.min_coverage``
     - The share of a tile that tissue must cover to keep the tile (default
       ``{tissue: 0.01}``).
   * - ``extraction.masks.pixel_mapping``
     - The values of the ``mask_path`` rasters (default
       ``{background: [0], tissue: [1]}``). With other classes, a tile is
       kept where those classes cover it, as in
       :ref:`preprocessing-source-masks`: for example, to predict inside a
       tumour mask only.

.. _whole-slide-plan:

The plan
~~~~~~~~

``soma predict`` first resolves the config into a plan, and writes it to
``<output_dir>/plan.yaml`` before it predicts any slide. In the plan:

- every path is absolute;
- a run directory is replaced by the list of its bundles;
- the extraction is spelled in full, with the tile size, the read spacing, the
  read tolerance and the slide reader taken from the bundles.

The plan is a config too: ``soma predict <output_dir>/plan.yaml`` replays it.

A slide is done when its record, ``<sample_id>.json``, exists, parses and names
a file that exists. A launch predicts only the slides that are not done, so a
stopped prediction resumes when you launch it again.

A launch into an existing output directory is refused when:

- the directory's ``plan.yaml`` differs from the new plan. The error names the
  keys that differ. ``predict.devices`` and ``predict.num_workers`` may change,
  because they decide where the work runs, not what it computes.
- a bundle changed since the directory was written. ``provenance.json`` records
  the SHA-256 of each bundle's files.

.. _whole-slide-tiling:

How a slide is tiled
~~~~~~~~~~~~~~~~~~~~

hs2p lays one fixed grid over the whole slide, from its top-left corner. A tile
starts at every multiple of the grid's step that falls inside the slide. The
grid does not follow the tissue, so neighbouring tiles always line up.

A tile is kept when tissue covers at least ``extraction.masks.min_coverage`` of
it. The tissue is the slide's ``mask_path`` raster when the records give one,
and hs2p's tissue segmentation otherwise. A tile with no tissue is never kept.

Each kept tile is read at the bundles' read spacing: from a pyramid level within
the tolerance, or resampled from a finer level. soma never upsamples a slide. A
slide whose level 0 is coarser than the read spacing, beyond the tolerance, is
refused before any slide is predicted.

The slides are read with the reader the bundles were trained with: the
``backend`` that each bundle's encoder recipe records. To use another reader,
set ``extraction.tiling.backend``; soma then warns that the pixels may decode
differently than in training. Bundles that record different readers are refused
unless ``extraction.tiling.backend`` names one. When no bundle records a
reader, the slides are read with ``auto``.

A slide with no kept tile gets a warning, and an all-zero label map or a points
file with no rows.

.. _whole-slide-stitching:

How tiles are combined
~~~~~~~~~~~~~~~~~~~~~~

With ``overlap: 0``, the tiles touch and do not overlap. Each tile contributes
its whole output.

With an overlap, each tile gives up a margin of ``tile * overlap / 2`` pixels
on each side and contributes the middle of its output. The tiles of the slide's
first row and first column also keep their outer margin, so the map starts at
pixel 0. Each map pixel comes from exactly one tile. No output is averaged.

.. figure:: /_static/figures/whole-slide-stitching.svg
   :figclass: soma-figure
   :alt: Left, one tile with overlap 0.5: a margin of a quarter of its side runs around its edge, and the middle square is the region it contributes. Right, three tiles of one grid row, each shifted by one stride; the first keeps its outer margin, and the map row takes each pixel from exactly one tile.

   With overlap 0.5, a tile gives up a quarter of its side on each side and
   contributes the middle. A tile on the slide's first row or column also
   keeps its outer margin.

The margin must be a whole number of pixels. With 512 px tiles, an overlap of
0.25 gives 64 px and 0.5 gives 128 px. Another value is refused, and the error
proposes one that works. An overlap of 0.5 gives about four times as many tiles,
and takes about four times as long.

A pixel that no kept tile contributes is 0, "not predicted". So is the margin a
tile gives up next to a tile that was not kept.

For detection, each tile keeps the peaks in the region it contributes, and
places them on the slide in level-0 pixels. Then one NMS per class over the
whole slide removes the duplicates that two tiles found on either side of a
seam. It uses the head's NMS distance, which is the matching distance unless the
head sets one. With an overlap, the margin must be at least the NMS distance,
so that a tile sees the whole neighbourhood of each peak it keeps. A smaller
margin is refused, and the error says which overlap works.

.. _whole-slide-outputs:

What it writes
--------------

Each slide gets one file and one record per bundle, in the bundle's directory
of the output. :ref:`whole-slide-prediction-outputs` gives the full layout and
the index.

.. _whole-slide-label-maps:

Label TIFF (segmentation)
~~~~~~~~~~~~~~~~~~~~~~~~~

``<output_dir>/<bundle>/<sample_id>.tif`` is a pyramidal, tiled TIFF of uint8
values, stored lossless (LZW). Its pixels lie on the grid of the tiles at the
read spacing, so its spacing is the read spacing, up to hs2p's rounding of the
grid step. The index gives the ``width``, ``height`` and ``spacing_um`` of each
map. Each coarser level keeps every other pixel of the level below, so it holds
stored values only, never an average.

The values are shifted by one:

- ``0`` means "not predicted": no kept tile contributed the pixel.
- Class ``k`` is stored as ``k + 1``. A map holds at most 255 classes.

The PNG label maps that a run writes for its ROIs (``predictions_<split>/``)
store class ``k`` as ``k``. An ROI is always predicted in full, so it needs no
"not predicted" value.

The description of the first page holds a JSON record of what the map means:

- ``values``: the name of each stored value, such as
  ``{"0": "not predicted", "1": "stroma", "2": "tumor"}``;
- ``classes``: the raw annotation values that form each class, in class order;
- ``ignore``: the raw annotation values left out of the loss and the metrics;
- ``pixel_mapping``: the vocabulary that hs2p reads the annotation rasters with;
- ``sample_id``, ``ensemble``, ``read_spacing_um``, ``tile_size_px``,
  ``overlap``, and ``bundles``: the name, path and SHA-256 of each bundle.

``soma evaluate`` scores the map with this class scheme. Any TIFF library reads
the map:

.. code-block:: python

   import json

   import tifffile

   with tifffile.TiffFile("predictions/beetle-test/fold_0/slide-01.tif") as tiff:
       metadata = json.loads(tiff.pages[0].description)
       overview = tiff.series[0].levels[-1].asarray()  # the coarsest level

   classes = overview.astype(int) - 1  # -1 where nothing was predicted

.. _whole-slide-points:

Points parquet (detection)
~~~~~~~~~~~~~~~~~~~~~~~~~~

``<output_dir>/<bundle>/<sample_id>.parquet`` has four columns: ``x`` and ``y``
in the slide's level-0 pixels, ``class`` (the class index) and ``score``. It
holds every peak above the head's ``min_score``. No threshold is applied, so a
threshold can be chosen later. Its schema metadata (key ``soma``) records:

- the class scheme the bundle was trained with: ``classes`` maps each class
  name to the annotated ids that form it, in class order, and ``drop`` lists
  the dropped ids (see :ref:`detection-classes`);
- the matching and NMS distances, in µm, and ``min_score``;
- the slide's level-0 spacing, the read spacing, the tile size and the overlap;
- the bundles: the name, path and SHA-256 of each;
- the bundle's tune threshold (``threshold``).

The scheme comes from the bundle's ``preprocessing.json``. A bundle trained
before soma recorded it has no ``class_scheme``, and ``soma predict`` refuses
it before it writes anything. Train the bundle again, or add ``class_scheme``
to its ``preprocessing.json`` from the run's config, for example
``{"classes": {"mnl": [0, 1]}, "drop": [2]}``. The class names must be the
bundle's own.

.. _whole-slide-bundles:

Several bundles
---------------

``predict.bundles`` may name several bundles, such as the folds of one run. They
must all be of one task family, and agree on the tile size, the read spacing and
the read tolerance. The bundles that share an encoder recipe share one encoder
pass per tile, so each extra bundle costs mostly its decoder.

By default, each bundle gets its own prediction, in its own directory. The
directory is named by the shortest end of the bundle's path that tells the
bundles apart, without a last ``bundle``: ``runs/beetle/fold_0/bundle`` writes
to ``fold_0/``. Score each directory on its own:

.. code-block:: bash

   soma evaluate predictions/beetle-test/fold_0 --cohort dataset.csv --splits splits.csv

With ``ensemble: true``, the bundles give one prediction per slide, in
``ensemble/``:

- segmentation: the class with the highest mean softmax over the bundles;
- detection: the peaks of the bundles' mean heatmap, found with the first
  bundle's head.

The bundles of an ensemble must share their class scheme. For segmentation, that
is the classes, the raw annotation values that form them and the values ignored.
For detection, it is the classes, their annotated ids, the dropped ids, both
distances and ``min_score``. An ensemble's points record no threshold (see
:ref:`whole-slide-points-scoring`).

Score
-----

``soma evaluate`` scores one bundle's predictions, or one ensemble's. Pass the
output directory when it holds one bundle, or else the bundle's directory. Pass
the split you predicted: ``--splits splits.csv`` selects the ``test`` split of
fold 0 unless ``--split`` and ``--fold`` say otherwise. Without ``--splits``,
every annotated row of the cohort is a target. Every target needs a prediction,
and every prediction a target.

.. _whole-slide-scoring:

Score label maps
~~~~~~~~~~~~~~~~

:class:`~soma.evaluation.SlideLabelMaps` is a handle on one bundle's directory
of label TIFFs, and :func:`soma.evaluate` scores it against a
:class:`~soma.data.LabelMapSource` of the slides' annotation rasters:

.. code-block:: python

   import soma
   from soma.data import AnnotationManifest, ImageManifest, LabelMapSource
   from soma.evaluation import SlideLabelMaps

   maps = SlideLabelMaps("predictions/beetle-test/fold_0")
   targets = LabelMapSource.from_manifests(
       AnnotationManifest.from_csv("dataset.csv"),
       ImageManifest.from_csv("dataset.csv"),  # aligns each raster to its slide
       size=(0, 0),
       backend=maps.mask_backend,
       image_backend=maps.backend,
   )
   evaluation = soma.evaluate(maps, targets.subset(maps.sample_ids))
   evaluation.per_sample  # one row of counts per slide

``soma evaluate predictions/beetle-test/fold_0 --cohort dataset.csv --splits
splits.csv`` scores the test split the same way. The cohort CSV gives each slide's annotation raster in its
``label_mask_path`` column.

The slides and the annotation rasters are read with the hs2p readers that the
prediction's ``plan.yaml`` records (``maps.backend`` and ``maps.mask_backend``).
To read with other readers, pass ``--backend`` for the slides and
``--mask-backend`` for the annotation rasters:

.. code-block:: bash

   soma evaluate predictions/beetle-test/fold_0 --cohort dataset.csv --splits splits.csv \
       --backend openslide --mask-backend tifffile

In Python, pass the readers to ``LabelMapSource.from_manifests`` as
``image_backend`` and ``backend``. Both options apply to a ``soma predict``
output only; ``soma evaluate`` refuses them for other predictions.

Each slide is read one window at a time on the map's pixel grid, so neither the
map nor the annotation raster is held in memory. hs2p aligns an annotation
raster at another resolution to the slide, and reads it at the map's spacing.
The class scheme is the one each map records from its bundle's head: the raw
annotation values of each class, the raw values to ignore, and the vocabulary
the raster is read with, as in training. The class scheme, spacing and size of
the target source are not used.

- A pixel whose annotation value is ignored is not scored.
- An annotated pixel of class ``c`` that was not predicted (value 0 in the map)
  is a false negative for ``c``, and a false positive for no class.
- A map pixel that lies partly off the slide is not scored.

Each slide gives one row of counts to the same reducers as
:class:`~soma.evaluation.LabelMapPredictions`. ``Evaluation.per_sample`` holds
the rows: ``annotated_px``, ``not_predicted_px``, ``not_predicted_fraction``
(the annotated pixels not predicted, over all annotated pixels) and
``tp_class_<c>``, ``fp_class_<c>``, ``fn_class_<c>`` per class. ``soma
evaluate`` prints them in a second table.

.. _whole-slide-points-scoring:

Score points
~~~~~~~~~~~~

:class:`~soma.evaluation.SlidePoints` is a handle on one bundle's directory of
points parquets, and :func:`soma.evaluate` scores it against a
:class:`~soma.data.PointSource` of the slides' annotated points:

.. code-block:: python

   import soma
   from soma.data import AnnotationManifest, PointSource
   from soma.evaluation import SlidePoints

   points = SlidePoints("predictions/midog-test/fold_0")
   targets = PointSource.from_manifest(AnnotationManifest.from_csv("dataset.csv"))
   evaluation = soma.evaluate(points, targets.subset(points.sample_ids))
   evaluation.per_sample  # one row of counts per slide

``soma evaluate predictions/midog-test/fold_0 --cohort dataset.csv --splits
splits.csv`` scores the test split the same way. The cohort CSV gives each slide's ``points_path``, a ``x,y,class`` file in
level-0 pixels that holds the dataset's own annotated ids, and optionally its
``ignore_mask_path``.

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

   tune = SlidePoints("predictions/tune/ensemble")  # an ensemble's points
   threshold, _ = sweep_threshold(tune, targets.subset(tune.sample_ids), per_class=True)
   test = SlidePoints("predictions/test/ensemble").with_threshold(threshold)
   evaluation = soma.evaluate(test, targets.subset(test.sample_ids))

Each slide gives one row of counts to the same reducers as
:class:`~soma.evaluation.PointPredictions`. ``Evaluation.per_sample`` holds the
rows: ``annotated_area_mm2`` (the annotated mask pixels times their area, or the
whole slide without a mask) and ``tp_class_<c>``, ``fp_class_<c>``,
``fn_class_<c>`` per class. The counts use the matching that all the metrics
name. If the metrics name none, or name different ones, the counts use the
default matching.

Matching lists only the pairs of points within the matching distance, so a
slide with many points takes little memory: 100,000 predictions against 20,000
annotations of one class match in about a second.

.. _whole-slide-not-comparable:

Slide scores are not run metrics
--------------------------------

A slide score is not comparable to the test metrics of the run that trained the
bundle. The two measure different things:

- A run scores only what it samples. For segmentation, these are the ROIs where
  an annotation class meets ``min_coverage``; for detection, the ROIs or tiles
  of its test split. A slide score covers every annotated pixel or point of the
  slide, and for detection every prediction outside the ignored region.
- A slide score counts the annotated pixels that were not predicted as misses.
  A run predicts its ROIs in full.
- A metric that averages over samples changes meaning: a sample is an ROI in a
  run and a slide here.

In the BEETLE check below, the pooled ``dataset_global_mean_dice`` was 0.883
over whole slides and 0.893 over the run's ROIs, and the Dice of each class
agreed within 0.020. But ``mean_dice``, which averages over samples, was 0.537
over slides and 0.761 over ROIs. 1.7% of the annotated pixels were not
predicted, mostly on two slides (8.1% and 2.1%).

Compare slide scores with slide scores, and run metrics with run metrics.

.. _whole-slide-cost:

Cost
----

These numbers come from one check on BEETLE breast resections, on a shared
machine: read them as orders of magnitude. The setup: one RTX 3080 Ti (12 GB),
Virchow2 in fp16, 512 px tiles at 0.5 µm, a ``lightweight_conv`` decoder with 4
classes, ``overlap: 0``, hsv tissue segmentation, batch size 8, and slides read
with openslide from a local disk.

One slide of 92,819 × 45,702 px, with 10,117 kept tiles:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - One slide
     - ``num_workers: 0``
     - ``num_workers: 8``
   * - Wall time
     - 24.7 min
     - 22.6 min
   * - Tiles per second
     - 6.8
     - 7.5
   * - Model (encoder and decoder)
     - 90% of the time
     - 99% of the time
   * - Tile reads
     - 9% (13.6 ms per tile)
     - hidden by the workers
   * - Stitching and TIFF writing
     - 1%
     - 1%
   * - Peak memory, main process
     - 3.1 GiB
     - 3.1 GiB
   * - Peak GPU memory
     - 2.95 GiB
     - 2.95 GiB

- **Speed.** The encoder sets the pace. Prediction runs at about the speed of
  feature extraction with the same encoder. With ``num_workers`` above 0, the
  tile reads overlap with the model.
- **Memory.** Peak memory grows with the slide's size in pixels, not with its
  tissue: hs2p's TIFF writer holds the coarser levels of the map in memory,
  about a third of the raw map. The largest slide (136,064 × 147,072 px, 2,052
  kept tiles) peaked at 7.2 GiB, 6.0 GiB above the level after the model
  loaded. Writing took 23% of its 6.0 min, because the writer processes the
  whole map, also where nothing was predicted.
- **File size.** The maps compress well. The slide above takes 53 MB, for a
  raw map of 3.95 GiB. The largest slide takes 181 MB.
- **A test set.** 10 slides (38,617 kept tiles) took 47.6 min on 3 GPUs, and
  622 MB of maps.
- **Overlap.** An overlap of 0.5 gave four times the tiles and the time on one
  small slide, and the same score within 0.010.
- **Scoring.** ``soma evaluate`` took 30 min for the 10 slides, in one process,
  with a peak memory of 0.9 GiB. It reads every window of the map and of the
  annotation raster, also the empty ones.

Detection, ensembles, several bundles and slides resampled from a finer level
were not measured.

Reference
---------

.. autofunction:: soma.predict_slides
