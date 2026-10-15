EVA
===

Reproduce the `kaiko-ai/eva <https://github.com/kaiko-ai/eva>`_ pathology
leaderboard with frozen encoders: patch classification with a linear head, nucleus
segmentation with a small convolutional decoder, and slide classification with
attention pooling over a fixed set of tiles per slide. Ten public datasets are
registered as ``eva/<dataset>``.

Data and task
-------------

Every dataset keeps EVA's own splits and reports on the split EVA reports on.

.. list-table::
   :header-rows: 1
   :widths: 22 24 38 16

   * - Dataset
     - Task
     - Samples
     - Metric
   * - ``bach``
     - Patch classification
     - Breast histology patches (ICIAR 2018), four classes
     - Balanced accuracy
   * - ``breakhis``
     - Patch classification
     - Breast tumour patches at 40×, EVA's class selection
     - Balanced accuracy
   * - ``crc``
     - Patch classification
     - Colorectal tissue patches, nine tissue types
     - Balanced accuracy
   * - ``gleason_arvaniti``
     - Patch classification
     - Prostate tissue-microarray patches, Gleason patterns
     - Balanced accuracy
   * - ``mhist``
     - Patch classification
     - Colorectal polyp patches, hyperplastic or sessile serrated
     - Balanced accuracy
   * - ``patch_camelyon``
     - Patch classification
     - Lymph-node patches, metastasis present or absent
     - Balanced accuracy
   * - ``consep``
     - Nucleus segmentation
     - Colorectal nuclei, four nucleus types and background
     - Foreground mean Dice
   * - ``monusac``
     - Nucleus segmentation
     - Multi-organ nuclei, four cell types and background
     - Foreground mean Dice
   * - ``camelyon16_small``
     - Slide classification
     - Lymph-node whole slides, metastasis present or absent
     - Balanced accuracy
   * - ``panda_small``
     - Slide classification
     - Prostate biopsy whole slides, six ISUP grades
     - Balanced accuracy

Protocol
--------

Each task group shares one recipe, transcribed from EVA's configs; the encoder is the
only varied axis, and every score is the mean over the canonical seeds.

Patch classification
~~~~~~~~~~~~~~~~~~~~

Labelled patches → frozen encoder → linear head → balanced accuracy.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Setting
     - Value
   * - Head
     - A linear probe on the patch embedding; each patch is its own bag
       (``aggregation: null``).
   * - Optimizer
     - AdamW, learning rate ``0.0003``, weight decay ``0.01``, batch size ``256``.
   * - Budget
     - ``12500`` optimizer steps, with EVA's per-dataset early-stopping patience.
   * - Seeds
     - ``0``–``4``, averaged.

Nucleus segmentation
~~~~~~~~~~~~~~~~~~~~

224 px tiles and masks → frozen dense grid, plus the tile's pixels →
``eva_conv_with_image`` decoder → foreground mean Dice.

The curators reproduce EVA's sample geometry. CoNSeP is cut into 250 px grid tiles at
its native 0.25 µm/px (16 per image), each resized to 224 px. MoNuSAC test images are
resized on their short side to 224 px and centre-cropped; its train images are kept
whole, and every training step draws a new random resized 224 px crop from them, as
EVA's online config does. CoNSeP merges HoVer-Net's seven nucleus types into other,
inflammatory, epithelial and spindle-shaped; MoNuSAC's test-only ``Ambiguous`` class is
excluded from the loss and from the metric's target area, while pixels predicted there
still count towards the predicted area, as in EVA.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Setting
     - Value
   * - Features
     - The last block's patch tokens before the backbone's final normalisation
       (``feature_kind: patch_features_prenorm``), where EVA taps them. CoNSeP uses
       cached dense grids; MoNuSAC re-encodes every step (``features.kind: live``)
       because its train crops are random.
   * - Decoder
     - ``eva_conv_with_image``, EVA's ``ConvDecoderWithImage``: nearest ×2 → 3×3
       conv-BN-ReLU (64) → bilinear to the tile size → concatenate the
       ImageNet-normalised tile → 2 × 3×3 conv-BN-ReLU (32) → 1×1 conv.
   * - Loss
     - Soft Dice over the softmax, background included (``eva_segmentation`` head).
   * - Optimizer
     - AdamW, learning rate ``0.002``, weight decay ``0.01``, batch size ``64``;
       EVA's ``ConstantLR`` warm-up (lr/3 for the first 5 epochs).
   * - Budget
     - ``2000`` optimizer steps.
   * - Metric
     - ``foreground_mean_dice``: per-image mean Dice over the foreground classes,
       empty targets skipped, as MONAI's ``DiceMetric(include_background=False)``.
       MoNuSAC is re-scored from the run's saved label maps with EVA's counting.
   * - Seeds
     - ``0``–``4``, averaged.

Slide classification
~~~~~~~~~~~~~~~~~~~~

EVA's tiles per slide → frozen encoder → ``eva_abmil`` attention pooling → MLP head →
balanced accuracy.

The curators choose the tiles EVA chooses and hand them to soma as ready-made
coordinates (see :ref:`bring your own coordinates <preprocessing-supplied-coordinates>`).
EVA lays a non-overlapping grid of 224 px cells at the target spacing over level 0,
shuffles the cells with a fixed seed and keeps the first ones whose foreground
fraction is at least 0.35. Camelyon16Small uses the 399 slides of the official release:
216 train, 54 tune (the slides EVA holds out for validation) and 129 test.
PANDASmall keeps the 9555 PANDA slides EVA does not flag as noisy and takes EVA's
stratified split by ISUP grade: 952 train, 475 tune and 475 test.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Setting
     - Value
   * - Tiles
     - At most 1000 tiles at 0.25 µm/px (``camelyon16_small``) or
       200 tiles at 0.5 µm/px (``panda_small``), read at level 0 and resized to 224 px.
   * - Aggregator
     - ``eva_abmil``, EVA's ``ABMIL``: a linear projection to 128, then gated
       attention pooling (``tanh`` × ``sigmoid``, hidden width 128), no dropout.
   * - Head
     - An MLP 128 → 128 → 64 → output with ReLU: ``eva_mil_binary`` (one logit,
       ``BCEWithLogitsLoss``) or ``eva_mil_multiclass`` (cross-entropy).
   * - Optimizer
     - AdamW, learning rate ``0.001``, weight decay ``0.01``, batch size ``32``
       bags; EVA's ``ConstantLR`` warm-up (lr/3 for the first 5 epochs).
   * - Budget
     - ``100`` epochs, early-stopping patience ``20`` counted as Lightning's
       ``EarlyStopping`` does; the best checkpoint on the tune balanced accuracy.
   * - Seeds
     - ``0``–``19``, averaged.

The decoder, loss, aggregator and heads above exist only for this benchmark and are
registered under ``eva_`` names; they are not general soma components. EVA resizes
each slide tile with bilinear, antialiased interpolation and slide2vec with area
interpolation; soma accepts this difference and measures its effect when the benchmark
is recorded.

Results
-------

soma's score next to the value EVA publishes for the same encoder. The reference band
is 2 % of the published value; a soma value outside it is shown in red. The published
values are the `kaiko-ai/eva pathology leaderboard
<https://github.com/kaiko-ai/eva/blob/main/tools/data/leaderboards/pathology.csv>`__.

Patch classification
~~~~~~~~~~~~~~~~~~~~

.. include:: _generated/eva-patch-results.rst

Nucleus segmentation
~~~~~~~~~~~~~~~~~~~~

.. include:: _generated/eva-segmentation-results.rst

Slide classification
~~~~~~~~~~~~~~~~~~~~

.. include:: _generated/eva-slide-results.rst

Reproduce
---------

**Prepare the data.** Download a dataset from its official source and unpack it in
the directory you will pass as ``--raw-root``. For a family run, put one such
directory per dataset under a common root, named as the dataset is.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Dataset and source
     - Raw-root contents
   * - `BACH <https://zenodo.org/records/3632035>`__ (``bach``)
     - ``ICIAR2018_BACH_Challenge/Photos/<class>/*.tif``
   * - `BreaKHis <https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/>`__ (``breakhis``)
     - ``BreaKHis_v1/histology_slides/…/40X/*.png``; soma selects EVA's classes
   * - `CRC <https://zenodo.org/records/1214456>`__ (``crc``)
     - ``NCT-CRC-HE-100K/`` and ``CRC-VAL-HE-7K/``
   * - `Gleason Arvaniti <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP>`__ (``gleason_arvaniti``)
     - the ``ZT{76_39,111_4,199_1,204_6}*.tar.gz`` TMA archives and ``Gleason_masks_train.tar.gz``
   * - `MHIST <https://bmirds.github.io/MHIST/#accessing-dataset>`__ (``mhist``)
     - ``images/*.png`` and ``annotations.csv``
   * - `PatchCamelyon <https://zenodo.org/records/2546921>`__ (``patch_camelyon``)
     - the six ``camelyonpatch_level_2_split_{train,valid,test}_{x,y}.h5`` files
   * - `CoNSeP <https://github.com/vqdang/hover_net/issues/267>`__ (``consep``)
     - ``Train/Images/*.png``, ``Train/Labels/*.mat``, ``Test/Images``, ``Test/Labels``
       (the HoVer-Net layout; the official download is login-walled, so use a public
       mirror such as the Kaggle ``consep`` dataset)
   * - `MoNuSAC <https://monusac-2020.grand-challenge.org/Data/>`__ (``monusac``)
     - ``MoNuSAC_images_and_annotations/`` and ``MoNuSAC Testing Data and Annotations/``
       (one ``.tif`` and ``.xml`` per image; CC BY-NC-SA 4.0)
   * - `Camelyon16 <https://camelyon17.grand-challenge.org/Data/>`__ (``camelyon16_small``)
     - ``images/*.tif`` (the 399 slides of the official release) and ``evaluation/reference.csv``
   * - `PANDA <https://www.kaggle.com/c/prostate-cancer-grade-assessment/data>`__ (``panda_small``)
     - ``train_images/*.tiff`` and EVA's noisy-label table ``train_with_noisy_labels.csv``
       (the ``train.csv`` of ``analokmaus/kaggle-panda-challenge-public``)

The slide-level curators read the slides with OpenSlide (``pip install
openslide-python openslide-bin``) and take a while on the full datasets.

**Run one dataset** or the whole family. Curated manifests are written under
``<raw-root>/curated`` (``--out-dir`` moves them; the segmentation curators write
tiles and masks there, so give it room)::

    soma reproduce eva/bach --encoder virchow2 --raw-root /path/to/eva/bach
    soma reproduce eva/consep --encoder virchow2 --raw-root /path/to/eva/consep
    soma reproduce eva --encoder virchow2 --raw-root /path/to/eva

Reuse a finished curation with ``--curated-dir`` in place of ``--raw-root``.

**Compare encoders** with ``--encoders``, and append the scores to the results ledger
with ``--record``. :doc:`benchmarking` describes the options shared by every
benchmark.
