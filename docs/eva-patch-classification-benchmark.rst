EVA
===

Reproduce the `kaiko-ai/eva <https://github.com/kaiko-ai/eva>`_
pathology leaderboard: patch classification with frozen tile encoders and
linear :doc:`classification` heads, patch :doc:`segmentation` with a
small convolutional decoder on the frozen dense feature grid, and slide
classification with an attention-MIL head over a fixed set of tiles per slide.

EVA provides 6 registered classification datasets: bach, breakhis, crc, gleason_arvaniti, mhist, and patch_camelyon, 2 segmentation datasets: consep and monusac, and 2 slide-level datasets: camelyon16_small and panda_small. Each group shares one protocol; see :doc:`benchmarking` for
the shared workflow.

**Pipeline (classification):** labelled patches → frozen encoder → linear head → balanced accuracy

**Pipeline (segmentation):** 224 px tiles and masks → frozen dense grid + the tile's pixels → ``eva_conv_with_image`` decoder → foreground mean Dice

**Pipeline (slide):** EVA's tiles per slide → frozen encoder → ``eva_abmil`` attention pooling → MLP head → balanced accuracy

Prepare the data
----------------

Download one EVA dataset from its official
source and unpack it in the directory you will pass as ``--raw-root``:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Dataset and source
     - Raw-root contents
   * - `BACH <https://zenodo.org/records/3632035>`__ (``bach``)
     - ``ICIAR2018_BACH_Challenge/Photos/<class>/*.tif``
   * - `BreaKHis <https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/>`__ (``breakhis``)
     - ``BreaKHis_v1/histology_slides/…/40X/*.png``; soma selects EVA classes
   * - `CRC <https://zenodo.org/records/1214456>`__ (``crc``)
     - ``NCT-CRC-HE-100K/`` and ``CRC-VAL-HE-7K/``
   * - `Gleason Arvaniti <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP>`__ (``gleason_arvaniti``)
     - the ``ZT{76_39,111_4,199_1,204_6}*.tar.gz`` TMA archives and ``Gleason_masks_train.tar.gz``
   * - `MHIST <https://bmirds.github.io/MHIST/#accessing-dataset>`__ (``mhist``)
     - ``images/*.png`` and ``annotations.csv``
   * - `PatchCamelyon <https://zenodo.org/records/2546921>`__ (``patch_camelyon``)
     - the six ``camelyonpatch_level_2_split_{train,valid,test}_{x,y}.h5`` files

The segmentation datasets use the same ``--raw-root`` convention:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Dataset and source
     - Raw-root contents
   * - `CoNSeP <https://github.com/vqdang/hover_net/issues/267>`__ (``consep``)
     - ``Train/Images/*.png``, ``Train/Labels/*.mat``, ``Test/Images``, ``Test/Labels`` (HoVer-Net layout; the official Warwick download is login-walled and the linked issue tracks that, so use a public mirror such as the Kaggle ``consep`` dataset)
   * - `MoNuSAC <https://monusac-2020.grand-challenge.org/Data/>`__ (``monusac``)
     - ``MoNuSAC_images_and_annotations/`` and ``MoNuSAC Testing Data and Annotations/`` (one ``.tif`` + ``.xml`` per image; CC BY-NC-SA 4.0)

So do the slide-level datasets. Their curators read the slides with OpenSlide
(``pip install openslide-python openslide-bin``):

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Dataset and source
     - Raw-root contents
   * - `Camelyon16 <https://camelyon17.grand-challenge.org/Data/>`__ (``camelyon16_small``)
     - ``images/*.tif`` (the 399 slides of the official release) and ``evaluation/reference.csv``
   * - `PANDA <https://www.kaggle.com/c/prostate-cancer-grade-assessment/data>`__ (``panda_small``)
     - ``train_images/*.tiff`` and EVA's noisy-label table ``train_with_noisy_labels.csv`` (the ``train.csv`` of ``analokmaus/kaggle-panda-challenge-public``)

Run the benchmark
-----------------

Choose a compatible tile-level :doc:`encoder <encoders>` and pass the downloaded
dataset directory as ``--raw-root``; curated manifests are written under
``<raw-root>/curated``. For example::

    soma reproduce eva/bach --encoder virchow2 --raw-root /path/to/eva/bach

To run the whole family, prepare one subdirectory per dataset under
``/path/to/eva``::

    soma reproduce eva --encoder virchow2 --raw-root /path/to/eva

A segmentation member runs the same way::

    soma reproduce eva/consep --encoder virchow2 --raw-root /path/to/eva/consep

Its curator writes the 224 px tiles and masks under ``--out-dir`` (default
``<raw-root>/curated``), so point ``--out-dir`` at a directory with room for them.

A slide-level member runs the same way::

    soma reproduce eva/panda_small --encoder virchow2 --raw-root /path/to/eva/panda_small

Its curator reads a low-resolution level of every slide, so it takes a while on
the full datasets. Reuse a finished curation with ``--curated-dir`` instead of
``--raw-root``. In a family run (``soma reproduce eva``), the raw roots are
``<raw-root>/camelyon16_small`` and ``<raw-root>/panda_small``.

Results
-------

Recorded balanced accuracy scores alongside the packaged EVA references.

.. list-table::
   :header-rows: 1

   * - Dataset
     - Encoder
     - soma (mean ± std)
     - EVA reference
   * - bach
     - uni2
     - 0.914 ± 0.008
     - 0.915
   * - bach
     - virchow2
     - 0.871 ± 0.011
     - 0.883
   * - breakhis
     - uni2
     - 0.855 ± 0.007
     - 0.859
   * - breakhis
     - virchow2
     - 0.812 ± 0.008
     - 0.821
   * - camelyon16_small
     - uni2
     - 0.854 ± 0.013
     - 0.849
   * - camelyon16_small
     - virchow2
     - 0.858 ± 0.011
     - 0.861
   * - crc
     - uni2
     - 0.966 ± 0.001
     - 0.965
   * - crc
     - virchow2
     - 0.966 ± 0.001
     - 0.967
   * - gleason_arvaniti
     - uni2
     - 0.779 ± 0.005
     - 0.775
   * - gleason_arvaniti
     - virchow2
     - 0.778 ± 0.010
     - 0.783
   * - mhist
     - uni2
     - 0.824 ± 0.003
     - 0.824
   * - mhist
     - virchow2
     - 0.861 ± 0.001
     - 0.861
   * - panda_small
     - uni2
     - 0.649 ± 0.017
     - 0.657
   * - panda_small
     - virchow2
     - 0.652 ± 0.017
     - 0.646

Recorded foreground mean Dice scores alongside the packaged EVA references.

.. list-table::
   :header-rows: 1

   * - Dataset
     - Encoder
     - soma (mean ± std)
     - EVA reference
   * - consep
     - uni2
     - 0.630 ± 0.003
     - 0.630
   * - consep
     - virchow2
     - 0.642 ± 0.001
     - 0.640
   * - monusac
     - uni2
     - 0.642 ± 0.004
     - 0.642
   * - monusac
     - virchow2
     - 0.673 ± 0.002
     - 0.669

See the `kaiko-ai/eva pathology leaderboard <https://github.com/kaiko-ai/eva/blob/main/tools/data/leaderboards/pathology.csv>`__ for the official reference leaderboard.

Protocol details
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Value
   * - head
     - linear probe (``aggregation: null`` — each patch is its own bag)
   * - optimizer
     - AdamW, lr ``0.0003``, weight_decay ``0.01``
   * - batch size
     - ``256``
   * - budget
     - fixed step budget: ``max_steps=12500`` optimizer updates
   * - metric
     - ``balanced_accuracy``
   * - varied axis
     - ``encoder``
   * - primary metric
     - ``test/balanced_accuracy`` (from ``summary.json``)
   * - canonical seeds
     - ``0, 1, 2, 3, 4`` (averaged)

Segmentation protocol
---------------------

The curators reproduce EVA's sample geometry: CoNSeP is cut into 250 px grid
tiles at its native 0.25 µm/px (16 per image) and each tile is resized to 224 px.
MoNuSAC test images are resized on their short side to 224 px and centre-cropped;
MoNuSAC train images are kept whole, and each training step draws a new random
resized 224 px crop from them, as EVA's online config does. CoNSeP
merges HoVer-Net's seven nucleus types into background, other, inflammatory,
epithelial and spindle-shaped; MoNuSAC's test-only ``Ambiguous`` class is
excluded from the loss and the metric. Both datasets report on EVA's validation
split, which is soma's ``test`` split (``tune_is_test``).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Value
   * - samples
     - 224 px tiles and class-index masks materialised by the curator (MoNuSAC train: whole images)
   * - features
     - CoNSeP: cached dense grids. MoNuSAC: re-encoded every step (``feature_mode: live``) because its train crops are random
   * - augmentation
     - MoNuSAC train only: ``random_resized_crop_scale: [0.08, 1.0]`` (torchvision's ``RandomResizedCrop(224)`` defaults). CoNSeP: none
   * - decoder
     - ``eva_conv_with_image`` (EVA's online ``ConvDecoderWithImage``, the leaderboard decoder): nearest ×2 → 3x3 conv-BN-ReLU (64) → bilinear to the tile size → concat the ImageNet-normalised RGB tile → 2 × 3x3 conv-BN-ReLU (32) → 1x1 conv (classes). ``eva_conv_ms`` (the offline ``ConvDecoderMS``) stays registered for reference
   * - loss
     - pure soft Dice over softmax, background included (``eva_segmentation`` head)
   * - optimizer
     - AdamW, lr ``0.002``, weight_decay ``0.01``, EVA's default ``ConstantLR`` warm-up (lr/3 for the first 5 epochs)
   * - batch size
     - ``64``
   * - budget
     - fixed step budget: ``max_steps=2000`` optimizer updates
   * - metric
     - ``foreground_mean_dice`` — per-image mean Dice over the foreground classes, empty targets skipped (MONAI ``DiceMetric(include_background=False)``)
   * - varied axis
     - ``encoder``
   * - primary metric
     - ``test/foreground_mean_dice`` (from ``summary.json``)
   * - canonical seeds
     - ``0, 1, 2, 3, 4`` (averaged)

The decoder, loss and confusion counting exist only for this benchmark and are
registered under ``eva_`` names; they are not general soma components. The dense
grid is tapped where EVA taps it: the last block's patch tokens *before* the
backbone's final normalisation layer (``feature_kind: patch_features_prenorm``,
timm's ``features_only`` output), so the decoder sees the same token scale as the
leaderboard decoders.

Slide-level protocol
--------------------

The curators choose the tiles EVA chooses and write them as one hs2p tiling
artifact per slide, named by the manifest's ``coordinates_path`` column (see
:ref:`bring your own coordinates <preprocessing-supplied-coordinates>`); soma skips its
own tiling. EVA lays a non-overlapping grid over level 0, with a cell side of
224 px at the target spacing, shuffles the cells with a fixed seed, and keeps the
first cells whose foreground fraction is at least 0.35. Foreground is HSV
saturation above 20 at a low-resolution level. The slide spacing is EVA's: the
OpenSlide ``mpp`` properties, else the TIFF resolution tags. The curator writes it
as ``spacing_at_level_0``.

Camelyon16Small uses the 399 slides of the official release. The 54 training slides
that EVA holds out for validation (the PatchCamelyon validation slides) are soma
``tune``, and the official test slides are ``test`` (216 / 54 / 129 slides).
PANDASmall keeps the 9555 PANDA slides that EVA does not filter as noisy and takes
EVA's stratified split by ISUP grade: 952 / 475 / 475 slides for train / tune / test.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Value
   * - tiles
     - chosen by the curator with EVA's sampler: at most 1000 at 0.25 µm/px (camelyon16_small), 200 at 0.5 µm/px (panda_small); read at level 0 and resized to 224 px by slide2vec
   * - aggregator
     - ``eva_abmil`` (EVA's ``ABMIL``): ``Linear(D → 128)`` projection, then gated attention pooling (``tanh`` × ``sigmoid``, hidden width 128). No dropout
   * - head
     - ``eva_mil_binary`` (one logit, ``BCEWithLogitsLoss``, positive when the logit is above 0) and ``eva_mil_multiclass`` (one logit per class, cross entropy): an MLP 128 → 128 → 64 → output with ReLU
   * - optimizer
     - AdamW, lr ``0.001``, weight_decay ``0.01``, EVA's default ``ConstantLR`` warm-up (lr/3 for the first 5 epochs)
   * - batch size
     - ``32`` bags, padded and masked, shuffled, last batch kept
   * - budget
     - ``100`` epochs, early-stopping patience ``20`` counted as Lightning's ``EarlyStopping`` does (an initial plateau counts too, ``hold_patience_until_monitor_moves: false``), best checkpoint on the validation balanced accuracy
   * - splits
     - EVA's validation split is soma ``tune``; the reported split is ``test``
   * - metric
     - ``balanced_accuracy``
   * - varied axis
     - ``encoder``
   * - primary metric
     - ``test/balanced_accuracy`` (from ``summary.json``)
   * - canonical seeds
     - ``0``–``19`` (20 seeds, averaged)

The aggregator and heads exist only for this benchmark and are registered under
``eva_`` names; they are not general soma components. EVA resizes each tile with
bilinear, antialiased interpolation; slide2vec uses area interpolation. soma
accepts this difference and measures its effect when the benchmark is recorded.
