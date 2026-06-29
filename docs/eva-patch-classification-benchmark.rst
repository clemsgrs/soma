EVA
===

*Maps to task:* :doc:`classification` — these are frozen-tile-probe runs of the
multiclass / binary classification heads on public patch-level benchmarks.

A suite of patch-level classification benchmarks reproduced on soma's
:doc:`classification` path under a single shared **frozen-tile-probe** protocol.
Each dataset reuses the same recipe — only the manifest and the head (binary vs
multiclass) change — so the table below differs by dataset, not by method.

.. note::

   This page is a **scaffold**. The reference and soma result cells are
   intentionally left as ``TBD`` and are populated by the run issues as the
   numbers are produced. Do not backfill them with estimates.

The frozen-tile-probe protocol
------------------------------

Stated once, shared by every dataset in the table:

* **Frozen encoder.** A pretrained foundation model (``uni`` in the shipped
  configs) extracts one feature vector per patch; the encoder is never
  fine-tuned, so its features are cached once and reused across epochs.
* **Linear probe.** With ``aggregation: null`` each patch *is* the bag — the
  classification head is a linear probe on the frozen patch embedding (binary or
  multiclass per dataset).
* **Metric.** ``balanced_accuracy`` throughout, so class imbalance does not
  inflate the headline.
* **Splits.** soma curates EVA's layout into ``train`` / ``tune`` / ``test`` (see
  :doc:`curation`). Where EVA ships only train/validation, the EVA validation
  split is reserved as soma ``test`` and the run sets ``tune_is_test: true`` to
  reproduce EVA's train-on-all-train / evaluate-on-validation protocol; where EVA
  ships a held-out test split (``patch_camelyon``) it is preserved as soma
  ``test``.

Per-dataset table
-----------------

Datasets and their per-dataset axes. The reference number is the published EVA
balanced-accuracy for the same encoder; the soma number is soma's reproduction —
both filled in by the run issues, never invented here.

.. list-table::
   :header-rows: 1
   :widths: 18 12 26 18 13 13

   * - Dataset
     - Classes
     - Metric
     - Eval split
     - Reference
     - soma
   * - ``bach``
     - 4
     - balanced accuracy
     - EVA validation (``tune_is_test``)
     - TBD
     - TBD
   * - ``breakhis``
     - 4
     - balanced accuracy
     - EVA validation (``tune_is_test``)
     - TBD
     - TBD
   * - ``crc``
     - 9
     - balanced accuracy
     - EVA validation (``tune_is_test``)
     - TBD
     - TBD
   * - ``mhist``
     - 2
     - balanced accuracy
     - EVA validation (``tune_is_test``)
     - TBD
     - TBD
   * - ``patch_camelyon``
     - 2
     - balanced accuracy
     - EVA test
     - TBD
     - TBD

Reproduce
---------

One recipe covers the whole suite — swap the dataset name and nothing else. The
per-dataset configs live under ``examples/eva/`` (one YAML per dataset:
``bach.yaml``, ``breakhis.yaml``, ``crc.yaml``, ``mhist.yaml``,
``patch_camelyon.yaml``); the curation step is documented on :doc:`curation`.

#. **Curate** the raw EVA layout into soma manifests (see :doc:`curation`)::

      from soma.curation import curate_eva_patch_dataset

      curate_eva_patch_dataset(
          "mhist",
          raw_root="/path/to/mhist",
          output_dir="data/eva/mhist",
      )

#. **Run** the shipped config for that dataset, repointing it at your curated
   manifests::

      python -m soma examples/eva/mhist.yaml \
        --set data.dataset_csv=data/eva/mhist/dataset.csv \
        --set data.splits_csv=data/eva/mhist/splits.csv

The headline ``balanced_accuracy`` is written to the run's ``metrics.json``;
record it in the soma column above.

.. seealso::

   * :doc:`classification` — the task heads the probe trains (binary, multiclass).
   * :doc:`curation` — the EVA curators and split policy.
   * ``examples/eva/`` — the per-dataset run configs.
