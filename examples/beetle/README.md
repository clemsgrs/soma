# BEETLE frozen-Virchow2 development protocol

This directory is the tracked Project protocol for the BEETLE rebuttal experiment.
BEETLE is not registered or described as a Benchmark or Built-in Benchmark: the
installable `soma` package stays dataset-neutral, while this bundle owns cohort policy,
the two sampling arms, reporting, and selection.

## 1. Curate the development cohort

Run from the repository root after placing the released files under `data/beetle`:

```bash
python -m examples.beetle.curate --beetle-root data/beetle
```

Publication curation requires all 587 slides / 527 patients and writes the canonical
Manifest (`dataset.csv`, `splits.csv`, `summary.json`). A four-slide, explicitly
non-publication curation check is available with `--slides 4`.

The organizer folds are preserved. Across the five Soma folds, every development patient
appears in held-out `tune` exactly once; that split both selects the fold checkpoint and
supplies its reported development prediction.

## 2. Complete hardware and encoder preflight

The shared `configs/base.yaml` intentionally contains
`${BEETLE_CAMPAIGN_BATCH_SIZE}` and is not runnable as-is. Each arm YAML is a minimal
overlay containing only sampling and arm-identifying output metadata. On the selected
node, preflight batch sizes in order `16`, `8`, `4`, then write the largest passing size
to a JSON record. The record must have this shape:

```json
{
  "schema_version": 1,
  "status": "completed",
  "scope": "campaign",
  "batch_size_candidates": [16, 8, 4],
  "batch_size_attempts": [
    {"batch_size": 16, "passed": false},
    {"batch_size": 8, "passed": true},
    {"batch_size": 4, "passed": true}
  ],
  "selected_batch_size": 8,
  "same_batch_every_arm_and_fold": true,
  "encoder": {
    "repository": "paige-ai/Virchow2",
    "revision": "3158645804b69e3f3bc4439d4116edddf0840a72",
    "weight_file": "model.safetensors",
    "weight_sha256": "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b",
    "patch_size": 14,
    "feature_channels": 1280,
    "weight_checksum_verified": true,
    "snapshot_path": "/absolute/hf/hub/models--paige-ai--Virchow2/snapshots/3158645804b69e3f3bc4439d4116edddf0840a72"
  }
}
```

The production preflight must obtain the repository revision from the pinned local
snapshot and compute the file digest itself (for example, with `sha256sum`). Keep the
gated weights local. `encoder_lock.json` records the expected identity; the resolver
requires the snapshot's Timm `config.json`, hashes the actual weight file in
`snapshot_path`, and rejects a revision, checksum, geometry, or batch-choice mismatch.
It creates an isolated, weight-free Hub view
whose sole `main` snapshot is a symlink to that validated local object. `launch run`
forces Hugging Face and Transformers offline against that view, so a moving repository
head cannot replace or invalidate the pinned encoder and no credential is needed after
preflight.

The node gate should additionally record GPU/CUDA facts, writable local and mirror
storage, clean-cache capacity, representative ordinary/native-spacing extraction,
fp16-versus-fp32 cache parity, and recovery/resume checks. These observations belong in
the preflight artifact; no GPU model or memory assumption is embedded in the protocol.

## 3. Resolve and launch both arms

```bash
python -m examples.beetle.launch resolve \
  --preflight data/beetle/hardware_preflight.json \
  --output-dir data/beetle/resolved

python -m examples.beetle.launch run \
  --preflight data/beetle/hardware_preflight.json \
  --output-dir data/beetle/resolved
```

The resolver writes `uniform.yaml`, `class_conditioned.yaml`, and
`protocol_resolution.json`. Both configs use one strict clean cache recipe: frozen
Virchow2 patch tokens, fp16 computation and storage, 224-pixel windows with 0.5 overlap,
patch size 14, and 1,280 feature channels. They use the same no-augmentation,
lightweight-convolutional-decoder recipe and the same 30-epoch Adam/cosine schedule.
The cache namespace includes the locked encoder revision and weight digest, and each
resolved YAML records the validated local snapshot. Only training-batch sampling and
arm-identifying output metadata differ between arms.

The preflight-selected decoder batch size is frozen across all ten arm/fold runs. Soma's
base seed 0 yields fold seeds 0 through 4. Resume through the pinned launcher so the
validated offline encoder binding remains active:

```bash
python -m examples.beetle.launch run-arm \
  --preflight data/beetle/hardware_preflight.json \
  --output-dir data/beetle/resolved \
  --arm uniform \
  --set run.run_id=<run-id>
```

Use `--set run.resume=true` instead only when exactly one compatible mirrored run exists.
The single-arm launcher accepts only these two run-lifecycle overrides; scientific recipe
changes require a newly reviewed protocol.

## 4. Report and select from development evidence

After all folds complete, pass the run-generated ROI sample-to-patient metadata and the
ten generic evidence files to the project reporter:

```bash
python -m examples.beetle.report_oof \
  --sample-patient-csv <uniform-run>/segmentation_rois/roi_manifest.csv \
  --uniform <uniform-run>/fold_{0,1,2,3,4}/confusion_evidence_tune.json \
  --class-conditioned <class-conditioned-run>/fold_{0,1,2,3,4}/confusion_evidence_tune.json \
  --output data/beetle/reports/oof_report.json

python -m examples.beetle.select_arm \
  --oof-report data/beetle/reports/oof_report.json \
  --output data/beetle/reports/arm_selection.json
```

The report includes both arms, five fold scores, pooled patient confusion evidence, the
fixed seed-0 10,000-draw patient bootstrap, and the evaluation-only 524-patient spacing
sensitivity view. The selector has no external-score argument: it recomputes each arm's
score as Fold-macro class Dice: the equal average of the five held-out tune
dataset-global mean class Dice
values and chooses exactly one (uniform is the predeclared exact-tie fallback). The
External evaluation set is used only after this file exists.

## Offline smoke

The complete development chain is executable without WSIs, gated weights, CUDA, or a
network:

```bash
python -m examples.beetle.smoke --output-dir /tmp/beetle-smoke
```

It resolves both templates from a clearly marked offline preflight, trains both samplers
over tiny cached fp16 grids, exports generic confusion evidence, assembles BEETLE OOF and
bootstrap reporting, mirrors verified recovery bundles, and selects one arm. Its
one-epoch synthetic results are operational evidence only, never publication results.

## Interpretation and publication evidence

Class-conditioned sampling balances requested-class opportunities, not pixels. Each
selected ROI stays intact and all annotated pixels enter cross-entropy plus soft Dice;
`roi_batch_sampling.json` records realized pixel exposure and repeated ROI exposure.

The three 0.657476 µm/px TCGA exceptions remain at native level 0. Soma never invents
finer tissue pixels. Their 524-patient exclusion view reuses the same predictions and is
not a separately trained result.

Fold-selected cross-validation is optimistic because each held-out tune fold selects its
own best epoch and reports that checkpoint. Comparing two arms on those same folds adds
arm-selection optimism. Patient-bootstrap intervals are conditional on the fixed OOF
predictions; they do not include training or selection uncertainty.

Publication evidence consists of the curated Manifest and checksums; hardware preflight;
encoder lock and protocol resolution; both resolved configs; dependency/GPU/run metadata;
cache metadata (not the cache payload); all ten decoder checkpoints; histories; sampler
audits; recovery manifests and hashes; generic fold confusion evidence; the two-arm OOF
and bootstrap report; and the arm-selection file. Do not redistribute Virchow2 weights.
