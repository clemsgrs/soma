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
CUDA node, run the production gate with a descending list sized for that GPU:

```bash
python -m examples.beetle.preflight \
  --snapshot-path <hf-cache>/models--paige-ai--Virchow2/snapshots/3158645804b69e3f3bc4439d4116edddf0840a72 \
  --dataset-csv data/beetle/curated_slide_manifest/dataset.csv \
  --splits-csv data/beetle/curated_slide_manifest/splits.csv \
  --batch-size-candidates 64 32 16 8 4 \
  --device cuda:0 \
  --output data/beetle/hardware_preflight.json
```

The candidate list is configurable, positive, and strictly descending. Each decoder
candidate performs complete optimizer steps; the largest passing batch is recorded and
frozen across both arms and all folds. The encoder extraction batch remains the separately
recorded protocol value. The command obtains the repository revision from the pinned local
snapshot and computes the weight digest itself. Keep the gated weights local.
`encoder_lock.json` records the expected identity; the resolver requires the snapshot's
Timm `config.json`, hashes the actual weight file in `snapshot_path`, and rejects a
revision, checksum, geometry, or batch-choice mismatch.
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

Populate and validate the shared dense cache before decoder training:

```bash
python -m examples.beetle.extract_cache \
  --config data/beetle/resolved/uniform.yaml \
  --work-dir data/beetle/cache_extraction \
  --output data/beetle/cache_extraction.json
```

Rerun the same command with a separate work directory and audit output to prove that ROI
sampling and dense extraction resume from the completed shared cache. The two audits must
match on config digest, feature directory, slide/ROI counts, geometry, and bytes; only the
work directory may differ. The completion audit enumerates tensor/sidecar coverage and
leaves the large per-file payload digest as an explicit companion-manifest step.

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

## 5. Build and validate the External submission

The paper lead supplies the External ROI directory and a reviewed schema-v1
ROI-to-WSI sidecar. The sidecar is JSON with exactly 170 rows:

```json
{
  "schema_version": 1,
  "rois": [
    {
      "roi_filename": "<exact organizer PNG basename>",
      "patient_id": "<independent patient ID>",
      "source_wsi": "<source WSI ID>",
      "native_spacing_um": 0.5,
      "width": 2048,
      "height": 1536
    }
  ]
}
```

Do not infer spacing, WSI identity, or patient identity from a filename. Run inference
only after `arm_selection.json` exists, against that arm's completed run directory:

```bash
python -m examples.beetle.external_submission infer \
  --selection data/beetle/reports/arm_selection.json \
  --run-dir <selected-arm-run> \
  --protocol-resolution data/beetle/resolved/protocol_resolution.json \
  --roi-dir data/beetle/external/rois \
  --roi-sidecar data/beetle/external/roi_to_wsi.json \
  --output-dir data/beetle/external/submission_pngs \
  --audit data/beetle/external/submission_audit.json \
  --zip data/beetle/external/submission.zip
```

The run must carry `beetle` and selected-arm tags and contain
`fold_0/best_model.pt` through `fold_4/best_model.pt`. The command loads one shared
frozen encoder plus all five decoders and averages their per-pixel softmaxes. The protocol
resolution is revalidated against `encoder_lock.json`, including the weight digest and
immutable local Hub binding, before the encoder loads in offline mode. Each flat
ROI inherits native spacing from the sidecar. The locked recipe uses a 10% relative
tolerance around 0.5 µm/px. Inputs within that tolerance stay native; finer inputs
outside tolerance may be area-downsampled; coarser inputs always stay native and are
never upsampled. Annotation-mask reads are pinned to OpenSlide because CuCIM can decode
sparse intermediate TIFF pyramid levels as background. Predictions are mapped back to
the exact sidecar width and height.

The generated masks are single-channel grayscale uint8 PNGs in the organizer vocabulary:
`1=other`, `2=non-invasive epithelium`, `3=invasive epithelium`, `4=necrosis`.
Validation requires exactly the sidecar's 170 basenames, checks dimensions, type, and
labels, and writes a deterministic ZIP whose entries have no directory prefix. To
revalidate existing masks without inference:

```bash
python -m examples.beetle.external_submission validate \
  --roi-sidecar data/beetle/external/roi_to_wsi.json \
  --output-dir data/beetle/external/submission_pngs \
  --zip data/beetle/external/submission.zip
```

`submission_audit.json` records the selected arm, five checkpoint digests, every ROI's
spacing decision and native/output dimensions, and the ZIP digest. It explicitly records
that hidden labels were not used.

Only after the submission predictions are fixed, if the paper lead supplies the
sequestered masks, compute the 54-patient confusion/bootstrap report:

```bash
python -m examples.beetle.external_submission evaluate \
  --roi-sidecar data/beetle/external/roi_to_wsi.json \
  --predictions-dir data/beetle/external/submission_pngs \
  --labels-dir <paper-lead-supplied-label-directory> \
  --output data/beetle/reports/external_report.json
```

All nested ROIs are grouped by the sidecar's `patient_id`; the report uses the same
seed-0, 10,000-draw whole-patient bootstrap as the development report. Hidden labels and
metrics are neither fabricated nor required to infer, validate, or ZIP the submission.

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
and bootstrap report; the arm-selection file; validated External ROI sidecar; all 170
submission PNGs; the flat submission ZIP; and `submission_audit.json`. If sequestered
labels are supplied, include `external_report.json`; otherwise it is absent by design and
submission generation remains complete. Do not redistribute Virchow2 weights.
