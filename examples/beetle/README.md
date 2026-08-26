# BEETLE Virchow2 validation

This directory is the tracked project protocol for the BEETLE rebuttal experiment. The
installable `soma` package supplies reusable mechanisms; this bundle owns the dataset
layout, cohort assertions, fold policy, experiment configs, reporting, and submission
rules.

BEETLE is not yet a Benchmark because its official manuscript remains under review. A
published, sufficiently fixed evaluation contract with reference evidence may make it a
Benchmark, but this repository will continue to treat it as a side project unless
maintainers separately promote it to a Built-in Benchmark.

Curate the strict development cohort:

```bash
python -m examples.beetle.curate --beetle-root data/beetle
```

Run a small non-publication curation smoke test:

```bash
python -m examples.beetle.curate --beetle-root data/beetle --slides 4
```

The base segmentation recipe is in `configs/segmentation.yaml`. `CONTEXT.md` defines
the project language and `decisions/` records project-level protocol decisions.

## OOF confusion report

The segmentation recipe enables Soma's dataset-neutral selected-checkpoint evidence
export. Each fold writes `confusion_evidence_tune.json`, containing one arbitrary-class
confusion matrix per held-out sample. Assemble the two sampling arms only after all five
folds have completed:

```bash
python -m examples.beetle.report_oof \
  --sample-patient-csv data/beetle/runs/uniform/segmentation_rois/roi_manifest.csv \
  --uniform data/beetle/runs/uniform/fold_{0,1,2,3,4}/confusion_evidence_tune.json \
  --class-conditioned data/beetle/runs/class_conditioned/fold_{0,1,2,3,4}/confusion_evidence_tune.json \
  --output data/beetle/reports/oof_report.json
```

The report script—not the installed `soma` package—maps samples to patients, asserts
the two named arms and folds 0–4, checks the 527-patient primary and 524-patient
spacing-sensitivity cohorts, removes the three declared native-spacing exceptions only
for the sensitivity view, and computes percentile intervals from 10,000 whole-patient
bootstrap draws with seed 0.

The `--sample-patient-csv` source is the run-generated ROI metadata written during
dense extraction; the reporter reads only its `sample_id` and `patient_id` columns.
The backing file retains Soma's historical `roi_manifest.csv` filename, but here it is
strictly sample-to-patient report metadata: it neither defines training samples/splits
nor acts as the canonical dataset Manifest.

### Migration from the earlier patient OOF artifact

`evaluation.patient_oof` has been replaced by the identity-neutral boolean
`evaluation.save_segmentation_confusion_evidence`. Core no longer accepts arm names,
patient IDs, cohort sizes, spacing exceptions, or bootstrap settings. The former
per-fold `patient_confusions_tune.json` and run-level `patient_oof_report.json` are
replaced by per-fold `confusion_evidence_tune.json` plus the project-owned report command
above. To migrate an existing completed run, enable the new boolean and resume the same
pinned run. Soma reloads each fold's selected checkpoint and evaluates only the effective
held-out split to create the missing evidence; it does not retrain, rewrite training
history, or treat this operational export flag as training identity. The former
patient-grouped artifacts are not converted because they cannot recover sample-level
sufficient statistics.
