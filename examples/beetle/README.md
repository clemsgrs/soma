# BEETLE Virchow2 validation

This directory is the tracked project protocol for the BEETLE rebuttal experiment. The
installable `soma` package supplies reusable mechanisms; this bundle owns the dataset
layout, cohort assertions, fold policy, experiment configs, reporting, and submission
rules.

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
