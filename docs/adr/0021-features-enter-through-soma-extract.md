# Features enter through `soma.extract`; no public type carries a path

Features enter soma through one verb, `soma.extract(images, *, encoder, cache, annotations=None, unit="slide", rois=None, execution=None, extraction=None, dense=None, spacing_policy="strict", output_dir)`. It reads an `ImageManifest` and, to sample ROIs from whole slides, an `AnnotationManifest`; it never reads records. It returns the source the extraction produced: a `SetSource`, or a `GridSource` when `dense` is given. `Pipeline` extracts through it. The legacy bridge is deleted whole: `soma/data/_legacy.py` (the fat `LegacyRecord`, `LegacySamples`, `LegacyFolds`, `records_for_pipeline`, `project_folds`) and `soma/pipeline/_legacy.py`, together with `FeatureExtractor`, `FeatureExtractionResult` and `attach_source`. There are no aliases.

## Why

Slice 1 took the file paths off `SampleRecord` (paths live in the manifests), but the extraction stack, the cache layer and the pipeline builders still read one record per sample with paths next to targets. A bridge rebuilt that record from the cohort and the manifests, and annotation-sampled ROIs added a second layer: each ROI's origin, its stored mask crop and its parent slide id were record fields, written to an ROI table, read back, and used to project the slide folds onto the ROIs. Every consumer of a path or an ROI address depended on that bridge, so the data contracts of slice 1 held only at the surface.

## Decisions

- **One verb, manifests in, a source out.** `soma.extract` takes the manifests and returns the source. The cache layer and the slide2vec adapter read `ImageEntry` fields (`image_path`, `mask_path`, `spacing_at_level_0`, `coordinates_path`, `patient_id`) straight from the manifest, and ROI sampling reads `label_mask_path` from the annotation manifest. The verb's keywords mirror the `features` section (ADR 0020); `unit` (whole slide or pre-cropped image) and `annotations` are what the plan derives from the cohort. `soma.extract` is also the namespace of `live_encoder`: the module is callable.
- **ROI sampling is extraction-internal.** A private `RoiEntry` holds an ROI's parent image row, its annotation raster, its level-0 origin and the mask crop sampling stored. The grid extractor encodes it and the label maps read its crop. What leaves extraction is a `GridSource` keyed by ROI id whose `rois` (a `RoiSet`) name each ROI's parent sample (`children`) and serve the ROIs' label maps and pixel windows. `rois=` passes such a set back to encode the same ROIs again without sampling. No public type exposes an ROI's geometry.
- **Splits are sample-level; children inherit.** `Cohort.derived(children)` replaces the fold projection: one record per child (tile, ROI) with its parent sample's targets, patient and metadata, in every split its parent sits in. A sample is usually a slide but can be a patient or anything else; `sample_id` is the key.
- **The builders use the public readers.** Label maps come from `LabelMapSource.from_manifests` (or, for ROIs, from the `RoiSet`), points from `PointSource.from_manifest`, flat tiles from `TileImageSource.from_manifest`, and attention heatmaps read the `ImageManifest`. Curators write the single CSV and are read back through `Cohort.from_csv`, `ImageManifest` and `AnnotationManifest`.
- **Provenance names the cohort CSV the run knows.** The dense source's provenance records the cohort CSV (and, for ROIs, the ROI table extraction writes); nothing else carries the CSV path.
- **Cache keys and feature identity are unchanged.** A test pins the literal keys and per-sample identities of a synthetic slide, tile and ROI, written before any code moved; the ROI identity is its ROI id over its slide's image row, as before.

## Considered and rejected

- **Keep the persisted ROI table as a cohort input.** Handing an extraction's ROI table to a run as its records is what required ROI addresses on records. A run is refused when its records carry ROI columns; it samples from the slide manifest, and the ROI sampling cache makes a relaunch free.
- **Return extraction artifacts beside the source.** The source is what a run consumes; `extraction_provenance.json` still records the extraction's kind, its zero-ROI slides and its artifacts for a reader who wants them.
- **Name the verb `soma.extract_features` to avoid a callable namespace.** The design names the verb and the namespace `soma.extract`; one name for both keeps the root surface small.

## Consequences

- `FeatureExtractor`, `FeatureExtractionResult`, `FeatureProvenance` and `ExtractionArtifacts` leave the root; extraction code calls `soma.extract`.
- `Pipeline.dataset` and `Pipeline.splits` are gone; `Pipeline.cohort` and the two manifests remain.
- The cache layer's resolvers take `images=` (and the ROI sampling cache `annotations=`) instead of `dataset=`.
- A composite run extracts each member through `soma.extract`; with the cache disabled, a member's features land under its member directory's `features/`.
