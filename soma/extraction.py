"""FeatureExtractor — preprocesses slides and extracts features."""

from __future__ import annotations

from pathlib import Path

from soma.config import EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset
from soma.encoders.distributed import SlideTask, extract_dataset
from soma.features import FeatureStore
from soma.preprocessing.io import load_tiling_result, save_tiling_result
from soma.preprocessing.tiling import generate_tiles
from soma.preprocessing.tissue import detect_contours, segment_tissue
from soma.wsi.reader import open_slide


class FeatureExtractor:
    """Preprocesses slides and extracts features for all samples in a dataset.

    Args:
        dataset: Dataset with sample records (image_path, sample_id).
        encoder: Encoder configuration (model name, precision, batch_size, etc.).
        preprocessing: Preprocessing configuration (tiling, tissue segmentation).
    """

    def __init__(
        self,
        dataset: Dataset,
        encoder: EncoderConfig,
        preprocessing: PreprocessingConfig = PreprocessingConfig(),
    ) -> None:
        self._dataset = dataset
        self._encoder = encoder
        self._preprocessing = preprocessing

    def preprocess(
        self,
        output_dir: str | Path,
        *,
        skip_existing: bool = True,
        backend: str = "auto",
    ) -> None:
        """Preprocess all slides: tissue segmentation -> contours -> tiles.

        Saves TilingResult artifacts per sample to output_dir.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = self._preprocessing

        for record in self._dataset.samples.values():
            npz_path = output_dir / f"{record.sample_id}.coordinates.npz"
            if skip_existing and npz_path.exists():
                continue

            slide = open_slide(record.image_path, backend)
            try:
                w, h = slide.dimensions
                ds = cfg.seg_downsample
                thumb = slide.get_thumbnail((w // ds, h // ds))

                mask = segment_tissue(thumb, method=cfg.tissue_method)

                contours = detect_contours(
                    mask,
                    slide_dimensions=slide.dimensions,
                    ref_tile_size_px=cfg.ref_tile_size_px,
                    requested_spacing_um=cfg.requested_spacing_um,
                    a_t=cfg.a_t,
                    a_h=cfg.a_h,
                    max_holes_per_contour=cfg.max_holes_per_contour,
                )

                tiling = generate_tiles(
                    slide_dimensions=slide.dimensions,
                    contours=contours,
                    requested_tile_size_px=cfg.requested_tile_size_px,
                    requested_spacing_um=cfg.requested_spacing_um,
                    base_spacing_um=slide.spacing,
                    level_downsamples=slide.level_downsamples,
                    overlap=cfg.overlap,
                    min_tissue_fraction=cfg.min_tissue_fraction,
                    tolerance=cfg.tolerance,
                )

                save_tiling_result(tiling, output_dir, record.sample_id)
            finally:
                slide.close()

    def extract(
        self,
        output_dir: str | Path,
        *,
        tiling_dir: str | Path | None = None,
        skip_existing: bool = True,
        num_gpus: int | None = None,
        backend: str = "auto",
    ) -> FeatureStore:
        """Extract features for all preprocessed slides.

        Args:
            output_dir: Directory to save feature .pt files.
            tiling_dir: Directory with tiling artifacts. If None, preprocesses
                in-memory (saves tiling to output_dir/.tiling/).
            skip_existing: Skip samples with existing .pt files.
            num_gpus: Number of GPUs for distributed extraction.
            backend: WSI backend for slide reading.

        Returns:
            FeatureStore pointing to output_dir.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if tiling_dir is None:
            tiling_dir = output_dir / ".tiling"
            self.preprocess(tiling_dir, skip_existing=skip_existing, backend=backend)

        tiling_dir = Path(tiling_dir)

        # Build SlideTask list from tiling artifacts
        slide_tasks = []
        for record in self._dataset.samples.values():
            npz_path = tiling_dir / f"{record.sample_id}.coordinates.npz"
            meta_path = tiling_dir / f"{record.sample_id}.coordinates.meta.json"
            tiling = load_tiling_result(npz_path, meta_path)
            slide_tasks.append(
                SlideTask(
                    slide_path=str(record.image_path),
                    tiling_result=tiling,
                    slide_id=record.sample_id,
                )
            )

        extract_dataset(
            encoder_name=self._encoder.name,
            slides=slide_tasks,
            output_dir=output_dir,
            batch_size=self._encoder.batch_size,
            num_workers=self._encoder.num_workers,
            precision=self._encoder.precision,
            skip_existing=skip_existing,
            num_gpus=num_gpus,
            backend=backend,
        )

        return FeatureStore(output_dir)

    def run(
        self,
        output_dir: str | Path,
        *,
        skip_existing: bool = True,
        num_gpus: int | None = None,
        backend: str = "auto",
    ) -> FeatureStore:
        """Preprocess + extract in one call.

        Tiling artifacts saved to output_dir/.tiling/.
        Feature .pt files saved to output_dir/.
        """
        output_dir = Path(output_dir)
        tiling_dir = output_dir / ".tiling"
        self.preprocess(tiling_dir, skip_existing=skip_existing, backend=backend)
        return self.extract(
            output_dir,
            tiling_dir=tiling_dir,
            skip_existing=skip_existing,
            num_gpus=num_gpus,
            backend=backend,
        )
