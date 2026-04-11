import os
from pathlib import Path
from slide2vec.utils.config import hf_login
from soma import Pipeline, FeatureExtractor
from soma import Dataset, Splits, FeatureStore, train
from soma import PipelineConfig, PreprocessingConfig, CacheConfig, EncoderConfig, AggregatorConfig, TaskConfig, TrainingConfig


def feature_extraction(
    dataset_csv: str,
    output_dir: Path,
    fm: str,
    cache_dir: Path | None,
    step_by_step: bool = False,
    backend: str = "auto",
):

    cache = CacheConfig(enabled=False)
    if cache_dir is not None:
        cache = CacheConfig(root_dir=cache_dir)

    tiling_dir = output_dir / "tiling"
    features_dir = output_dir / "features"

    dataset = Dataset(dataset_csv)
    preprocessing = PreprocessingConfig(tissue_threshold=0.1, backend=backend)
    encoder = EncoderConfig(name=fm, spacing_um=0.5)

    extractor = FeatureExtractor(
        dataset=dataset,
        encoder=encoder,
        preprocessing=preprocessing,
        cache=cache,
    )

    if step_by_step:
        # step by step
        # 1- preprocessing (tiling)
        # 2- feature encoding
        extractor.preprocess(tiling_dir)
        store = extractor.extract(features_dir, tiling_dir=tiling_dir)
    else:
        # full pipeline: tiling + encoding
        store = extractor.run(features_dir)

    return store


def run_training(
    dataset_csv: str,
    splits_csv: str,
    output_dir: Path,
    fm: str,
    cache_dir: Path | None,
    step_by_step: bool = False,
    backend: str = "auto",
):

    dataset = Dataset(dataset_csv)
    splits = Splits(splits_csv, dataset)

    store = feature_extraction(
        dataset_csv=dataset_csv,
        output_dir=output_dir,
        fm=fm,
        cache_dir=cache_dir,
        step_by_step=step_by_step,
        backend=backend,
    )

    # train all folds + summarize
    result = train(
        feature_store=store,
        dataset=dataset,
        splits=splits,
        aggregator=AggregatorConfig(name="abmil"),
        task=TaskConfig(name="ordinal_classification"),
        training=TrainingConfig(epochs=50),
        output_dir=output_dir,
    )


def run_pipeline(
    dataset_csv: str,
    splits_csv: str,
    output_dir: Path,
    fm: str,
    cache_dir: Path | None,
    backend: str = "auto",
):
    cache = CacheConfig(enabled=False)
    if cache_dir is not None:
        cache = CacheConfig(root_dir=cache_dir)

    encoder = EncoderConfig(name=fm)

    config = PipelineConfig(
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        output_dir=output_dir,
        dataset_type="slide",
        cache=cache,
        encoder=encoder,
        preprocessing=PreprocessingConfig(backend=backend),
        aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 256}),
        training=TrainingConfig(learning_rate=1e-4, epochs=50),
    )

    results = Pipeline(config).run()


if __name__ == "__main__":
    os.environ["HF_TOKEN"] = "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"
    hf_login()

    dataset_csv = "/data/pathology/projects/clement/notebooks/pathauto/datasets/panda-safe.sample.csv"
    splits_csv = "/data/pathology/projects/clement/notebooks/pathauto/datasets/panda-safe.splits.csv"

    output_dir = Path("output/panda-ordinal").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path("shared/feature_cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    fm = "virchow2"
    backend = "cucim"

    print("*=*"*10)
    print("TRAINING RUN")
    print("*=*"*10)
    run_training(
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        output_dir=output_dir,
        fm=fm,
        cache_dir=cache_dir,
        step_by_step=True,
        backend=backend,
    )

    # print("*=*"*10)
    # print("PIPELINE RUN")
    # print("*=*"*10)
    # run_pipeline(
    #     dataset_csv=dataset_csv,
    #     splits_csv=splits_csv,
    #     output_dir=output_dir,
    #     fm=fm,
    #     cache_dir=cache_dir,
    #     backend=backend,
    # )
