from soma import (
    AggregatorConfig,
    CacheConfig,
    EncoderConfig,
    Pipeline,
    PipelineConfig,
    TrainingConfig,
)


def main() -> None:
    dataset_csv = "/path/to/dataset.csv"
    splits_csv = "/path/to/splits.csv"
    output_dir = "output/panda"

    config = PipelineConfig(
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        output_dir=output_dir,
        cache=CacheConfig(root_dir="output/cache"),
        encoder=EncoderConfig(name="h0-mini"),
        aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 256}),
        training=TrainingConfig(learning_rate=1e-4, epochs=50),
    )
    Pipeline(config).run()


if __name__ == "__main__":
    main()
