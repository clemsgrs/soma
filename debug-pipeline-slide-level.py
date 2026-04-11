import os
from pathlib import Path
from slide2vec.utils.config import hf_login
from soma import Dataset, Splits, Pipeline
from soma import PipelineConfig, PreprocessingConfig, EncoderConfig, TaskConfig, TrainingConfig, EvalConfig


def main():
    os.environ["HF_TOKEN"] = "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"
    hf_login()

    dataset_csv = "/data/pathology/projects/clement/notebooks/pathauto/datasets/panda-safe.sample.debug.csv"
    splits_csv = "/data/pathology/projects/clement/notebooks/pathauto/datasets/panda-safe.splits.debug.csv"

    output_root = Path("output").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    fm = "prism"
    backend = "cucim"
    task_type = "multiclass_classification"
    requested_spacing_um = 0.5
    # task_type = "ordinal_classification"
    metrics = ["qwk"]

    config = PipelineConfig(
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        output_root=output_root,
        encoder=EncoderConfig(name=fm),
        preprocessing=PreprocessingConfig(backend=backend),
        aggregator=None,
        task=TaskConfig(name=task_type),
        training=TrainingConfig(learning_rate=1e-4, epochs=50, batch_size=1),
        eval=EvalConfig(metrics=metrics),
    )

    results = Pipeline(config).run()


if __name__ == "__main__":
    main()
