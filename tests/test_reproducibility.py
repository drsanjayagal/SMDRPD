# tests/test_reproducibility.py
import pandas as pd
from pandas.testing import assert_frame_equal


def test_same_seed_same_data(small_dataset, second_run):
    first_dir, _ = small_dataset
    second_dir, _ = second_run
    for name in ("psychometric_features", "facial_expression_features",
                 "multimodal_features", "synthetic_metadata"):
        left = pd.read_parquet(first_dir / f"{name}.parquet")
        right = pd.read_parquet(second_dir / f"{name}.parquet")
        assert_frame_equal(left, right, check_like=False)


def test_different_seed_different_data(small_dataset, tmp_path):
    import copy

    from dataset_generator.generate_dataset import DEFAULT_CONFIG, generate_dataset

    first_dir, _ = small_dataset
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["target"]["calibration_size"] = 20000
    generate_dataset(config, tmp_path, n_records=300, seed=9999, batch_size=300,
                     formats=("parquet",), sample_records=50)
    left = pd.read_parquet(first_dir / "psychometric_features.parquet").head(300)
    right = pd.read_parquet(tmp_path / "psychometric_features.parquet")
    assert not left["P01"].equals(right["P01"])