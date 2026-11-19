# tests/test_dataset_generator.py
import numpy as np
import pandas as pd


def test_record_count(small_dataset):
    out_dir, summary = small_dataset
    frame = pd.read_parquet(out_dir / "multimodal_features.parquet")
    assert len(frame) == 1500
    assert summary["records"] == 1500


def test_no_missing_values(small_dataset):
    out_dir, _ = small_dataset
    frame = pd.read_parquet(out_dir / "multimodal_features.parquet")
    assert not frame.isna().any().any()


def test_target_values(small_dataset):
    out_dir, _ = small_dataset
    frame = pd.read_parquet(out_dir / "multimodal_features.parquet")
    assert set(frame["depression_risk"].unique()) <= {0, 1, 2}
    assert frame["depression_risk"].nunique() == 3


def test_psychometric_range(small_dataset):
    out_dir, _ = small_dataset
    frame = pd.read_parquet(out_dir / "psychometric_features.parquet")
    psych_columns = [c for c in frame.columns if c.startswith("P")]
    values = frame[psych_columns].to_numpy()
    assert values.min() >= 0 and values.max() <= 3
    assert np.allclose(values, np.round(values))


def test_batch_boundaries_preserve_rows(small_dataset):
    out_dir, _ = small_dataset
    frame = pd.read_parquet(out_dir / "multimodal_features.parquet")
    ids = frame["synthetic_observation_id"]
    assert ids.is_unique
    assert ids.iloc[0] == "O00000000"
    assert ids.iloc[-1] == "O00001499"