# tests/test_feature_dimensions.py
import pandas as pd

from dataset_generator.facial_feature_generator import FACIAL_COLUMNS
from dataset_generator.psychometric_generator import PSYCHOMETRIC_COLUMNS


def test_psychometric_columns(small_dataset):
    out_dir, _ = small_dataset
    frame = pd.read_parquet(out_dir / "psychometric_features.parquet")
    assert len(PSYCHOMETRIC_COLUMNS) == 10
    assert all(c in frame.columns for c in PSYCHOMETRIC_COLUMNS)


def test_facial_columns(small_dataset):
    out_dir, _ = small_dataset
    frame = pd.read_parquet(out_dir / "facial_expression_features.parquet")
    assert len(FACIAL_COLUMNS) == 512
    assert all(c in frame.columns for c in FACIAL_COLUMNS)


def test_multimodal_dimension_is_522(small_dataset):
    out_dir, _ = small_dataset
    frame = pd.read_parquet(out_dir / "multimodal_features.parquet")
    feature_columns = [c for c in frame.columns if c[0] in {"P", "F"} and c[1:].isdigit()]
    assert len(feature_columns) == 522