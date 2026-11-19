# tests/test_data_integrity.py
import pandas as pd

from dataset_generator.utils import sha256_file


def test_checksums_match(small_dataset):
    out_dir, _ = small_dataset
    lines = (out_dir / "SHA256SUMS.txt").read_text().splitlines()
    assert lines
    for line in lines:
        digest, name = line.split(maxsplit=1)
        assert sha256_file(out_dir / name.strip()) == digest


def test_modality_alignment(small_dataset):
    out_dir, _ = small_dataset
    psych = pd.read_parquet(out_dir / "psychometric_features.parquet")
    facial = pd.read_parquet(out_dir / "facial_expression_features.parquet")
    multi = pd.read_parquet(out_dir / "multimodal_features.parquet")
    key = "synthetic_observation_id"
    assert (psych[key].to_numpy() == facial[key].to_numpy()).all()
    assert (psych[key].to_numpy() == multi[key].to_numpy()).all()


def test_identifiers_do_not_encode_target(small_dataset):
    out_dir, _ = small_dataset
    meta = pd.read_parquet(out_dir / "synthetic_metadata.parquet")
    grouped = meta.groupby("synthetic_subject_id")["depression_risk"].nunique()
    assert grouped.max() == 1  # one observation per subject in the default configuration