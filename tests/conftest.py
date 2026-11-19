# tests/conftest.py
from __future__ import annotations

import copy

import pytest

from dataset_generator.generate_dataset import DEFAULT_CONFIG, generate_dataset


@pytest.fixture(scope="session")
def small_dataset(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("smdrpd_small")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["target"]["calibration_size"] = 20000
    summary = generate_dataset(
        config,
        out_dir,
        n_records=1500,
        seed=4242,
        batch_size=400,
        scenario="moderate_imbalance",
        formats=("parquet",),
        sample_records=100,
    )
    return out_dir, summary


@pytest.fixture(scope="session")
def second_run(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("smdrpd_small_repeat")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["target"]["calibration_size"] = 20000
    summary = generate_dataset(
        config,
        out_dir,
        n_records=1500,
        seed=4242,
        batch_size=400,
        scenario="moderate_imbalance",
        formats=("parquet",),
        sample_records=100,
    )
    return out_dir, summary