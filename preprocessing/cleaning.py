# preprocessing/cleaning.py
"""Cleaning helpers (no silent data destruction: everything is reported)."""
from __future__ import annotations

import pandas as pd

from dataset_generator.multimodal_fusion import ID_COLUMNS


def drop_duplicate_observations(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(frame)
    cleaned = frame.drop_duplicates(subset=list(ID_COLUMNS), keep="first")
    return cleaned, before - len(cleaned)


def report_missing(frame: pd.DataFrame) -> pd.Series:
    return frame.isna().sum().sort_values(ascending=False)


def enforce_feature_dtypes(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out[columns] = out[columns].astype("float32")
    return out