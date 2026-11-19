# preprocessing/feature_selection.py
"""Feature filters for ablation experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd


def variance_filter(frame: pd.DataFrame, threshold: float = 1e-6) -> list[str]:
    variances = frame.var(numeric_only=True)
    return variances[variances > threshold].index.tolist()


def correlation_filter(frame: pd.DataFrame, threshold: float = 0.98) -> list[str]:
    corr = frame.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [c for c in upper.columns if not (upper[c] > threshold).any()]