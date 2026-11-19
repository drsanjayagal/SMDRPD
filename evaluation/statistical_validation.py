# evaluation/statistical_validation.py
"""Descriptive statistics, cross-class tests and outlier reporting for synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def descriptive_statistics(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    block = frame[columns]
    return pd.DataFrame(
        {
            "mean": block.mean(),
            "median": block.median(),
            "std": block.std(),
            "variance": block.var(),
            "min": block.min(),
            "max": block.max(),
            "skewness": block.skew(),
            "kurtosis": block.kurtosis(),
            "outliers_iqr": [
                int(((block[c] < block[c].quantile(0.25) - 1.5 * (block[c].quantile(0.75) - block[c].quantile(0.25)))
                     | (block[c] > block[c].quantile(0.75) + 1.5 * (block[c].quantile(0.75) - block[c].quantile(0.25)))).sum())
                for c in columns
            ],
        }
    )


def cross_class_separation(frame: pd.DataFrame, columns: list[str], target: str = "depression_risk") -> pd.DataFrame:
    """Kruskal-Wallis p-values: separation is a property of the generator, not evidence of validity."""
    rows = []
    for column in columns:
        groups = [g[column].to_numpy() for _, g in frame.groupby(target)]
        statistic, p_value = stats.kruskal(*groups)
        rows.append({"feature": column, "kruskal_H": float(statistic), "p_value": float(p_value)})
    return pd.DataFrame(rows)