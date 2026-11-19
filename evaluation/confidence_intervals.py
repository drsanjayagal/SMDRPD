# evaluation/confidence_intervals.py
"""Bootstrap confidence intervals for a scalar metric."""
from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        values.append(metric_fn(y_true[idx], y_pred[idx]))
    return {
        "point": float(metric_fn(y_true, y_pred)),
        "lower": float(np.quantile(values, alpha / 2)),
        "upper": float(np.quantile(values, 1 - alpha / 2)),
    }