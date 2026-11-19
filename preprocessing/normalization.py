# preprocessing/normalization.py
"""Reusable standardisation transform with persistence."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


class StandardizeTransform:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardizeTransform":
        X = np.asarray(X, dtype=np.float64)
        self.mean = X.mean(axis=0)
        self.scale = X.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("Transform is not fitted")
        return ((np.asarray(X, dtype=np.float64) - self.mean) / self.scale).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str | Path) -> None:
        joblib.dump({"mean": self.mean, "scale": self.scale}, path)

    @classmethod
    def load(cls, path: str | Path) -> "StandardizeTransform":
        data = joblib.load(path)
        obj = cls()
        obj.mean, obj.scale = data["mean"], data["scale"]
        return obj