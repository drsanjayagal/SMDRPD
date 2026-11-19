# explainability/permutation_importance.py
"""Permutation importance on any fitted scikit-learn compatible estimator."""
from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance


def compute_permutation_importance(model, X, y, feature_names, n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )