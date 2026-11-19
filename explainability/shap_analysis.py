# explainability/shap_analysis.py
"""Optional SHAP wrapper. Import errors are raised with an actionable message."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_shap_values(model, X, feature_names, max_samples: int = 1000, seed: int = 0):
    try:
        import shap
    except ImportError as exc:  # pragma: no cover
        raise ImportError("shap is not installed; pip install shap") from exc

    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    if X.shape[0] > max_samples:
        X = X[rng.choice(X.shape[0], max_samples, replace=False)]

    estimator = model
    if hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]
        X = model[:-1].transform(X) if len(model.named_steps) > 1 else X

    try:
        explainer = shap.TreeExplainer(estimator)
    except Exception:
        explainer = shap.Explainer(estimator.predict, X)
    values = explainer(X).values if hasattr(explainer(X), "values") else explainer.shap_values(X)
    values = np.asarray(values)
    if values.ndim == 3:  # (samples, features, classes)
        values = np.abs(values).mean(axis=2)
    return pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": np.abs(values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)