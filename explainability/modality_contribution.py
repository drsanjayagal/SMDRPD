# explainability/modality_contribution.py
"""Modality-level contribution by ablating modality blocks (association, not causation)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataset_generator.multimodal_fusion import FACIAL_COLUMNS, PSYCHOMETRIC_COLUMNS
from evaluation.metrics import compute_metrics


def modality_contribution(model, X_test, y_test) -> pd.DataFrame:
    n_psych = len(PSYCHOMETRIC_COLUMNS)
    variants = {
        "full": X_test,
        "psychometric_only": np.hstack([X_test[:, :n_psych], np.zeros_like(X_test[:, n_psych:])]),
        "facial_only": np.hstack([np.zeros_like(X_test[:, :n_psych]), X_test[:, n_psych:]]),
    }
    rows = []
    for name, X in variants.items():
        proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        metrics = compute_metrics(y_test, model.predict(X), proba)
        rows.append({"variant": name, "macro_f1": metrics["f1_macro"], "accuracy": metrics["accuracy"]})
    return pd.DataFrame(rows)