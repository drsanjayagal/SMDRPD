"""Ten synthetic psychometric assessment features on a documented 0-3 scale.

These are simulated research variables generated from an ordinal Gaussian
copula. They are NOT validated clinical questionnaire items and must not be
interpreted as PHQ-9 or any other validated instrument.
"""
from __future__ import annotations

import numpy as np

# (feature_id, name, description, severity_loading, difficulty_shift)
PSYCHOMETRIC_FEATURES: tuple[tuple[str, str, str, float, float], ...] = (
    ("P01", "Depressed Mood", "Simulated depressed mood intensity", 0.80, -0.10),
    ("P02", "Loss of Interest", "Simulated reduction in interest", 0.74, 0.05),
    ("P03", "Sleep Disturbance", "Simulated sleep-related disturbance", 0.66, 0.00),
    ("P04", "Fatigue", "Simulated fatigue level", 0.70, -0.05),
    ("P05", "Concentration Difficulty", "Simulated concentration problems", 0.72, 0.08),
    ("P06", "Appetite Disturbance", "Simulated appetite changes", 0.58, 0.12),
    ("P07", "Self-Esteem Concerns", "Simulated negative self-perception", 0.76, 0.02),
    ("P08", "Psychomotor Changes", "Simulated psychomotor variation", 0.55, 0.15),
    ("P09", "Social Withdrawal", "Simulated social disengagement", 0.68, 0.06),
    ("P10", "Emotional Distress", "Simulated emotional distress", 0.82, -0.08),
)

# Ordinal cut points on the standardised latent response (0/1/2/3 boundaries).
BASE_THRESHOLDS = (-0.62, 0.14, 0.92)
# Shared residual factor -> residual inter-item correlation of 0.35**2 = 0.1225.
RESIDUAL_COMMON = 0.35

PSYCHOMETRIC_COLUMNS: tuple[str, ...] = tuple(f[0] for f in PSYCHOMETRIC_FEATURES)


def generate_psychometric(rng: np.random.Generator, severity: np.ndarray) -> np.ndarray:
    """Map a synthetic severity factor to ten correlated ordinal features."""
    severity = np.asarray(severity, dtype=np.float64)
    n = severity.shape[0]
    common = rng.standard_normal(n)
    out = np.empty((n, len(PSYCHOMETRIC_FEATURES)), dtype=np.float32)
    for j, (_, _, _, loading, shift) in enumerate(PSYCHOMETRIC_FEATURES):
        residual = np.sqrt(max(1.0 - loading**2, 1e-6)) * (
            RESIDUAL_COMMON * common
            + np.sqrt(1.0 - RESIDUAL_COMMON**2) * rng.standard_normal(n)
        )
        latent = loading * severity + residual
        cuts = np.asarray(BASE_THRESHOLDS) + shift
        out[:, j] = np.digitize(latent, cuts).astype(np.float32)
    return out


def describe_features() -> list[dict[str, object]]:
    return [
        {
            "feature_id": fid,
            "feature_name": name,
            "description": desc,
            "severity_loading": loading,
            "difficulty_shift": shift,
            "scale": "0-3 ordinal (synthetic)",
        }
        for fid, name, desc, loading, shift in PSYCHOMETRIC_FEATURES
    ]