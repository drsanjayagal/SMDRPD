"""Probabilistic synthetic depression-risk label generation.

The label is a noisy, nonlinear function of the latent severity factor, the
latent emotional profile and the psychometric summary — never a threshold on a
single feature. Labels are SIMULATED RESEARCH LABELS; they are not diagnoses.
"""
from __future__ import annotations

import numpy as np

from .latent_variable_model import sample_latents
from .psychometric_generator import generate_psychometric
from .utils import derive_rng

CLASS_LABELS: dict[int, str] = {
    0: "lower_synthetic_risk",
    1: "moderate_synthetic_risk",
    2: "higher_synthetic_risk",
}

# Target proportions per documented scenario (lower, moderate, higher).
CLASS_SCENARIOS: dict[str, tuple[float, float, float]] = {
    "balanced": (1 / 3, 1 / 3, 1 / 3),
    "moderate_imbalance": (0.25, 0.40, 0.35),
    "high_imbalance": (0.68, 0.22, 0.10),
}


def raw_risk_score(
    rng: np.random.Generator,
    severity: np.ndarray,
    emotions: np.ndarray,
    psychometric: np.ndarray,
    nonlinear_strength: float = 0.45,
    noise_sd: float = 0.55,
) -> np.ndarray:
    """Unstandardised synthetic risk score (documented in GENERATION_METHODOLOGY.md)."""
    negative_affect = (
        -emotions[:, 0] + emotions[:, 1] + emotions[:, 3] + emotions[:, 4] + emotions[:, 6]
    ) / np.sqrt(5.0)
    psych_summary = (psychometric.mean(axis=1) - 1.33) / 0.95
    nonlinear = nonlinear_strength * np.tanh(1.2 * severity * negative_affect)
    return (
        0.85 * severity
        + 0.40 * negative_affect
        + 0.35 * psych_summary
        + nonlinear
        + rng.normal(0.0, noise_sd, size=severity.shape[0])
    )


class TargetGenerator:
    """Calibrated probabilistic three-class assignment."""

    def __init__(
        self,
        base_seed: int,
        scenario: str = "moderate_imbalance",
        calibration_size: int = 200_000,
        nonlinear_strength: float = 0.45,
        noise_sd: float = 0.55,
    ) -> None:
        if scenario not in CLASS_SCENARIOS:
            raise ValueError(f"Unknown class scenario: {scenario}")
        self.scenario = scenario
        self.nonlinear_strength = nonlinear_strength
        self.noise_sd = noise_sd
        self.target_proportions = CLASS_SCENARIOS[scenario]

        rng = derive_rng(base_seed, 991)
        severity, emotions = sample_latents(rng, calibration_size)
        psychometric = generate_psychometric(derive_rng(base_seed, 992), severity)
        scores = raw_risk_score(
            rng, severity, emotions, psychometric, nonlinear_strength, noise_sd
        )
        self.location = float(scores.mean())
        self.scale = float(scores.std() + 1e-12)
        standardised = (scores - self.location) / self.scale
        cumulative = np.cumsum(self.target_proportions)[:-1]
        self.cutpoints = np.quantile(standardised, cumulative)

    def assign(
        self,
        rng: np.random.Generator,
        severity: np.ndarray,
        emotions: np.ndarray,
        psychometric: np.ndarray,
    ) -> np.ndarray:
        scores = raw_risk_score(
            rng, severity, emotions, psychometric, self.nonlinear_strength, self.noise_sd
        )
        standardised = (scores - self.location) / self.scale
        return np.digitize(standardised, self.cutpoints).astype(np.int8)